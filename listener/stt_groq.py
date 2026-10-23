"""Fast cloud speech-to-text: Whisper on Groq, with Moonshine as the fallback.

Goes in: listener/stt_groq.py   (new file)

Groq runs whisper-large-v3-turbo fast -- typically well under a second for a
spoken command -- and more accurately than the local model. main.py tries it
first and falls back to Moonshine (offline, listener/stt_moonshine.py) when
it isn't configured, the network is down, or Groq refuses.

ITS OWN KEY
-----------
Uses GROQ_TTS_API_KEY (or GROQ_STT_API_KEY), NOT the GROQ_API_KEY the router
uses, so transcription and command parsing draw on separate accounts' limits.
Unset: Moonshine only, nothing leaves the machine.

PRIVACY
-------
With a key set, what you say after "hey Jarvis" is sent to Groq to be
transcribed. STT_ENGINE=moonshine keeps hearing fully offline even with a key.

WHISPER AND SILENCE
-------------------
Whisper invents text for near-silence ("Thank you.", "Thanks for watching!").
Audio that's too quiet isn't sent at all, and those stock phrases are dropped
when the clip is short -- so a false wake word returns "" like Moonshine does.
"""

import io
import os
import threading
import time
import wave

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo").strip() or "whisper-large-v3-turbo"
LANGUAGE = os.getenv("STT_LANGUAGE", "en").strip() or None
SAMPLE_RATE = 16000
TIMEOUT_SECONDS = 8.0
RATE_LIMIT_COOLDOWN = 60.0
MIN_RMS = 120.0             # quieter than this (int16 RMS) is treated as silence

_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks for watching", "thanks for watching!",
    "thank you for watching", "thank you for watching.", "you", "bye", "bye.",
    "so", ".", "okay.", "uh", "um",
}

_client = None
_lock = threading.Lock()
_cool_until = 0.0
_disabled = ""


def _key():
    return (os.getenv("GROQ_TTS_API_KEY") or os.getenv("GROQ_STT_API_KEY") or "").strip()


def _forced_offline():
    return os.getenv("STT_ENGINE", "").strip().lower() == "moonshine"


def configured():
    """A key is set and hearing isn't forced offline -- Groq is the main ear,
    even while it's briefly cooling down after a rate limit."""
    return bool(_key()) and not _forced_offline() and not _disabled


def available():
    """configured(), and not cooling down after a rate limit right now."""
    return configured() and time.time() >= _cool_until


def status():
    if _forced_offline():
        return "off (STT_ENGINE=moonshine)"
    if not _key():
        return "off (no GROQ_TTS_API_KEY)"
    if _disabled:
        return "off ({})".format(_disabled)
    return "on ({})".format(MODEL)


def _get_client():
    global _client
    with _lock:
        if _client is None:
            from groq import Groq
            # No retries: a slow retry is worse than Moonshine answering now.
            _client = Groq(api_key=_key(), max_retries=0, timeout=TIMEOUT_SECONDS)
        return _client


def _wav(samples):
    """int16 mono 16 kHz samples -> an in-memory WAV file (no temp files)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def transcribe(audio):
    """The transcript, "" for silence. Raises when Groq can't answer, so the
    caller falls back to Moonshine."""
    global _cool_until, _disabled
    samples = np.asarray(audio, dtype=np.int16).reshape(-1)
    if not len(samples):
        return ""
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    if rms < MIN_RMS:
        return ""

    from listener.record import normalize
    samples = normalize(samples)  # after the silence check, so quiet stays quiet

    try:
        result = _get_client().audio.transcriptions.create(
            file=("speech.wav", _wav(samples)),
            model=MODEL,
            language=LANGUAGE,
            response_format="json",
            temperature=0.0,
        )
    except Exception as e:
        text = str(e).lower()
        if "401" in text or "invalid api key" in text or "403" in text:
            _disabled = "key rejected"
            print("stt: Groq rejected GROQ_TTS_API_KEY -- using Moonshine for this session")
        elif "429" in text or "rate limit" in text:
            _cool_until = time.time() + RATE_LIMIT_COOLDOWN
        raise

    text = (getattr(result, "text", "") or "").strip()
    seconds = len(samples) / float(SAMPLE_RATE)
    if text.lower().strip(" !") in _HALLUCINATIONS and seconds < 4.0:
        return ""
    return text
