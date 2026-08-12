"""Local, fully offline TTS fallback using Kokoro-ONNX.

Goes in: listener/tts_local.py

Used as an offline fallback path when cloud TTS fails. Kokoro provides
near-cloud neural voice quality directly on CPU via ONNX without
requiring C++ dependencies like misaki or espeak-ng binaries.
"""

import os
import queue
import re
import threading
import numpy as np

from listener.audio_out import StreamingPlayer

# Kokoro outputs 24kHz audio natively
SAMPLE_RATE = 24000

MODEL_PATH = os.path.join("models", "tts", "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join("models", "tts", "voices-v1.0.bin")

# Available female voices:
# 'af_bella'  -> Warm, expressive US female (Default)
# 'af_sarah'  -> Clear, natural US female
# 'af_sky'    -> Soft US female
# 'af_nicole' -> Crisp US female
VOICE_NAME = "af_sky"

_kokoro = None
_kokoro_lock = threading.Lock()

_DONE = object()  # Sentinel value indicating synthesis completion


def load_model():
    """Loads and caches the Kokoro ONNX model. Safe to call repeatedly."""
    global _kokoro
    with _kokoro_lock:
        if _kokoro is None:
            from kokoro_onnx import Kokoro

            if not os.path.exists(MODEL_PATH) or not os.path.exists(VOICES_PATH):
                raise FileNotFoundError(
                    f"Kokoro model files missing. Ensure '{MODEL_PATH}' and "
                    f"'{VOICES_PATH}' are in your root directory."
                )

            print(f"tts_local: loading Kokoro ONNX ({VOICE_NAME})...")
            _kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
            print("tts_local: ready")
        return _kokoro


def warm():
    """Pre-loads the model on a background thread at startup."""
    threading.Thread(target=load_model, daemon=True).start()


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text):
    """Splits input text into rough sentence chunks for streaming synthesis."""
    text = " ".join(str(text or "").split())
    if not text:
        return []
    pieces = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    return pieces or [text]


def _to_pcm16(samples):
    """Converts Kokoro float32 audio samples [-1.0, 1.0] to int16 PCM bytes."""
    samples_np = np.asarray(samples, dtype=np.float32)
    return np.clip(samples_np * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def _synthesise(pieces, out, stop):
    """Worker thread that synthesizes text sentence-by-sentence ahead of playback."""
    try:
        kokoro = load_model()
    except Exception as e:
        print(f"tts_local: couldn't load Kokoro-ONNX ({e})")
        out.put(_DONE)
        return

    try:
        for piece in pieces:
            if stop.is_set():
                break
            try:
                # Kokoro returns (samples, sample_rate)
                samples, _ = kokoro.create(
                    piece, voice=VOICE_NAME, speed=1.0, lang="en-us"
                )
                pcm = _to_pcm16(samples)
                if pcm:
                    out.put(pcm)
            except Exception as e:
                print(f"tts_local: synthesis failed on segment ({e})")
    finally:
        out.put(_DONE)


def speak(text, on_first_audio=None):
    """Synthesizes and plays text offline.

    Mirrors tts.speak()'s interface and behavior using StreamingPlayer.
    """
    if not text or not str(text).strip():
        return None

    pieces = _sentences(text)
    if not pieces:
        return None

    out = queue.Queue()
    stop = threading.Event()
    player = None

    worker = threading.Thread(
        target=_synthesise, args=(pieces, out, stop), daemon=True
    )
    worker.start()

    try:
        while True:
            item = out.get()
            if item is _DONE:
                break
            if player is None:
                player = StreamingPlayer(
                    samplerate=SAMPLE_RATE,
                    speed=1.0,
                    on_first_audio=on_first_audio,
                )
                player.start()
            player.feed(item)
    except Exception as e:
        print(f"tts_local: playback failed ({e})")
    finally:
        stop.set()
        try:
            while True:
                out.get_nowait()
        except queue.Empty:
            pass
        if player is not None:
            try:
                player.finish(timeout=40.0)
            except Exception as e:
                print(f"tts_local: playback didn't finish cleanly ({e})")

    if player is None:
        print("tts_local: could not produce any audio for this reply")

    return player


def speak_async(text, on_first_audio=None):
    """Fire-and-forget speech execution on a background thread."""
    thread = threading.Thread(
        target=speak,
        args=(text,),
        kwargs={"on_first_audio": on_first_audio},
        daemon=True,
    )
    thread.start()
    return thread


if __name__ == "__main__":
    # Test execution: python -m listener.tts_local
    load_model()
    speak("Kokoro female offline voice is working successfully.... Gimme that Tomato sauce! With the Potato! and that Bottle of Water!")