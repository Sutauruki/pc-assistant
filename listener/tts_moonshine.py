"""Local, fully offline TTS using the Moonshine C API (Kokoro/Piper voices).

WHY A SECOND LOCAL TIER
-----------------------
listener/tts_local.py (Kokoro-ONNX) is the classic offline fallback. Moonshine
ships the same Kokoro voices plus Piper and ZipVoice through its own portable
C core -- one shared native binary, one asset cache, GPU-free -- which makes it
a natural primary for JARVIS_TTS_LOCAL_FIRST=1 while Kokoro-ONNX remains the
deeper fallback.

PLACEMENT
---------
By default it replaces nothing. When JARVIS_TTS_LOCAL_FIRST=1, listener/tts.py
tries THIS module first (Moonshine -> Kokoro-ONNX -> cloud), matching the
JARVIS_STT_LOCAL_FIRST pattern on the STT side.

THE VOICE
---------
Moonshine resolves voices through a language+voice catalog (`kokoro_af_sky`
matches the classic Kokoro-ONNX af_sky default). Assets download on first use
into the Moonshine cache and are reused afterwards; voice() can be overridden
with MOONSHINE_VOICE.

`TextToSpeech.synthesize()` returns (samples, sample_rate) where samples are
mono float32 in roughly -1..1. That is the raw TTS output: we convert it to
int16 PCM and hand it to audio_out's StreamingPlayer, which now plays it
straight through with no time-stretch, no speed governor, no dry-buffer
machinery -- the audio that reaches the speakers is exactly what the model
produced.
"""

import os
import queue
import threading

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - .env is optional
    pass

from listener.audio_out import StreamingPlayer

DEFAULT_LANGUAGE = "en-us"

VOICE = os.getenv("MOONSHINE_VOICE", "kokoro_af_sky")

_tts = None
_tts_lock = threading.Lock()

_DONE = object()


def available():
    """True when the package is installed. The assets themselves download
    lazily on first use, so this is cheap and never downloads."""
    try:
        import moonshine_voice  # noqa: F401
        return True
    except ImportError:
        return False


def load_model():
    """Loads (and on the very first use downloads) the Moonshine TTS
    synthesizer once, then keeps it for the life of the process."""
    global _tts
    with _tts_lock:
        if _tts is not None:
            return _tts

        from moonshine_voice import TextToSpeech

        print("tts_moonshine: loading Moonshine TTS ({})...".format(VOICE))
        _tts = TextToSpeech().language(DEFAULT_LANGUAGE).voice(VOICE)
        _tts.load()
        print("tts_moonshine: ready")
        return _tts


def warm():
    """Pre-loads the synthesizer on a background thread at startup.

    No-op when the package is missing -- the same available() check speak()
    would hit anyway, just without paying the ImportError on the first reply.
    """
    if not available():
        return
    threading.Thread(target=load_model, daemon=True).start()


def _sentences(text):
    from moonshine_voice.tts import split_say_utterances
    return split_say_utterances(str(text or ""))


# Replies up to this many characters are synthesized as ONE moonshine C call.
# On this machine every call carries ~3s of fixed overhead (G2P + init) and
# runs slower than realtime, so splitting a short reply on its sentence
# boundaries makes it SLOWER: "That's great! Let me open it." is two calls
# (~19s) versus one call (~11s), and the second piece lands after a dead gap.
# Short replies are the normal case (parser replies are one sentence), so one
# call is the right trade. Only long text gets chunked, to bound call size.
_MAX_ONE_CALL_CHARS = 180


def _chunks(text):
    """Sentence pieces, coalesced so short replies stay a single C call."""
    pieces = _sentences(text)
    if not pieces:
        return []
    if len(text) <= _MAX_ONE_CALL_CHARS:
        return [str(text).strip()]
    chunks, buf = [], ""
    for piece in pieces:
        if buf and len(buf) + 1 + len(piece) > _MAX_ONE_CALL_CHARS:
            chunks.append(buf)
            buf = piece
        else:
            buf = "{} {}".format(buf, piece).strip() if buf else piece
    if buf:
        chunks.append(buf)
    return chunks


def _to_pcm16(samples):
    """Moonshine float32 (-1..1) -> int16 PCM bytes."""
    arr = np.asarray(samples, dtype=np.float32)
    return np.clip(arr * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def speak(text, on_first_audio=None):
    """Synthesizes and plays text offline. Mirrors tts_local.speak()'s
    interface: returns the StreamingPlayer, or None when nothing could be
    produced.

    The text is coalesced into the fewest possible moonshine C calls
    (_chunks): a short reply is ONE call, so a mid-reply "!" no longer means
    two fixed-overhead synthesis passes (which, on this machine, is what made
    the pause after an exclamation feel awful). Chunks are synthesized in a
    background worker while the player streams each one as it arrives, so a
    long reply still starts talking before the tail is synthesized.
    """
    if not text or not str(text).strip():
        return None

    chunks = _chunks(text)
    if not chunks:
        return None

    try:
        tts = load_model()
    except Exception as e:
        print("tts_moonshine: couldn't load Moonshine TTS ({})".format(e))
        return None

    out = queue.Queue()
    stop = threading.Event()
    player = None

    def synthesise():
        try:
            for chunk in chunks:
                if stop.is_set():
                    break
                try:
                    # (samples, sample_rate) -- raw floats straight from the
                    # Moonshine core, no post-processing.
                    samples, sr = tts.synthesize(chunk)
                    if samples is None or len(samples) == 0:
                        continue
                    rate = int(sr)
                    pcm = _to_pcm16(samples)
                    if pcm:
                        out.put((rate, pcm))
                except Exception as e:
                    print("tts_moonshine: synthesis failed on chunk ({})".format(e))
        finally:
            out.put(_DONE)

    worker = threading.Thread(target=synthesise, daemon=True)
    worker.start()

    try:
        while True:
            item = out.get()
            if item is _DONE:
                break
            rate, pcm = item
            if player is None:
                player = StreamingPlayer(
                    samplerate=rate,
                    on_first_audio=on_first_audio,
                )
                player.start()
            player.feed(pcm)
    except Exception as e:
        print("tts_moonshine: playback failed ({})".format(e))
    finally:
        stop.set()
        try:
            while True:
                out.get_nowait()
        except queue.Empty:
            pass
        if player is not None:
            try:
                player.finish(timeout=60.0)
            except Exception as e:
                print("tts_moonshine: playback didn't finish cleanly ({})".format(e))

    if player is None:
        print("tts_moonshine: could not produce any audio for this reply")

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
    # Test execution: python -m listener.tts_moonshine
    load_model()
    speak("Moonshine offline voice is working successfully. "
          "Gimme that Tomato sauce! With the Potato! And that Bottle of Water!")