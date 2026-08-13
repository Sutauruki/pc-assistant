"""Local ear: Moonshine STT, an entirely offline transcription tier.

Goes in: listener/stt_moonshine.py   (new file)

WHY A SECOND LOCAL TIER
-----------------------
Vosk (listener/stt_local.py) uses a 26-word command grammar, which is exactly
right for the wake-word era of this project and exactly wrong for a full
conversation -- "where is chrome located on my system" survives contact with
the cloud tiers, but vanishes to the one word of it Vosk knows when the router
is unplugged. Moonshine is a real open-vocabulary STT model that runs on the
CPU, no network, no key, no per-minute budget. That makes it the tier that
still works with the router unplugged *and* hears a normal sentence.

PLACEMENT
---------
By default it replaces nothing: Gemini -> OpenRouter -> Vosk stays the chain,
and Moonshine comes in as a local fallback only when JARVIS_STT_LOCAL_FIRST=1,
which puts it FIRST -- local, private, no rate limit -- and demotes the cloud
tiers to backups. Setting that env var in .env is the whole switch:

    JARVIS_STT_LOCAL_FIRST=1

THE MODEL
---------
Moonshine models are downloaded on first use and cached forever afterwards
(%LOCALAPPDATA%\\moonshine_voice\\moonshine_voice\\Cache by default; override
with MOONSHINE_VOICE_CACHE). get_model_for_language("en") resolves to the
highest-quality phone-size model for English (MEDIUM_STREAMING, a few hundred
megabytes) and verifies the download by checksum, skipping files that are
already present -- so calling it on every run is cheap.

MOONSHINE_COMPRESSION=small / tiny is supported for older machines, with
compression running overnight. Values: tiny, base, tiny_streaming,
base_streaming, small_streaming, medium_streaming.

Transcriber wants a List[float] in -1.0..1.0, not the int16 numpy array the
microphone produces -- the one thing said out loud below that matters is that
conversion, because feeding raw int16 in is how you get a wall of noise.
"""

import os
import threading
import time

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - .env is optional
    pass

SAMPLE_RATE = 16000

# Quality ladder, best first. MOONSHINE_COMPRESSION picks one; "" means the
# library's own default (highest available for the language).
_ARCHES = {
    "medium_streaming": 5,
    "small_streaming": 4,
    "base_streaming": 3,
    "tiny_streaming": 2,
    "base": 1,
    "tiny": 0,
}

_model = None
_model_arch = None
_lock = threading.Lock()


def _cached_on_disk(arch):
    """Deterministic disk check: the model's own download URL maps 1:1 to its
    cache path (cache_root / url minus scheme). Handles MOONSHINE_VOICE_CACHE
    overrides, because the library does too."""
    from moonshine_voice import download
    info = download.find_model_info("en", arch)
    url = info["download_url"]
    cached = os.path.join(
        str(download.get_cache_dir()), url.replace("https://", "")
    )
    return os.path.exists(cached)


def _transcriber():
    """Loads (and on the very first run downloads) the MoonShine model once,
    then keeps it for the life of the process. Safe to call on every turn:
    already-cached downloads are just a checksum check."""
    global _model, _model_arch
    with _lock:
        if _model is not None:
            return _model

        from moonshine_voice import download
        from moonshine_voice.transcriber import Transcriber

        wanted = os.getenv("MOONSHINE_COMPRESSION", "").strip().lower()
        arch = _ARCHES.get(wanted) if wanted else None

        already = _cached_on_disk(arch)
        if not already:
            print(
                "stt: moonshine model not cached yet -- one-time download"
                " (a few hundred MB)..."
            )

        model_path, model_arch = download.get_model_for_language(
            "en", arch, include_word_timestamps=False,
        )
        _model_arch = model_arch
        _model = Transcriber(model_path=model_path, model_arch=model_arch)
        if not already:
            print("stt: moonshine model downloaded")
        print("stt: moonshine loaded ({})".format(model_arch.name))
        return _model


def available():
    """True when the package is installed. The model itself downloads lazily
    on first transcribe, so this is cheap and never downloads."""
    try:
        import moonshine_voice  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe(audio):
    """audio: int16 numpy array, mono, 16kHz -- the recorder's own format.

    Returns the transcript, "" for silence. Raises on failure so main.py can
    fall through to the cloud tiers.
    """
    started = time.time()

    # int16 (-32768..32767) -> float (-1.0..1.0). MoonShine wants the float
    # range; feeding it raw int16 is not "close enough", it's garbage.
    # The recorder hands back a (N, 1) mono array, so flatten before
    # converting -- the API takes a flat List[float], and a tall list of
    # one-element lists makes ctypes raise "must be real number, not list".
    floats = (audio.astype(np.float32).reshape(-1) / 32768.0).tolist()

    result = _transcriber().transcribe_without_streaming(
        audio_data=floats, sample_rate=SAMPLE_RATE
    )
    text = " ".join(line.text for line in result.lines).strip()
    print("stt: moonshine in {:.1f}s".format(time.time() - started))
    return text