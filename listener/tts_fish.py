"""Cloud TTS through Fish Audio (S2.1 Pro, free developer tier).

WHY THIS IS THE PRIMARY VOICE
-----------------------------
Fish Audio's S2.1 Pro is the same model as the paid tier, offered free under
the `s2.1-pro-free` model header (fair-use, no SLA). It sounds far better
than the local Kokoro/Piper voices and starts talking quicker: the response
is a chunked stream of raw 16-bit mono PCM, which feeds straight into
audio_out's StreamingPlayer as it arrives, with no decoding.

listener/tts.py tries this module first and falls back to the offline
Moonshine voice (listener/tts_moonshine.py) whenever Fish can't produce any
audio: no key, no network, rate limited, or the free window has closed.

CONFIG (.env)
-------------
FISH_API_KEY    required to enable this voice (fish.audio/app/api-keys)
FISH_VOICE_ID   optional voice model id (`reference_id`) from fish.audio;
                unset uses Fish's default voice
FISH_MODEL      optional model header; default s2.1-pro-free
FISH_LATENCY    low | normal | balanced (default balanced); low starts
                talking sooner at a slight cost in voice quality
FISH_CACHE      0 turns off the on-disk phrase cache (default on)

A 401/402/403 (bad key, out of credit, free tier ended) turns this voice off
for the rest of the session, so every later reply doesn't pay a doomed
round trip before falling back.
"""

import hashlib
import os
import queue
import threading
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - .env is optional
    pass

from listener.audio_out import DEFAULT_SAMPLE_RATE, StreamingPlayer

API_URL = "https://api.fish.audio/v1/tts"
API_KEY = os.getenv("FISH_API_KEY", "").strip()
VOICE_ID = os.getenv("FISH_VOICE_ID", "").strip()
MODEL = os.getenv("FISH_MODEL", "s2.1-pro-free").strip() or "s2.1-pro-free"

_LATENCIES = ("low", "normal", "balanced")
LATENCY = os.getenv("FISH_LATENCY", "balanced").strip().lower() or "balanced"
if LATENCY not in _LATENCIES:
    print("tts_fish: FISH_LATENCY={!r} isn't one of {}; using balanced".format(
        LATENCY, ", ".join(_LATENCIES)))
    LATENCY = "balanced"

# Same rate the shared output stream is opened at (and the earcons use), so
# a reply never forces the device stream to close and reopen.
SAMPLE_RATE = DEFAULT_SAMPLE_RATE

# Status codes that won't fix themselves within a session.
_FATAL_STATUS = (401, 402, 403)

_DONE = object()

_disabled = False
_disabled_reason = ""
_client = None
_client_lock = threading.Lock()


def available():
    """True when a key is configured and the voice hasn't been turned off."""
    if _disabled or not API_KEY:
        return False
    try:
        import httpx  # noqa: F401
        return True
    except ImportError:
        return False


def disabled_reason():
    return _disabled_reason


def _disable(reason):
    global _disabled, _disabled_reason
    _disabled = True
    _disabled_reason = reason
    print("tts_fish: disabled for this session ({}); using Moonshine".format(reason))


def _get_client():
    """One pooled client for the life of the process, so the TLS handshake
    happens once (warm() does it at startup) instead of on every reply.

    Connect fast so a dead network falls back quickly; allow a generous read
    gap because the server may pause between synthesised chunks."""
    global _client
    with _client_lock:
        if _client is None:
            import httpx
            _client = httpx.Client(
                timeout=httpx.Timeout(10.0, connect=5.0, read=20.0),
                headers={
                    "Authorization": "Bearer {}".format(API_KEY),
                    "model": MODEL,
                },
            )
        return _client


def warm():
    """Opens the pooled connection on a background thread at startup."""
    if not available():
        return

    def run():
        try:
            # Any response at all (even a 405) leaves a warm keep-alive
            # connection in the pool; failures here are irrelevant.
            _get_client().head(API_URL)
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


def _body(text):
    body = {
        "text": str(text).strip(),
        "format": "pcm",
        "sample_rate": SAMPLE_RATE,
        "latency": LATENCY,
        "normalize": True,
    }
    if VOICE_ID:
        body["reference_id"] = VOICE_ID
    return body


# ---------------------------------------------------------------------------
# Phrase cache
# ---------------------------------------------------------------------------
# Command replies are short and repeat constantly ("Done.", "Opening Chrome.").
# Their PCM is kept on disk, keyed by everything that shapes the audio, so a
# repeat plays instantly, costs no quota, and still works with Fish down.
# Long chat replies almost never repeat and aren't stored.

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "tts_fish"
_CACHE_ON = os.getenv("FISH_CACHE", "1").strip().lower() not in ("0", "false", "no", "off")
_CACHE_MAX_CHARS = 120
_CACHE_MAX_FILES = 300  # ~3s of 24kHz PCM each is ~150KB, so tens of MB at most


def _cache_path(text):
    key = "|".join((MODEL, VOICE_ID, str(SAMPLE_RATE), LATENCY,
                    " ".join(str(text).split())))
    return _CACHE_DIR / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".pcm")


def _cacheable(text):
    return _CACHE_ON and len(str(text).strip()) <= _CACHE_MAX_CHARS


def _cache_get(text):
    if not _cacheable(text):
        return None
    path = _cache_path(text)
    try:
        data = path.read_bytes()
        os.utime(path)  # recently used -- pruned last
        return data or None
    except OSError:
        return None


def _cache_put(text, pcm):
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(text)
        tmp = path.with_suffix(".tmp{}".format(threading.get_ident()))
        tmp.write_bytes(pcm)
        os.replace(tmp, path)
        files = sorted(_CACHE_DIR.glob("*.pcm"), key=lambda f: f.stat().st_mtime)
        for old in files[:max(0, len(files) - _CACHE_MAX_FILES)]:
            old.unlink(missing_ok=True)
    except OSError as e:
        print("tts_fish: couldn't write phrase cache ({})".format(e))


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

class _Failed(Exception):
    """Fish produced no (complete) audio for a piece of text."""


class _Output:
    """Opens the StreamingPlayer on the first audio, not before, so a request
    that fails up front leaves no player behind and the caller can fall back."""

    def __init__(self, on_first_audio):
        self._on_first_audio = on_first_audio
        self.player = None

    def feed(self, pcm):
        if not pcm:
            return
        if self.player is None:
            self.player = StreamingPlayer(
                samplerate=SAMPLE_RATE,
                on_first_audio=self._on_first_audio,
            )
            self.player.start()
        self.player.feed(pcm)

    def finish(self):
        if self.player is not None:
            try:
                self.player.finish(timeout=60.0)
            except Exception as e:
                print("tts_fish: playback didn't finish cleanly ({})".format(e))


def _synth(text, feed):
    """Feeds `text`'s PCM to feed() as it arrives -- from the phrase cache
    when it's there, otherwise from one streamed /v1/tts request. Raises
    _Failed (or a network error) when the audio couldn't be completed."""
    cached = _cache_get(text)
    if cached:
        feed(cached)
        return
    if not available():
        raise _Failed("not available")

    keep = bytearray() if _cacheable(text) else None
    with _get_client().stream("POST", API_URL, json=_body(text)) as res:
        if res.status_code != 200:
            detail = res.read().decode("utf-8", "replace")[:200]
            reason = "HTTP {}: {}".format(res.status_code, detail)
            if res.status_code in _FATAL_STATUS:
                _disable(reason)
            raise _Failed(reason)
        for chunk in res.iter_bytes():
            if not chunk:
                continue
            feed(chunk)
            if keep is not None:
                keep += chunk
    if keep:
        _cache_put(text, bytes(keep))


def speak(text, on_first_audio=None):
    """Synthesises and plays `text` through Fish Audio.

    Returns the StreamingPlayer once playback has finished, or None when no
    audio was produced at all -- the caller's cue to fall back to Moonshine.
    A failure after audio has started returns the player: the reply was
    partly spoken, and repeating it from the top in another voice would be
    worse than a clipped ending.
    """
    if not text or not str(text).strip():
        return None

    out = _Output(on_first_audio)
    try:
        _synth(text, out.feed)
    except _Failed as e:
        if str(e) != "not available":
            print("tts_fish: {}".format(e))
    except Exception as e:
        print("tts_fish: request failed ({})".format(e))
    finally:
        out.finish()
    return out.player


# Upper bound on one request's text when batching sentences that arrived
# while the previous request was downloading (Fish's own chunk_length cap).
_MAX_SEGMENT_CHARS = 300


def speak_stream(pieces, on_first_audio=None):
    """Speaks text while it is still being generated.

    The first sentence goes to Fish the moment it arrives, so the reply
    starts after one sentence instead of after the whole answer. Each request
    downloads faster than it plays, so the next one -- every sentence that
    arrived in the meantime, batched -- is requested while the previous one is
    still on the speakers, and all of it feeds one continuous player.

    Returns (player, unspoken): player is None when nothing was spoken;
    unspoken is the text Fish couldn't voice (the failed piece plus the rest
    of the stream), for the caller to hand to Moonshine.
    """
    incoming = queue.Queue()

    def pump():
        try:
            for piece in pieces:
                if piece and str(piece).strip():
                    incoming.put(str(piece).strip())
        finally:
            incoming.put(_DONE)

    threading.Thread(target=pump, daemon=True).start()

    def rest():
        parts = []
        while True:
            item = incoming.get()
            if item is _DONE:
                return parts
            parts.append(item)

    out = _Output(on_first_audio)
    unspoken = []
    try:
        segment = incoming.get()
        finished = segment is _DONE
        while not finished:
            try:
                _synth(segment, out.feed)
            except Exception as e:
                if str(e) != "not available":
                    print("tts_fish: streaming failed ({})".format(e))
                unspoken = [segment] + rest()
                break

            # Next segment: wait for at least one sentence, then take
            # whatever else is already waiting, up to one request's worth.
            item = incoming.get()
            if item is _DONE:
                break
            parts, size = [item], len(item)
            while size < _MAX_SEGMENT_CHARS:
                try:
                    item = incoming.get_nowait()
                except queue.Empty:
                    break
                if item is _DONE:
                    finished = True
                    break
                parts.append(item)
                size += len(item) + 1
            segment = " ".join(parts)
            if finished:
                # The stream closed during batching; speak the last batch.
                try:
                    _synth(segment, out.feed)
                except Exception as e:
                    print("tts_fish: streaming failed ({})".format(e))
                    unspoken = [segment]
    finally:
        out.finish()

    return out.player, " ".join(unspoken)


if __name__ == "__main__":
    # Test execution: python -m listener.tts_fish
    if not available():
        print("tts_fish: set FISH_API_KEY in .env first")
    else:
        speak("If it is your wish, I will follow you anywhere, even if your throne crumbles, and your shiny crown turns to rust. Even if the bodies pile up endlessly above the bottomless pile of corpses. Beside you as you lie down softly, I will be, until I hear the words 'Check Mate'!")
