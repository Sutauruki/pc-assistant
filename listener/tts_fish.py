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
FISH_VOLUME_DB  loudness boost in dB, applied by Fish (prosody.volume);
                default 3. 0 = Fish's own level; 6 is about 1.4x louder again.
FISH_SAMPLE_RATE  8000 | 16000 | 24000 | 32000 | 44100 (default 44100, Fish's
                native rate -- nothing is downsampled). main.py opens the
                speaker at this rate from the start, so it never has to
                switch rates mid-reply (that switch is what can click).
FISH_CACHE      0 turns off the on-disk phrase cache (default on)

A 401/402/403 (bad key, out of credit, free tier ended) turns this voice off
for the rest of the session, so every later reply doesn't pay a doomed
round trip before falling back.
"""

import hashlib
import os
import queue
import threading
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - .env is optional
    pass

from listener.audio_out import StreamingPlayer

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

# Fish evens out loudness (normalize_loudness) and Jarvis plays it as-is,
# which came out quieter than the Fish website's player. prosody.volume asks
# Fish to raise it server-side, where it's mastered, rather than amplifying
# samples here and risking clipping. Clamped to a sane range.
try:
    VOLUME_DB = float(os.getenv("FISH_VOLUME_DB", "3").strip() or 3)
except ValueError:
    VOLUME_DB = 3.0
VOLUME_DB = max(-12.0, min(12.0, VOLUME_DB))

# The rates Fish offers for raw PCM. 44100 is what the model produces, so
# it's sent untouched; lower rates are downsampled on Fish's side. The speaker
# stream is opened at this same rate at startup (tts.output_rate()), and the
# earcons render at whatever rate the stream is open at, so nothing ever
# forces the device to close and reopen mid-conversation.
_RATES = (8000, 16000, 24000, 32000, 44100)
try:
    SAMPLE_RATE = int(os.getenv("FISH_SAMPLE_RATE", "44100").strip() or 44100)
except ValueError:
    SAMPLE_RATE = 0
if SAMPLE_RATE not in _RATES:
    print("tts_fish: FISH_SAMPLE_RATE must be one of {}; using 44100".format(
        ", ".join(str(r) for r in _RATES)))
    SAMPLE_RATE = 44100

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
    if VOLUME_DB:
        body["prosody"] = {"volume": VOLUME_DB}
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
_CACHE_MAX_FILES = 300  # ~3s of 44.1kHz PCM is ~260KB, so ~80MB at the very most


def _cache_path(text):
    key = "|".join((MODEL, VOICE_ID, str(SAMPLE_RATE), LATENCY, str(VOLUME_DB),
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


# --- Streaming a reply that is still being written --------------------------
#
# Each request is a separate clip, and two things used to make the joins
# audible -- right after the "." or "?" where one clip ended:
#   1. the next request only started once the previous one had finished
#      downloading, and on the free tier that's barely faster than playback,
#      so its start-up time became a gap;
#   2. every clip carries Fish's own lead-in and trailing silence.
# So: fewer, bigger requests; the next one is fetched while the current one
# plays (_IN_FLIGHT at a time, well inside Fish's concurrency limit); and the
# silence where two clips meet is trimmed to one natural pause.

_MAX_SEGMENT_CHARS = 300   # Fish's own chunk_length cap
_FIRST_MIN_CHARS = 80      # gather this much before the first request...
_BATCH_MIN_CHARS = 160     # ...and this much for each later one
_GATHER_WAIT = 0.6         # but never wait longer than this for more text
_IN_FLIGHT = 2             # requests running at once
_JOIN_PAUSE = 0.22         # quiet kept where two clips meet: one natural sentence break
_SILENCE = 350             # |sample| below this (about -40 dBFS) is silence


class _Segment(object):
    def __init__(self, text, fetch=True):
        self.text = text
        self.fetch = fetch
        self.chunks = queue.Queue()
        self.error = None


def _fetch(segment, slots):
    try:
        _synth(segment.text, segment.chunks.put)
    except Exception as e:
        segment.error = e
    finally:
        segment.chunks.put(_DONE)
        slots.release()


def _quiet_head(samples, keep):
    """Index of the first sample worth keeping: leading silence trimmed to
    `keep` samples."""
    import numpy as np
    loud = np.flatnonzero(np.abs(samples) > _SILENCE)
    if not len(loud):
        return None  # all silence so far
    return max(0, int(loud[0]) - keep)


def _quiet_tail(samples, keep):
    """Length to keep: trailing silence trimmed to `keep` samples."""
    import numpy as np
    loud = np.flatnonzero(np.abs(samples) > _SILENCE)
    if not len(loud):
        return 0
    return min(len(samples), int(loud[-1]) + 1 + keep)


def _play_segment(segment, out):
    """Feeds one clip to the player with its edge silence trimmed. Returns
    True if any of its audio was played."""
    import numpy as np
    keep = int(SAMPLE_RATE * _JOIN_PAUSE / 2)
    hold_samples = int(SAMPLE_RATE * 0.5)  # held back until we know it's the tail
    carry = b""
    held = np.zeros(0, dtype=np.int16)
    started = False
    played = False
    while True:
        item = segment.chunks.get()
        if item is _DONE:
            break
        data = carry + item
        cut = len(data) - (len(data) % 2)
        carry = data[cut:]
        held = np.concatenate([held, np.frombuffer(data[:cut], dtype=np.int16)])
        if not started:
            head = _quiet_head(held, keep)
            if head is None:
                held = held[-keep:] if len(held) > keep else held
                continue
            held = held[head:]
            started = True
        if len(held) > hold_samples:
            out.feed(held[:-hold_samples].tobytes())
            played = True
            held = held[-hold_samples:]
    if started and len(held):
        held = held[:_quiet_tail(held, keep)]
        if len(held):
            out.feed(held.tobytes())
            played = True
    return played


def speak_stream(pieces, on_first_audio=None):
    """Speaks text while it is still being generated.

    Sentences are gathered into a few bigger requests (_FIRST_MIN_CHARS for
    the first, _BATCH_MIN_CHARS after, never waiting more than _GATHER_WAIT
    for more text). Each request starts as soon as its text is ready -- up to
    _IN_FLIGHT at once -- so the next clip is downloading while the current
    one plays, and the clips are joined with their edge silence trimmed into
    one continuous player.

    Returns (player, unspoken): player is None when nothing was spoken;
    unspoken is the text Fish couldn't voice (from the first clip that
    failed before any of it played, to the end), for Moonshine.
    """
    incoming = queue.Queue()
    segments = queue.Queue()
    abort = threading.Event()
    slots = threading.Semaphore(_IN_FLIGHT)

    def pump():
        try:
            for piece in pieces:
                if piece and str(piece).strip():
                    incoming.put(str(piece).strip())
        finally:
            incoming.put(_DONE)

    def gather(min_chars):
        """(text, stream_finished). text is None when nothing is left."""
        item = incoming.get()
        if item is _DONE:
            return None, True
        parts, size = [item], len(item)
        deadline = time.time() + _GATHER_WAIT
        while size < min_chars:
            wait = deadline - time.time()
            if wait <= 0:
                break
            try:
                item = incoming.get(timeout=wait)
            except queue.Empty:
                break
            if item is _DONE:
                return " ".join(parts), True
            parts.append(item)
            size += len(item) + 1
        while size < _MAX_SEGMENT_CHARS:  # whatever else is already here
            try:
                item = incoming.get_nowait()
            except queue.Empty:
                break
            if item is _DONE:
                return " ".join(parts), True
            parts.append(item)
            size += len(item) + 1
        return " ".join(parts), False

    def segmenter():
        try:
            minimum, finished = _FIRST_MIN_CHARS, False
            while not finished:
                text, finished = gather(minimum)
                if text is None:
                    break
                minimum = _BATCH_MIN_CHARS
                if abort.is_set():
                    segments.put(_Segment(text, fetch=False))  # text only, for Moonshine
                    continue
                slots.acquire()
                segment = _Segment(text)
                segments.put(segment)
                threading.Thread(target=_fetch, args=(segment, slots), daemon=True).start()
        finally:
            segments.put(_DONE)

    threading.Thread(target=pump, daemon=True).start()
    threading.Thread(target=segmenter, daemon=True).start()

    out = _Output(on_first_audio)
    unspoken = []
    try:
        while True:
            segment = segments.get()
            if segment is _DONE:
                break
            if abort.is_set() or not segment.fetch:
                unspoken.append(segment.text)
                if segment.fetch:
                    _drain(segment)
                continue
            played = _play_segment(segment, out)
            if segment.error is not None:
                if str(segment.error) != "not available":
                    print("tts_fish: streaming failed ({})".format(segment.error))
                if not played:
                    # Nothing of this clip was heard: hand it, and everything
                    # after it, to Moonshine.
                    abort.set()
                    unspoken.append(segment.text)
    finally:
        out.finish()

    return out.player, " ".join(unspoken)


def _drain(segment):
    while segment.chunks.get() is not _DONE:
        pass


if __name__ == "__main__":
    # Test execution: python -m listener.tts_fish
    if not available():
        print("tts_fish: set FISH_API_KEY in .env first")
    else:
        speak("If it is your wish, I will follow you anywhere, even if your throne crumbles, and your shiny crown turns to rust. Even if the bodies pile up endlessly above the bottomless pile of corpses. Beside you as you lie down softly, I will be, until I hear the words 'Check Mate'!")
