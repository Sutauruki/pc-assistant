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

A 401/402/403 (bad key, out of credit, free tier ended) turns this voice off
for the rest of the session, so every later reply doesn't pay a doomed
round trip before falling back.
"""

import os
import threading

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

# Same rate the shared output stream is opened at (and the earcons use), so
# a reply never forces the device stream to close and reopen.
SAMPLE_RATE = DEFAULT_SAMPLE_RATE

# Status codes that won't fix themselves within a session.
_FATAL_STATUS = (401, 402, 403)

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
        "latency": "balanced",
        "normalize": True,
    }
    if VOICE_ID:
        body["reference_id"] = VOICE_ID
    return body


def speak(text, on_first_audio=None):
    """Synthesises and plays `text` through Fish Audio.

    Returns the StreamingPlayer once playback has finished, or None when no
    audio was produced at all -- the caller's cue to fall back to Moonshine.
    A failure after audio has started returns the player: the reply was
    partly spoken, and repeating it from the top in another voice would be
    worse than a clipped ending.
    """
    if not text or not str(text).strip() or not available():
        return None

    player = None
    try:
        client = _get_client()
        with client.stream("POST", API_URL, json=_body(text)) as res:
            if res.status_code != 200:
                detail = res.read().decode("utf-8", "replace")[:200]
                if res.status_code in _FATAL_STATUS:
                    _disable("HTTP {}: {}".format(res.status_code, detail))
                else:
                    print("tts_fish: HTTP {} ({})".format(res.status_code, detail))
                return None

            for chunk in res.iter_bytes():
                if not chunk:
                    continue
                if player is None:
                    player = StreamingPlayer(
                        samplerate=SAMPLE_RATE,
                        on_first_audio=on_first_audio,
                    )
                    player.start()
                player.feed(chunk)
    except Exception as e:
        print("tts_fish: request failed ({})".format(e))
    finally:
        if player is not None:
            try:
                player.finish(timeout=60.0)
            except Exception as e:
                print("tts_fish: playback didn't finish cleanly ({})".format(e))

    return player


if __name__ == "__main__":
    # Test execution: python -m listener.tts_fish
    if not available():
        print("tts_fish: set FISH_API_KEY in .env first")
    else:
        speak("Fish Audio voice is working. Gimme that tomato sauce! "
              "With the potato! And that bottle of water!")
