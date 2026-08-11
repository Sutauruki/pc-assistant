"""Streaming Groq TTS.

Goes in: listener/tts.py   (replaces existing)

Gemini keeps doing transcription in listener/stt_gemini.py -- untouched.
Speech goes through Groq on a SEPARATE key so TTS volume doesn't burn down
the same quota the parser uses.

    .env
    GROQ_API_KEY=...          # parser / intent
    GROQ_TTS_API_KEY=...      # speech only  (falls back to GROQ_API_KEY)

Groq returns a container format rather than bare PCM, so the WAV header is
parsed off the front of the stream before samples reach the device.
"""

import os
import struct
import threading

import numpy as np
from dotenv import load_dotenv
from groq import Groq

from listener.audio_out import StreamingPlayer

load_dotenv()

MODEL_NAME = "canopylabs/orpheus-v1-english"

# Voices differ in natural pace as well as timbre -- if 'autumn' still drags,
# 'austin' and 'daniel' read noticeably brisker at the same stretch factor,
# which is the free way to sound quicker without asking the player for it.
# Full set: autumn, diana, hannah, austin, daniel, troy
VOICE_NAME = "autumn"

RESPONSE_FORMAT = "wav"

# Playback rate, applied locally in audio_out via pitch-preserving WSOLA.
#
# The `speed` field in the Groq SDK signature is a no-op for Orpheus -- the
# model accepts it and ignores it. Groq's own answer is inline "[fast paced]"
# tags, but the docs admit the model applies unrecognised directions
# inconsistently, and a tag that sometimes works is not much use on every
# reply. Stretching the samples ourselves is deterministic and costs ~1ms per
# chunk.
#
# 1.25 -> 1.15. This is the single biggest lever on the remaining chop, and
# it is worth understanding why rather than just taking the number.
#
# Groq generates roughly at realtime. Playing at 1.25x drains the buffer at
# 1.25 played-seconds per wall second against ~1.0 arriving, so the player
# loses 0.25s of cushion for every second it plays -- a *structural* deficit
# that no amount of buffering removes, only delays. Over a 4-second reply that
# is a full second of audio that has to be pre-funded up front or made up
# somewhere. At 1.15 the deficit is 0.15s/s, 40% smaller, and it lands inside
# what the player's speed governor can absorb without a pause.
#
# As of this round audio_out treats this as a *target*, not a promise: when
# the buffer runs thin it eases the stretch down toward 1.0 and back up again,
# which is heard as a slight change of pace instead of a gap. So the cost of
# this number being slightly optimistic is now "one sentence reads a little
# slower", not "the sentence has a hole in it".
#
# Past ~1.6 the WSOLA seams get audible on sibilants. Below ~1.05 there is no
# point stretching at all -- set it to 1.0 and the stretcher is skipped
# entirely, which is the cheapest and smoothest option if you don't mind the
# natural pace.
SPEECH_SPEED = 1.15

# Rough speaking rate of the *source* audio, used only to tell the player how
# much is coming so it can size its prebuffer and cap its recovery pauses.
#
# 14.5 -> 13.0. Erring low is the safe direction: it overstates how much audio
# is still to come, which makes the prebuffer slightly generous and recovery
# pauses slightly shorter. Erring high understates the reply and under-buffers
# it, which is audible. audio_out clamps the result either way.
CHARS_PER_SECOND = 13.0

# Orpheus accepts 8000/16000/22050/24000/32000/44100/48000 and defaults to
# 48000. Pinning it means audio_out never has to reopen the device mid-run,
# and 24k is plenty for a voice reply -- half the bytes of 48k over the wire.
SAMPLE_RATE = 24000

# Orpheus rejects input longer than this.
MAX_INPUT_CHARS = 200

# Network read size. Small enough that the first block lands quickly.
#
# 4096 -> 2048. At 24kHz mono int16 a 4096-byte read is ~85ms of audio, so the
# player's queue only ever moved in 85ms steps and its depth sawtoothed by
# that much between reads -- enough on its own to dip a thin buffer under the
# low-water mark. Halving it halves the sawtooth and gets the first sample out
# sooner, at the cost of twice as many (already trivial) socket reads.
CHUNK_BYTES = 2048

# Params the API takes but older SDK builds may not expose.
_OPTIONAL_KEYS = ("sample_rate",)

_client = None
_client_lock = threading.Lock()
_warned_terms = False


def get_client():
    """Speech client, keyed separately from the parser's client."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.getenv("GROQ_TTS_API_KEY") or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_TTS_API_KEY (or GROQ_API_KEY) not found in .env"
                )
            if not os.getenv("GROQ_TTS_API_KEY"):
                print("tts: GROQ_TTS_API_KEY unset, sharing GROQ_API_KEY quota")
            _client = Groq(api_key=api_key)
        return _client


def _trim(text):
    """Keeps input under the model's hard limit, cutting on a word boundary."""
    text = text.strip()
    if len(text) <= MAX_INPUT_CHARS:
        return text

    cut = text[:MAX_INPUT_CHARS]
    spaced = cut.rsplit(" ", 1)[0]
    trimmed = spaced if len(spaced) > MAX_INPUT_CHARS * 0.6 else cut
    print("tts: input trimmed to {} chars".format(len(trimmed)))
    return trimmed


class WavStreamDecoder:
    """Strips the RIFF header off a WAV byte stream and emits mono int16 PCM.

    The header can't be assumed to be 44 bytes -- providers are free to emit
    LIST/fact chunks before `data`, and a network boundary can land anywhere,
    including mid-header. So we buffer until `data` is located, then pass
    everything after it straight through.
    """

    def __init__(self):
        self.samplerate = None
        self.channels = 1
        self.bits = 16
        self._buf = b""
        self._in_data = False
        self._odd = b""

    def feed(self, data):
        """Returns mono int16 PCM bytes, or b"" while still reading header."""
        if self._in_data:
            return self._to_mono(data)

        self._buf += data
        if len(self._buf) < 12:
            return b""

        if self._buf[0:4] != b"RIFF" or self._buf[8:12] != b"WAVE":
            # not a RIFF container -- assume the provider sent raw PCM
            self._in_data = True
            body, self._buf = self._buf, b""
            return self._to_mono(body)

        pos = 12
        while True:
            if pos + 8 > len(self._buf):
                return b""  # need more bytes to read the next chunk header

            chunk_id = self._buf[pos:pos + 4]
            chunk_size = struct.unpack_from("<I", self._buf, pos + 4)[0]
            payload = pos + 8

            if chunk_id == b"data":
                self._in_data = True
                body = self._buf[payload:]
                self._buf = b""
                return self._to_mono(body)

            if chunk_id == b"fmt ":
                if payload + 16 > len(self._buf):
                    return b""
                (_fmt, channels, samplerate, _byterate,
                 _align, bits) = struct.unpack_from("<HHIIHH", self._buf, payload)
                self.channels = channels or 1
                self.samplerate = samplerate
                self.bits = bits or 16
                if self.bits != 16:
                    print("tts: unexpected {}-bit WAV".format(self.bits))

            pos = payload + chunk_size + (chunk_size % 2)  # chunks are word-aligned

    def _to_mono(self, data):
        if self.channels <= 1:
            return data

        # Downmix interleaved frames; audio_out plays mono.
        data = self._odd + data
        frame = 2 * self.channels
        usable = len(data) - (len(data) % frame)
        self._odd = data[usable:]
        if usable <= 0:
            return b""

        frames = np.frombuffer(data[:usable], dtype=np.int16).reshape(-1, self.channels)
        return frames.mean(axis=1).astype(np.int16).tobytes()


def _create_kwargs(text):
    # No `speed` here on purpose -- Orpheus ignores it. Pacing is handled
    # downstream by the time-stretcher in audio_out.
    return {
        "model": MODEL_NAME,
        "voice": VOICE_NAME,
        "response_format": RESPONSE_FORMAT,
        "sample_rate": SAMPLE_RATE,
        "input": text,
    }


def _stream_chunks(streaming, kwargs):
    with streaming.create(**kwargs) as response:
        for chunk in response.iter_bytes(CHUNK_BYTES):
            if chunk:
                yield chunk


def _iter_response_bytes(client, text):
    """Yields audio bytes as they arrive.

    Prefers the streaming response so playback can start before synthesis
    finishes; falls back to a buffered read on SDK versions without it.
    """
    kwargs = _create_kwargs(text)

    streaming = getattr(client.audio.speech, "with_streaming_response", None)
    if streaming is not None:
        try:
            for chunk in _stream_chunks(streaming, kwargs):
                yield chunk
            return
        except TypeError as e:
            # older SDK that doesn't know sample_rate/speed -- drop them and
            # retry rather than losing speech entirely
            print("tts: retrying without {} ({})".format(
                ", ".join(_OPTIONAL_KEYS), e
            ))
            for key in _OPTIONAL_KEYS:
                kwargs.pop(key, None)
            for chunk in _stream_chunks(streaming, kwargs):
                yield chunk
            return
        except AttributeError as e:
            print("tts: streaming unavailable ({}), buffering".format(e))

    response = client.audio.speech.create(**kwargs)
    if hasattr(response, "iter_bytes"):
        for chunk in response.iter_bytes(CHUNK_BYTES):
            if chunk:
                yield chunk
    elif hasattr(response, "read"):
        yield response.read()
    else:
        yield bytes(response.content)


def _explain_failure(err):
    """Turns the noisy SDK exceptions into one actionable line."""
    global _warned_terms
    msg = str(err)

    if "model_terms_required" in msg or "requires terms acceptance" in msg:
        if not _warned_terms:
            _warned_terms = True
            print(
                "\ntts: {} needs a one-time terms acceptance.\n"
                "     Accept it here, then re-run:\n"
                "     https://console.groq.com/playground?model="
                "canopylabs%2Forpheus-v1-english\n".format(MODEL_NAME)
            )
        return

    if "invalid_api_key" in msg or "Invalid API Key" in msg:
        print("tts: GROQ_TTS_API_KEY rejected -- check the key in .env")
        return

    if "rate_limit" in msg or "429" in msg:
        print("tts: rate limited on the speech key")
        return

    print("tts: speech failed ({})".format(err))


def speak(text, on_first_audio=None):
    """Streams Groq TTS straight to the speakers -- nothing touches disk.

    Everything after the WAV `data` marker is fed to the player as it lands,
    so the first words play while the rest is still being synthesised.

    on_first_audio: called when the first sample actually hits the device.

    Never raises. A speech failure shouldn't take down the assistant loop --
    the actions still ran, so it degrades to silent-but-working.
    """
    if not text or not text.strip():
        return

    text = _trim(text)
    decoder = WavStreamDecoder()
    player = None

    try:
        client = get_client()
        for raw in _iter_response_bytes(client, text):
            pcm = decoder.feed(raw)
            if not pcm:
                continue

            if player is None:
                rate = decoder.samplerate or SAMPLE_RATE
                player = StreamingPlayer(
                    samplerate=rate,
                    speed=SPEECH_SPEED,
                    on_first_audio=on_first_audio,
                    # Lets the player work out how much to hold back before it
                    # starts, and how long a mid-reply refill is allowed to
                    # take, instead of guessing fixed figures.
                    expected_seconds=len(text) / CHARS_PER_SECOND,
                )
                player.start()

            player.feed(pcm)
    except Exception as e:
        _explain_failure(e)
    finally:
        # Guarded because this sits in the `finally`, outside the `try` that
        # makes speak() safe. An exception raised here would walk straight out
        # past the handler above and break the never-raises contract the wake
        # loop depends on.
        if player is not None:
            try:
                player.finish()
            except Exception as e:
                print("tts: playback didn't finish cleanly ({})".format(e))

    return player


def speak_async(text, on_first_audio=None):
    """Fire-and-forget speak() on its own thread. Returns the thread."""
    thread = threading.Thread(
        target=speak,
        args=(text,),
        kwargs={"on_first_audio": on_first_audio},
        daemon=True,
    )
    thread.start()
    return thread