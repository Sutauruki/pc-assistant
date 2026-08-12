"""Streaming Groq TTS, with sentence packing for long replies.

Goes in: listener/tts.py   (replaces existing)

Gemini keeps doing transcription in listener/stt_gemini.py -- untouched.
Speech goes through Groq on a SEPARATE key so TTS volume doesn't burn down
the same quota the parser uses.

    .env
    GROQ_API_KEY=...          # parser / intent
    GROQ_TTS_API_KEY=...      # speech only  (falls back to GROQ_API_KEY)

Groq returns a container format rather than bare PCM, so the WAV header is
parsed off the front of the stream before samples reach the device.

WHAT CHANGED, AND WHY IT HAD TO
-------------------------------
Orpheus rejects input over 200 characters. Until now `_trim()` quietly cut
anything longer on a word boundary and moved on. For command acknowledgements
("Opening Chrome.") that never mattered -- 200 characters is about 35 words and
no acknowledgement is that long.

Conversation breaks it immediately. A real answer to "what's the difference
between TCP and UDP" is three or four sentences, comfortably past the cap, so
under the old code you would hear the first two sentences and then silence,
with no indication anything had been cut. So speech is now segmented: the text
is split on sentence boundaries and packed into requests that each fit under
the cap.

    ONE player, MANY synthesis requests.

That distinction is the whole trick. Feeding several requests into one
StreamingPlayer means its buffer bridges the joins, so the reply plays as a
single continuous stretch of audio. A player per segment would start, drain and
stop for each one, and you would hear the pause at every seam -- which is
exactly the chopping this file has spent two rounds getting rid of.

AND WHY THERE IS A PREFETCH THREAD
----------------------------------
Segments are synthesised one after another, so while request N+1 is in flight
nothing is arriving: no bytes for the length of a round trip, typically
300-700ms. The player is draining at SPEECH_SPEED the whole time. On a
four-segment answer that is four hard stalls, each one long enough to hear --
the structural version of the same fault, reintroduced by the fix for the
truncation.

So synthesis runs on its own thread, one segment ahead, pushing decoded PCM
into a bounded queue that the feeding side drains. By the time segment N stops
producing, segment N+1's first bytes are already waiting. The queue bound is
what keeps a long answer from buffering itself into memory -- roughly seventeen
seconds of audio in flight, then the synthesiser blocks until playback catches
up.

THE SPLITTER DOES NOT SPLIT ON "."
----------------------------------
"Dr. Who aired in 1963." naively splits into "Dr." and "Who aired in 1963."
Two requests, and because Orpheus picks intonation per request you hear two
unrelated readings joined mid-name. So boundaries are checked against an
abbreviation list, single-letter initials, and whether the next word starts
lowercase.

commands/chat.py has its own splitter for streaming answers out sentence by
sentence. This one is deliberately not shared: listener/ importing from
commands/ inverts the layering, and thirty duplicated lines is a cheaper price
than that.

LOCAL FALLBACK
--------------
If the Groq path produces zero audio for a reply -- bad key, no terms
acceptance, rate limited, offline -- speak() and speak_stream() now hand the
text to listener/tts_local.py (Kitten TTS, fully offline) instead of going
silent. This is the same STT-side pattern stt_local.py/main.py already use
for Gemini: a cloud tier failing should cost quality, not the reply
entirely. See tts_local.py's docstring for why Kitten specifically.
"""

import os
import queue
import re
import struct
import threading

import numpy as np
from dotenv import load_dotenv
from groq import Groq

from listener.audio_out import StreamingPlayer
from listener import tts_local

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
# audio_out treats this as a *target*, not a promise: when the buffer runs thin
# it eases the stretch down toward 1.0 and back up again, which is heard as a
# slight change of pace instead of a gap. So the cost of this number being
# slightly optimistic is "one sentence reads a little slower", not "the
# sentence has a hole in it".
#
# Past ~1.6 the WSOLA seams get audible on sibilants. Below ~1.05 there is no
# point stretching at all -- set it to 1.0 and the stretcher is skipped
# entirely, which is the cheapest and smoothest option if you don't mind the
# natural pace.
SPEECH_SPEED = 1.15

# Rough speaking rate of the *source* audio, used only to tell the player how
# much is coming so it can size its prebuffer and cap its recovery pauses.
#
# Erring low is the safe direction: it overstates how much audio is still to
# come, which makes the prebuffer slightly generous and recovery pauses
# slightly shorter. Erring high understates the reply and under-buffers it,
# which is audible. audio_out clamps the result either way.
CHARS_PER_SECOND = 13.0

# Orpheus accepts 8000/16000/22050/24000/32000/44100/48000 and defaults to
# 48000. Pinning it means audio_out never has to reopen the device mid-run,
# 24k is plenty for a voice reply, and -- new consideration -- every segment
# comes back at the same rate, so one player can take all of them.
SAMPLE_RATE = 24000

# Orpheus rejects input longer than this. Hard limit, not a preference.
MAX_INPUT_CHARS = 200

# Target size for a packed segment. Under the hard cap so a sentence that
# lands slightly over the estimate still fits without a mid-word cut.
SEGMENT_CHARS = 180

# The first segment is kept shorter than the rest. Synthesis is roughly
# realtime, so a shorter first request reaches its first samples sooner and the
# reply starts talking sooner. Everything after it is already covered by the
# prefetch, so only the first one benefits from being small.
FIRST_SEGMENT_CHARS = 110

# Network read size. Small enough that the first block lands quickly.
#
# At 24kHz mono int16 a 4096-byte read is ~85ms of audio, so the player's queue
# would only ever move in 85ms steps and its depth would sawtooth by that much
# between reads -- enough on its own to dip a thin buffer under the low-water
# mark. 2048 halves the sawtooth and gets the first sample out sooner, at the
# cost of twice as many (already trivial) socket reads.
CHUNK_BYTES = 2048

# Decoded PCM chunks allowed in flight between the synthesiser and the player.
# At ~23 chunks/second of audio this is roughly 17 seconds, about 800KB. Past
# that the synthesis thread blocks, which is the intended back pressure: a long
# answer should not buffer itself into memory just because it can.
PREFETCH_CHUNKS = 400

# Longest a single reply is allowed to hold the speech path. Scaled by the
# estimated length in _finish_timeout -- a fixed 40s would truncate a genuinely
# long conversational answer, which is the fault this whole change exists to
# fix, arriving by a different route.
MIN_FINISH_TIMEOUT = 40.0

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
    """Last-resort cut for something with no usable split point at all.

    Kept because the splitter can be handed a single 300-character token -- a
    URL, a base64 blob read off the clipboard -- where there is no word
    boundary to break on and the cap still has to be respected.
    """
    text = text.strip()
    if len(text) <= MAX_INPUT_CHARS:
        return text

    cut = text[:MAX_INPUT_CHARS]
    spaced = cut.rsplit(" ", 1)[0]
    trimmed = spaced if len(spaced) > MAX_INPUT_CHARS * 0.6 else cut
    print("tts: input trimmed to {} chars".format(len(trimmed)))
    return trimmed


# --- Segmentation ----------------------------------------------------------

# Words that end in a period without ending a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "rev", "hon",
    "vs", "etc", "eg", "ie", "al", "fig", "approx", "est", "no", "vol",
    "inc", "ltd", "co", "corp", "dept", "univ", "min", "max", "avg",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thur", "thurs",
    "fri", "sat", "sun", "am", "pm", "ph", "sq", "ft", "lb", "oz",
}

# Punctuation that can end a sentence, any trailing quote or bracket, then
# whitespace or the end of the string. The trailing whitespace requirement is
# what keeps "3.5" and "example.com" from being treated as boundaries.
_BOUNDARY_RE = re.compile(r"([.!?…]+)([\"'”’)\]]*)(\s+|$)")

# Weaker break points, used only when a run has no sentence punctuation in it.
_CLAUSE_RE = re.compile(r"([;:,—-]|\s–\s)(\s+)")

_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z0-9'’]*)\s*$")


def _is_boundary(text, punct_start, punct, after_index):
    """Whether the punctuation at punct_start really ends a sentence."""
    # "!" and "?" are unambiguous. Only "." carries the abbreviation problem.
    if punct.strip("."):
        return True

    before = text[:punct_start]
    match = _WORD_BEFORE_RE.search(before)
    if match:
        word = match.group(1)
        if len(word) == 1:
            return False                      # initial: "J. R. R. Tolkien"
        if word.lower() in _ABBREVIATIONS:
            return False

    # A following lowercase letter almost always means the period was not a
    # full stop -- an abbreviation this list doesn't know, or a file extension.
    tail = text[after_index:].lstrip()
    if tail and tail[0].islower():
        return False

    return True


def _sentences(text):
    """Text -> list of sentence-ish strings, abbreviation-aware."""
    text = " ".join(str(text or "").split())
    if not text:
        return []

    out = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        end = match.end(2)          # include the punctuation and any quotes
        if not _is_boundary(text, match.start(1), match.group(1), end):
            continue
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _split_long(piece, cap):
    """One over-long sentence -> several pieces, each within `cap`.

    Tries clause punctuation first because a break at a comma sounds like a
    breath; falls back to word boundaries, which sound like a slightly odd
    pause; falls back to _trim only for a single token with no spaces in it.
    """
    piece = piece.strip()
    if len(piece) <= cap:
        return [piece] if piece else []

    # Clause-level breaks, keeping the punctuation with the left-hand side.
    points = [m.end(1) for m in _CLAUSE_RE.finditer(piece)]
    if points:
        out, start = [], 0
        for point in points:
            if point - start >= cap * 0.5:
                out.append(piece[start:point].strip())
                start = point
        rest = piece[start:].strip()
        if rest:
            out.append(rest)
        if out and all(len(p) <= cap for p in out):
            return [p for p in out if p]

    # Word boundaries.
    out, current = [], ""
    for word in piece.split(" "):
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= cap:
            current += " " + word
        else:
            out.append(current)
            current = word
        while len(current) > cap:
            # A single token longer than the cap. Nothing to break on.
            out.append(_trim(current))
            current = current[len(out[-1]):].lstrip()
    if current:
        out.append(current)
    return [p for p in out if p]


def _segments(text, first_cap=FIRST_SEGMENT_CHARS, cap=SEGMENT_CHARS):
    """Text -> synthesis-sized pieces, each guaranteed within MAX_INPUT_CHARS.

    Packs consecutive sentences together rather than sending one request per
    sentence: fewer requests means fewer seams to hide and one consistent
    reading of the whole clause group.
    """
    pieces = []
    limit = max(40, min(first_cap, cap))

    for sentence in _sentences(text):
        for part in _split_long(sentence, cap):
            if pieces and len(pieces[-1]) + 1 + len(part) <= limit:
                pieces[-1] = pieces[-1] + " " + part
                continue
            if pieces:
                # Past the opening pack, so the short cap has done its job.
                limit = cap
            pieces.append(part)

    # Belt and braces: the cap is the model's, not ours, and a bug here would
    # surface as an API rejection mid-reply.
    return [p if len(p) <= MAX_INPUT_CHARS else _trim(p) for p in pieces if p]


# --- WAV decoding ----------------------------------------------------------

class WavStreamDecoder:
    """Strips the RIFF header off a WAV byte stream and emits mono int16 PCM.

    The header can't be assumed to be 44 bytes -- providers are free to emit
    LIST/fact chunks before `data`, and a network boundary can land anywhere,
    including mid-header. So we buffer until `data` is located, then pass
    everything after it straight through.

    One decoder per synthesis request: every response carries its own header,
    so reusing a decoder across segments would feed the second header into the
    player as samples, which is audible as a click.
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


# --- Requests -------------------------------------------------------------

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


# --- The pipeline ---------------------------------------------------------

_DONE = object()          # sentinel: no more audio is coming


def _synthesise(pieces, out, stop):
    """Synthesis worker. Decoded PCM goes into `out` as (rate, bytes).

    Runs one whole segment ahead of playback -- that lead is what covers the
    round trip for the next request, so the player never sees a gap at a seam.

    Errors are reported and the loop stops, but whatever already played stays
    played: half an answer is better than an answer that vanished.
    """
    try:
        client = get_client()
    except Exception as e:
        _explain_failure(e)
        out.put(_DONE)
        return

    try:
        for piece in pieces:
            if stop.is_set():
                break
            # A piece arriving mid-answer gets the full segment size -- the
            # short first cap only helps the very first request of a reply.
            for text in ([piece] if len(piece) <= MAX_INPUT_CHARS
                         else _segments(piece, first_cap=SEGMENT_CHARS,
                                        cap=SEGMENT_CHARS)):
                if stop.is_set():
                    break
                decoder = WavStreamDecoder()
                try:
                    for raw in _iter_response_bytes(client, text):
                        if stop.is_set():
                            break
                        pcm = decoder.feed(raw)
                        if pcm:
                            # Blocks when the queue is full. That back pressure
                            # is the point -- it throttles synthesis to
                            # roughly playback rate instead of buffering a
                            # whole long answer into memory.
                            out.put((decoder.samplerate or SAMPLE_RATE, pcm))
                except Exception as e:
                    _explain_failure(e)
                    stop.set()
                    break
    finally:
        out.put(_DONE)


def _finish_timeout(expected_seconds):
    """How long finish() may block. Scales with the reply.

    A fixed timeout truncates long answers, which is the exact fault the
    segmentation was added to fix -- so it cannot be fixed with a constant.
    """
    if not expected_seconds:
        return MIN_FINISH_TIMEOUT
    return max(MIN_FINISH_TIMEOUT, (expected_seconds / SPEECH_SPEED) + 20.0)


def _play(pieces, on_first_audio=None, expected_chars=None):
    """Runs the pipeline. One player, however many requests it takes.

    `pieces` may be a list or a lazy iterator -- speak_stream() passes one that
    yields sentences as a language model produces them.
    """
    out = queue.Queue(maxsize=PREFETCH_CHUNKS)
    stop = threading.Event()
    player = None
    rate = None

    worker = threading.Thread(
        target=_synthesise, args=(pieces, out, stop), daemon=True
    )
    worker.start()

    expected_seconds = (
        (expected_chars / CHARS_PER_SECOND) if expected_chars else None
    )

    try:
        while True:
            item = out.get()
            if item is _DONE:
                break

            chunk_rate, pcm = item

            if player is None:
                rate = chunk_rate or SAMPLE_RATE
                player = StreamingPlayer(
                    samplerate=rate,
                    speed=SPEECH_SPEED,
                    on_first_audio=on_first_audio,
                    # Lets the player work out how much to hold back before it
                    # starts, and how long a mid-reply refill is allowed to
                    # take, instead of guessing fixed figures.
                    expected_seconds=expected_seconds,
                )
                player.start()
            elif chunk_rate and chunk_rate != rate:
                # Shouldn't happen -- sample_rate is pinned on every request --
                # but a rate change mid-player would play at the wrong pitch,
                # and knowing that from a log line beats guessing at it.
                print("tts: segment came back at {}Hz, expected {}Hz".format(
                    chunk_rate, rate
                ))

            player.feed(pcm)
    except Exception as e:
        print("tts: playback failed ({})".format(e))
    finally:
        stop.set()
        # Drain whatever the worker still holds so a blocked put() can return
        # and the thread can exit. Without this a full queue plus a dead
        # consumer leaks a thread per reply.
        try:
            while True:
                out.get_nowait()
        except queue.Empty:
            pass

        if player is not None:
            try:
                player.finish(timeout=_finish_timeout(expected_seconds))
            except Exception as e:
                print("tts: playback didn't finish cleanly ({})".format(e))

    return player


# --- Public API -----------------------------------------------------------

def _consuming(pieces):
    """Wraps an iterable of text pieces, capturing everything drawn from it
    as it's consumed.

    speak_stream()'s `pieces` is often a live, one-shot iterator (a
    SentenceFeeder draining as commands/chat.py generates text) rather than a
    list -- by the time _play() finishes there is no going back to replay it.
    If the Groq path turns out to have produced zero audio, this is what lets
    the local fallback speak whatever text actually came through instead of
    nothing.
    """
    captured = []

    def gen():
        for piece in pieces:
            captured.append(piece)
            yield piece

    return gen(), captured


def speak(text, on_first_audio=None):
    """Streams Groq TTS straight to the speakers -- nothing touches disk.

    Any length. Text over the model's input cap is split on sentence
    boundaries and played through a single player, so a long answer sounds
    continuous rather than arriving in chunks with holes between them.

    on_first_audio: called when the first sample actually hits the device.

    Never raises. If Groq produces no audio at all for this reply (offline,
    rate limited, bad key, terms not accepted), falls back to
    listener/tts_local.py (Kitten TTS, offline) rather than going silent.
    """
    if not text or not text.strip():
        return None

    try:
        pieces = _segments(text)
    except Exception as e:
        # A splitter bug must not cost the reply. Fall back to the old
        # behaviour: say the first 200 characters.
        print("tts: couldn't split that ({}), speaking the opening".format(e))
        pieces = [_trim(text)]

    if not pieces:
        return None

    if len(pieces) > 1:
        print("tts: {} chars -> {} segments".format(len(text), len(pieces)))

    player = _play(pieces, on_first_audio=on_first_audio,
                    expected_chars=sum(len(p) for p in pieces))

    if player is None:
        # `player` only stays None when zero chunks of audio ever arrived --
        # _play() isn't reached with empty pieces (checked above), so this
        # means every request failed, not that there was nothing to say.
        print("tts: cloud speech unavailable, falling back to local voice")
        player = tts_local.speak(text, on_first_audio=on_first_audio)

    return player


def speak_stream(pieces, on_first_audio=None, expected_chars=None):
    """Speaks text that is still being generated.

    `pieces` is any iterable of strings -- typically a queue drain fed by
    commands/chat.py's on_sentence callback. Synthesis starts on the first
    sentence instead of waiting for the whole answer, which takes several
    seconds off a conversational reply.

    The caller decides when the answer is over by ending the iterator.
    Never raises. Falls back to the local voice on total Groq failure, same
    as speak() -- see _consuming() for how the already-generated text
    survives that handoff even though `pieces` itself can only be drained
    once.
    """
    if pieces is None:
        return None

    wrapped, captured = _consuming(pieces)
    player = _play(wrapped, on_first_audio=on_first_audio,
                    expected_chars=expected_chars)

    if player is None and captured:
        text = " ".join(str(p) for p in captured if p)
        print("tts: cloud speech unavailable, falling back to local voice")
        player = tts_local.speak(text, on_first_audio=on_first_audio)

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


class SentenceFeeder:
    """A hand-off between something producing sentences and speak_stream().

        feeder = tts.SentenceFeeder()
        threading.Thread(target=tts.speak_stream, args=(feeder,)).start()
        chat.answer(question, on_sentence=feeder.put)
        feeder.close()

    Iterating blocks until the next sentence arrives or close() is called, so
    the synthesiser stays exactly one sentence behind the model with no polling
    and no sleeps.
    """

    def __init__(self):
        self._q = queue.Queue()
        self._closed = threading.Event()

    def put(self, sentence):
        if sentence and str(sentence).strip() and not self._closed.is_set():
            self._q.put(str(sentence).strip())

    def close(self):
        if not self._closed.is_set():
            self._closed.set()
            self._q.put(_DONE)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is _DONE:
                return
            yield item


if __name__ == "__main__":
    # python -m listener.tts  -- segmentation only, nothing is synthesised.
    samples = [
        "Opening Chrome.",
        "Dr. Who first aired in 1963. It ran for 26 years. Mr. Smith presented "
        "the first episode, i.e. the pilot, on Nov. 23rd.",
        "TCP is connection-oriented, which means it sets up a session, "
        "guarantees delivery and puts packets back in order for you. UDP does "
        "none of that. It fires datagrams and forgets them, so it is faster but "
        "you handle loss yourself. Use TCP for files, UDP for live audio.",
        "https://example.com/" + "x" * 260,
    ]
    for sample in samples:
        print("\n{} chars:".format(len(sample)))
        for i, piece in enumerate(_segments(sample), 1):
            print("  {}. [{:>3}] {}".format(i, len(piece), piece))