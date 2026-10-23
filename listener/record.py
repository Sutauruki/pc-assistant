"""Fast, forgiving endpointing for the mic.

Goes in: listener/record.py

WHAT WAS WRONG
--------------
1. The "room noise" was measured from the first half-second of each
   recording -- which, right after "hey Jarvis", is usually you already
   talking. The speech gate was then set at 3x your own voice, so normal
   speech didn't count as speech: you had to shout to start it, and shout
   again before it would notice you'd stopped.
2. One threshold did two jobs. Soft syllables and trailing words dipped
   under it, and 0.7s later the recording ended mid-sentence.

WHAT IT DOES NOW
----------------
- The room is learned while Jarvis waits for the wake word: main.py hands
  every listening chunk to note_ambient(), which tracks a low percentile of
  recent levels (the quiet moments, not your voice). Recordings start with
  the gate already set, and keep learning from the quiet chunks inside them.
- Two gates (hysteresis): START_RATIO x room to decide you've started,
  the lower KEEP_RATIO x room to decide you're still talking -- so a soft
  ending doesn't count as stopping.
- SILENCE_SECONDS (1.2s) of quiet ends it, which rides over natural pauses.
- normalize() lifts quiet recordings before speech-to-text, without
  boosting near-silence into fake words.

Tune in .env if needed:
    MIC_SENSITIVITY=1.0        higher = hears quieter speech (1.5 for a
                               soft voice or far mic; 0.7 for a loud room)
    RECORD_SILENCE_SECONDS=1.2 how long a pause ends the recording
"""

import collections
import os
import time

import numpy as np

from listener import ui


def _env_float(name, default):
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


SILENCE_SECONDS = max(0.4, _env_float("RECORD_SILENCE_SECONDS", 1.2))
SENSITIVITY = min(4.0, max(0.25, _env_float("MIC_SENSITIVITY", 1.0)))
GRACE_PERIOD = 3.0          # never end in the first moments, before you've begun
MIN_SPEECH_SECONDS = 0.25   # a click or a door slam isn't speech
MAX_RECORD_SECONDS = 60

# Speech gates, relative to the room -- divided by MIC_SENSITIVITY.
START_RATIO = 2.6
KEEP_RATIO = 1.6
# Absolute bounds (int16 RMS). The old 2000 ceiling is what made you shout.
MIN_START_RMS = 140.0
MAX_START_RMS = 1100.0
MIN_KEEP_RMS = 90.0

# Room tracking: a low percentile over the last few seconds of listening.
_AMBIENT_WINDOW = 60        # chunks (~5s at 80ms)
_AMBIENT_PERCENTILE = 20
_DEFAULT_AMBIENT = 60.0
_levels = collections.deque(maxlen=_AMBIENT_WINDOW)


def _rms(block):
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def note_ambient(block):
    """Called with every chunk the wake loop reads. Cheap: one RMS."""
    _levels.append(_rms(block))


def ambient():
    """The room's background level: the quiet end of recent listening."""
    if len(_levels) < 8:
        return _DEFAULT_AMBIENT
    return max(10.0, float(np.percentile(np.asarray(_levels), _AMBIENT_PERCENTILE)))


def gates(room=None):
    """(start, keep) RMS thresholds for this room."""
    room = ambient() if room is None else room
    start = min(MAX_START_RMS, max(MIN_START_RMS, room * START_RATIO)) / SENSITIVITY
    keep = max(MIN_KEEP_RMS, room * KEEP_RATIO) / SENSITIVITY
    return start, min(keep, start * 0.8)


def normalize(samples, target_peak=0.6, max_gain=6.0):
    """Lifts a quiet recording so its loudest part reaches target_peak of
    full scale -- a soft voice transcribes far better at a normal level. The
    gain is capped, and recordings whose loudest moment is barely above the
    room are left alone, so silence isn't boosted into something Whisper
    "hears" as words."""
    samples = np.asarray(samples, dtype=np.int16).reshape(-1)
    if not len(samples):
        return samples
    floats = samples.astype(np.float32)
    peak = float(np.percentile(np.abs(floats), 99.9))
    if peak < max(200.0, ambient() * 3.0):
        return samples
    gain = min(max_gain, (32767.0 * target_peak) / peak)
    if gain <= 1.05:
        return samples
    return np.clip(floats * gain, -32768, 32767).astype(np.int16)


def record_until_silence(stream, chunk_size, samplerate=16000, on_endpoint=None,
                         give_up_after=None):
    """Reads from an already-open sounddevice stream (int16, mono) until the
    speaker stops. Returns an int16 numpy array (or None, see give_up_after).

    Does NOT open its own stream -- that would make two streams fight over
    the mic.

    on_endpoint: optional callback fired the instant we decide speech ended,
    before frames are concatenated.

    give_up_after: optional seconds. If nothing has crossed the start gate by
    then, returns None instead of waiting out MAX_RECORD_SECONDS. Only useful
    where silence is a legitimate answer -- a follow-up question rather than a
    command that was deliberately triggered.
    """
    ui.set_state(ui.RECORDING)
    print("Recording... (speak now)")

    room = ambient()
    start_gate, keep_gate = gates(room)

    frames = []
    started = False
    speech_seconds = 0.0
    loudest = 0.0
    silence_start = None
    start_time = time.time()
    chunk_seconds = chunk_size / float(samplerate)

    while True:
        data, overflowed = stream.read(chunk_size)
        if overflowed:
            print("input overflow (recording)")

        frames.append(data.copy())
        level = _rms(data)
        loudest = max(loudest, level)
        elapsed = time.time() - start_time
        ui.push_rms(level)

        gate = keep_gate if started else start_gate
        if level > gate:
            speech_seconds += chunk_seconds
            if speech_seconds >= MIN_SPEECH_SECONDS:
                started = True
            silence_start = None
        else:
            if level < start_gate * 0.5:
                _levels.append(level)  # quiet chunks keep teaching the room
            # Only a pause AFTER you've started can end it, and never in the
            # opening moments of a short phrase.
            if started and elapsed > min(GRACE_PERIOD, speech_seconds + SILENCE_SECONDS):
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_SECONDS:
                    break

        # Nobody started talking.
        if give_up_after is not None and elapsed > give_up_after and not started:
            print("Nothing said.")
            _log(room, start_gate, keep_gate, loudest, speech_seconds, elapsed, "nothing said")
            return None

        if elapsed > MAX_RECORD_SECONDS:
            print("Max recording time reached.")
            break

    if on_endpoint is not None:
        try:
            on_endpoint()
        except Exception:
            pass

    took = time.time() - start_time
    print("Done recording ({:.2f}s).".format(took))
    _log(room, start_gate, keep_gate, loudest, speech_seconds, took, "ok")
    return np.concatenate(frames, axis=0)


def _log(room, start_gate, keep_gate, loudest, speech, took, outcome):
    """One dev-log row per recording, so the gates can be tuned from real
    numbers instead of guesses."""
    try:
        from commands import devlog
        devlog.stt("mic", "record", "INFO", detail=(
            "{} | room {:.0f} start {:.0f} keep {:.0f} loudest {:.0f} speech {:.1f}s "
            "sens {:.2f}".format(outcome, room, start_gate, keep_gate, loudest, speech,
                                 SENSITIVITY)), seconds=took)
    except Exception:
        pass
