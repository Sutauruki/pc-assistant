"""Earcons -- the two sounds that bracket a recording.

Goes in: listener/cues.py   (new file)

    ready()     rising  G5 -> C6   "mic is live, talk now"
    done()      falling F5 -> C5   "got it, thinking"

Same interval both times, a perfect fourth, in opposite directions. That is
the whole design: matched enough to be obviously one pair of sounds from one
program, opposite enough that you never have to think about which one you
just heard. Rising reads as "go", falling as "received" -- the convention
every shutter, scanner and door chime already trained everyone on.

Synthesised, not loaded from disk. Two hundred milliseconds of numpy costs
nothing to generate, there is no asset to lose, and the pitch and volume are
constants you can edit instead of files you have to re-record.

WHY THEY DON'T CLICK
--------------------
A sine wave switched on and off abruptly is a step discontinuity, and a step
is broadband -- you hear a click at each edge, which is exactly what makes
hand-rolled notification tones sound cheap. Three things prevent it here: a
raised-cosine attack, an explicit release ramp to true zero, and starting at
zero phase. The exponential decay in between is what gives it a struck,
bell-like quality rather than a test tone.
"""

import os
import threading
import time

import numpy as np

from listener import audio_out

# Peak amplitude, 0..1. "Soft" was the brief -- this sits around -16 dBFS,
# well under a speaking voice, audible in another tab without being a jolt.
CUE_VOLUME = 0.16

# Harmonic mix. A pure sine reads as a diagnostic beep; a little second and
# third partial reads as a small struck bell.
HARMONICS = ((1.0, 1.0), (2.0, 0.30), (3.0, 0.11))

# How fast each note dies away. Higher is shorter and more percussive.
DECAY = 5.5

ATTACK_SECONDS = 0.006
RELEASE_SECONDS = 0.012

# (frequency, start offset, duration, gain). The notes overlap on purpose --
# the second arrives while the first is still ringing, which is what makes it
# one gesture instead of two separate beeps.
READY_NOTES = (
    (783.99, 0.000, 0.100, 1.00),   # G5
    (1046.50, 0.045, 0.130, 0.90),  # C6
)

DONE_NOTES = (
    (698.46, 0.000, 0.130, 1.00),   # F5
    (523.25, 0.060, 0.180, 0.95),   # C5
)


def _settle_seconds():
    """Extra quiet time after samples are written before calling sound finished.
    
    Evaluated lazily at call time to avoid circular import errors with audio_out.
    """
    return audio_out.OUTPUT_LATENCY + 0.02


_cache = {}
_lock = threading.Lock()
_enabled = os.environ.get("JARVIS_NO_CUES") is None


def _current_rate():
    """The rate the shared output stream is already open at."""
    rate = getattr(audio_out, "_stream_rate", None)
    try:
        return int(rate) if rate else int(audio_out.DEFAULT_SAMPLE_RATE)
    except (TypeError, ValueError):
        return int(audio_out.DEFAULT_SAMPLE_RATE)


def _envelope(count, rate):
    """Struck-and-decaying, with both ends pinned to silence."""
    if count <= 0:
        return np.zeros(0)

    t = np.arange(count) / float(rate)
    seconds = count / float(rate)

    env = np.exp(-DECAY * t / seconds)

    attack = min(int(ATTACK_SECONDS * rate), count)
    if attack > 1:
        # Raised cosine slope starts at zero to avoid wave corners/clicks
        env[:attack] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, attack))

    release = min(int(RELEASE_SECONDS * rate), count)
    if release > 1:
        env[-release:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, release))

    return env


def _note(freq, seconds, rate):
    count = int(seconds * rate)
    if count <= 0:
        return np.zeros(0)

    t = np.arange(count) / float(rate)
    # Zero phase at t=0 for every partial
    wave = np.zeros(count)
    for multiple, amplitude in HARMONICS:
        wave += amplitude * np.sin(2.0 * np.pi * freq * multiple * t)

    return wave * _envelope(count, rate)


def _render(notes, rate):
    total = max(start + duration for _f, start, duration, _g in notes)
    buf = np.zeros(int((total + 0.02) * rate))

    for freq, start, duration, gain in notes:
        segment = _note(freq, duration, rate) * gain
        offset = int(start * rate)
        buf[offset:offset + len(segment)] += segment

    # Normalize then scale for consistent perceived levels
    peak = float(np.max(np.abs(buf))) if len(buf) else 0.0
    if peak > 0:
        buf /= peak

    return (buf * CUE_VOLUME * 32767.0).astype(np.int16)


def _samples(name, notes):
    rate = _current_rate()
    key = (name, rate)
    with _lock:
        if key not in _cache:
            _cache[key] = _render(notes, rate)
        return _cache[key], rate


def _play(name, notes, wait):
    """Never raises. A cue failing is not a reason to lose the turn."""
    if not _enabled:
        return 0.0

    try:
        samples, rate = _samples(name, notes)
        audio_out.play_blocking(samples, rate)
        seconds = len(samples) / float(rate)
        settle = _settle_seconds()
        if wait:
            time.sleep(seconds + settle)
        return seconds + settle
    except Exception as e:
        print("cue failed ({}) -- continuing without it".format(e))
        return 0.0


def ready(wait=True):
    """Mic is live. Blocks until the sound has actually finished[cite: 6]."""
    return _play("ready", READY_NOTES, wait)


def done(wait=False):
    """Recording closed, thinking now[cite: 6]. Fire and forget by default[cite: 6]."""
    if not _enabled:
        return
    if wait:
        _play("done", DONE_NOTES, True)
        return
    threading.Thread(
        target=_play, args=("done", DONE_NOTES, False), daemon=True
    ).start()


def warm():
    """Renders both cues up front so the first one isn't late[cite: 6]."""
    if not _enabled:
        return
    try:
        _samples("ready", READY_NOTES)
        _samples("done", DONE_NOTES)
    except Exception as e:
        print("couldn't prepare cues ({})".format(e))


if __name__ == "__main__":
    audio_out.get_stream()
    print("ready cue (rising -- talk now)")
    ready()
    time.sleep(0.7)
    print("done cue (falling -- thinking)")
    done(wait=True)
    time.sleep(0.3)
    audio_out.close_stream()