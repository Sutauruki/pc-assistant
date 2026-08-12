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

# Extra quiet time after the samples are written before we call the sound
# finished. stream.write() returns once the data is *buffered*, not once it
# has been heard, so the device still has up to OUTPUT_LATENCY of audio in
# hand. Without this the ready cue is still sounding when the mic opens.
#
# Deliberately NOT a module-level constant anymore. audio_out.py imports
# this module (`from listener import ui` pulls in ui.py, which imports
# cues.py, which imports audio_out.py) *before* audio_out.py finishes
# defining OUTPUT_LATENCY further down its own file -- so reading it here at
# import time only worked when something else happened to import ui.py (or
# cues.py) before it imported audio_out.py directly. Whichever module a
# process imports *first* decides whether that accident holds -- main.py's
# import order happened to keep it true, but `python -m listener.tts_local`
# imports audio_out.py first and broke it: "partially initialized module
# 'listener.audio_out' has no attribute 'OUTPUT_LATENCY'". Reading the value
# lazily, at call time instead of import time, means audio_out.py has always
# finished loading by the time anyone asks -- correct regardless of which
# module happens to be the process's entry point.
def _settle_seconds():
    return audio_out.OUTPUT_LATENCY + 0.02

_cache = {}
_lock = threading.Lock()
_enabled = os.environ.get("JARVIS_NO_CUES") is None


def _current_rate():
    """The rate the shared output stream is already open at.

    get_stream() closes and reopens the device when asked for a different
    rate, and a reopen mid-turn is both slow and audible. Matching whatever
    is open means a cue never causes one -- tts pins 24kHz, so in practice
    these agree, but reading it keeps that from being an assumption.
    """
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
        # Raised cosine rather than a straight line: the slope starts at zero,
        # so there is no corner in the waveform to hear.
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
    # Zero phase at t=0 for every partial, so the note begins at amplitude
    # zero and the attack ramp has nothing to fight.
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

    # Normalise then scale, so both cues land at the same perceived level no
    # matter how their partials happened to add up.
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
    """Mic is live. Blocks until the sound has actually finished.

    Blocking is the point. record.py calibrates its noise floor from the
    opening chunks of the recording, so a cue still ringing when the mic
    opens gets measured as room noise and pushes the speech gate up -- quiet
    speech then fails to register as speech at all.
    """
    return _play("ready", READY_NOTES, wait)


def done(wait=False):
    """Recording closed, thinking now.

    Fire and forget by default: transcription starts immediately and there is
    nothing left to protect from bleed.
    """
    if not _enabled:
        return
    if wait:
        _play("done", DONE_NOTES, True)
        return
    threading.Thread(
        target=_play, args=("done", DONE_NOTES, False), daemon=True
    ).start()


def warm():
    """Renders both cues up front so the first one isn't late.

    Only matters once -- generation is sub-millisecond -- but the first cue
    is also the first thing the speaker plays, and doing it at startup means
    any device wake-up happens before it is standing between you and talking.
    """
    if not _enabled:
        return
    try:
        _samples("ready", READY_NOTES)
        _samples("done", DONE_NOTES)
    except Exception as e:
        print("couldn't prepare cues ({})".format(e))


if __name__ == "__main__":
    # Hear them without running the assistant. Tune CUE_VOLUME and the note
    # tables against this rather than against a live turn.
    audio_out.get_stream()
    print("ready cue (rising -- talk now)")
    ready()
    time.sleep(0.7)
    print("done cue (falling -- thinking)")
    done(wait=True)
    time.sleep(0.3)
    audio_out.close_stream()