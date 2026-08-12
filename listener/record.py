"""Fast endpointing for the mic.

Goes in: listener/record.py  (replaces existing)

One addition this round: give_up_after. The loop only ends on silence that
*follows* speech, which is right for a command -- you pressed the wake word, so
something is coming. It is wrong for the follow-up question after "should I
delete these?", where the honest possibility is that nobody answers. Without a
give-up path that case holds the microphone for the full MAX_RECORD_SECONDS and
the assistant is deaf for fifteen seconds, twice over.
"""

import time

import numpy as np

from listener import ui

# --- Endpointing -----------------------------------------------------------
# The old config waited 4.5s of silence after you stopped speaking before it
# even stopped recording. That one constant was the largest single chunk of
# the delay before Jarvis said anything.
#
# 700ms rides over natural mid-sentence pauses without making you wait.
SILENCE_HANGOVER = 0.7
GRACE_PERIOD = 3          # ignore silence in the opening moments
MIN_SPEECH_SECONDS = 0.25    # don't endpoint on a click or a door slam
MAX_RECORD_SECONDS = 60

# --- Adaptive gating -------------------------------------------------------
# A fixed RMS threshold breaks both ways: in a noisy room it never endpoints,
# on a quiet mic it endpoints mid-word. Measure the room during the first few
# chunks instead and set the speech gate relative to that.
NOISE_FLOOR_CHUNKS = 6
SPEECH_MULTIPLIER = 3.0
MIN_SPEECH_RMS = 250.0
MAX_SPEECH_RMS = 2000.0


def _rms(block):
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def record_until_silence(stream, chunk_size, samplerate=16000, on_endpoint=None,
                         give_up_after=None):
    """Reads from an already-open sounddevice stream (int16, mono) until the
    speaker stops. Returns an int16 numpy array.

    Does NOT open its own stream -- that would make two streams fight over
    the mic.

    on_endpoint: optional callback fired the instant we decide speech ended,
    before frames are concatenated.

    give_up_after: optional seconds. If nothing has crossed the speech gate by
    then, returns None instead of waiting out MAX_RECORD_SECONDS. Only useful
    where silence is a legitimate answer -- a follow-up question rather than a
    command that was deliberately triggered.
    """
    ui.set_state(ui.RECORDING)
    print("Recording... (speak now)")

    frames = []
    noise_samples = []
    gate = MIN_SPEECH_RMS

    silence_start = None
    speech_seconds = 0.0
    start_time = time.time()
    chunk_seconds = chunk_size / float(samplerate)

    while True:
        data, overflowed = stream.read(chunk_size)
        if overflowed:
            print("input overflow (recording)")

        frames.append(data.copy())
        level = _rms(data)
        elapsed = time.time() - start_time

        ui.push_rms(level)

        # calibrate the gate off the opening chunks
        if len(noise_samples) < NOISE_FLOOR_CHUNKS:
            noise_samples.append(level)
            if len(noise_samples) == NOISE_FLOOR_CHUNKS:
                floor = float(np.median(noise_samples))
                gate = min(
                    MAX_SPEECH_RMS,
                    max(MIN_SPEECH_RMS, floor * SPEECH_MULTIPLIER),
                )
            continue

        if level > gate:
            speech_seconds += chunk_seconds
            silence_start = None
        elif elapsed > GRACE_PERIOD and speech_seconds >= MIN_SPEECH_SECONDS:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= SILENCE_HANGOVER:
                break

        # Nobody started talking. Checked against the same speech_seconds the
        # endpointer uses, so a cough that never became words doesn't count as
        # an answer and hold the mic open.
        if (give_up_after is not None
                and elapsed > give_up_after
                and speech_seconds < MIN_SPEECH_SECONDS):
            print("Nothing said.")
            return None

        if elapsed > MAX_RECORD_SECONDS:
            print("Max recording time reached.")
            break

    if on_endpoint is not None:
        try:
            on_endpoint()
        except Exception:
            pass

    print("Done recording ({:.2f}s).".format(time.time() - start_time))
    return np.concatenate(frames, axis=0)