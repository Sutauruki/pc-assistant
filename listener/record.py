import numpy as np
import time

SILENCE_THRESHOLD = 500      # RMS amplitude below this = silence (tune later)
SILENCE_DURATION = 4.5       # seconds of silence before stopping (gives room to think)
GRACE_PERIOD = 1.0           # seconds at start where silence is ignored entirely
MAX_RECORD_SECONDS = 20      # hard cap so it never hangs forever


def record_until_silence(stream, chunk_size):
    """Reads from an already-open sounddevice stream (int16, mono)
    until it detects sustained silence. Does NOT open its own stream —
    this avoids two streams fighting over the mic."""
    print("Recording... (speak now)")
    frames = []
    silence_start = None
    start_time = time.time()

    while True:
        data, overflowed = stream.read(chunk_size)
        if overflowed:
            print("input overflow (recording)")

        frames.append(data.copy())

        elapsed = time.time() - start_time
        rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

        if elapsed < GRACE_PERIOD:
            pass
        elif rms < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > SILENCE_DURATION:
                break
        else:
            silence_start = None

        if elapsed > MAX_RECORD_SECONDS:
            print("Max recording time reached.")
            break

    print("Done recording.")
    audio = np.concatenate(frames, axis=0)
    return audio