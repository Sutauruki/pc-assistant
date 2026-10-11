"""Wake-word detection for openWakeWord, with loudness normalisation.

WHY THE INPUT IS NORMALISED
---------------------------
openWakeWord's scores track input loudness: the same word spoken quietly
scores a fraction of the same word shouted. A mic that transcribes fine at a
normal speaking voice (Moonshine normalises internally) still needs shouting
at the wake word, because the raw int16 amplitudes are what the model sees.
So before every prediction we push the chunk toward a fixed RMS level with a
slow-attack/slow-release gain (a proper AGC, not per-chunk scaling -- the
model keeps state across chunks, so wildly different levels between frames
would break its feature buffer). A normal voice then scores like a loud one.

Chunks that are effectively silence are skipped entirely: no gain applied,
no model call. That is both a false-trigger guard and the main CPU saving
for a 24/7 listener -- most of the day the mic hears nothing, and near-
silence needs no inference.

Tuning via environment:
  WAKE_THRESHOLD  score needed to trigger (default 0.40)
  WAKE_SILENCE_RMS  int16 RMS below which a chunk is skipped (default 60)
  WAKE_TARGET_RMS   RMS level input is pushed toward (default 2500)
"""

import os

import numpy as np

from openwakeword.model import Model

# Score needed to trigger. 0.5 assumes audio at the model's training level;
# combined with the gain normalisation, 0.4 catches a normal speaking voice
# without opening the door to room noise.
THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.40"))

# Int16 RMS below which a chunk is treated as silence and skipped. Speech is
# generally >150 RMS; a quiet room reads single digits.
SILENCE_RMS = float(os.getenv("WAKE_SILENCE_RMS", "60.0"))

# Int16 RMS we push speech toward before the model sees it. Typical speech
# RMS is 1000-4000, so 2500 sits mid-range.
TARGET_RMS = float(os.getenv("WAKE_TARGET_RMS", "2500.0"))

# Clamp the gain so a whisper isn't boosted to the point of amplifying
# background noise into a trigger, and a scream isn't distorted.
MIN_GAIN = 0.25
MAX_GAIN = 6.0

# How fast the AGC adapts. Attack (needing more gain) is fast so a quiet
# voice is caught within a few chunks; release (needing less) is slow so a
# single loud word doesn't flatten the following syllables.
_ATTACK = 0.25
_RELEASE = 0.05

_gain = 1.0


def _autogain(chunk):
    """Slow-track a loudness gain and return an int16 copy scaled by it.

    Returns None when the chunk is effectively silence -- the caller skips
    the model call entirely in that case.
    """
    global _gain
    rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
    if rms < SILENCE_RMS:
        return None

    wanted = min(MAX_GAIN, max(MIN_GAIN, TARGET_RMS / rms)) if rms > 0 else 1.0
    alpha = _ATTACK if wanted > _gain else _RELEASE
    _gain += alpha * (wanted - _gain)

    scaled = chunk.astype(np.float32) * _gain
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def load_wake_word_model(wake_word="hey_jarvis"):
    print("Loading wake word model...")
    return Model(wakeword_models=[wake_word])


def check_wake_word(model, audio_chunk, threshold=None):
    """audio_chunk should be int16 mono numpy array.
    Returns the detected wakeword name if triggered, else None."""
    prepared = _autogain(audio_chunk)
    if prepared is None:
        return None

    thr = THRESHOLD if threshold is None else threshold
    prediction = model.predict(prepared)
    for wakeword, score in prediction.items():
        if score > thr:
            print("Wake word detected: {} (score: {:.2f})".format(wakeword, score))
            return wakeword
    return None
