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

WHICH WAKE WORD
---------------
A custom model wins when there is one: WAKE_WORD_MODEL (a path), or else
models/hey_<name>.onnx for the assistant's name (commands/identity.py) --
models/hey_novyx.onnx for Novyx. Train it free with openWakeWord's notebook
(see the README, "Custom wake word"). Without one, the built-in "Hey Jarvis"
model is used and the UI says so.

Tuning via environment:
  WAKE_THRESHOLD  score needed to trigger (default 0.40)
  WAKE_SILENCE_RMS  int16 RMS below which a chunk is skipped (default 60)
  WAKE_TARGET_RMS   RMS level input is pushed toward (default 2500)
"""

import os
import re
from pathlib import Path

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


_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _custom_model():
    """Path to a custom wake-word model, or None."""
    from commands import identity
    configured = os.getenv("WAKE_WORD_MODEL", "").strip().strip('"')
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = _MODELS_DIR.parent / path
        if path.exists():
            return path
        print("wake word: WAKE_WORD_MODEL {} not found, using the default".format(configured))
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", identity.NAME.lower()).strip("_")
    for candidate in (_MODELS_DIR / "hey_{}.onnx".format(slug),
                      _MODELS_DIR / "{}.onnx".format(slug)):
        if candidate.exists():
            return candidate
    return None


def _phrase_from(path):
    """'hey_novyx.onnx' -> 'Hey Novyx'."""
    words = re.split(r"[_\-\s]+", Path(path).stem)
    return " ".join(w.capitalize() for w in words if w and not re.fullmatch(r"v?\d+", w))


def load_wake_word_model(wake_word=None):
    """The custom model for the assistant's name when there is one, else the
    built-in "hey_jarvis". Tells commands/identity.py which phrase is live."""
    from commands import identity
    custom = None if wake_word else _custom_model()
    if custom is not None:
        print("Loading wake word model {}...".format(custom.name))
        try:
            framework = "onnx" if custom.suffix == ".onnx" else "tflite"
            model = Model(wakeword_models=[str(custom)], inference_framework=framework)
            identity.set_wake_phrase(_phrase_from(custom))
            return model
        except Exception as e:
            print("wake word: couldn't load {} ({}), using Hey Jarvis".format(custom.name, e))
    print("Loading wake word model...")
    identity.set_wake_phrase("Hey Jarvis")
    return Model(wakeword_models=[wake_word or "hey_jarvis"])


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
