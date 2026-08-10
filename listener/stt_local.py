import json
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # swap to lgraph/full model later

_model = None  # loaded once, reused across calls

# Restrict recognition to words we actually expect from commands.
# This is a *word list*, not exact phrases — Vosk can still combine
# them in any order/combination. "[unk]" lets it fall back gracefully
# for anything outside this list instead of guessing wildly.
# Expand this as we add more commands to the registry.
COMMAND_VOCAB = [
    "open", "close", "set", "check", "take",
    "chrome", "notepad", "calculator", "spotify",
    "volume", "screenshot", "weather",
    "to", "the", "percent",
    "zero", "ten", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred",
    "[unk]"
]


def load_stt_model():
    global _model
    if _model is None:
        print("Loading local fallback STT model (Vosk)...")
        _model = Model(MODEL_PATH)
    return _model


def transcribe(audio):
    """Local, fully offline fallback transcription. Used only when
    Gemini STT fails (rate limit, no internet, API error)."""
    model = load_stt_model()
    grammar = json.dumps(COMMAND_VOCAB)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)
    recognizer.SetWords(True)

    recognizer.AcceptWaveform(audio.tobytes())
    result = json.loads(recognizer.FinalResult())

    return result.get("text", "")