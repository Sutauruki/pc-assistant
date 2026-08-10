import os
import io
import wave
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

SAMPLE_RATE = 16000
MODEL_NAME = "gemini-3.6-flash"

_client = None  # created once, reused across calls


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        _client = genai.Client(api_key=api_key)
    return _client


def _audio_to_wav_bytes(audio):
    """Convert int16 numpy array to in-memory WAV bytes (no disk write needed)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def transcribe(audio):
    """audio: int16 numpy array, mono, 16kHz (same format our recorder produces).
    Sends it to Gemini for transcription and returns plain text."""
    client = get_client()
    wav_bytes = _audio_to_wav_bytes(audio)

    prompt = (
        "Transcribe the spoken words in this audio exactly as said. "
        "Return ONLY the transcribed text, no punctuation commentary, "
        "no explanation, nothing else — just the raw spoken words."
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            prompt,
        ],
    )

    text = response.text.strip() if response.text else ""
    return text