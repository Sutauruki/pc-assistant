import os
import io
import wave
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-3.1-flash-tts-preview"
VOICE_NAME = "Fenrir"  # options: Puck, Charon, Kore, Fenrir, Zephyr
SAMPLE_RATE = 24000  # Gemini TTS always returns 24kHz PCM

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        _client = genai.Client(api_key=api_key)
    return _client


def _pcm_to_wav_bytes(pcm_bytes, sample_rate=SAMPLE_RATE):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def speak(text):
    """Converts text to speech via Gemini and plays it immediately."""
    if not text or not text.strip():
        return

    client = get_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=VOICE_NAME
                    )
                )
            ),
        ),
    )

    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    wav_bytes = _pcm_to_wav_bytes(pcm_data)

    # play directly from memory
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()  # block until playback finishes