import sounddevice as sd
import numpy as np
import wave
import time
import os
import threading
import comtypes

from listener.wake_word import load_wake_word_model, check_wake_word
from listener.record import record_until_silence
from listener import stt_gemini
from listener import stt_local
from commands.parser import parse_command, generate_reply
from commands.executor import execute_actions
from listener.tts import speak

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
COOLDOWN_SECONDS = 2.0
RECORDINGS_DIR = "recordings"


def save_wav(audio, filename):
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    filepath = os.path.join(RECORDINGS_DIR, filename)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    print(f"Saved to {filepath}")


def main():
    model = load_wake_word_model("hey_jarvis")

    print("Listening for 'Hey Jarvis'... (Ctrl+C to stop)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
    ) as stream:
        while True:
            data, overflowed = stream.read(CHUNK_SIZE)
            if overflowed:
                print("input overflow (listening)")

            audio_chunk = data[:, 0]
            wakeword = check_wake_word(model, audio_chunk)

            if wakeword:
                audio = record_until_silence(stream, CHUNK_SIZE)
                filename = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.wav"
                save_wav(audio, filename)

                try:
                    text = stt_gemini.transcribe(audio)
                    print(f"Heard (Gemini): \"{text}\"")
                except Exception as e:
                    print(f"Gemini STT failed ({e}), falling back to local...")
                    text = stt_local.transcribe(audio)
                    print(f"Heard (local): \"{text}\"")

                actions, reply = parse_command(text)
                print(f"Parsed actions: {actions}")

                # Run execution and speech concurrently — execution is fast
                # (local), speaking takes real time (network + playback),
                # so overlapping them cuts total wait roughly in half.
                exec_results = {}

                def run_execution():
                    comtypes.CoInitialize()  # needed for pycaw on this thread
                    try:
                        exec_results["results"] = execute_actions(actions)
                    finally:
                        comtypes.CoUninitialize()

                exec_thread = threading.Thread(target=run_execution)
                exec_thread.start()

                print(f"Jarvis says: \"{reply}\"")
                speak(reply)

                exec_thread.join()
                results = exec_results.get("results", [])

                # only pay for a second Groq+TTS round trip if something
                # actually failed — rare path, spoken as a follow-up
                failed = any(
                    "Failed" in r or "Couldn't" in r or "Didn't" in r
                    for r in results
                )
                if failed:
                    correction = generate_reply(results)
                    print(f"Jarvis corrects: \"{correction}\"")
                    speak(correction)

                # Flush any backlog audio that built up in the stream buffer
                # while transcription was running (STT takes real time on this
                # CPU, and the mic keeps feeding the buffer meanwhile).
                # Without this, stale audio bleeds into the next recording.
                if stream.read_available > 0:
                    stream.read(stream.read_available)

                # reset the wake word model's internal state so it doesn't
                # think it's still mid-detection, and pause briefly before
                # listening again
                model.reset()
                time.sleep(COOLDOWN_SECONDS)
                print("Listening for 'Hey Jarvis'... (Ctrl+C to stop)")


if __name__ == "__main__":
    main()