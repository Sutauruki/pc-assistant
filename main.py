"""Wake word loop and turn orchestration.

Goes in: main.py  (repo root)
"""

import os
import threading
import time
import traceback
import wave

import comtypes
import numpy as np
import sounddevice as sd

from listener.wake_word import load_wake_word_model, check_wake_word
from listener.record import record_until_silence
from listener import stt_gemini
from listener import stt_local
from listener import ui
from listener.tts import speak
from listener.audio_out import get_stream as get_output_stream, close_stream
from commands.parser import parse_command, generate_reply
from commands.executor import execute_actions

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280

# Was 2.0s. The wake word model is reset explicitly, so a long sleep here
# just makes back-to-back commands feel sluggish.
COOLDOWN_SECONDS = 0.4

RECORDINGS_DIR = "recordings"
SAVE_RECORDINGS = True  # set False to skip disk writes entirely

# Speaking and acting at the same time is what removed the dead air -- the
# reply starts while apps are still launching. Set False to go back to
# strictly speak-then-act, at the cost of the latency win.
OVERLAP_SPEECH_AND_ACTIONS = True

# ...with one exception. set_volume drives the same output device the reply is
# playing through, so running it mid-sentence audibly jumps the voice's volume
# partway through the word. These wait for the sentence to finish regardless.
DEFERRED_ACTIONS = {"set_volume"}

# Upper bound on waiting for a reply to finish playing. Long enough that a
# normal sentence is never cut off (the model caps input at 200 chars, so
# ~15s of speech at most), short enough that a hung request doesn't end the
# session in silence.
SPEECH_JOIN_TIMEOUT = 45.0


def _write_wav(audio, filepath):
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def save_wav_async(audio):
    """Archives the clip on a background thread. The old code wrote to disk
    *before* transcribing, putting file I/O on the critical path for no
    reason -- STT works off the in-memory array."""
    if not SAVE_RECORDINGS:
        return

    filename = "recording_{}.wav".format(time.strftime("%Y%m%d_%H%M%S"))

    def run():
        try:
            os.makedirs(RECORDINGS_DIR, exist_ok=True)
            _write_wav(audio, os.path.join(RECORDINGS_DIR, filename))
        except Exception as e:
            print("Failed to archive recording: {}".format(e))

    threading.Thread(target=run, daemon=True).start()


def transcribe(audio):
    try:
        text = stt_gemini.transcribe(audio)
        print('Heard (Gemini): "{}"'.format(text))
    except Exception as e:
        print("Gemini STT failed ({}), falling back to local...".format(e))
        text = stt_local.transcribe(audio)
        print('Heard (local): "{}"'.format(text))
    ui.set_heard(text)
    return text


def run_actions(actions):
    """Executes on this thread with COM initialised (pycaw needs it)."""
    if not actions:
        return []
    comtypes.CoInitialize()
    try:
        return execute_actions(actions)
    finally:
        comtypes.CoUninitialize()


def split_actions(actions):
    """Separates what can run under the voice from what has to wait for it."""
    if not OVERLAP_SPEECH_AND_ACTIONS:
        return [], list(actions)

    immediate, deferred = [], []
    for action in actions:
        target = deferred if action.get("action") in DEFERRED_ACTIONS else immediate
        target.append(action)
    return immediate, deferred


def handle_utterance(audio):
    t_start = time.perf_counter()
    marks = {}

    # STT is the long pole (several seconds), so the spinner goes up the
    # moment recording ends rather than waiting for the parse step.
    ui.set_state(ui.THINKING)
    save_wav_async(audio)  # off the critical path

    text = transcribe(audio)
    marks["stt"] = time.perf_counter() - t_start

    speech = {}

    def on_first_audio():
        marks["first_audio"] = time.perf_counter() - t_start

    def speak_guarded(reply):
        # speak() already swallows its own failures, but it does network I/O
        # and this is a bare thread target -- anything it misses would reach
        # threading.excepthook and print through the panel.
        try:
            speak(reply, on_first_audio=on_first_audio)
        except Exception as e:
            print("speech failed: {}".format(e))

    def on_reply(reply):
        # Fires mid-stream, before the actions array has finished generating.
        # This is where the latency actually goes away.
        marks["reply"] = time.perf_counter() - t_start
        print('Jarvis says: "{}"'.format(reply))
        ui.set_reply(reply)
        speech["thread"] = threading.Thread(
            target=speak_guarded,
            args=(reply,),
            daemon=True,
        )
        speech["thread"].start()

    actions, reply = parse_command(text, on_reply=on_reply)
    marks["parse"] = time.perf_counter() - t_start
    print("Parsed actions: {}".format(actions))

    # Safety net: if the model ignored the key ordering, on_reply never fired.
    if "thread" not in speech and reply:
        on_reply(reply)

    immediate, deferred = split_actions(actions)
    exec_box = {}

    def run_immediate():
        # An exception here used to disappear: the thread died, the traceback
        # printed, exec_box stayed empty, and exec_box.get("results", [])
        # handed back a clean empty list. The turn then carried on as though
        # nothing had been asked of it -- no error, no spoken correction.
        try:
            exec_box["results"] = run_actions(immediate)
        except Exception as e:
            exec_box["results"] = ["Failed to run that: {}".format(e)]
            print("action execution failed: {}".format(e))
            traceback.print_exc()

    exec_thread = threading.Thread(target=run_immediate, daemon=True)
    exec_thread.start()
    exec_thread.join()

    # Let the sentence land before touching the output device. Bounded: a
    # wedged TTS request would otherwise hold the wake loop forever, and a
    # daemon thread that never returns is invisible -- no error, no reply,
    # no more listening.
    if "thread" in speech:
        speech["thread"].join(timeout=SPEECH_JOIN_TIMEOUT)
        if speech["thread"].is_alive():
            print("speech is taking too long -- moving on")
    marks["spoken"] = time.perf_counter() - t_start

    results = list(exec_box.get("results", []))
    if deferred:
        # Same guard as above, but this one runs on the wake loop's own
        # thread -- an escape here would end the whole session.
        try:
            results.extend(run_actions(deferred))
        except Exception as e:
            results.append("Failed to run that: {}".format(e))
            print("deferred execution failed: {}".format(e))
            traceback.print_exc()

    timing = (
        "stt {:.2f}s | reply {:.2f}s | first audio {:.2f}s"
        " | parse {:.2f}s | spoken {:.2f}s".format(
            marks.get("stt", 0.0),
            marks.get("reply", 0.0),
            marks.get("first_audio", 0.0),
            marks.get("parse", 0.0),
            marks.get("spoken", 0.0),
        )
    )
    ui.set_timing(timing)
    print("  timing: " + timing)

    # Only pay for a second Groq + TTS round trip if something actually
    # failed -- rare path, spoken as a follow-up. Matched case-insensitively
    # so a failure phrase mid-sentence ("...but couldn't close the rest")
    # counts the same as one at the start.
    failed = any(
        any(word in r.lower() for word in ("failed", "couldn't", "didn't"))
        for r in results
    )
    if failed:
        correction = generate_reply(results)
        print('Jarvis corrects: "{}"'.format(correction))
        ui.set_reply(correction)
        speak(correction)


def _banner():
    """The panel already shows the state, so the banner is only useful when
    the dashboard isn't running."""
    if not ui.is_active():
        print("Listening for 'Hey Jarvis'... (Ctrl+C to stop)")


def main():
    # Load first, draw second. openwakeword and PortAudio both emit startup
    # chatter from native code, which goes to the real file descriptors and
    # sails straight past the stdout proxy -- so anything drawn before this
    # point gets written over and leaves a fragment of a panel on screen.
    model = load_wake_word_model("hey_jarvis")
    get_output_stream()  # open the speaker up front so the first reply isn't slow

    ui.start()  # no-op when stdout isn't a terminal, or JARVIS_NO_UI is set
    ui.set_state(ui.LISTENING)
    _banner()

    try:
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

                # Feed the idle wave off the same chunks the wake word model
                # sees -- the panel twitches with the room, which is the cheap
                # way to show the mic is actually live.
                ui.push_rms(
                    float(np.sqrt(np.mean(
                        data[:, 0].astype(np.float32) ** 2
                    )))
                )

                if not check_wake_word(model, data[:, 0]):
                    continue

                audio = record_until_silence(
                    stream, CHUNK_SIZE, samplerate=SAMPLE_RATE
                )

                # Last net. One bad turn -- a dropped network call, an app
                # that won't die -- should cost that turn and nothing more.
                # Before this, any escape from a turn ended the session and
                # left the terminal holding a half-drawn panel.
                try:
                    handle_utterance(audio)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print("turn failed: {}".format(e))
                    traceback.print_exc()

                # Flush backlog that piled up in the input buffer while we
                # were transcribing and speaking. Without this, stale audio
                # bleeds into the next recording.
                pending = stream.read_available
                if pending > 0:
                    stream.read(pending)

                # Reset the wake word model's internal state so it doesn't
                # think it's still mid-detection.
                model.reset()
                time.sleep(COOLDOWN_SECONDS)
                ui.set_state(ui.LISTENING)
                _banner()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        # ui.stop() last and unconditionally: it un-hides the cursor and puts
        # sys.stdout/stderr back. If close_stream() raised first, the terminal
        # would be left with no cursor, both streams still proxied, and the
        # traceback printing through a panel nobody is redrawing.
        try:
            close_stream()
        except Exception as e:
            print("couldn't close the audio device: {}".format(e))
        ui.stop()


if __name__ == "__main__":
    main()