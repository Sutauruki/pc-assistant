"""Wake word loop and turn orchestration.

Goes in: main.py  (repo root -- replaces existing)

WHAT CHANGED: A TURN NOW HAS A ROUTE
------------------------------------
The parser used to answer one question -- "what should I do?" -- and an empty
action list was hardcoded to mean failure. Ask "what's the capital of France"
and you got "I didn't catch anything to do there", because the code had no
concept of a turn that isn't a command. It now answers two questions, and the
first one is the route:

  command  -- do it, say the one-liner. Unchanged from before.
  chat     -- don't do anything, answer the question. The router's stream is
              abandoned the moment "chat" is read, ~1s before it finishes, and
              the question goes to commands/chat.py.
  both     -- act AND answer. "What's my IP address" runs ipconfig, then hands
              the real output to the chat brain, so the spoken answer is the
              address rather than "3 lines of output, they're in the log".

Chat turns skip the router's filler line on purpose. A filler plus an answer is
two TTS calls with a hole between them; the earcon and the spinner already say
it's working.

CONFIRMATIONS ARRIVE AS THE NEXT UTTERANCE
------------------------------------------
An action that needs a yes can't wait for one -- the executor runs on a worker
thread and the answer arrives through a microphone this loop owns. So the
action parks itself in commands/confirm.py and the *next* turn resolves it. Two
consequences are visible here:

  - _run_turn checks for a pending question before parsing anything, so "yes"
    is never sent to the router as a fresh command;
  - _voice_loop re-records without the wake word while one is pending, because
    "Hey Jarvis, yes" to confirm a delete is a silly thing to ask of anyone.
    Bounded by MAX_CONFIRM_FOLLOWUPS, then the question is dropped.

ONE PLAYER PER TURN
-------------------
Every path through a turn speaks through _Voice, which joins the previous
thread before starting another. Two players on one device don't queue, they
interleave.
"""

import os
import subprocess
import sys
import threading
import time
import traceback
import wave

import comtypes
import numpy as np
import sounddevice as sd

from listener.wake_word import load_wake_word_model, check_wake_word
from listener.record import record_until_silence
from listener import console_input
from listener import cues
from listener import stt_moonshine
from listener import tts_fish
from listener import tts_moonshine
from listener import tts
from listener import ui
from listener import window_ui
from listener.audio_out import get_stream as get_output_stream, close_stream
from commands import agent
from commands import chat
from commands import cmd_control
from commands import confirm
from commands import devlog
from commands import logbook
from commands import memory
from commands import notify
from commands import recall
from commands.parser import parse_command, generate_reply, remember
from commands.executor import execute_actions

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280

# Was 2.0s. The wake word model is reset explicitly, so a long sleep here
# just makes back-to-back commands feel sluggish.
COOLDOWN_SECONDS = 0.4

# Typed turns skip the cooldown entirely. It exists to stop the wake word
# model re-triggering on the tail of its own detection; a keystroke has no
# such problem, and waiting 0.4s after pressing Enter just feels broken.
TYPED_COOLDOWN_SECONDS = 0.0

def _recordings_dir():
    """Where clips live: per-user app data, not the project tree.

    A recordings folder next to the code is something a user finds, opens, and
    worries about. %LOCALAPPDATA%\\Jarvis is where Windows programs put state
    nobody should need to see, and a voice assistant storing what it hears in
    a browseable project folder is a bad look on top of being bad practice.
    Falls back to a local folder when LOCALAPPDATA isn't set (non-Windows).
    """
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return os.path.join(base, "Jarvis", "recordings")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")


RECORDINGS_DIR = _recordings_dir()
SAVE_RECORDINGS = True  # set False to skip disk writes entirely.

# Recordings exist to be inspected for a few seconds after a turn (did it
# hear what I think I said?), not for the long haul. Deleting each clip this
# long after it was written keeps the folder tens of megabytes instead of
# gigabytes. Anything left over (a crash, a kill -9) is swept on startup.
RECORDING_RETENTION_SECONDS = 60.0

# Safety net on top of the retention timer: even if a clip can't be deleted
# (file locked, transient failure), the oldest ones make room. The retention
# timer is the plan; this is the roof.
MAX_RETAINED_RECORDINGS = 100

# Speaking and acting at the same time is what removed the dead air -- the
# reply starts while apps are still launching. Set False to go back to
# strictly speak-then-act, at the cost of the latency win.
OVERLAP_SPEECH_AND_ACTIONS = True

# ...with one exception. set_volume drives the same output device the reply is
# playing through, so running it mid-sentence audibly jumps the voice's volume
# partway through the word. These wait for the sentence to finish regardless.
DEFERRED_ACTIONS = {"set_volume"}

# Floor for waiting on speech to finish. Conversational answers scale past it
# -- see _speech_timeout -- but a one-line command reply never needs more.
SPEECH_JOIN_TIMEOUT = 45.0

# Actions whose result IS the answer. For everything else the result is a
# status line and the router's reply already covered it; for these, staying
# quiet means the user asked a question and got told "okay".
SPOKEN_RESULTS = {
    "list_windows", "list_files", "find_file", "where_is",
    "read_clipboard", "run_command", "explain_command",
    "list_memory", "what_do_you_remember", "show_memory",
    "understand_screen",
    # Tier 6: the result is the answer, a read that comes back as prose.
    "analyze_file", "youtube_summary", "web_search", "browser_run",
    "stop_preview",
    # The computer agent speaks its own result (and its own approval
    # questions, mid-task); research_explain's result is the explanation.
    "computer_task", "research_explain", "system_info",
}

# Actions whose result is already a finished spoken answer. In mode "both"
# these skip the second pass through the chat model that would only
# paraphrase them -- a whole extra model round trip for nothing.
DIRECT_ANSWERS = {"computer_task", "browser_run", "research_explain",
                  "research_save", "understand_screen", "system_info"}

# How many times to re-record for an answer to a pending confirmation before
# giving up on it. Two is enough for one misheard reply; more than that and
# something is wrong that repeating the question won't fix.
MAX_CONFIRM_FOLLOWUPS = 2

# How long to hold the mic open waiting for that answer before deciding nobody
# is going to give one. Well inside confirm.EXPIRY_SECONDS (45s), so the loop
# gives up before the request expires underneath it.
CONFIRM_SILENCE_SECONDS = 6.0

# And how long after that to keep watching the keyboard. The question is on
# screen as well as spoken, so somebody who reads it and reaches for the
# keyboard is slower than somebody who just answers out loud -- giving up on
# them at the same 6s throws away an answer that was on its way.
# 6 + 8 = 14s, still well inside confirm.EXPIRY_SECONDS.
CONFIRM_TYPED_SECONDS = 8.0

_FAILURE_WORDS = ("failed", "couldn't", "didn't")

# Recover from a vanishing input device (an unplugged headset raises
# PortAudioError on the next stream.read). These bound how hard _listen_loop
# tries to bring the mic back before surrendering to typed-only mode.
MIC_RECONNECT_ATTEMPTS = 10
MIC_RECONNECT_DELAY = 1.0

def _write_wav(audio, filepath):
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def _hide_dir(path):
    """Best-effort Windows 'hidden' attribute on a directory. Fails silently:
    a visible recordings folder is an aesthetic issue, not an error."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["attrib", "+h", path], check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def save_wav_async(audio):
    """Archives the clip on a background thread. The old code wrote to disk
    *before* transcribing, putting file I/O on the critical path for no
    reason -- STT works off the in-memory array. The clip self-deletes after
    RECORDING_RETENTION_SECONDS, so the folder stays small on its own."""
    if not SAVE_RECORDINGS:
        return

    filename = "recording_{}.wav".format(time.strftime("%Y%m%d_%H%M%S"))
    filepath = os.path.join(RECORDINGS_DIR, filename)

    def run():
        try:
            os.makedirs(RECORDINGS_DIR, exist_ok=True)
            _hide_dir(RECORDINGS_DIR)
            _write_wav(audio, filepath)
            time.sleep(RECORDING_RETENTION_SECONDS)
            try:
                os.remove(filepath)
            except OSError:
                pass
        except Exception as e:
            print("Failed to archive recording: {}".format(e))

    threading.Thread(target=run, daemon=True).start()


def sweep_recordings():
    """Called at startup: clear clips no process will ever delete.

    A clip's deletion is scheduled on the thread that wrote it, so a crash or
    a force-kill leaves the whole backlog behind. Retention is a guarantee
    only for a healthy process, and this sweeps the folder back to before the
    retention window -- and, as a roof, to MAX_RETAINED_RECORDINGS even if
    timestamps are broken or clock-skewed.
    """
    if not SAVE_RECORDINGS:
        return
    if not os.path.isdir(RECORDINGS_DIR):
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
    _hide_dir(RECORDINGS_DIR)
    if not os.path.isdir(RECORDINGS_DIR):
        return

    cutoff = time.time() - RECORDING_RETENTION_SECONDS
    retained = []
    try:
        for name in os.listdir(RECORDINGS_DIR):
            if not name.endswith(".wav"):
                continue
            path = os.path.join(RECORDINGS_DIR, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                os.remove(path)
                continue
            if mtime < cutoff:
                try:
                    os.remove(path)
                except OSError:
                    pass
            else:
                retained.append((mtime, path))

        retained.sort()
        while len(retained) > MAX_RETAINED_RECORDINGS:
            _, old = retained.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass
    except OSError as e:
        print("couldn't sweep recordings: {}".format(e))


def transcribe(audio):
    """Moonshine -- the only ear, and a fully offline one.

    Gemini, OpenRouter and Vosk are gone; nothing heard here has to leave the
    machine. Moonshine is open-vocabulary, so a long free-form question
    survives contact with the transcript just as well as a one-word command.
    """
    if not stt_moonshine.available():
        devlog.stt("moonshine", "transcribe", "SKIP", detail="not installed")
        print("Moonshine STT not installed")
        return ""

    t0 = time.perf_counter()
    try:
        text = stt_moonshine.transcribe(audio)
    except Exception as e:
        devlog.stt("moonshine", "transcribe", "FAIL",
                   detail=str(e)[:120], seconds=time.perf_counter() - t0)
        print("Moonshine STT failed ({}).".format(e))
        return ""

    if text:
        devlog.stt("moonshine", "transcribe", "OK",
                   detail="...{}".format(text[-80:]) if len(text) > 80 else text,
                   seconds=time.perf_counter() - t0)
        print('Heard (Moonshine): "{}"'.format(text))
        ui.set_heard(text)
    else:
        # Moonshine returned "" (silence detected). "Nothing heard" is a real
        # answer, not a failure -- _run_turn already knows what to do with it.
        devlog.stt("moonshine", "transcribe", "SKIP", detail="silence",
                   seconds=time.perf_counter() - t0)
        print("Heard (Moonshine): silence")
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


def _format_timing(marks):
    """Typed turns have no STT step, so the field is left out rather than
    printed as 0.00s -- a zero there reads like a bug in the timer."""
    parts = []
    if "stt" in marks:
        parts.append("stt {:.2f}s".format(marks["stt"]))
    parts.append("reply {:.2f}s".format(marks.get("reply", 0.0)))
    parts.append("first audio {:.2f}s".format(marks.get("first_audio", 0.0)))
    parts.append("parse {:.2f}s".format(marks.get("parse", 0.0)))
    parts.append("spoken {:.2f}s".format(marks.get("spoken", 0.0)))
    if "answer" in marks:
        parts.append("answer {:.2f}s".format(marks["answer"]))
    return " | ".join(parts)


def _speech_timeout(text):
    """How long to wait for a reply to finish playing.

    A fixed bound was fine when everything spoken was one short sentence. A
    conversational answer can be two hundred words, and the old 45s ceiling
    would cut it off mid-thought -- reintroducing, by a completely different
    route, the truncation the segmenting work just fixed. Roughly 11 chars a
    second at the playback speed, doubled for slack.
    """
    seconds = len(text or "") / 11.0 * 2.0
    return max(SPEECH_JOIN_TIMEOUT, seconds + 15.0)


def _failed(results):
    """True if any result reports a failure.

    Matched case-insensitively so a failure phrase mid-sentence ("...but
    couldn't close the rest") counts the same as one at the start. Every action
    module is written to put one of these words in its failure strings and none
    in its successes -- that convention is the entire error-detection mechanism.
    """
    return any(
        any(word in str(r).lower() for word in _FAILURE_WORDS) for r in results
    )


class _Voice:
    """Serialises everything a turn says.

    A turn can speak up to three times: the router's line, then results, then
    a correction. Each one is a network stream feeding a player that owns the
    output device, so starting a second before the first has drained gives you
    two voices interleaved sample-by-sample. Every call joins the previous
    thread first, which makes that impossible by construction rather than by
    everyone remembering to join.
    """

    def __init__(self, marks, t_start):
        self._marks = marks
        self._t_start = t_start
        self._thread = None
        self._text = ""

    def _on_first_audio(self):
        self._marks.setdefault("first_audio", time.perf_counter() - self._t_start)

    def say(self, text, background=True):
        """Speak `text`. background=False waits for it to finish."""
        if not text or not str(text).strip():
            return
        self.wait()
        spoken = str(text)
        self._text = spoken
        logbook.jarvis(spoken)

        def run():
            # speak() swallows its own failures, but it does network I/O on a
            # bare thread target -- anything it misses would reach
            # threading.excepthook and print straight through the panel.
            #
            # `spoken` rather than self._text: if a previous wait() timed out
            # and this attribute is reassigned underneath a still-running
            # thread, the closure should still say what it was handed.
            t0 = time.perf_counter()
            try:
                tts.speak(spoken, on_first_audio=self._on_first_audio)
                ui.set_engine(tts=tts.last_engine())
                devlog.tts(tts.last_engine(), "speak", "OK",
                           detail=spoken, seconds=time.perf_counter() - t0)
            except Exception as e:
                print("speech failed: {}".format(e))
                devlog.tts(tts.last_engine(), "speak", "FAIL",
                           detail="{}: {}".format(e, spoken),
                           seconds=time.perf_counter() - t0)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        if not background:
            self.wait()

    def stream(self, produce):
        """Speak an answer while it is still being written.

        `produce(put)` is called on this thread and should invoke put(sentence)
        for each finished sentence; it returns the complete text. Synthesis
        starts on the first sentence rather than the last, which is most of the
        latency win on a conversational turn.
        """
        self.wait()
        feeder = tts.SentenceFeeder()
        t_s0 = time.perf_counter()

        def run():
            try:
                tts.speak_stream(feeder, on_first_audio=self._on_first_audio)
            except Exception as e:
                print("speech failed: {}".format(e))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self._thread = thread

        try:
            text = produce(feeder.put)
        finally:
            # Unconditional: the synthesiser blocks on the queue until this
            # lands, so an exception in produce() would otherwise hang the
            # thread and then the join below.
            feeder.close()

        self._text = str(text or "")
        logbook.jarvis(self._text)
        if self._text.strip():
            ui.set_engine(tts=tts.last_engine())
            devlog.tts(tts.last_engine(), "speak", "OK",
                       detail=self._text, seconds=time.perf_counter() - t_s0)
        return self._text

    def wait(self):
        """Bounded join. A daemon thread that never returns is invisible -- no
        error, no reply, no more listening -- so this reports and moves on."""
        if self._thread is None:
            return
        self._thread.join(timeout=_speech_timeout(self._text))
        if self._thread.is_alive():
            print("speech is taking too long -- moving on")
        self._thread = None


def _resolve_confirmation(text, voice, typed=False):
    """Handles a turn that is an answer to a parked question.

    Returns True if the text was consumed as an answer. None from interpret()
    means it was neither -- the question is dropped and the words run as a
    normal turn, because someone who ignores "should I?" and says "open Chrome"
    wants Chrome opened, not the question repeated.
    """
    question = confirm.question()
    verdict = confirm.interpret(text)

    if verdict is True and confirm.needs_physical() and not typed:
        # A spoken yes can't approve this one (confirm.py, PHYSICAL
        # CONFIRMATIONS). Keep it pending and say how to approve it.
        line = ("For safety I can't take a spoken yes for that. Type yes in my "
                "window, or click Yes on the notification.")
        print("confirm: spoken yes refused for a physical confirmation")
        ui.set_reply(line)
        voice.say(line, background=False)
        return True

    if verdict is True:
        print("confirm: yes -> {}".format(question))
        parked = confirm.pending() or {}
        # The parked callable is whatever asked the question, and the common one
        # is a delete -- send2trash on Windows drives IFileOperation, which is
        # COM. run_actions() initialises COM around the executor, but this runs
        # on the wake loop thread, which is not that thread. Without this a
        # confirmed delete fails on "CoInitialize has not been called".
        comtypes.CoInitialize()
        try:
            result = confirm.accept()
        finally:
            comtypes.CoUninitialize()
        if result and "run_command" in parked.get("kinds", ()) and parked.get("asked"):
            # An approved shell command's output is raw -- "FreePhysicalMemory=
            # 2677600" -- so answer the original question from it, the same
            # way a "both" turn does, instead of reading it out.
            answer = _answer_parked(parked["asked"], result, voice)
            if answer:
                return True
        if result:
            print(result)
            ui.set_reply(result)
            voice.say(result, background=False)
        return True

    if verdict is False:
        print("confirm: no -> {}".format(question))
        result = confirm.reject()
        if result:
            ui.set_reply(result)
            voice.say(result, background=False)
        return True

    print("confirm: unrelated reply, dropping the question")
    confirm.clear()
    return False


def _answer_parked(asked, result, voice):
    """Answers `asked` in plain words from an approved command's output.
    Returns the spoken answer, or None so the caller falls back to reading
    the result out."""
    question = _chat_context(asked, [{"action": "run_command"}], [result])
    ui.set_state(ui.THINKING)
    try:
        answer = voice.stream(lambda put: chat.answer(question, on_sentence=put))
    except Exception as e:
        print("couldn't explain the result ({})".format(e))
        return None
    if not answer:
        return None
    print('Jarvis says: "{}"'.format(answer))
    ui.set_reply(answer)
    remember(asked, answer)
    voice.wait()
    return answer


def _answer_from_toast(text):
    """Resolves a parked confirmation with an answer that came from a toast
    (a [Yes]/[No] button or text typed into the notification).

    Runs on the toast's background thread, not the wake loop, so it does its
    own COM initialisation (the parked callable is often a delete) and reports
    through a fresh voice. It resolves whichever confirmation is pending; the
    wake loop's own follow-up listener sees is_pending() go false and moves on
    without double-running the callable, because accept()/reject() pop it under
    the same lock.
    """
    text = (text or "").strip()
    if not text or not confirm.is_pending():
        return
    verdict = confirm.interpret(text)

    comtypes.CoInitialize()
    try:
        if verdict is True:
            result = confirm.accept()
        elif verdict is False:
            result = confirm.reject()
        else:
            # Neither clearly a yes nor a no (e.g. dismissed or gibberish).
            # Leave it pending rather than inventing an answer -- it can still
            # be answered by speaking or typing, and expires on its own.
            return
    finally:
        comtypes.CoUninitialize()

    if result:
        print(result)
        ui.set_reply(result)
        _Voice({}, time.perf_counter()).say(result, background=False)
        notify.notify("Jarvis", "Done: {}".format(result))


def _chat_context(text, actions, results):
    """Builds what the chat brain sees on a "both" turn.

    The results list is written to be *heard*: run_command returns "Done -- 3
    lines of output, they're in the log", which is right for a speaker and
    useless to a model that has just been asked what the IP address is. So for
    run_command the real captured output is substituted instead. Clipboard
    needs no equivalent -- read_clipboard already returns the full text.
    """
    lines = []
    for action, result in zip(actions, results):
        kind = action.get("action") if isinstance(action, dict) else None
        if kind == "run_command":
            output = cmd_control.last_output()
            lines.append(output if output else str(result))
        else:
            lines.append(str(result))

    if not lines:
        return text

    return (
        "{}\n\n[I just ran this on the user's PC. Answer their question using "
        "these results. Don't describe the commands, just answer.]\n{}".format(
            text, "\n".join(lines)
        )
    )


def _partition_results(actions, results):
    """(already-spoken, never-spoken) results for this turn.

    One classification pass, two callers, so they cannot disagree about which
    bucket a result landed in -- and that agreement is the point. The failure
    correction and the spoken read-out used to be decided independently, which
    meant a failing `run_command` got read out by _spoken_results and then
    paraphrased again by generate_reply: the same sentence twice, a model call
    and a TTS round trip apart.

    Zipping recovers which result came from which action -- execute_actions
    returns a bare list of strings, in order.
    """
    # An action that parked a confirmation returned the question, not a result.
    # It gets spoken on its own at the end of the turn, so it belongs in
    # neither bucket -- speaking it here would ask twice, and correcting it
    # would treat an unanswered question as a failure.
    parked = (confirm.question() or "").strip()

    spoken, unspoken = [], []
    for action, result in zip(actions, results):
        text = str(result).strip() if result else ""
        if not text or text == parked:
            continue
        kind = action.get("action") if isinstance(action, dict) else None
        (spoken if kind in SPOKEN_RESULTS else unspoken).append(text)

    # zip() stops at the shorter sequence. A result with no matching action
    # shouldn't be silently dropped, so call it unspoken: the cost of that
    # guess is one extra correction, versus swallowing a failure whole.
    for extra in results[len(actions):]:
        text = str(extra).strip() if extra else ""
        if text and text != parked:
            unspoken.append(text)

    return spoken, unspoken


def _spoken_results(actions, results):
    """The subset of results that are themselves the answer.

    Joined into one string and spoken once: three separate speak() calls would
    be three requests, three players and two gaps. "list my downloads and
    delete the temp file" still reads the listing out -- only the parked
    question is held back.
    """
    return " ".join(_partition_results(actions, results)[0])


def _finish(marks, t_start):
    """One place where a turn records how long it took. Every exit path from
    _run_turn goes through here exactly once."""
    marks["spoken"] = time.perf_counter() - t_start
    timing = _format_timing(marks)
    ui.set_timing(timing)
    devlog.turn("done", detail=timing, seconds=marks["spoken"])
    print("  timing: " + timing)


def _run_turn(text, t_start, marks):
    """Everything downstream of having the words: route, speak, act, report.

    Shared by the spoken and typed paths so the two can never disagree about
    what a turn does. The only difference upstream is where `text` came from
    and whether marks["stt"] exists.
    """
    voice = _Voice(marks, t_start)
    devlog.turn("start", detail="...{}".format(text[-80:]) if len(text) > 80 else text)

    # Nothing was said. Stop here, before the parked-question check and before
    # the router.
    #
    # Both of those treat an empty string as a turn. confirm.interpret("")
    # returns None, which means "that wasn't an answer" -- and the rule for a
    # non-answer is to drop the question and run the words instead. Correct for
    # someone who ignores "should I?" and says "open Chrome"; wrong for someone
    # who said nothing, because it throws away a pending delete confirmation
    # and then asks the router to route silence.
    #
    # This got more likely when the STT tier's contract became "return "" for
    # silence rather than raise" -- an empty transcript is a real answer, not
    # a failure, and routing it as a turn would ask the router to route
    # silence.
    if not text or not str(text).strip():
        print("nothing heard")
        return

    logbook.user(text)
    # Long-term facts relevant to these words go into this turn's prompts.
    recall.set_focus(text)

    # A parked question owns the next thing said, whatever it was going to be.
    if confirm.is_pending():
        if _resolve_confirmation(text, voice, typed=bool(marks.get("typed"))):
            _finish(marks, t_start)
            return

    route = {"mode": None, "spoke": False}

    def on_mode(mode):
        route["mode"] = mode
        devlog.route("mode", "OK", mode)
        print("route: {}".format(mode))
        # True aborts the parse. Only for pure chat: there are no actions to
        # wait for, so every remaining token of the router's response is
        # latency spent on something that will be thrown away.
        return mode == "chat"

    def on_reply(reply):
        # Fires mid-stream, before the actions array has finished generating.
        # This is where the latency actually goes away.
        #
        # Suppressed for chat, where the answer is the reply -- speaking "let
        # me think" first means two synthesis requests with a hole between
        # them, and the earcon already said the same thing.
        if route["mode"] == "chat":
            return
        marks["reply"] = time.perf_counter() - t_start
        route["spoke"] = True
        print('Jarvis says: "{}"'.format(reply))
        ui.set_reply(reply)
        voice.say(reply)

    mode, actions, reply = parse_command(text, on_reply=on_reply, on_mode=on_mode)
    marks["parse"] = time.perf_counter() - t_start
    kinds = [a.get("action") if isinstance(a, dict) else "?" for a in actions]
    devlog.route("parse", "OK", "mode={} actions={} {}".format(
        mode, len(actions), kinds
    ), seconds=marks["parse"])
    print("Mode: {} | actions: {}".format(mode, actions))

    if mode == "chat":
        _chat_turn(text, t_start, marks, voice)
        return

    kinds_here = {a.get("action") for a in actions if isinstance(a, dict)}
    if actions and kinds_here == {"understand_screen"}:
        if reply and not route["spoke"]:
            route["spoke"] = True
            ui.set_reply(reply)
            voice.say(reply)
        _screen_turn(text, actions, t_start, marks, voice)
        return
    if mode == "both" and actions and kinds_here <= DIRECT_ANSWERS:
        mode = "command"  # the results are the answer; speak them as-is
    if kinds_here & {"computer_task", "browser_run"}:
        _agent_hooks(voice)

    # Safety net for a model that ignored the key ordering, or that routed to
    # chat and then changed its mind -- either way on_reply never spoke. The
    # flag is what keeps this from saying the line a second time.
    if reply and not route["spoke"]:
        route["spoke"] = True
        print('Jarvis says: "{}"'.format(reply))
        ui.set_reply(reply)
        voice.say(reply)

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
            devlog.exec_event("-", "run", "FAIL", str(e)[:120])
            print("action execution failed: {}".format(e))
            traceback.print_exc()

    exec_thread = threading.Thread(target=run_immediate, daemon=True)
    exec_thread.start()
    exec_thread.join()

    # Let the sentence land before touching the output device again.
    voice.wait()

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

    # Deferred actions ran last but were split out of the middle of the list,
    # so rebuild the pairing rather than assuming actions[i] made results[i].
    ordered = immediate + deferred

    if mode == "both":
        answer = _answer_from_results(text, ordered, results, t_start, marks, voice)
        # The answer is what's worth keeping, not the filler said while the
        # actions ran -- "and how tall is it?" next turn needs to see what "it"
        # referred to. Falls back to the filler when there was no answer.
        remember(text, answer or reply)
        unspoken = []
    else:
        spoken, unspoken = _partition_results(ordered, results)
        # The router's history gets what was actually said back -- for the
        # agent that's its result or its question ("which window has the
        # search box?"), and the next turn's answer only makes sense with it.
        remember(text, " ".join(spoken) if spoken else reply)
        if spoken:
            line = " ".join(spoken)
            print('Jarvis reports: "{}"'.format(line))
            ui.set_reply(line)
            voice.say(line, background=False)

    # A toast whenever a task actually ran -- the spoken "done" is easy to miss
    # when you're not looking at the terminal. Pure chat is skipped (the answer
    # was already spoken), and so are failures (those get their own toast in
    # the correction branch). When an action parked a confirmation, its result
    # is the question itself, filtered out here so the toast doesn't read its
    # own question back.
    if mode != "chat" and not _failed(unspoken):
        parked = (confirm.question() or "").strip()
        done_lines = [
            str(r).strip() for r in results
            if str(r).strip() and str(r).strip() != parked
        ]
        if done_lines:
            notify.notify("Done", " ".join(done_lines))

    # A parked question is the last word of the turn -- it has to be, since the
    # answer is the next thing the user says.
    if confirm.is_pending():
        confirm.set_context(text, [a.get("action") for a in ordered if isinstance(a, dict)])
        question = confirm.question()
        print('Jarvis asks: "{}"'.format(question))
        ui.set_reply(question)
        voice.say(question, background=False)
        # An answerable toast: [Yes]/[No] buttons plus a text box, so the
        # question can be answered from the notification without speaking or
        # typing into the terminal. Falls back to a plain toast showing the
        # question when no interactive backend is installed.
        # Never let the toast take the question down with it: a failure in
        # here used to raise out of the turn, and _guarded_turn's clean-up
        # then cleared the very confirmation it was showing.
        try:
            notify.confirm("Jarvis", question, _answer_from_toast)
        except Exception as e:
            print("couldn't show the confirmation toast ({})".format(e))
        _finish(marks, t_start)
        return

    # Only pay for a second synthesis + TTS round trip if something actually failed
    # AND the user hasn't already heard it. `run_command` is in SPOKEN_RESULTS,
    # so a failing command's output was just read out verbatim; paraphrasing it
    # here said the same thing twice, several seconds apart. The correction is
    # for failures that would otherwise be silent -- a move that couldn't find
    # its file, a close that didn't take -- where the router's cheerful
    # "Moving it to Documents." was the only thing said and it wasn't true.
    # Skipped on "both" too, where the failure was already in the context the
    # chat brain answered from.
    if mode != "both" and _failed(unspoken):
        correction = generate_reply(unspoken)
        print('Jarvis corrects: "{}"'.format(correction))
        ui.set_reply(correction)
        voice.say(correction, background=False)
        notify.notify("Heads up", correction)

    _finish(marks, t_start)


def _screen_turn(text, actions, t_start, marks, voice):
    """"What's on my screen": the vision model's answer streams straight into
    the voice, sentence by sentence, instead of arriving whole and then being
    paraphrased by the chat model -- that was two model calls back to back
    before the first word."""
    from commands import screen_actions

    ui.set_state(ui.THINKING)
    question = " ".join(
        str(a.get("question") or a.get("query") or a.get("text") or "")
        for a in actions if isinstance(a, dict)
    ).strip() or text

    def produce(put):
        return screen_actions.understand_screen(question, on_sentence=put)

    try:
        answer = voice.stream(produce)
    except Exception as e:
        print("screen understanding failed: {}".format(e))
        traceback.print_exc()
        answer = "I couldn't read the screen just now."
        voice.say(answer)

    marks["answer"] = time.perf_counter() - t_start
    print('Jarvis says: "{}"'.format(answer))
    ui.set_reply(answer)
    remember(text, answer)
    voice.wait()
    _finish(marks, t_start)


def _agent_hooks(voice):
    """How the computer agent talks mid-task: blocking speech (it waits for
    your F8 right after asking) and a progress line on the panel."""
    def say(line):
        ui.set_reply(line)
        voice.say(line, background=False)

    def progress(line):
        print(line)
        ui.set_reply(line)

    agent.set_hooks(say=say, progress=progress)


def _chat_turn(text, t_start, marks, voice):
    """A question with nothing to do about it.

    Runs on the wake loop's thread and blocks it for the length of the answer,
    which is correct -- listening for a new wake word while mid-sentence would
    just make the assistant talk over itself.
    """
    ui.set_state(ui.THINKING)

    def produce(put):
        return chat.answer(text, on_sentence=put)

    try:
        answer = voice.stream(produce)
    except Exception as e:
        # chat.answer is documented never to raise, but this is the whole turn.
        print("chat failed: {}".format(e))
        traceback.print_exc()
        answer = "I couldn't work that one out, sorry."
        voice.say(answer)

    marks["answer"] = time.perf_counter() - t_start
    print('Jarvis says: "{}"'.format(answer))
    ui.set_reply(answer)

    # chat.py keeps its own history, but the router keeps a separate one and
    # needs this too -- otherwise "and how tall is it?" arrives at the router
    # with no idea what "it" is, and gets routed as a command.
    remember(text, answer)

    voice.wait()
    _finish(marks, t_start)


def _answer_from_results(text, actions, results, t_start, marks, voice):
    """The "both" tail: actions have run, now answer the question with them.

    Returns the spoken answer, or None if there was nothing to answer from.

    The router already said something short while they ran ("Let me check"), so
    the wait here is covered. chat.answer keeps its own history, and the
    question it stores is the augmented one -- which is what you want, since
    "what's my IP" without the output is a question the model can't answer if
    it comes back as context next turn.
    """
    if confirm.is_pending():
        # Nothing ran yet -- it's waiting on a yes. Answering now would mean
        # answering from results that don't exist.
        return None

    ui.set_state(ui.THINKING)
    question = _chat_context(text, actions, results)

    def produce(put):
        return chat.answer(question, on_sentence=put)

    try:
        answer = voice.stream(produce)
    except Exception as e:
        print("chat failed: {}".format(e))
        traceback.print_exc()
        # The actions did run, so fall back to reading their results out --
        # a worse answer, but not a silent turn.
        answer = _spoken_results(actions, results) or "That's done."
        voice.say(answer)

    marks["answer"] = time.perf_counter() - t_start
    print('Jarvis says: "{}"'.format(answer))
    ui.set_reply(answer)

    voice.wait()
    return answer


def handle_utterance(audio):
    """Spoken turn: transcribe, then run it."""
    t_start = time.perf_counter()
    marks = {}

    # STT is the long pole (several seconds), so the spinner goes up the
    # moment recording ends rather than waiting for the parse step.
    ui.set_state(ui.THINKING)
    save_wav_async(audio)  # off the critical path

    text = transcribe(audio)
    marks["stt"] = time.perf_counter() - t_start

    _run_turn(text, t_start, marks)


def handle_typed(text):
    """Typed turn: identical to a spoken one, minus the microphone.

    Worth noting how much this skips -- no recording, no silence detection,
    no STT round trip. A typed command reaches the parser roughly two to
    four seconds sooner than the same words spoken, which is why this is the
    fastest way to test a change.
    """
    t_start = time.perf_counter()
    # "typed" isn't a timing: it tells _run_turn the words came off the
    # keyboard, which is what a physical confirmation needs (confirm.py).
    marks = {"typed": True}
    devlog.wake("typed", detail="...{}".format(text[-80:]) if len(text) > 80 else text)

    ui.set_state(ui.THINKING)
    print('Typed: "{}"'.format(text))
    ui.set_heard(text)

    _run_turn(text, t_start, marks)


def _guarded_turn(fn, arg):
    """Last net. One bad turn -- a dropped network call, an app that won't
    die -- should cost that turn and nothing more. Before this, any escape
    from a turn ended the session and left the terminal holding a half-drawn
    panel."""
    try:
        fn(arg)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print("turn failed: {}".format(e))
        traceback.print_exc()
        # A question parked by the turn that just blew up will never be
        # answered sensibly -- and leaving it pending would swallow the next
        # utterance as a yes/no.
        confirm.clear()


def _banner(typing=False):
    """The panel already shows the state, so the banner is only useful when
    the dashboard isn't running."""
    if ui.is_active():
        return
    if typing:
        print("Listening for 'Hey Jarvis' -- or just type. (Ctrl+C to stop)")
    else:
        print("Listening for 'Hey Jarvis'... (Ctrl+C to stop)")


def _settle(stream, model, typed):
    """Puts the loop back into a clean listening state after a turn."""
    if stream is not None:
        # Flush backlog that piled up in the input buffer while we were
        # transcribing and speaking. Without this, stale audio bleeds into
        # the next recording.
        pending = stream.read_available
        if pending > 0:
            stream.read(pending)

    if model is not None:
        # Reset the wake word model's internal state so it doesn't think
        # it's still mid-detection.
        model.reset()

    cooldown = TYPED_COOLDOWN_SECONDS if typed else COOLDOWN_SECONDS
    if cooldown > 0:
        time.sleep(cooldown)

    # Anything typed while Jarvis was talking is dropped rather than queued.
    # Someone typing over a reply is nearly always waiting for it to finish,
    # not stacking up three more commands to fire the instant it ends.
    #
    # Except when a question is waiting: a typed "yes" is the one keystroke
    # that is definitely a reply to what was just said, and dropping it means
    # the confirmation silently expires while the user waits for the delete
    # they already approved.
    if not confirm.is_pending():
        console_input.drain()

    ui.set_state(ui.LISTENING)


def _open_input_stream():
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
    )
    stream.start()
    return stream


def _record_turn(stream, give_up_after=None):
    """Cue, flush, record, run. Returns True if there was anything to run.

    The order of the first three matters:

    cues.ready() blocks until the sound has finished playing, so the tone is
    over before the recorder starts measuring the room. Without that, the cue
    lands inside record.py's noise-floor window and lifts the speech gate --
    quiet speech then never registers as speech.

    The flush drops what the mic captured *during* the cue: the wake word tail
    ("...jarvis") plus the cue leaking back in over the speakers. Both would
    otherwise be the first thing transcribed.
    """
    cues.ready()

    pending = stream.read_available
    if pending > 0:
        stream.read(pending)

    audio = record_until_silence(
        stream, CHUNK_SIZE, samplerate=SAMPLE_RATE,
        # Fires the instant endpointing decides speech ended, before the
        # frames are even concatenated -- so the "thinking" cue is heard
        # at the moment it becomes true rather than after STT has begun.
        on_endpoint=cues.done,
        give_up_after=give_up_after,
    )
    if audio is None:
        # Nobody spoke. Only reachable on a follow-up, where that is a real
        # answer rather than a fault, so it is not worth a spoken complaint.
        cues.done()
        return False

    _guarded_turn(handle_utterance, audio)
    return True


def _follow_up(stream, model, typing):
    """Listens for the answer to a parked question without a wake word.

    Making someone say "Hey Jarvis, yes" to confirm a delete they were just
    asked about is a silly thing to ask of anyone, so this re-records directly.
    Typed input is checked first and on every pass, because the question is on
    screen as well as spoken and typing "yes" should work exactly as well as
    saying it.

    The order matters for the last pass: when the mic gives up (CONFIRM_SILENCE
    _SECONDS of nothing), the loop keeps watching the keyboard for another
    CONFIRM_TYPED_SECONDS. Someone reading the question off the screen and
    reaching for the keyboard is slower than someone who just answers out loud,
    and cutting them off at the same six seconds throws away an answer that was
    on its way.
    """
    # Whether the pass that just ended left the loop in a clean listening
    # state. Every branch that runs a turn settles it, so the tidy-up at the
    # bottom only has to settle the paths that didn't -- otherwise a resolved
    # confirmation pays the cooldown twice.
    settled = True

    for _ in range(MAX_CONFIRM_FOLLOWUPS):
        if not confirm.is_pending():
            return

        settled = False

        if typing:
            typed = console_input.poll()
            if typed:
                _guarded_turn(handle_typed, typed)
                _settle(stream, model, typed=True)
                settled = True
                continue

        if _record_turn(stream, give_up_after=CONFIRM_SILENCE_SECONDS):
            _settle(stream, model, typed=False)
            settled = True
            continue

        # The mic gave up. That is not the same as nobody answering: reading
        # the question off the screen and reaching for the keyboard takes
        # longer than saying "yes" does, so watch the keyboard for a while
        # before deciding the question went unanswered.
        if typing:
            typed = console_input.wait(timeout=CONFIRM_TYPED_SECONDS)
            if typed:
                _guarded_turn(handle_typed, typed)
                _settle(stream, model, typed=True)
                settled = True
                continue

        break

    if confirm.is_pending():
        # Out of patience. Dropping it beats leaving a live yes/no hanging over
        # the next unrelated thing said -- and confirm.py would expire it in
        # EXPIRY_SECONDS regardless, just less predictably.
        print("confirm: no clear answer, dropping the question")
        confirm.clear()

    if not settled:
        # Reached by giving up rather than by answering, including the case
        # where the request expired underneath this loop. Up to fourteen
        # seconds of room went into the input buffer while nobody answered,
        # and handing that backlog to the wake word model is how you get a
        # detection nobody triggered.
        _settle(stream, model, typed=False)


def _voice_loop(model, stream, typing):
    while True:
        # Checked before the blocking read, so a typed line waits at most one
        # chunk -- 1280 frames at 16kHz, i.e. 80ms. Not perceptible, and it
        # keeps the mic read blocking, which is what stops this loop from
        # spinning the CPU.
        if typing:
            typed = console_input.poll()
            if typed:
                _guarded_turn(handle_typed, typed)
                _settle(stream, model, typed=True)
                _follow_up(stream, model, typing)
                _banner(typing)
                continue

        data, overflowed = stream.read(CHUNK_SIZE)
        if overflowed:
            print("input overflow (listening)")

        # Feed the idle wave off the same chunks the wake word model sees --
        # the panel twitches with the room, which is the cheap way to show
        # the mic is actually live.
        ui.push_rms(
            float(np.sqrt(np.mean(
                data[:, 0].astype(np.float32) ** 2
            )))
        )

        if not check_wake_word(model, data[:, 0]):
            continue

        devlog.wake("detected", detail="hey_jarvis")
        _record_turn(stream)
        _settle(stream, model, typed=False)
        _follow_up(stream, model, typing)
        _banner(typing)


def _typed_only_loop():
    """Fallback when there's no usable microphone.

    Losing the mic used to end the process at startup with a PortAudio
    traceback. Everything downstream of the words still works, so it is a
    much better outcome to keep running with one input instead of none.
    """
    while True:
        text = console_input.wait(timeout=0.25)
        if not text:
            continue
        _guarded_turn(handle_typed, text)
        _settle(None, None, typed=True)


def _listen_loop(model, typing):
    # Attempts to bring the microphone back after it disappears. Unplugging a
    # headset makes PortAudio raise on the next stream.read() -- previously
    # that was a full crash. Instead: close, wait for the device to come back
    # (Windows re-enumerates a replugged headset), and retry the open. Only
    # give up to typed-input mode when nothing recovers.
    def _open_with_retry():
        last = None
        for attempt in range(MIC_RECONNECT_ATTEMPTS):
            try:
                return _open_input_stream()
            except Exception as e:
                last = e
                if attempt < MIC_RECONNECT_ATTEMPTS - 1:
                    time.sleep(MIC_RECONNECT_DELAY)
        raise last

    try:
        stream = _open_with_retry()
    except Exception as e:
        if not typing:
            raise
        print("microphone unavailable ({}) -- typed input only".format(e))
        ui.set_state(ui.LISTENING)
        _typed_only_loop()
        return

    while True:
        try:
            _voice_loop(model, stream, typing)
            return
        except sd.PortAudioError as e:
            # The device vanished mid-read (headset unplugged, driver blip).
            # Close it and try to reopen; typed input still works meanwhile
            # only if we come back to a fresh stream, so do that in-loop.
            print("microphone lost ({}); reopening...".format(e))
        except KeyboardInterrupt:
            raise
        finally:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception as e:
                print("couldn't close the microphone: {}".format(e))

        try:
            stream = _open_with_retry()
        except Exception as e:
            if typing:
                print("microphone didn't come back ({}) -- typed input only".format(e))
                ui.set_state(ui.LISTENING)
                _typed_only_loop()
                return
            raise


def _restore_history():
    """Brings back the last few turns from before a restart (recall.py), so
    "and close it" or "yes" after a restart still has something to refer to.
    Loaded straight into the router's and chat's short-term history -- not
    through remember(), which would log them a second time."""
    from commands import parser as router
    try:
        turns = recall.recent(max(router.ROUTER_HISTORY_TURNS, chat.HISTORY_TURNS))
    except Exception as e:
        print("couldn't restore history ({})".format(e))
        return
    with router._history_lock:
        for user, jarvis in turns[-router.ROUTER_HISTORY_TURNS:]:
            router._history.append((user[:router.HISTORY_CHARS],
                                    jarvis[:router.HISTORY_CHARS]))
    for user, jarvis in turns[-chat.HISTORY_TURNS:]:
        chat.remember(user, jarvis)
    if turns:
        print("restored {} recent turn{}".format(len(turns), "" if len(turns) == 1 else "s"))


def _clear_terminal():
    """Wipes the console -- screen and scrollback -- so Jarvis's panel sits
    alone, with none of the shell history or startup chatter above it.
    Only when stdout is a real terminal; a pipe or log file is left alone."""
    try:
        if not sys.stdout.isatty():
            return
    except Exception:
        return
    if os.name == "nt":
        os.system("cls")
    # \x1b[3J clears the scrollback too (Windows Terminal, VS Code and most
    # POSIX terminals); \x1b[2J\x1b[H the visible screen and cursor.
    sys.stdout.write("\x1b[3J\x1b[2J\x1b[H")
    sys.stdout.flush()


def main():
    _clear_terminal()
    if os.name == "nt":
        os.system("title Jarvis")
    print("Starting Jarvis...")

    # A provider reply is spoken, and it is also printed. The Windows console
    # defaults to cp1252, and model text carries characters it cannot encode
    # (curly quotes, non-breaking hyphens) -- a print() that raises on that
    # would turn a successful generation into a reported failure. Replace
    # unencodable characters rather than crash the turn.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass

    # Load first, draw second. openwakeword and PortAudio both emit startup
    # chatter from native code, which goes to the real file descriptors and
    # sails straight past the stdout proxy -- so anything drawn before this
    # point gets written over and leaves a fragment of a panel on screen.
    model = load_wake_word_model("hey_jarvis")
    # Open the speaker up front so the first reply isn't slow -- at the
    # voice's own rate (FISH_SAMPLE_RATE), so it never has to reopen.
    get_output_stream(tts.output_rate())
    cues.warm()  # render the earcons now, not while you're waiting to talk
    chat.warm()  # fetch the free-model list on a thread, off the first question
    # Warm the voice the first reply will use, on a thread: Fish Audio's
    # connection when it's configured, otherwise the Moonshine synthesizer
    # (several seconds on first use). With Fish on, Moonshine loads only if
    # a reply ever falls back to it -- see tts.warm().
    tts.warm()

    # Read the preferences file now rather than on the first turn that needs
    # it. It's a few hundred bytes, and doing it here means the "loaded N
    # preferences" line lands in the startup log where it can be noticed --
    # a preferences file that silently failed to load looks exactly like an
    # assistant that ignored you.
    try:
        memory.load()
    except Exception as e:
        print("couldn't load preferences: {}".format(e))

    sweep_recordings()  # clear clips orphaned by a crash or force-kill
    _restore_history()  # pick up where the last run left off

    devlog.sys_event("startup", "OK", "dev log on")
    devlog.stt("moonshine", "setup", "INFO",
               detail="available={} compression={}".format(
                   stt_moonshine.available(),
                   os.getenv("MOONSHINE_COMPRESSION", "medium_streaming")))
    devlog.tts("fish", "setup", "INFO",
               detail="available={} model={} voice={}".format(
                   tts_fish.available(), tts_fish.MODEL,
                   tts_fish.VOICE_ID or "default"))
    devlog.tts("moonshine", "setup", "INFO",
               detail="available={} voice={}".format(
                   tts_moonshine.available(), tts_moonshine.VOICE))

    ui.set_engine(tts=tts.primary_engine())
    # Second wipe: model loading above prints native chatter (openwakeword,
    # PortAudio) that nothing else can catch. The panel starts on a clean
    # screen. Without the panel (JARVIS_NO_UI) the startup log is kept.
    if not os.environ.get("JARVIS_NO_UI"):
        _clear_terminal()
    ui.start()  # no-op when stdout isn't a terminal, or JARVIS_NO_UI is set

    # After ui.start(), because it draws the prompt row through the panel.
    # Returns False when there's no console to read keystrokes from (piped
    # output, IDE run pane, JARVIS_NO_TYPING set) -- voice still works.
    typing = console_input.start()
    # The desktop window (listener/window_ui.py): its input box feeds the same
    # queue as the console, so typed turns work even with no console reader.
    if window_ui.start():
        typing = True

    ui.set_state(ui.LISTENING)
    _banner(typing)

    try:
        _listen_loop(model, typing)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        # Order matters. console_input.stop() first: on POSIX it restores the
        # terminal modes it took, and it talks to ui, so it has to run while
        # ui is still up.
        try:
            console_input.stop()
        except Exception as e:
            print("couldn't stop the keyboard reader: {}".format(e))

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