"""One pending question, waiting for a yes.

Goes in: commands/confirm.py   (new file)

The brake on the two surfaces that can actually hurt: shell commands outside
the allowlist, and deletes big enough that getting it wrong matters. Both work
the same way -- instead of running, they park a callable here and hand back a
question. main.py notices something is parked, speaks the question, reopens the
mic, and either runs it or throws it away.

WHY A STORE AND NOT A CALLBACK
------------------------------
The executor runs on a worker thread while speech is playing on another. It
cannot block waiting for an answer -- the answer arrives through the microphone,
which the wake loop owns, and the wake loop is what's waiting for the executor.
Parking the decision breaks that circle: the executor finishes normally, the
turn ends, and the confirmation becomes its own small turn afterwards.

Only one thing can be pending at a time. Two stacked confirmations would mean
speaking two questions and hoping the yes lands on the right one, so a second
request replaces the first and says so.

PHYSICAL CONFIRMATIONS
----------------------
request(..., physical=True) marks a question that a spoken "yes" can't
answer: it needs a typed "yes" in Jarvis's window or the toast's Yes button.
Used for shell commands off the safe list and for opening protected system
apps -- the things where audio from a video or a call saying "hey Jarvis...
yes" would otherwise be enough to change your machine. main.py enforces it.

FAIL-SAFE, NOT FAIL-QUIET
-------------------------
An answer that isn't clearly yes is not a yes. interpret() returns None for
anything ambiguous so the caller can ask once more, and a request left sitting
past EXPIRY_SECONDS is discarded on its own -- a pending `rmdir` should not
still be armed twenty minutes later when you say "yeah" to something else
entirely.
"""

import threading
import time

# A confirmation older than this is dead. Long enough to think about it, short
# enough that a stray "yes" much later can't reach back and trigger it.
EXPIRY_SECONDS = 45.0

_lock = threading.Lock()
_pending = None


def request(question, run, label=None, physical=False):
    """Park `run` until someone says yes. Returns the question to speak.

    question: a full spoken sentence ending in a question mark.
    run:      zero-argument callable returning a result string.
    label:    short description for logs and the dashboard.
    physical: only a typed yes or the toast button counts, never voice.
    """
    global _pending

    entry = {
        "question": question,
        "run": run,
        "label": label or question,
        "physical": bool(physical),
        "created": time.time(),
    }

    with _lock:
        replaced = _pending
        _pending = entry

    if replaced is not None:
        print("confirm: replaced pending request ({})".format(replaced["label"]))
    print("confirm: waiting on -- {}".format(entry["label"]))
    return question


def _live():
    """Returns the pending entry, dropping it first if it has gone stale."""
    global _pending
    if _pending is None:
        return None
    if time.time() - _pending["created"] > EXPIRY_SECONDS:
        stale, _pending = _pending, None
        print("confirm: request expired ({})".format(stale["label"]))
        return None
    return _pending


def pending():
    with _lock:
        entry = _live()
        return dict(entry) if entry else None


def is_pending():
    with _lock:
        return _live() is not None


def question():
    with _lock:
        entry = _live()
        return entry["question"] if entry else None


def set_context(text, kinds=()):
    """What the user originally asked, and which actions were in that turn --
    so main.py can answer "how much RAM am I using" in words once the
    approved command has run, instead of reading its raw output aloud."""
    with _lock:
        entry = _live()
        if entry is not None:
            entry["asked"] = str(text or "")
            entry["kinds"] = [k for k in kinds if k]


def needs_physical():
    """True when the pending question can't be answered by voice."""
    with _lock:
        entry = _live()
        return bool(entry and entry.get("physical"))


def clear():
    global _pending
    with _lock:
        _pending = None


def accept():
    """Runs the parked callable. Returns its result string, or None if the
    request had already expired or been cleared.

    Never raises -- a confirmed action that blows up is reported the same way
    any other failed action is, using one of the words the turn loop scans for.
    """
    global _pending
    with _lock:
        entry = _live()
        _pending = None

    if entry is None:
        return None

    try:
        return entry["run"]()
    except Exception as e:
        msg = "Failed to run that: {}".format(e)
        print(msg)
        return msg


def reject():
    """Throws the parked callable away. Returns a line to speak."""
    global _pending
    with _lock:
        entry = _live()
        _pending = None

    if entry is None:
        return None
    print("confirm: cancelled ({})".format(entry["label"]))
    return "Alright, cancelled."


# --- Reading a yes out of a sentence ---------------------------------------
#
# Token sets, not substrings. "no" lives inside "notepad", "yes" inside
# "yesterday", and a substring test on either turns a request to open Notepad
# into a confirmed shell command.

_YES = {
    "yes", "yeah", "yep", "yup", "ya", "yah", "sure", "ok", "okay", "okey",
    "affirmative", "confirm", "confirmed", "approve", "approved", "correct",
    "right", "please", "definitely", "absolutely", "certainly", "proceed",
    "continue", "do", "go", "run", "send", "send it", "fine",
}

_NO = {
    "no", "nope", "nah", "naw", "negative", "cancel", "cancelled", "stop",
    "abort", "dont", "don't", "never", "nevermind", "forget", "skip", "leave",
    "quit", "exit", "wait", "hold", "nothing",
}

# Multi-word forms worth catching before tokenising, because the tokens on
# their own are ambiguous ("go" is a yes, "go back" is not).
_YES_PHRASES = ("go ahead", "go for it", "do it", "yes please", "that's right",
                "thats right", "sounds good", "carry on", "make it so")
_NO_PHRASES = ("never mind", "forget it", "go back", "not now", "leave it",
               "hold on", "hold off", "don't", "do not", "no thanks",
               "no thank you")

_STRIP = ".,!?;:\"'()[]"


def interpret(text):
    """True for yes, False for no, None if it isn't an answer.

    Phrases are checked before tokens because "don't" tokenises into words
    that read as agreement, and negation has to win.
    """
    if not text:
        return None

    lowered = " ".join(text.lower().split())

    for phrase in _NO_PHRASES:
        if phrase in lowered:
            return False
    for phrase in _YES_PHRASES:
        if phrase in lowered:
            return True

    tokens = [word.strip(_STRIP) for word in lowered.split()]
    tokens = [word for word in tokens if word]

    # First decisive token wins, so "no, cancel that" and "yes, run it" both
    # land the obvious way and a trailing filler word can't flip the answer.
    for word in tokens:
        if word in _NO:
            return False
        if word in _YES:
            return True

    return None


if __name__ == "__main__":
    # python -m commands.confirm  -- check the yes/no reader against the kinds
    # of sentence speech-to-text actually produces.
    probes = [
        "yes", "yeah go ahead", "do it", "sure, run it", "okay",
        "no", "nope", "cancel that", "never mind", "don't", "not now",
        "open notepad", "yesterday was fine", "what did you say",
        "hold on", "go back", "no thanks", "affirmative",
    ]
    for probe in probes:
        answer = interpret(probe)
        shown = {True: "YES", False: "NO", None: "-- not an answer"}[answer]
        print("{:<24} {}".format(probe, shown))