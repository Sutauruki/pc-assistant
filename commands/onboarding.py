"""'Get to know me': a short interview that fills in what Jarvis knows.

Goes in: commands/onboarding.py   (new file)

Say "get to know me" and Jarvis asks a handful of quick questions, one per
turn -- like a confirmation, the next thing you say (or type) is the answer,
no wake word needed (main.py routes it here first). "skip" passes on one,
"stop" ends it early and keeps what you've said so far. If you go quiet,
it pauses; "get to know me" again picks up at the same question.

At the end, ONE model call turns the answers into clean memory: the profile
(name, what to call you, where you are, what you do -- in every prompt from
then on), lasting facts (people you message and how they're saved, what
you're into), and a preference for how you like answers. If that call fails,
the answers are stored as they were said, so nothing you told it is lost.
Nothing that looks like a password, code or card number is kept.
"""

import threading

from commands import recall

QUESTIONS = (
    ("name", "Let's do this quickly -- say skip for anything you'd rather not "
             "share. First, what's your name, and what should I call you?"),
    ("location", "Where are you based? A city is plenty."),
    ("work", "What do you do -- work, study, or what you're building?"),
    ("people", "Who do you message most? Give me names and how they're saved, "
               "like my sister Tolu, saved as Tolu Ade."),
    ("style", "How do you like my answers -- short and quick, or more detail?"),
    ("interests", "What are you into, or working on lately?"),
    ("anything", "Last one: anything else I should always keep in mind?"),
)
_KEYS = [q[0] for q in QUESTIONS]

_SKIP = {"skip", "pass", "next", "no", "nope", "none", "nothing", "rather not",
         "next one", "skip it", "skip that", "no thanks", "nah"}
_STOP = {"stop", "that's enough", "thats enough", "enough", "later", "cancel",
         "done", "quit", "end", "finish", "not now"}

_lock = threading.Lock()
_state = {"active": False, "index": 0, "answers": {}}


def active():
    with _lock:
        return _state["active"]


def start():
    """Begins the interview, or resumes a paused one. Returns the question."""
    with _lock:
        _state["active"] = True
        index = _state["index"]
        if index:
            return "Picking up where we left off. " + QUESTIONS[index][1]
        return QUESTIONS[0][1]


def pause():
    """Stops listening for answers without losing them (you went quiet)."""
    with _lock:
        _state["active"] = False


def _norm(text):
    return " ".join(str(text or "").lower().strip(" .!?,").split())


def answer(text):
    """Takes the answer to the current question. Returns what to say next:
    the next question, or the wrap-up. None if no interview is running."""
    said = _norm(text)
    with _lock:
        if not _state["active"]:
            return None
        stopped = said in _STOP
        if not stopped:
            key = QUESTIONS[_state["index"]][0]
            if said and said not in _SKIP and recall.secret_reason(text) is None:
                _state["answers"][key] = " ".join(str(text).split())
            _state["index"] += 1
            if _state["index"] < len(QUESTIONS):
                return QUESTIONS[_state["index"]][1]
        # Finished or stopped: save, and start fresh next time.
        answers = dict(_state["answers"])
        _state.update(active=False, index=0, answers={})

    saved = _save(answers)
    lead = "Okay, we'll stop there." if stopped else "That's everything, thanks."
    if not saved:
        return lead + " I didn't get anything to remember, though."
    p = recall.profile()
    name = p.get("call_me") or p.get("name")
    return "{} I'll remember all that{}.".format(lead, ", " + name if name else "")


_SHAPE = (
    "a JSON object with keys: name (string, the user's name, or empty), "
    "call_me (string, what they want to be called, or empty), location "
    "(string, or empty), work (string, a short phrase like 'software "
    "developer building a voice assistant', or empty), facts (array of short "
    "third-person sentences worth remembering, e.g. \"The user's sister is "
    "Tolu, saved as Tolu Ade on WhatsApp.\" -- one per person or interest), "
    "and answer_style (string: a one-line instruction for how to answer, "
    "e.g. 'Keep answers short.', or empty)"
)


def _save(answers):
    """Turns the answers into profile fields, facts and a preference.
    Returns how many things were stored."""
    if not answers:
        return 0
    stored = 0
    try:
        from commands import genbrain
        lines = "\n".join("Q: {}\nA: {}".format(QUESTIONS[_KEYS.index(k)][1], v)
                          for k, v in answers.items())
        data = genbrain.structured(
            _SHAPE,
            "Turn these interview answers into memory about the user. Use only "
            "what they said; leave anything they didn't say empty. Never "
            "include passwords, codes, or account or card numbers.\n\n" + lines,
            max_tokens=800,
        )
    except Exception as e:
        print("onboarding: couldn't tidy the answers ({}), saving them as said".format(e))
        data = None

    if isinstance(data, dict):
        for field in ("name", "call_me", "location", "work"):
            value = str(data.get(field) or "").strip()
            if value and recall.set_profile(field, value).startswith("Got it"):
                stored += 1
        for fact in data.get("facts") or []:
            if isinstance(fact, str) and fact.strip():
                if recall.remember(fact).startswith(("I'll", "Updated")):
                    stored += 1
        style = str(data.get("answer_style") or "").strip()
        if style:
            try:
                from commands import memory
                memory.remember_preference(style)
                stored += 1
            except Exception:
                pass
        if stored:
            return stored

    # Fallback: keep what was said, word for word.
    for key, value in answers.items():
        if key in ("name", "location", "work"):
            recall.set_profile(key, value)
        else:
            recall.remember("The user said ({}): {}".format(key, value))
        stored += 1
    return stored
