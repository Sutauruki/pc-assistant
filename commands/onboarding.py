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

import re
import threading
import time

from commands import recall

# How long the interview waits for an answer after asking. Long enough to
# read the question in the window and type a reply; after that the next
# thing said is treated as a normal command again.
ANSWER_SECONDS = 180

# An "answer" that starts like this is a command -- "open Chrome", "what's the
# weather" -- so it pauses the interview and runs normally instead of being
# saved as your occupation.
_COMMANDY = re.compile(
    r"^(hey (jarvis|novyx|novix|nova x)\b|open |close |launch |play |search |google |delete |move |"
    r"copy |rename |minimi[sz]e|maximi[sz]e|turn |set the |take a |text |"
    r"message |send |call |remind |what'?s (on|the)|how (much|many|do i)|can you |"
    r"could you |please )", re.I)

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
_state = {"active": False, "index": 0, "answers": {}, "asked": 0.0}

# Questions the profile can already answer, and how to say they're known.
_KNOWN = {
    "name": (lambda p: p["name"] or p["call_me"], "your name"),
    "location": (lambda p: p["location"], "where you're based"),
    "work": (lambda p: p["work"], "what you do"),
}


def active():
    with _lock:
        if _state["active"] and time.time() - _state["asked"] > ANSWER_SECONDS:
            _state["active"] = False  # gone quiet: pause, keep the place
        return _state["active"]


def _skip_known(index, profile):
    """Moves past questions the profile already answers. Returns (index,
    names of what was skipped)."""
    skipped = []
    while index < len(QUESTIONS) and QUESTIONS[index][0] in _KNOWN:
        known, label = _KNOWN[QUESTIONS[index][0]]
        if not known(profile):
            break
        skipped.append(label)
        index += 1
    return index, skipped


def _question(index, lead=""):
    text = QUESTIONS[index][1]
    if index and text.startswith("Let's"):
        text = text.split("First, ", 1)[-1]
    return (lead + " " + text).strip() if lead else text


def start():
    """Begins the interview, or resumes a paused one. Returns the question."""
    profile = recall.profile()
    with _lock:
        resumed = _state["index"] > 0
        index, skipped = _skip_known(_state["index"], profile)
        _state.update(active=True, index=index, asked=time.time())
        if index >= len(QUESTIONS):
            _state.update(active=False, index=0)
            return "I've already got everything I'd ask. Tell me anything new anytime."
        lead = "Picking up where we left off." if resumed else ""
        if skipped:
            lead = (lead + " I already know {}, so I'll skip {}.".format(
                " and ".join(skipped), "that" if len(skipped) == 1 else "those")).strip()
            if not resumed:
                lead = "Say skip for anything you'd rather not share. " + lead
        return _question(index, lead)


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
    if not active():
        return None
    if _COMMANDY.match(said):
        pause()
        return None  # let it run as the command it is
    profile = recall.profile()
    with _lock:
        stopped = said in _STOP
        if not stopped:
            key = QUESTIONS[_state["index"]][0]
            if said and said not in _SKIP and recall.secret_reason(text) is None:
                _state["answers"][key] = " ".join(str(text).split())
            index, _skipped = _skip_known(_state["index"] + 1, profile)
            _state["index"] = index
            if index < len(QUESTIONS):
                _state["asked"] = time.time()
                return _question(index)
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
