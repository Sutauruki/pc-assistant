"""Who Jarvis is, and how it learns to talk like you.

Goes in: commands/persona.py   (new file)

Two layers, both added to every prompt that speaks (the router's reply, chat
answers, failure corrections) through memory.prompt_block():

CHARACTER -- fixed. Classic Jarvis: calm, composed, dry understated wit,
quietly loyal, never gushing. Uses your name now and then (the profile's
"call me", else your name), never "sir".

YOUR STYLE -- learned. Every LEARN_EVERY of your messages, one small model
call reads what you've said to Jarvis lately and updates a short note on how
you talk: length, slang, energy, humour, what you want from answers. Jarvis
drifts toward that while keeping its own composure. The note lives in
data/style.json, is built only from what you said to Jarvis (anything that
looks like a password or card number is skipped), and you can see it ("how
do you think I talk?") or wipe it ("forget my style").

Things you tell it directly ("too long", "stop using my name") are
preferences (commands/memory.py) and always win over both layers.
"""

import json
import os
import threading
import time

from commands import identity

LEARN_EVERY = 30        # your messages between style updates
FIRST_LEARN = 12        # the first note is written once there's this much
SAMPLE_MESSAGES = 80    # recent messages read per update
MAX_STYLE_CHARS = 600

CHARACTER = """WHO YOU ARE
You are {NAME}: calm, composed and quietly capable, with a dry, understated
wit -- think a trusted friend who happens to run the computer, not a
customer-service bot. You answer first and skip filler ("Great question!",
"Certainly!", "I'd be happy to"). Warm but never gushing; confident but
honest about what you don't know. A light touch of humour when the moment
is relaxed; none when something failed, when they're stressed or in a
hurry, or when it would slow the answer down. Safety and honesty come
before personality: never pretend something worked.
{ADDRESS}"""

_lock = threading.Lock()
_learning = threading.Event()
_state = None


def _path():
    from commands import recall
    return recall.data_dir() / "style.json"


def _load():
    global _state
    if _state is None:
        try:
            _state = json.loads(_path().read_text(encoding="utf-8"))
            if not isinstance(_state, dict):
                raise ValueError("not an object")
        except FileNotFoundError:
            _state = {}
        except Exception as e:
            print("persona: couldn't read style.json ({}); starting fresh".format(e))
            _state = {}
        _state.setdefault("style", "")
        _state.setdefault("since", 0)
    return _state


def _save():
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _user_name():
    try:
        from commands import recall
        p = recall.profile()
        return (p.get("call_me") or p.get("name") or "").strip()
    except Exception:
        return ""


def _address():
    name = _user_name()
    if name:
        return ("Call the user {} -- naturally, now and then (a greeting, a "
                "finished task, a heads-up), not in every reply. Never call "
                "them \"sir\".".format(name))
    return ("You don't know the user's name yet; don't guess one, and never "
            "call them \"sir\".")


def style():
    with _lock:
        return _load()["style"]


def prompt_block():
    """Character plus the learned style, as prompt text."""
    text = CHARACTER.replace("{NAME}", identity.NAME).replace("{ADDRESS}", _address())
    learned = style()
    if learned:
        text += ("\n\nHOW THE USER TALKS (learned from their messages)\n" + learned +
                 "\nDrift toward this: match their length and energy, and their "
                 "slang where it comes naturally -- but stay yourself. Don't "
                 "copy their typos or overdo the slang, and don't mention that "
                 "you've noticed how they talk.")
    return text


# --- Learning ----------------------------------------------------------------

def note_turn():
    """Called once per conversation turn (recall.log_turn). Starts a style
    update on a background thread when enough new messages have built up."""
    with _lock:
        state = _load()
        state["since"] = int(state.get("since") or 0) + 1
        due = state["since"] >= (LEARN_EVERY if state["style"] else FIRST_LEARN)
        if not due or _learning.is_set():
            # Written every few turns rather than every turn, so frequent
            # restarts still reach the next update.
            if state["since"] % 5 == 0:
                try:
                    _save()
                except OSError:
                    pass
            return
        _learning.set()
    threading.Thread(target=_learn_quietly, name="persona", daemon=True).start()


def _learn_quietly():
    try:
        learn()
    except Exception as e:
        print("persona: couldn't update your style ({})".format(e))
    finally:
        _learning.clear()


def _recent_messages(limit=SAMPLE_MESSAGES):
    from commands import recall
    path = recall.data_dir() / "conversation.jsonl"
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 128 * 1024))
            lines = f.read().decode("utf-8", "ignore").splitlines()
    except OSError:
        return []
    said = []
    for line in lines:
        try:
            text = str(json.loads(line).get("user") or "").strip()
        except (ValueError, AttributeError):
            continue
        if text and recall.secret_reason(text) is None:
            said.append(text[:200])
    return said[-limit:]


_SHAPE = ("a JSON object with one key, style (string): 2 to 4 short sentences "
          "describing how this person talks and what they want from replies -- "
          "typical message length, formality, slang or local expressions they "
          "actually use (quote a few), energy, humour, and whether they like "
          "short or detailed answers. About how they talk, not what they talk "
          "about: no names, places, secrets or personal facts.")


def learn():
    """Rewrites the style note from recent messages. Returns the new note, or
    "" when there isn't enough to go on yet."""
    said = _recent_messages()
    if len(said) < 6:
        return ""
    from commands import genbrain
    previous = style()
    prompt = ("Here are the most recent things a user said to their voice "
              "assistant (some spoken and transcribed, some typed), oldest "
              "first:\n\n" + "\n".join("- " + s for s in said))
    if previous:
        prompt += ("\n\nThe current note about how they talk is below. Refine it "
                   "with what the messages show; keep what still holds.\n" + previous)
    data = genbrain.structured(_SHAPE, prompt, max_tokens=400)
    note = " ".join(str((data or {}).get("style") or "").split())[:MAX_STYLE_CHARS]
    if not note:
        return ""
    with _lock:
        state = _load()
        state.update(style=note, since=0, updated=time.time(), learned_from=len(said))
        _save()
    print("persona: updated how you talk ({} messages)".format(len(said)))
    return note


# --- Voice commands (executor "my_style") ---------------------------------------

def describe():
    note = style()
    if not note:
        with _lock:
            left = max(0, FIRST_LEARN - int(_load().get("since") or 0))
        return ("I haven't got a feel for how you talk yet -- give me about {} more "
                "messages.".format(left) if left else
                "I'm still working out how you talk -- ask me again in a bit.")
    return "Here's my read on how you talk: " + note


def forget():
    with _lock:
        state = _load()
        state.update(style="", since=0, updated=time.time())
        _save()
    return "Done -- I've forgotten how you talk and I'll start learning fresh."


def relearn():
    try:
        note = learn()
    except Exception as e:
        return "I couldn't update that just now ({}).".format(e)
    return ("Updated. " + note) if note else "I need a few more of your messages first."


def handle(do="show"):
    do = str(do or "show").strip().lower()
    if do in ("forget", "reset", "clear", "wipe"):
        return forget()
    if do in ("learn", "update", "relearn", "refresh"):
        return relearn()
    return describe()
