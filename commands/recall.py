"""Long-term memory: facts that last, and a conversation that survives restarts.

Goes in: commands/recall.py   (new file)

TWO THINGS commands/memory.py DOESN'T DO
----------------------------------------
memory.py keeps preferences (how to behave) and session facts (gone on
restart). This adds:

  facts         things you told Jarvis that stay true: "my sister is Tolu",
                "mum is saved as Mama on WhatsApp", "my project folder is
                D:\\work\\motive". Saved forever (data/facts.json), up to
                MAX_FACTS.
  profile       the few things that are ALWAYS relevant -- your name, what to
                call you, where you live, what you do (data/profile.json).
                Unlike facts, these go into every prompt, keyword match or
                not: "what do you know about me" has no words in common with
                "the user's name is Ezekiel". About 50 tokens.
  conversation  every turn, appended to data/conversation.jsonl. On start-up
                the last few turns from the past HISTORY_MAX_AGE come back,
                so restarting mid-task doesn't make Jarvis forget it.

FAST BECAUSE IT'S SMALL AND LOCAL
---------------------------------
Every prompt gets only the facts RELEVANT to what you just said -- the few
whose words overlap yours most (the same crude stemming memory.py uses).
Scoring a few hundred short lines is well under a millisecond, there's no
embedding model to load and no extra network call, and the prompts stay
small. main.py calls set_focus(text) at the start of each turn; memory.py's
prompt_block() pulls relevant() into the chat and router prompts, and the
agent asks for its own.

NEVER STORES SECRETS
--------------------
Anything that looks like a password, PIN, one-time code, card number, seed
phrase or API key is refused -- and so is any of your .env keys. A memory
file is plain text on disk; it must never be worth stealing.
"""

import json
import os
import re
import threading
import time
from pathlib import Path

MAX_FACTS = 500
MAX_FACT_CHARS = 240
RELEVANT_FACTS = 5          # injected per prompt
HISTORY_MAX_AGE = 6 * 3600  # older turns don't come back after a restart
LOG_KEEP_LINES = 500        # conversation.jsonl is trimmed to this

_lock = threading.RLock()
_facts = None               # [{"text", "added", "used"}], loaded lazily
_focus = ""
_data_override = None
_appends = 0


def data_dir():
    if _data_override is not None:
        return Path(_data_override)
    return Path(__file__).resolve().parent.parent / "data"


def use_dir(path):
    """Point the stores somewhere else. For tests."""
    global _data_override, _facts
    with _lock:
        _data_override = path
        _facts = None


_token_cache = {}


def _tokens(text):
    """Content words of `text`, cached -- facts are scored on every turn."""
    text = text or ""
    found = _token_cache.get(text)
    if found is None:
        from commands.memory import _content_tokens
        found = _content_tokens(text)
        if len(_token_cache) > 5000:
            _token_cache.clear()
        _token_cache[text] = found
    return found


# --- Secrets ----------------------------------------------------------------

_SECRET_WORDS = re.compile(
    r"\b(password|passcode|passphrase|pin|otp|one[- ]time code|cvv|cvc|"
    r"card number|seed phrase|recovery phrase|private key|api key|secret key|"
    r"access token|bvn)\b", re.I)
_CARDISH = re.compile(r"\b(?:\d[ -]?){12,19}\b")


def secret_reason(text):
    """Why `text` must not be stored, or None."""
    if _SECRET_WORDS.search(text or ""):
        return "that looks like a password or code"
    if _CARDISH.search(text or ""):
        return "that looks like a card or account number"
    try:
        from commands import pc_policy
        if any(s in (text or "") for s in pc_policy._secrets()):
            return "that contains one of your keys"
    except Exception:
        pass
    return None


# --- Facts ------------------------------------------------------------------

def _facts_path():
    return data_dir() / "facts.json"


def _load():
    global _facts
    if _facts is not None:
        return
    _facts = []
    try:
        raw = json.loads(_facts_path().read_text(encoding="utf-8"))
        for entry in raw.get("facts", []) if isinstance(raw, dict) else []:
            if isinstance(entry, dict) and entry.get("text"):
                _facts.append({"text": str(entry["text"])[:MAX_FACT_CHARS],
                               "added": float(entry.get("added") or 0),
                               "used": float(entry.get("used") or 0)})
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        print("recall: couldn't read facts ({}), starting empty".format(e))


def _save():
    path = _facts_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"version": 1, "facts": _facts}, indent=1,
                                   ensure_ascii=False), encoding="utf-8")
        os.replace(str(temp), str(path))
    except OSError as e:
        print("recall: couldn't save facts ({})".format(e))


def facts():
    with _lock:
        _load()
        return [f["text"] for f in _facts]


def remember(text):
    """Stores a lasting fact. Returns a spoken result."""
    text = " ".join(str(text or "").split())[:MAX_FACT_CHARS].strip()
    if not text:
        return "I didn't catch what to remember."
    reason = secret_reason(text)
    if reason:
        return "I won't store that -- {}. Keep it somewhere safe instead.".format(reason)

    new = _tokens(text)
    with _lock:
        _load()
        # The same fact again, or an update of it, replaces rather than
        # piling up.
        for index, entry in enumerate(_facts):
            old = _tokens(entry["text"])
            if old and new and len(old & new) / float(len(old | new)) >= 0.8:
                _facts[index] = {"text": text, "added": time.time(), "used": time.time()}
                _save()
                return "Updated: {}".format(text)
        _facts.append({"text": text, "added": time.time(), "used": 0.0})
        if len(_facts) > MAX_FACTS:
            # Drop the least recently useful, not just the oldest.
            _facts.sort(key=lambda f: max(f["used"], f["added"]))
            del _facts[0]
        _save()
    print("recall: remembered -- {}".format(text))
    return "I'll remember that."


def forget(text=""):
    """Removes the best-matching fact. Returns a spoken result."""
    probe = _tokens(text)
    if not probe:
        return "Tell me which thing to forget."
    with _lock:
        _load()
        best, score = None, 0.0
        for entry in _facts:
            have = _tokens(entry["text"])
            if not have:
                continue
            overlap = len(probe & have) / float(len(probe))
            if overlap > score:
                best, score = entry, overlap
        if best is None or score < 0.5:
            return "I couldn't find that in what I remember."
        _facts.remove(best)
        _save()
    return "Forgotten: {}".format(best["text"])


def relevant(query, k=RELEVANT_FACTS):
    """The k facts that share the most words with `query`, best first."""
    probe = _tokens(query)
    if not probe:
        return []
    with _lock:
        _load()
        scored = []
        for entry in _facts:
            shared = len(probe & _tokens(entry["text"]))
            if shared:
                scored.append((shared, entry["used"], entry))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        picked = [s[2] for s in scored[:k]]
        now = time.time()
        for entry in picked:
            entry["used"] = now  # in memory; saved with the next change
    return [e["text"] for e in picked]


def set_focus(text):
    """What the current turn is about -- main.py sets it from your words."""
    global _focus
    _focus = str(text or "")


def prompt_block(query=None):
    """The profile (always) plus the facts relevant to this turn, as prompt
    text, or "" when there's nothing."""
    head = profile_block()
    found = relevant(_focus if query is None else query)
    if not found:
        return head
    lines = ([head, ""] if head else []) + ["THINGS THE USER HAS TOLD YOU BEFORE",
             "Remembered from earlier conversations. Use them when they help;"
             " don't recite them."]
    lines += ["- {}".format(t) for t in found]
    return "\n".join(lines)


# --- Profile ----------------------------------------------------------------

PROFILE_FIELDS = ("name", "call_me", "location", "work", "about")
_FIELD_ALIASES = {
    "name": "name", "full name": "name",
    "call me": "call_me", "call_me": "call_me", "nickname": "call_me",
    "address me": "call_me",
    "location": "location", "city": "location", "live": "location",
    "lives": "location", "where": "location", "based": "location",
    "work": "work", "job": "work", "occupation": "work", "study": "work",
    "school": "work", "profession": "work",
    "about": "about", "bio": "about",
}
_profile = None


def _profile_path():
    return data_dir() / "profile.json"


def profile():
    """{"name", "call_me", "location", "work", "about"} -- empty strings for
    anything not yet known."""
    global _profile
    with _lock:
        if _profile is None:
            _profile = {f: "" for f in PROFILE_FIELDS}
            try:
                raw = json.loads(_profile_path().read_text(encoding="utf-8"))
                for f in PROFILE_FIELDS:
                    _profile[f] = str(raw.get(f) or "")[:MAX_FACT_CHARS]
            except FileNotFoundError:
                pass
            except (OSError, ValueError, AttributeError) as e:
                print("recall: couldn't read profile ({})".format(e))
        return dict(_profile)


def set_profile(field, value):
    """Sets one profile field. Returns a spoken result."""
    global _profile
    key = _FIELD_ALIASES.get(str(field or "").strip().lower().replace("_", " "),
                             str(field or "").strip().lower())
    if key not in PROFILE_FIELDS:
        return "I don't keep a profile field called {}.".format(field)
    value = " ".join(str(value or "").split())[:MAX_FACT_CHARS].strip(" .")
    reason = secret_reason(value) if value else None
    if reason:
        return "I won't store that -- {}.".format(reason)
    with _lock:
        current = profile()
        current[key] = value
        _profile = current
        try:
            path = _profile_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(current, indent=1, ensure_ascii=False), encoding="utf-8")
            os.replace(str(temp), str(path))
        except OSError as e:
            return "I couldn't save that: {}".format(e)
    if not value:
        return "Forgotten."
    labels = {"name": "your name", "call_me": "what to call you", "location": "where you are",
              "work": "what you do", "about": "that about you"}
    return "Got it -- I'll remember {}.".format(labels[key])


def profile_block():
    """The profile as prompt text, or "" when nothing is known yet."""
    p = profile()
    parts = []
    if p["name"]:
        parts.append("Name: {}".format(p["name"]))
    if p["call_me"]:
        parts.append("Call them: {}".format(p["call_me"]))
    if p["location"]:
        parts.append("Lives in: {}".format(p["location"]))
    if p["work"]:
        parts.append("Does: {}".format(p["work"]))
    if p["about"]:
        parts.append("Also: {}".format(p["about"]))
    if not parts:
        return ""
    return "WHO THE USER IS\n" + "\n".join("- " + x for x in parts)


# --- Conversation -----------------------------------------------------------

def _log_path():
    return data_dir() / "conversation.jsonl"


def log_turn(user, jarvis):
    """Appends one turn. Cheap: one line, no rewrite (trimmed now and then)."""
    global _appends
    user, jarvis = str(user or "").strip(), str(jarvis or "").strip()
    if not user and not jarvis:
        return
    line = json.dumps({"t": time.time(), "user": user[:1000], "jarvis": jarvis[:1000]},
                      ensure_ascii=False)
    path = _log_path()
    with _lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            _appends += 1
            if _appends % 50 == 0:
                _trim(path)
        except OSError as e:
            print("recall: couldn't log the turn ({})".format(e))
    if user:
        try:
            from commands import persona
            persona.note_turn()  # learns how you talk, now and then
        except Exception as e:
            print("recall: couldn't note the turn for style ({})".format(e))


def _trim(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > LOG_KEEP_LINES * 2:
            temp = path.with_suffix(".tmp")
            temp.write_text("\n".join(lines[-LOG_KEEP_LINES:]) + "\n", encoding="utf-8")
            os.replace(str(temp), str(path))
    except OSError:
        pass


def recent(limit, max_age=HISTORY_MAX_AGE):
    """The last `limit` turns from the past `max_age` seconds, oldest first,
    as (user, jarvis) pairs."""
    path = _log_path()
    try:
        with open(path, "rb") as f:
            # Only the tail matters; don't read a big file to get 6 lines.
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            tail = f.read().decode("utf-8", "ignore").splitlines()
    except OSError:
        return []
    cutoff = time.time() - max_age
    turns = []
    for line in tail:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and float(entry.get("t") or 0) >= cutoff:
            turns.append((entry.get("user") or "", entry.get("jarvis") or ""))
    return turns[-limit:]


def describe():
    """One line for "what do you remember", or "" when nothing's known."""
    p = profile()
    who = []
    if p["name"]:
        who.append("you're {}".format(p["name"]) + (
            ", and I call you {}".format(p["call_me"]) if p["call_me"] and p["call_me"] != p["name"] else ""))
    elif p["call_me"]:
        who.append("I call you {}".format(p["call_me"]))
    if p["location"]:
        who.append("you're in {}".format(p["location"]))
    if p["work"]:
        who.append("what you do: {}".format(p["work"]))
    known = facts()
    parts = []
    if who:
        parts.append("I know " + ", ".join(who))
    if known:
        head = "; ".join(known[-5:])
        more = len(known) - 5
        parts.append("Things you've told me: {}{}".format(
            head, ", and {} more".format(more) if more > 0 else ""))
    return ". ".join(parts)
