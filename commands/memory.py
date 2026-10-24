"""What Jarvis remembers: this session, and forever.

Goes in: commands/memory.py   (new file)

TWO STORES, TWO LIFETIMES, ON PURPOSE
-------------------------------------
    session     facts about right now.      Gone on restart.
    preferences how you like to be spoken to. Written to disk.

They are separated because they fail differently. A wrong session fact is
self-correcting -- restart and it's gone. A wrong preference follows you around
until you notice and say something, so it has a higher bar to get written and
an easy way to remove.

    session:     "I'm working in the pc-assistant folder", "the file we were
                 just talking about is idea.txt"
    preference:  "keep answers short", "call me Sutauruki", "always ask before
                 deleting anything"

WHY A FLAT FILE AND NOT A DATABASE
----------------------------------
You should be able to open this, read it, and fix it with a text editor. That
matters more than query power for a store that will hold a dozen lines. It's
JSON at commands/../jarvis_prefs.json -- next to the code, not buried in
AppData, because a preferences file you cannot find is a preferences file you
cannot correct.

Writes are atomic (temp file then os.replace) because the alternative is a
truncated JSON file after a power cut, which reads as "no preferences at all"
and silently loses the lot.

HOW PREFERENCES GET SET
-----------------------
Two ways, and the second is the interesting one.

  1. You say so. "always keep it brief" -> set_preference action.
  2. You correct Jarvis. "no, shorter" / "stop doing that" / "I said don't
     read the whole list out". The router is told that a correction is worth
     recording, and emits set_preference alongside whatever else it does.

The second is why this is worth building. A preference you had to remember to
state is a settings menu with extra steps; a preference inferred from the
correction you were going to make anyway is the thing that makes an assistant
feel like it's paying attention.

CAPS ARE NOT DECORATION
-----------------------
MAX_PREFERENCES and MAX_SESSION_FACTS are small, and the oldest entry is
dropped when they're hit. Every remembered line is injected into two prompts on
every single turn, so an unbounded store is an unbounded token bill and a
slowly-degrading router. Twenty lines is plenty for how one person likes to be
spoken to.
"""

import json
import os
import threading
import time
from pathlib import Path

# Small on purpose -- see the note above. These go into every prompt.
MAX_PREFERENCES = 20
MAX_SESSION_FACTS = 12

# Nothing spoken should be able to write a paragraph into a prompt.
MAX_ENTRY_CHARS = 160

PREFS_FILENAME = "jarvis_prefs.json"

_lock = threading.RLock()

# [{"text": str, "added": float}] -- newest last.
_preferences = []
_session = []

_loaded = False
_path_override = None


# --- Where the file lives --------------------------------------------------

def prefs_path():
    """commands/memory.py -> commands -> repo root -> jarvis_prefs.json"""
    if _path_override is not None:
        return Path(_path_override)
    return Path(__file__).resolve().parent.parent / PREFS_FILENAME


def use_path(path):
    """Point the store somewhere else. For tests; not used at runtime."""
    global _path_override, _loaded
    with _lock:
        _path_override = path
        _loaded = False


# --- Normalising what gets stored ------------------------------------------

def _clean(text):
    """One line, trimmed, capped. Returns "" for anything not worth keeping.

    Newlines are collapsed rather than rejected: speech-to-text won't produce
    them but the set_preference action carries model output, and a multi-line
    preference would break the one-per-line format the prompt injection
    depends on.
    """
    text = " ".join((text or "").split())
    if len(text) > MAX_ENTRY_CHARS:
        text = text[:MAX_ENTRY_CHARS].rstrip() + "..."
    return text.strip()


def _same(a, b):
    """Loose equality, so "keep it brief" doesn't get stored twice with
    different punctuation or casing."""
    strip = ".,!?;:'\"-"
    return a.lower().strip(strip) == b.lower().strip(strip)


# Function words, plus the meta-words that the phrasing "forget that I prefer
# X" wraps around the actual preference. "always" and "never" are deliberately
# NOT here -- "always ask before deleting" and "never ask before deleting" are
# different preferences and the distinguishing word is the one being stripped.
_STOPWORDS = frozenset("""
a an the my me i you your it its that this these those to of for and or but in
on at by with from is are was were be been am do does did don't dont please
just very really so too my mine i'd i'll want wants wanted like likes liked
prefer prefers preferred preference preferences setting settings jarvis thing
things when
""".split())


def _content_tokens(text):
    """Meaning-carrying words, crudely stemmed, as a set.

    The stemming is one rule -- drop a trailing "s" from words over three
    characters -- which is enough for the singular/plural mismatch that actually
    happens ("short answer" vs "short answers") and cheap enough to be obviously
    correct. It is applied to both sides of every comparison, so even where it
    stems wrongly ("always" -> "alway") it stems consistently.
    """
    words = "".join(
        ch if (ch.isalnum() or ch == "'") else " " for ch in text.lower()
    ).split()
    tokens = set()
    for word in words:
        word = word.strip("'")
        if not word or word in _STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.add(word)
    return tokens


# --- Loading and saving ----------------------------------------------------

def _load_locked():
    global _loaded, _preferences
    if _loaded:
        return
    _loaded = True

    path = prefs_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except OSError as e:
        print("memory: couldn't read {} ({})".format(path.name, e))
        return

    try:
        data = json.loads(raw)
    except ValueError as e:
        # A corrupt file is kept, not overwritten. Losing preferences silently
        # is worse than starting empty and saying why -- and the file is the
        # only copy.
        print("memory: {} isn't valid JSON ({}), starting with none".format(
            path.name, e
        ))
        return

    entries = data.get("preferences") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return

    found = []
    for entry in entries:
        if isinstance(entry, str):
            text, added = _clean(entry), 0.0
        elif isinstance(entry, dict):
            text, added = _clean(entry.get("text")), entry.get("added") or 0.0
        else:
            continue
        if not text:
            continue
        try:
            added = float(added)
        except (TypeError, ValueError):
            added = 0.0
        if not any(_same(text, e["text"]) for e in found):
            found.append({"text": text, "added": added})

    _preferences = found[-MAX_PREFERENCES:]
    print("memory: loaded {} preference{} from {}".format(
        len(_preferences), "" if len(_preferences) == 1 else "s", path.name
    ))


def _save_locked():
    path = prefs_path()
    payload = {
        "version": 1,
        "saved": time.time(),
        "preferences": _preferences,
    }

    # Atomic: a half-written file reads as no preferences at all.
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        temp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(str(temp), str(path))
        return True
    except OSError as e:
        print("memory: couldn't save preferences ({})".format(e))
        try:
            temp.unlink()
        except OSError:
            pass
        return False


def load():
    with _lock:
        _load_locked()


# --- Preferences -----------------------------------------------------------

def preferences():
    with _lock:
        _load_locked()
        return [entry["text"] for entry in _preferences]


def remember_preference(text):
    """Store a preference. Returns a spoken result string.

    An existing near-match is moved to the end rather than duplicated, so
    saying the same thing twice is a no-op that still reads as agreement.
    """
    text = _clean(text)
    if not text:
        return "I didn't catch what to remember."

    with _lock:
        _load_locked()

        for index, entry in enumerate(_preferences):
            if _same(entry["text"], text):
                _preferences.append(_preferences.pop(index))
                _save_locked()
                return "Already had that one, but noted."

        _preferences.append({"text": text, "added": time.time()})
        dropped = ""
        while len(_preferences) > MAX_PREFERENCES:
            gone = _preferences.pop(0)
            dropped = gone["text"]

        saved = _save_locked()

    print("memory: preference -- {}".format(text))
    if not saved:
        # Held for this session but not written. Say so: a preference the user
        # thinks is permanent and isn't is worse than one that failed loudly.
        return "Noted for now, but couldn't write it to disk, so it won't survive a restart."
    if dropped:
        return "Got it, I'll remember that. Dropped the oldest one to make room."
    return "Got it, I'll remember that."


def forget_preference(text=None):
    """Drop one preference, or all of them when text is empty.

    WHY THIS ISN'T A SUBSTRING MATCH
    -------------------------------
    It was, and it almost never fired. Preferences don't go in the way you say
    them: the router is told to write them as a standing instruction, so "I
    prefer short answers" is stored as something like "Keep answers short -- a
    sentence or two." Then you say "forget that I prefer short answers", the
    router sends text="short answers", and `"short answers" in "keep answers
    short..."` is False. The words are all there; the order isn't. A feature you
    can set and can't unset is worse than one you never had.

    So the match is on content words, ignoring order: at least half the words
    you named have to appear in the stored line, and the best-scoring line wins.

    Matching loosely is safe here specifically because the result is spoken
    back -- "Forgotten: Keep answers short" tells you exactly what went, so a
    wrong guess is one sentence away from being noticed and re-stated. That
    read-back is doing real work; don't quietly drop it.
    """
    with _lock:
        _load_locked()

        if not (text or "").strip():
            count = len(_preferences)
            if not count:
                return "I wasn't remembering any preferences."
            _preferences[:] = []
            _save_locked()
            return "Forgotten all {} of them.".format(count)

        wanted = _clean(text)

        # Exact and substring first: unambiguous, and cheap.
        for index, entry in enumerate(_preferences):
            if _same(entry["text"], wanted) or wanted.lower() in entry["text"].lower():
                gone = _preferences.pop(index)
                _save_locked()
                return "Forgotten: {}".format(gone["text"])

        wanted_tokens = _content_tokens(wanted)
        if not wanted_tokens:
            # "forget that" / "forget it" -- nothing to match on. With exactly
            # one preference stored there's no ambiguity, so take it; otherwise
            # guessing is worse than asking.
            if len(_preferences) == 1:
                gone = _preferences.pop(0)
                _save_locked()
                return "Forgotten: {}".format(gone["text"])
            return (
                "Couldn't tell which preference you meant. Say a few words from "
                "it, or say forget all my preferences."
            )

        best_index, best_score = -1, 0.0
        for index, entry in enumerate(_preferences):
            shared = wanted_tokens & _content_tokens(entry["text"])
            score = len(shared) / len(wanted_tokens)
            # >= so that on a tie the later (newer) entry wins, which matches
            # remember_preference moving a re-stated preference to the end.
            if score >= best_score:
                best_index, best_score = index, score

        if best_score >= 0.5:
            gone = _preferences.pop(best_index)
            _save_locked()
            return "Forgotten: {}".format(gone["text"])

    return "Couldn't find a preference like that to forget."



# --- Session facts ---------------------------------------------------------

def session_facts():
    with _lock:
        return [entry["text"] for entry in _session]


def remember_session(text):
    """A fact about right now. Never written to disk."""
    text = _clean(text)
    if not text:
        return "I didn't catch what to remember."

    with _lock:
        for index, entry in enumerate(_session):
            if _same(entry["text"], text):
                _session.append(_session.pop(index))
                return "Already noted."
        _session.append({"text": text, "added": time.time()})
        while len(_session) > MAX_SESSION_FACTS:
            _session.pop(0)

    print("memory: session -- {}".format(text))
    return "Noted for this session."


def clear_session():
    with _lock:
        _session[:] = []


# --- Prompt injection ------------------------------------------------------

def prompt_block(include_session=True):
    """The remembered context as prompt text, or "" when there's nothing.

    Returning "" rather than an empty header matters: an empty "things to
    remember" section reads to a model as a section it should fill in, and
    invites invention.
    """
    # Who Jarvis is and how the user talks (commands/persona.py) -- first,
    # so the preferences below can override it.
    try:
        from commands import persona
        character = persona.prompt_block()
    except Exception as e:
        print("memory: couldn't build the personality ({})".format(e))
        character = ""

    prefs = preferences()
    facts = session_facts() if include_session else []
    # Long-term facts relevant to this turn (commands/recall.py).
    try:
        from commands import recall
        lasting = recall.prompt_block()
    except Exception as e:
        print("memory: couldn't read long-term facts ({})".format(e))
        lasting = ""

    if not prefs and not facts and not lasting:
        return character

    lines = [character, ""] if character else []
    if prefs:
        lines.append("HOW THIS USER LIKES YOU TO BEHAVE")
        lines.append("They told you these directly. Follow them over your own"
                     " defaults, and don't mention them unless asked.")
        for text in prefs:
            lines.append("- {}".format(text))

    if facts:
        if lines:
            lines.append("")
        lines.append("WHAT'S GOING ON IN THIS SESSION")
        lines.append("Context from earlier in this run. May be stale; the"
                     " machine itself is the authority.")
        for text in facts:
            lines.append("- {}".format(text))

    if lasting:
        if lines:
            lines.append("")
        lines.append(lasting)

    return "\n".join(lines)


SPOKEN_LIMIT = 5


def describe():
    """What's remembered, as something worth hearing out loud.

    Caps at SPOKEN_LIMIT and counts the rest. Twenty preferences read aloud is
    a monologue, and the full list goes to the log anyway -- which is the only
    place a list of that length is actually useful.
    """
    prefs = preferences()
    facts = session_facts()
    try:
        from commands import recall
        lasting = recall.describe()
    except Exception:
        lasting = ""

    if not prefs and not facts and not lasting:
        return ("I'm not remembering anything yet. Tell me how you'd like me to"
                " do things and I'll keep it.")

    print("memory: {}".format(prefs_path()))
    for text in prefs:
        print("  preference -- {}".format(text))
    for text in facts:
        print("  session    -- {}".format(text))

    parts = []
    if prefs:
        head = "; ".join(prefs[:SPOKEN_LIMIT])
        extra = len(prefs) - SPOKEN_LIMIT
        if extra > 0:
            head += ", and {} more".format(extra)
        parts.append("You've asked me to: {}".format(head))
    if facts:
        head = "; ".join(facts[:SPOKEN_LIMIT])
        extra = len(facts) - SPOKEN_LIMIT
        if extra > 0:
            head += ", and {} more".format(extra)
        parts.append("For this session: {}".format(head))
    if lasting:
        parts.append(lasting)

    return ". ".join(parts)


def status():
    prefs = preferences()
    return "memory: {} preference{}, {} session fact{} ({})".format(
        len(prefs), "" if len(prefs) == 1 else "s",
        len(session_facts()), "" if len(session_facts()) == 1 else "s",
        prefs_path().name,
    )


if __name__ == "__main__":
    # python -m commands.memory              -- shows what's stored, writes nothing.
    # python -m commands.memory --selftest   -- exercises forget matching in a
    #                                           temp file, never touches yours.
    # python -m commands.memory <words...>   -- adds that preference.
    import sys

    if "--selftest" in sys.argv[1:]:
        import tempfile

        use_path(os.path.join(tempfile.mkdtemp(), PREFS_FILENAME))

        store = [
            "Keep answers short -- a sentence or two.",
            "Call the user Sutauruki.",
            "Always ask before deleting anything.",
            "Read file listings out in full rather than summarising.",
        ]

        # (phrase you'd say, the stored line it should remove or None).
        # Note that none of the first four are substrings of what's stored --
        # that is the whole point, see forget_preference's docstring.
        cases = [
            ("short answers", store[0]),
            ("short answer", store[0]),
            ("answers short", store[0]),
            ("that I prefer short answers", store[0]),
            ("keep answers short", store[0]),
            ("call me Sutauruki", store[1]),
            ("Sutauruki", store[1]),
            ("asking before deleting", store[2]),
            ("ask before delete", store[2]),
            ("reading file listings", store[3]),
            ("file listing", store[3]),
            ("the weather in Lagos", None),
            ("my name", None),
        ]

        bad = 0
        for phrase, want in cases:
            with _lock:
                _loaded = True
                _preferences[:] = [
                    {"text": text, "added": index}
                    for index, text in enumerate(store)
                ]
            out = forget_preference(phrase)
            got = out[11:] if out.startswith("Forgotten: ") else None
            ok = got == want
            bad += not ok
            print("  {} {!r:32s} -> {}".format(
                "ok  " if ok else "FAIL", phrase, out
            ))

        print("\n{}/{} as specified".format(len(cases) - bad, len(cases)))
        raise SystemExit(1 if bad else 0)

    print(status())
    print("file: {}".format(prefs_path()))

    if len(sys.argv) > 1:
        print("\n" + remember_preference(" ".join(sys.argv[1:])))

    block = prompt_block()
    print("\nprompt block:\n{}".format(block or "(nothing remembered yet)"))