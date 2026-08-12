"""Where something lives on this machine.

Goes in: commands/locate.py   (new file)

WHY THIS IS AN ACTION AND NOT A CONVERSATION
--------------------------------------------
"where is chrome located on my system" went to the chat brain, which answered
from general knowledge about where Chrome usually installs -- or, when a
classifier had been picked off the free roster, with "User Safety: safe". Both
are wrong in the same way: the question is about THIS computer, and the chat
brain has never seen it.

The router already had a rule saying questions about the user's own machine are
answered with actions rather than from memory. It had no action to point at.
list_files and find_file cover files; nothing covered "where is this app". So
the rule was unenforceable for exactly the question that exposed it.

TWO KINDS OF "WHERE"
--------------------
An app and a file are different lookups and people say them the same way.
"where is chrome" wants the install path; "where is idea.txt" wants the folder
it is sitting in. This module tries the app catalog first and falls back to the
file search, because app names are a small closed set that either matches or
doesn't, while a filename search is fuzzy and will happily return something
irrelevant for a query that was never a filename.

SPEAKING A PATH
---------------
Full paths are miserable to listen to. C:\\Program Files\\Google\\Chrome\\
Application\\chrome.exe read aloud is fifteen seconds of punctuation. The
result string leads with the folder in a speakable form and prints the full
path to the log, which is where you would copy it from anyway.
"""

import os
from pathlib import Path

from commands import app_index
from commands import paths

# How many file hits are worth naming out loud before it becomes a list nobody
# can follow.
SPOKEN_HITS = 3


def _speakable(path):
    """A path a synthesiser can read without it turning into an inventory.

    paths.display already shortens the known folders ("Downloads\\report.pdf").
    Beyond those, the last two components carry almost all the information --
    "Chrome\\Application" tells you where you are, the eight parents above it
    do not.
    """
    path = Path(path)
    shown = paths.display(path)
    if not str(shown).startswith(str(path.anchor)) or len(path.parts) <= 3:
        return shown

    tail = Path(*path.parts[-3:])
    return "{}...{}{}".format(path.anchor, os.sep, tail)


def _describe_app(entry):
    """Result string for a catalog hit."""
    if entry.path:
        target = Path(entry.path)
        folder = target.parent if target.suffix else target
        print("{} -> {}".format(entry.name, target))
        return "{} is in {}".format(entry.name, _speakable(folder))

    if entry.appid:
        # A Store app has no filesystem path worth speaking -- it lives under
        # WindowsApps with a versioned, hyphenated folder name that is useless
        # read aloud, and is not something you can navigate to anyway.
        print("{} -> AppID {}".format(entry.name, entry.appid))
        return ("{} is a Store app, so it doesn't have a normal folder --"
                " Windows launches it by ID".format(entry.name))

    return ""


def where_is(target):
    """Locate an installed app, or a file, by name.

    Apps first. A query that matches the catalog is almost certainly about an
    app, and the file search would otherwise return the app's shortcut, its
    uninstaller, or a log file with a similar name.
    """
    query = (target or "").strip()
    if not query:
        return "I didn't catch what you're looking for."

    try:
        entry, _score = app_index.resolve(query)
    except Exception as e:
        # A catalog that failed to build shouldn't take the file search with
        # it -- "where is my notes file" still works without it.
        print("locate: app lookup failed ({})".format(e))
        entry = None

    if entry is not None:
        described = _describe_app(entry)
        if described:
            return described

    # Not an app, or an app with nothing locatable. Try it as a file.
    try:
        hits = paths.search(query, limit=SPOKEN_HITS + 3)
    except Exception as e:
        return "Couldn't search for {}: {}".format(query, e)

    if not hits:
        return ("Couldn't find anything called {} -- it's not an installed app"
                " and it's not in your usual folders.".format(query))

    print("Matches for {!r}:".format(query))
    for hit in hits:
        print("  {}".format(hit))

    if len(hits) == 1:
        return "{} is in {}".format(hits[0].name, _speakable(hits[0].parent))

    named = ", ".join(_speakable(h.parent) for h in hits[:SPOKEN_HITS])
    extra = ""
    if len(hits) > SPOKEN_HITS:
        extra = ", and {} more".format(len(hits) - SPOKEN_HITS)
    return "Found {} of those -- in {}{}".format(len(hits), named, extra)


if __name__ == "__main__":
    # python -m commands.locate  -- read-only, nothing is opened or changed.
    import sys

    probes = sys.argv[1:] or ["chrome", "vs code", "notepad", "screenshot"]
    for probe in probes:
        print("\n{!r}".format(probe))
        print("  {}".format(where_is(probe)))