"""Processes Jarvis is not allowed to kill.

Goes in: commands/self_guard.py   (new file)

WHY THIS EXISTS
---------------
"Close and reopen Claude Code" closed Claude Code, VS Code, and the terminal
Jarvis itself was running in. VS Code survived being asked; it did not survive
being killed, and Jarvis went down with it mid-sentence.

The old guard in app_index.py was:

    pids = {os.getpid()} | {p.pid for p in psutil.Process().parents()}

which is an ANCESTOR check. That is the wrong shape for the problem. A modern
editor is not one process: VS Code runs a main process, a renderer, a shared
process, an extension host, a pty host and one child per terminal. Jarvis lives
under exactly one of those. Every other Code.exe on the machine is a SIBLING or
a COUSIN -- not an ancestor -- so the ancestor set does not contain it, and
_pids_by_exe_name / _pids_by_fuzzy_name happily hand them all to terminate().

Killing a sibling renderer tears down the whole editor, which takes the pty
host with it, which takes Jarvis with it. So an ancestor-only guard cannot fix
this no matter how carefully it is written.

WHAT THIS DOES INSTEAD
----------------------
Protection by EXECUTABLE, not by PID. Walk the ancestor chain once, record the
exe path and the exe filename of every process on it, and refuse to kill
anything that matches either. There are eight Code.exe processes and only one
of them is an ancestor -- but all eight share that executable, so all eight are
covered.

Both keys are kept because they fail in opposite directions. The full path is
exact and cannot collide, but proc.exe() raises AccessDenied on a process
running at higher integrity than us, and returns nothing at all for some
system processes. The bare filename always works and is never denied, but it is
coarse: it also protects an unrelated program that happens to be called
code.exe. Coarse is the correct direction to fail here. The cost of a false
positive is one honest "I can't close that"; the cost of a false negative is
the user's editor.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not look at command lines. `python jarvis.py` and `python setup.py` are
the same executable, and a user can name a script anything, so a command-line
rule is both spoofable and unstable. The executable is ground truth.

It does not try to be clever about "well, that Code.exe has no windows, so it's
probably safe". Guessing which of an editor's processes is load-bearing is
exactly the kind of reasoning that produced the bug.

    python -m commands.self_guard      <- prints what's protected on this
                                          machine, and why
"""

import os
import sys
import threading

import psutil

# Some launchers are the same program under a different filename. Folded so
# that being started from pythonw.exe still protects python.exe and vice versa.
_EXE_ALIASES = {
    "pythonw": "python",
    "pyw": "python",
    "py": "python",
    "wt": "windowsterminal",
    "openconsole": "conhost",
}

_lock = threading.Lock()
_cache = None


def _fold(name):
    """An exe filename reduced to its comparison key: no extension, lowercase,
    aliases collapsed."""
    name = (name or "").strip().lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return _EXE_ALIASES.get(name, name)


def _exe_of(proc):
    """(full path, folded filename) for one process, blanks where unavailable.

    Never raises. Every field here is optional -- a process can deny us its
    path, exit between the two calls, or be a kernel object with neither. A
    blank simply means that key can't contribute to the protection, not that
    the process is safe.
    """
    path, name = "", ""
    try:
        path = (proc.exe() or "").lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError):
        path = ""
    try:
        name = _fold(proc.name())
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError):
        name = ""
    if not name and path:
        name = _fold(os.path.basename(path))
    return path, name


def _ancestry():
    """This process and every ancestor, outermost last. Best effort.

    psutil.parents() is one call but is missing on older psutil and raises if
    any link in the chain has already exited, which would leave us with no
    protection at all. So the walk is manual, bounded, and keeps whatever it
    managed to collect before something went wrong.
    """
    chain = []
    try:
        proc = psutil.Process()
    except Exception:
        return chain

    seen = set()
    for _ in range(64):  # bounded: a PID-reuse cycle must not spin forever
        if proc is None or proc.pid in seen:
            break
        seen.add(proc.pid)
        chain.append(proc)
        try:
            proc = proc.parent()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            break
    return chain


def _build():
    pids, paths, names = set(), set(), set()

    for proc in _ancestry():
        pids.add(proc.pid)
        path, name = _exe_of(proc)
        if path:
            paths.add(path)
        if name:
            names.add(name)

    # sys.executable is the interpreter running this code. Usually already in
    # the set via our own process, but not when Jarvis is frozen or launched
    # through a stub, and it is free to add.
    if sys.executable:
        paths.add(sys.executable.lower())
        names.add(_fold(os.path.basename(sys.executable)))

    names.discard("")
    paths.discard("")
    return {"pids": pids, "paths": paths, "names": names}


def _snapshot():
    """Computed once. The ancestor chain does not change for the life of the
    process, and rebuilding it per candidate would mean a process walk inside
    a loop over processes."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = _build()
        return _cache


def refresh():
    """Drop the cached snapshot. Only useful in a REPL while tuning."""
    global _cache
    with _lock:
        _cache = None


def reason(proc):
    """Why `proc` must not be killed, or "" if it may be.

    The wording is the spoken explanation -- it ends up in the sentence the
    user hears, so it says what the process IS to Jarvis rather than quoting a
    rule at them.
    """
    snap = _snapshot()

    try:
        pid = proc.pid
    except Exception:
        return "that process went away"

    if pid == os.getpid():
        return "that's me"
    if pid in snap["pids"]:
        return "that's what I'm running inside"

    path, name = _exe_of(proc)
    if path and path in snap["paths"]:
        return "that's the app I'm running inside"
    if name and name in snap["names"]:
        return "that's the app I'm running inside"
    return ""


def is_protected(proc):
    return bool(reason(proc))


def filter_pids(pids):
    """(pids we may kill, refusal reason or "").

    Call this BEFORE terminate(), not inside it. The distinction matters to the
    caller: a set that came back empty because everything in it was protected
    is a refusal to report, while a set that came back empty because nothing
    matched is a miss to keep searching for. Collapsing those two is what let a
    precise hit on the editor fall through to a fuzzier strategy that killed
    something else.
    """
    allowed, why = set(), ""

    for pid in pids:
        try:
            proc = psutil.Process(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        blocked = reason(proc)
        if blocked:
            why = why or blocked
            continue
        allowed.add(pid)

    return allowed, why


def safe_to_kill(proc):
    """Convenience wrapper for the per-process check."""
    return not is_protected(proc)


def describe():
    """Human-readable summary of the protection on this machine."""
    snap = _snapshot()
    lines = [
        "protected pids : {}".format(
            ", ".join(str(p) for p in sorted(snap["pids"])) or "(none)"
        ),
        "protected names: {}".format(
            ", ".join(sorted(snap["names"])) or "(none)"
        ),
    ]
    for path in sorted(snap["paths"]):
        lines.append("protected exe  : {}".format(path))
    return "\n".join(lines)


if __name__ == "__main__":
    # python -m commands.self_guard
    #
    # Run this from the same terminal you run Jarvis in -- that is the whole
    # point. If your editor's executable is not in this list, Jarvis can still
    # kill your editor.
    print(describe())
    print()

    print("processes that would be refused right now:")
    shown = 0
    for candidate in psutil.process_iter(["pid", "name"]):
        why = reason(candidate)
        if not why:
            continue
        print("  {:>7}  {:<28} {}".format(
            candidate.info["pid"], (candidate.info["name"] or "?")[:28], why
        ))
        shown += 1
    if not shown:
        print("  (none -- that is suspicious, expected at least this process)")