"""Background command tasks and saved recipes.

Goes in: commands/tasks.py   (new file)

    "in pc-assistant, git pull then pip install -r requirements.txt, and
     tell me when it's done"            -> plan("pc-assistant", [...])
    "is the task done?" / "cancel it"    -> status() / cancel()
    "save that as update jarvis"         -> save_recipe("update jarvis")
    "run update jarvis"                  -> run_recipe("update jarvis")

(Not commands/agent.py's run_task -- that one drives the mouse and
keyboard. This one runs shell commands in a folder.)

A TASK
------
A folder (found by name: "pc-assistant", "downloads", a full path) and one
or more commands run in order, in the background, with no 25-second limit
(STEP_TIMEOUT per step). A step runs only if the one before it succeeded --
a failed git pull doesn't go on to install. Output goes to a log in
Documents\\Jarvis\\Logs ("send me the task log" from Telegram). When it
ends you get a notification and a Telegram message with what happened.

APPROVAL
--------
Every step is classified by commands/cmd_control.py first: anything on its
deny list refuses the whole plan. A plan of only read-only commands runs
straight away; anything else is read back once and needs a physical yes --
typed, the notification button, or from Telegram your JARVIS_REMOTE_PIN
(listener/telegram_bot.py). The approved commands are fixed: what runs is
exactly what was read back.

RECIPES
-------
A plan that ran can be saved under a name, at the PC only. Running a recipe
at the PC needs no new approval -- you approved those exact commands when
you saved them. From the phone it needs your PIN. A recipe is re-checked
against the deny list every time it runs, in case the file was edited.
"""

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from commands import cmd_control, confirm

STEP_TIMEOUT = 3600          # seconds per command
MAX_COMMANDS = 8
MAX_RECIPES = 30

_lock = threading.Lock()
_current = None              # the running task
_last_plan = None            # the most recent plan that ran, for "save that"


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

def find_folder(spec):
    """An existing folder for `spec`, or None: a path, a known folder
    ("downloads"), or a folder found by name under your home folders."""
    from commands import paths
    spec = str(spec or "").strip().strip('"')
    if not spec:
        return None
    if os.path.isabs(spec) and os.path.isdir(spec):
        return Path(spec)
    try:
        path = paths.resolve(spec)
        if path.is_dir():
            return path
    except Exception:
        pass
    name = Path(paths.normalise(spec)).name
    folders = paths.known_folders()
    roots = [folders["home"]] + [folders[k] for k in ("desktop", "documents", "downloads")
                                 if k in folders]
    try:
        hits = [h for h in paths.search(name, roots=roots, limit=20) if h.is_dir()]
    except Exception:
        hits = []
    exact = [h for h in hits if h.name.lower() == name.lower()]
    return (exact or hits or [None])[0]


def _short(folder):
    from commands import paths
    try:
        return paths.display(folder)
    except Exception:
        return str(folder)


# ---------------------------------------------------------------------------
# Planning and approval
# ---------------------------------------------------------------------------

def _clean_commands(commands):
    if isinstance(commands, str):
        commands = commands.replace("&&", "\n").splitlines()
    return [" ".join(str(c).split()) for c in (commands or []) if str(c).strip()]


def _check(commands):
    """(needs approval?, refusal or None)."""
    needs = False
    for command in commands:
        verdict, reason = cmd_control.classify(command)
        if verdict == "deny":
            return True, "I won't run that -- {} involves {}.".format(command, reason)
        if verdict != "allow" or cmd_control.PARANOID:
            needs = True
    return needs, None


def _describe(folder, commands):
    return "In {}: {}".format(_short(folder), ", then ".join(commands))


def _busy():
    with _lock:
        return _current["what"] if _current is not None else None


def plan(folder_spec, commands, label=None, require_approval=False):
    """Checks a plan and either starts it or parks it for approval. Returns
    what to say."""
    commands = _clean_commands(commands)
    if not commands:
        return "I didn't catch which commands to run."
    if len(commands) > MAX_COMMANDS:
        return "That's more than {} commands -- split it up.".format(MAX_COMMANDS)
    folder = find_folder(folder_spec) if folder_spec else None
    if folder_spec and folder is None:
        return "I couldn't find a folder called {}.".format(folder_spec)
    folder = folder or Path.home()
    needs, refusal = _check(commands)
    if refusal:
        return refusal
    busy = _busy()
    if busy:
        return "A task is already running ({}). Say cancel the task first.".format(busy)
    what = _describe(folder, commands)
    if not needs and not require_approval:
        return _start(folder, commands, label)
    question = ("{}. Should I? It needs a physical yes: type yes in my window or "
                "click Yes on the notification.".format(what))
    return confirm.request(question, lambda: _start(folder, commands, label),
                           label="task: " + what, physical=True)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _log_path():
    from commands import paths
    folder = paths.jarvis_dir("Logs")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "task {}.log".format(time.strftime("%Y-%m-%d %H-%M-%S"))


def _start(folder, commands, label=None):
    global _current, _last_plan
    with _lock:
        if _current is not None:
            return "A task is already running ({}).".format(_current["what"])
        what = _describe(folder, commands)
        task = {"folder": folder, "commands": commands, "label": label, "what": what,
                "step": 0, "process": None, "cancelled": False, "started": time.time(),
                "log": _log_path()}
        _current = task
        _last_plan = {"folder": str(folder), "commands": list(commands)}
    threading.Thread(target=_run, args=(task,), name="task", daemon=True).start()
    try:
        from listener import telegram_bot
        how = "message your phone" if telegram_bot.paired() else "let you know"
    except Exception:
        how = "let you know"
    return "Started. {}. I'll {} when it's done.".format(what, how)


def _run(task):
    global _current
    results = []                    # (command, exit code)
    try:
        with open(task["log"], "w", encoding="utf-8", errors="replace") as log:
            log.write("{}\n\n".format(task["what"]))
            for i, command in enumerate(task["commands"]):
                task["step"] = i
                if task["cancelled"]:
                    break
                log.write("$ {}\n".format(command))
                log.flush()
                try:
                    process = subprocess.Popen(
                        command, shell=True, cwd=str(task["folder"]),
                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        **cmd_control._popen_flags())
                except Exception as e:
                    log.write("couldn't start: {}\n".format(e))
                    results.append((command, -1))
                    break
                task["process"] = process
                try:
                    code = process.wait(timeout=STEP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    cmd_control._kill_tree(process)
                    log.write("\n(stopped after {} minutes)\n".format(STEP_TIMEOUT // 60))
                    code = -2
                task["process"] = None
                log.write("\n(exit code {})\n\n".format(code))
                log.flush()
                results.append((command, code))
                if code != 0:
                    break
    except Exception as e:
        results.append(("the task", -1))
        print("tasks: task failed ({})".format(e))
    finally:
        with _lock:
            _current = None
    _announce(_report(task, results))


def _tail(path, chars=3000):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-chars:]
    except OSError:
        return ""


def _minutes(seconds):
    if seconds < 60:
        return "{:.0f} seconds".format(seconds)
    return "{:.0f} minute{}".format(seconds / 60, "" if seconds < 90 else "s")


def _report(task, results):
    from commands import app_watch
    took = _minutes(time.time() - task["started"])
    name = " ({})".format(task["label"]) if task["label"] else ""
    where = _short(task["folder"])
    if task["cancelled"]:
        return "Cancelled the task in {}{}.".format(where, name)
    failed = next((r for r in results if r[1] != 0), None)
    tail = app_watch.scrub(_tail(task["log"]))
    if failed:
        lines = [ln.strip() for ln in tail.splitlines()
                 if ln.strip() and not ln.startswith("(exit code")]
        last = lines[-1][:200] if lines else ""
        why = "it ran out of time" if failed[1] == -2 else "exit code {}".format(failed[1])
        line = "{} failed in {}{} ({}){}.".format(failed[0], where, name, why,
                                                  ": " + last if last else "")
        done = [r[0] for r in results if r[1] == 0]
        if done:
            line += " {} worked before that.".format(" and ".join(done))
        return line
    who = _who()
    lead = "Done{}{}. ".format(", " + who if who else "", name)
    summary = _summarise(task["commands"], tail)
    if summary:
        return "{}{} Took {}.".format(lead, summary, took)
    return "{}{} finished in {}.".format(lead, " and ".join(task["commands"]), took)


def _who():
    try:
        from commands import recall
        p = recall.profile()
        return (p.get("call_me") or p.get("name") or "").strip()
    except Exception:
        return ""


def _summarise(commands, tail):
    """One or two spoken sentences on what the output says, or None."""
    if not tail.strip():
        return None
    try:
        from commands import genbrain
        text = genbrain.summarise(
            tail, "These commands just finished successfully: {}. In one or two "
            "short spoken sentences, say what they did -- e.g. how many commits "
            "were pulled or that it was already up to date, whether the install, "
            "build or tests passed. No paths, no code, no markdown.".format(
                "; ".join(commands)))
        text = " ".join(str(text or "").split())
        return text[:400] or None
    except Exception as e:
        print("tasks: couldn't summarise the output ({})".format(e))
        return None


def _announce(line):
    print("tasks: " + line)
    try:
        from commands import notify
        notify.notify("Task finished", line)
    except Exception:
        pass
    try:
        from listener import telegram_bot
        telegram_bot.send(line)
    except Exception:
        pass


def status():
    with _lock:
        task = _current
    if task is None:
        return "No task is running."
    line = "Still running: {} -- step {} of {}, {}, for {} so far.".format(
        task["what"], task["step"] + 1, len(task["commands"]),
        task["commands"][task["step"]], _minutes(time.time() - task["started"]))
    last = _last_line(task["log"])
    return line + (" Last line: {}".format(last) if last else "")


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _last_line(path):
    """The newest meaningful line of output so far, scrubbed and short, or
    "". Progress bars redraw with \\r, so only the last redraw counts."""
    from commands import app_watch
    for raw in reversed(_tail(path, 4000).splitlines()):
        text = _ANSI.sub("", raw.split("\r")[-1]).strip()
        if not text or text.startswith("$ ") or text.startswith("(exit code"):
            continue
        text = app_watch.scrub(text)
        return text if len(text) <= 150 else text[:147] + "..."
    return ""


def cancel():
    with _lock:
        task = _current
    if task is None:
        return "No task is running."
    task["cancelled"] = True
    process = task["process"]
    if process is not None:
        cmd_control._kill_tree(process)
    return "Cancelling it."


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

def _recipes_path():
    from commands import recall
    return recall.data_dir() / "recipes.json"


def recipes():
    try:
        data = json.loads(_recipes_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print("tasks: couldn't read recipes.json ({})".format(e))
        return {}


def _write(data):
    path = _recipes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _key(name):
    return " ".join(str(name or "").lower().replace("recipe", "").split())


def save_recipe(name):
    """Saves the last plan that ran. PC only (main.py refuses it remotely)."""
    key = _key(name)
    if not key:
        return "What should I call it?"
    with _lock:
        last = dict(_last_plan) if _last_plan else None
    if not last:
        return "There's no task to save yet -- run one first, then say save that as a name."
    data = recipes()
    if key not in data and len(data) >= MAX_RECIPES:
        return "You've got {} recipes already -- delete one first.".format(MAX_RECIPES)
    last["saved"] = time.strftime("%Y-%m-%d %H:%M")
    data[key] = last
    _write(data)
    return "Saved as {}: {}.".format(key, _describe(Path(last["folder"]), last["commands"]))


def _match(name):
    data = recipes()
    key = _key(name)
    if key in data:
        return key, data[key]
    close = [k for k in data if key and (key in k or k in key)]
    return (close[0], data[close[0]]) if len(close) == 1 else (None, None)


def run_recipe(name, require_approval=False):
    key, recipe = _match(name)
    if recipe is None:
        names = ", ".join(sorted(recipes())) or "none yet"
        return "I don't have a recipe called {}. Saved ones: {}.".format(name, names)
    folder = Path(recipe.get("folder") or "")
    if not folder.is_dir():
        return "The folder for {} ({}) isn't there any more.".format(key, folder)
    commands = _clean_commands(recipe.get("commands"))
    _needs, refusal = _check(commands)   # re-checked every run
    if refusal:
        return refusal
    if not require_approval:
        return _start(folder, commands, label=key)
    busy = _busy()
    if busy:
        return "A task is already running ({}).".format(busy)
    question = "Run {}? {}.".format(key, _describe(folder, commands))
    return confirm.request(question, lambda: _start(folder, commands, label=key),
                           label="recipe: " + key, physical=True)


def list_recipes():
    data = recipes()
    if not data:
        return "No recipes saved yet. Run a task, then say save that as a name."
    parts = ["{} ({})".format(k, ", then ".join(v.get("commands") or []))
             for k, v in sorted(data.items())]
    return "Your recipes: " + "; ".join(parts) + "."


def delete_recipe(name):
    key, recipe = _match(name)
    if recipe is None:
        return "I don't have a recipe called {}.".format(name)
    data = recipes()
    data.pop(key, None)
    _write(data)
    return "Deleted the {} recipe.".format(key)
