"""Action execution.

Goes in: commands/executor.py  (replaces existing)

open_app / close_app delegate to commands/app_index.py, which builds a catalog
of what's actually installed instead of relying on hardcoded APP_PATHS /
PROCESS_NAMES tables. Those tables are gone -- add nothing here when you
install new software, it gets picked up by the next scan.

This round adds files, clipboard, notifications and raw shell. The dispatch
table is now long enough to be worth saying what it deliberately is NOT: there
is no policy in this file. Whether a delete is safe lives in commands/paths.py,
whether a command may run lives in commands/cmd_control.py, and whether
something needs asking first lives in commands/confirm.py. This module reads a
dict and calls a function. Keeping it that way is what makes the safety
properties reviewable -- they are in three small files, not spread through a
twenty-branch if-chain.

TWO THINGS TO KNOW ABOUT THE RETURN STRINGS
-------------------------------------------
1. main.py scans them for "failed", "couldn't" and "didn't" to decide whether
   to spend a second model round trip on a spoken correction. Every failure
   message must contain one of those words and no success message may.

2. An action that parks a confirmation returns the *question*, not a result.
   The caller finds out by asking commands/confirm.is_pending(), not by
   inspecting the string.
"""

import os
import time

import mss
import mss.tools
from pycaw.pycaw import AudioUtilities

from commands import agent
from commands import app_index
from commands import clipboard_actions
from commands import cmd_control
from commands import docs
from commands import file_actions
from commands import file_analysis
from commands import locate
from commands import memory
from commands import notify as notifications
from commands import pdf_export
from commands import research
from commands import screen_actions
from commands import sheets
from commands import slides
from commands import web_search
from commands import website_builder
from commands import window_actions
from commands import youtube_video

SCREENSHOTS_DIR = "screenshots"

# Kick the app scan off now rather than on the first spoken command. This
# module is imported at startup, so the ~1-3s PowerShell round trip overlaps
# with loading the wake word model and never lands on the latency path.
app_index.preload()


def open_app(target):
    msg = app_index.open_app(target)
    print(msg)
    return msg


def close_app(target):
    msg = app_index.close_app(target)
    print(msg)
    return msg


def refresh_apps():
    msg = app_index.rescan()
    print(msg)
    return msg


def set_volume(value):
    try:
        value = max(0, min(100, int(value)))
        device = AudioUtilities.GetSpeakers()
        volume = device.EndpointVolume
        volume.SetMasterVolumeLevelScalar(value / 100, None)

        actual = round(volume.GetMasterVolumeLevelScalar() * 100)
        msg = f"Volume set to {actual} percent"
    except Exception as e:
        msg = f"Failed to set volume: {e}"

    print(msg)
    return msg


def take_screenshot():
    try:
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[0])
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=filepath)
        msg = "Screenshot taken"
    except Exception as e:
        msg = f"Failed to take screenshot: {e}"

    print(msg)
    return msg


def _call(fn, *args):
    """Call it, log it, hand the string back. Every action module already
    returns a spoken-English result string, so there is nothing else to do."""
    msg = fn(*args)
    print(msg)
    return msg


def _first(action, *keys, default=""):
    """First non-empty value among `keys`.

    The model is consistent about the schema roughly nine times in ten. The
    tenth writes "target" where the prompt asked for "source", or "query"
    where it asked for "target". Accepting both spellings costs one line here
    and saves a whole failed turn -- and unlike a retry, it costs nothing when
    the model got it right.
    """
    for key in keys:
        value = action.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _run_one(action):
    kind = action.get("action")

    # --- apps ---------------------------------------------------------------
    if kind == "open_app":
        return open_app(_first(action, "target", "app", "name"))
    if kind == "close_app":
        return close_app(_first(action, "target", "app", "name"))
    if kind == "set_volume":
        return set_volume(_first(action, "value", "level", default=50))
    if kind == "take_screenshot":
        return take_screenshot()
    if kind == "refresh_apps":
        return refresh_apps()

    # --- windows ------------------------------------------------------------
    if kind == "focus_window":
        return _call(window_actions.focus, action.get("target", ""))
    if kind == "minimize_window":
        return _call(window_actions.minimize, action.get("target", ""))
    if kind == "maximize_window":
        return _call(window_actions.maximize, action.get("target", ""))
    if kind == "restore_window":
        # No target is meaningful here -- it means "put back what you just
        # minimized" -- so this one passes None through rather than "".
        return _call(window_actions.restore, action.get("target") or None)
    if kind == "arrange_windows":
        targets = action.get("targets") or None
        if isinstance(targets, str):
            # The model occasionally emits a bare string where the shape asks
            # for a list. Cheaper to accept it than to spend a retry on it.
            targets = [targets]
        return _call(
            window_actions.arrange, action.get("layout", "grid"), targets
        )
    if kind == "list_windows":
        return _call(window_actions.list_windows)

    # --- files --------------------------------------------------------------
    if kind == "move_file":
        return _call(
            file_actions.move,
            _first(action, "source", "target", "file"),
            _first(action, "destination", "dest", "to", "folder"),
        )
    if kind == "copy_file":
        return _call(
            file_actions.copy,
            _first(action, "source", "target", "file"),
            _first(action, "destination", "dest", "to", "folder"),
        )
    if kind == "rename_file":
        return _call(
            file_actions.rename,
            _first(action, "target", "source", "file"),
            _first(action, "name", "new_name", "to", "destination"),
        )
    if kind == "delete_file":
        return _call(
            file_actions.delete, _first(action, "target", "source", "file")
        )
    if kind == "organize_folder":
        return _call(
            file_actions.organize,
            _first(action, "target", "folder", "path"),
            _first(action, "by", "mode", default="type"),
        )
    if kind == "make_folder":
        return _call(
            file_actions.make_folder,
            _first(action, "target", "name", "folder", "path"),
        )
    if kind == "list_files":
        return _call(
            file_actions.list_files, _first(action, "target", "folder", "path")
        )
    if kind == "find_file":
        return _call(
            file_actions.find,
            _first(action, "query", "target", "name"),
            _first(action, "where", "folder", "in", default=None) or None,
        )
    if kind == "open_path":
        return _call(
            file_actions.open_path, _first(action, "target", "path", "file")
        )
    if kind == "where_is":
        return _call(
            locate.where_is, _first(action, "target", "query", "name", "app")
        )

    # --- clipboard ----------------------------------------------------------
    if kind == "read_clipboard":
        return _call(clipboard_actions.read)
    if kind == "write_clipboard":
        return _call(
            clipboard_actions.write, _first(action, "text", "content", "value")
        )
    if kind == "append_clipboard":
        return _call(
            clipboard_actions.append, _first(action, "text", "content", "value")
        )
    if kind == "clear_clipboard":
        return _call(clipboard_actions.clear)
    if kind == "open_link":
        return _call(clipboard_actions.open_link)

    # --- notifications ------------------------------------------------------
    if kind == "notify":
        return _call(
            notifications.notify,
            _first(action, "title", "heading", default="Jarvis"),
            _first(action, "message", "body", "text"),
        )

    # --- screen -------------------------------------------------------------
    if kind == "understand_screen":
        return _call(
            screen_actions.understand_screen,
            _first(action, "question", "query", "text"),
        )

    # --- memory -------------------------------------------------------------
    if kind in ("set_preference", "remember_preference", "save_preference"):
        return _call(
            memory.remember_preference,
            _first(action, "text", "preference", "value", "content"),
        )
    if kind in ("forget_preference", "remove_preference"):
        # No _first default: an empty text means "forget everything", and that
        # branch is inside forget_preference where it can say how many went.
        return _call(
            memory.forget_preference,
            _first(action, "text", "preference", "value", default=""),
        )
    if kind in ("remember_this", "remember", "note_this"):
        return _call(
            memory.remember_session,
            _first(action, "text", "fact", "value", "content"),
        )
    if kind in ("list_memory", "what_do_you_remember", "show_memory"):
        return _call(memory.describe)

    # --- shell --------------------------------------------------------------
    if kind == "run_command":
        return _call(
            cmd_control.run, _first(action, "command", "cmd", "text")
        )
    if kind == "explain_command":
        return _call(
            cmd_control.explain, _first(action, "command", "cmd", "text")
        )

    # --- Tier 6: content & productivity -------------------------------------
    if kind == "make_doc":
        return _call(
            docs.make_docx,
            _first(action, "prompt", "text", "request"),
            _first(action, "target", "name", "where", default=""),
        )
    if kind == "make_pptx":
        return _call(
            slides.make_pptx,
            _first(action, "prompt", "text", "request"),
            _first(action, "target", "name", "where", default=""),
        )
    if kind == "make_xlsx":
        return _call(
            sheets.make_xlsx,
            _first(action, "prompt", "text", "request"),
            _first(action, "target", "name", "where", default=""),
        )
    if kind == "make_pdf":
        return _call(
            pdf_export.make_pdf,
            _first(action, "prompt", "text", "request"),
            _first(action, "target", "name", "where", default=""),
        )
    if kind == "analyze_file":
        return _call(
            file_analysis.analyze_file,
            _first(action, "target", "file", "path", "source"),
            _first(action, "question", "query", "ask", default=""),
        )
    if kind == "youtube_summary":
        return _call(
            youtube_video.youtube_summary,
            _first(action, "target", "url", "query", "video"),
            _first(action, "question", "ask", "query", default=""),
        )
    if kind == "web_search":
        return _call(
            web_search.web_search,
            _first(action, "query", "target", "text"),
            _first(action, "question", "ask", "detail", default=""),
        )

    # --- computer control (commands/agent.py) -------------------------------
    # browser_run is the old name; the agent drives your real browser now.
    if kind in ("computer_task", "browser_run"):
        return _call(
            agent.run_task,
            _first(action, "task", "instruction", "text", "prompt"),
        )
    if kind == "research_explain":
        return _call(
            research.explain, _first(action, "question", "query", default="")
        )
    if kind == "research_save":
        return _call(
            research.save,
            _first(action, "format", "as", "type", default="doc"),
            _first(action, "target", "name", "where", default=""),
        )
    if kind == "build_website":
        return _call(
            website_builder.build_website,
            _first(action, "prompt", "text", "request"),
            _first(action, "target", "name", "where", default=""),
        )
    if kind == "stop_preview":
        return _call(website_builder.stop_preview_voice)

    if kind == "unknown":
        msg = f"Didn't understand: \"{action.get('raw', '')}\""
    else:
        msg = f"Unrecognized action type: {kind}"

    print(msg)
    return msg


def execute_actions(actions):
    """Takes the list of parsed action dicts, runs each one, and returns
    a list of result message strings (used to generate a spoken reply).

    Each action is isolated. "Close notepad and close task manager" used to
    lose *both* results when the second one raised -- the exception unwound
    the whole loop, so the successful Notepad close was discarded along with
    it and the turn went by in silence.
    """
    from commands import devlog  # local import: this is hot enough to be lazy

    results = []
    for action in actions:
        # Read the kind up front. Doing it inside the except block means a
        # malformed action -- anything that isn't a dict -- raises the same
        # AttributeError a second time from the handler, which throws away
        # the isolation this loop exists to provide.
        try:
            kind = action.get("action")
        except AttributeError:
            kind = None

        t0 = time.time()
        try:
            results.append(_run_one(action))
            devlog.exec_event(kind or "unknown", "run", "OK",
                              seconds=time.time() - t0)
        except Exception as e:
            msg = "Failed to {}: {}".format(kind or "run that", e)
            devlog.exec_event(kind or "unknown", "run", "FAIL",
                              str(e)[:120], seconds=time.time() - t0)
            print(msg)
            results.append(msg)

    return results