"""Action execution.

Goes in: commands/executor.py  (replaces existing)

open_app / close_app delegate to commands/app_index.py, which builds a catalog
of what's actually installed instead of relying on hardcoded APP_PATHS /
PROCESS_NAMES tables. Those tables are gone -- add nothing here when you
install new software, it gets picked up by the next scan.

New this round: the window actions, in commands/window_actions.py. Note the
split of responsibility -- close_app kills a process, close_window is not
offered at all, and everything here only moves windows around. Nothing in the
window path can end a program, which is why it needs no protected list.
"""

import os
import time

import mss
import mss.tools
from pycaw.pycaw import AudioUtilities

from commands import app_index
from commands import window_actions

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


def _window_action(fn, *args):
    msg = fn(*args)
    print(msg)
    return msg


def _run_one(action):
    kind = action.get("action")

    if kind == "open_app":
        return open_app(action.get("target", ""))
    if kind == "close_app":
        return close_app(action.get("target", ""))
    if kind == "set_volume":
        return set_volume(action.get("value", 50))
    if kind == "take_screenshot":
        return take_screenshot()
    if kind == "refresh_apps":
        return refresh_apps()

    if kind == "focus_window":
        return _window_action(window_actions.focus, action.get("target", ""))
    if kind == "minimize_window":
        return _window_action(window_actions.minimize, action.get("target", ""))
    if kind == "maximize_window":
        return _window_action(window_actions.maximize, action.get("target", ""))
    if kind == "restore_window":
        # No target is meaningful here -- it means "put back what you just
        # minimized" -- so this one passes None through rather than "".
        return _window_action(window_actions.restore, action.get("target") or None)
    if kind == "arrange_windows":
        targets = action.get("targets") or None
        if isinstance(targets, str):
            # The model occasionally emits a bare string where the shape asks
            # for a list. Cheaper to accept it than to spend a retry on it.
            targets = [targets]
        return _window_action(
            window_actions.arrange, action.get("layout", "grid"), targets
        )
    if kind == "list_windows":
        return _window_action(window_actions.list_windows)

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

        try:
            results.append(_run_one(action))
        except Exception as e:
            msg = "Failed to {}: {}".format(kind or "run that", e)
            print(msg)
            results.append(msg)

    return results