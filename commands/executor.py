"""Action execution.

Goes in: commands/executor.py  (replaces existing)

open_app / close_app now delegate to commands/app_index.py, which builds a
catalog of what's actually installed instead of relying on the hardcoded
APP_PATHS / PROCESS_NAMES tables. Those tables are gone -- add nothing here
when you install new software, it gets picked up by the next scan.

set_volume and take_screenshot are unchanged.
"""

import os
import time

import mss
import mss.tools
from pycaw.pycaw import AudioUtilities

from commands import app_index

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