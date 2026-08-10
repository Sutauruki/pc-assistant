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


def execute_actions(actions):
    """Takes the list of parsed action dicts, runs each one, and returns
    a list of result message strings (used to generate a spoken reply)."""
    results = []
    for action in actions:
        kind = action.get("action")

        if kind == "open_app":
            results.append(open_app(action.get("target", "")))
        elif kind == "close_app":
            results.append(close_app(action.get("target", "")))
        elif kind == "set_volume":
            results.append(set_volume(action.get("value", 50)))
        elif kind == "take_screenshot":
            results.append(take_screenshot())
        elif kind == "refresh_apps":
            results.append(refresh_apps())
        elif kind == "unknown":
            msg = f"Didn't understand: \"{action.get('raw', '')}\""
            print(msg)
            results.append(msg)
        else:
            msg = f"Unrecognized action type: {kind}"
            print(msg)
            results.append(msg)

    return results