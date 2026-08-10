import os
import subprocess
import time
import psutil
import mss
import mss.tools
from pycaw.pycaw import AudioUtilities

SCREENSHOTS_DIR = "screenshots"

# Known app -> launch path/command. Add more here as needed.
APP_PATHS = {
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
}

# Known app -> actual Windows process name (for closing).
# Some apps' process name differs from their launch command.
PROCESS_NAMES = {
    "chrome": "chrome.exe",
    "notepad": "notepad.exe",
    "calculator": "CalculatorApp.exe",  # modern Windows 11 calculator
}


def open_app(target):
    target = target.lower().strip()
    path = APP_PATHS.get(target)

    if path:
        try:
            subprocess.Popen(path)
            msg = f"Opened {target}"
        except Exception as e:
            msg = f"Failed to open {target}: {e}"
    else:
        try:
            os.startfile(target)
            msg = f"Opened {target}"
        except Exception as e:
            msg = f"Couldn't open '{target}' — not a known app"

    print(msg)
    return msg


def close_app(target):
    target = target.lower().strip()
    process_name = PROCESS_NAMES.get(target, f"{target}.exe")

    found = False
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] and proc.info["name"].lower() == process_name.lower():
                proc.terminate()
                found = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    msg = f"Closed {target}" if found else f"Couldn't find '{target}' running"
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
        elif kind == "unknown":
            msg = f"Didn't understand: \"{action.get('raw', '')}\""
            print(msg)
            results.append(msg)
        else:
            msg = f"Unrecognized action type: {kind}"
            print(msg)
            results.append(msg)

    return results