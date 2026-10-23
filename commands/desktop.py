"""Small facts about the desktop: what's in front, where, and on which screen.

Goes in: commands/desktop.py   (new file)

Shared by the screen reader (commands/screen_actions.py) and the computer
agent (commands/agent.py). Reuses the ctypes bindings commands/window_actions.py
already declares, and that module's import is also what makes the process
per-monitor DPI aware -- so the window rectangles here, the UI Automation
rectangles in commands/uia.py, the mss screenshot and the cursor position all
speak the same physical pixels. Mixing scaled and unscaled coordinates is how
a click lands 25% off on a laptop at 125% scaling.

The process name comes from the OS (PID -> psutil), never from the window
title: commands/pc_policy.py decides what is protected by process, and a title
is something any web page can set.
"""

import ctypes
from ctypes import wintypes

import psutil

from commands import window_actions as wa

_user32 = wa._user32
_user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetClassNameW.restype = ctypes.c_int
_user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
_user32.GetCursorPos.restype = wintypes.BOOL

MONITOR_DEFAULTTONEAREST = 2


def _title(hwnd):
    length = _user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _class(hwnd):
    buf = ctypes.create_unicode_buffer(256)
    _user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _rect(hwnd):
    rect = wa.RECT()
    if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return (rect.left, rect.top, rect.right, rect.bottom)


def _process(pid):
    try:
        return psutil.Process(pid).name()
    except Exception:
        return ""


def window_info(hwnd):
    """{"hwnd", "pid", "process", "title", "class", "rect"} for a window, or
    None. Store apps report the app inside ApplicationFrameHost, not the host."""
    if not hwnd:
        return None
    pid = wa._window_pid(hwnd)
    process = _process(pid)
    if process.lower() == "applicationframehost.exe":
        try:
            real = wa._real_owner_pid(hwnd, pid, {pid: "applicationframehost.exe"})
            if real and real != pid:
                pid, process = real, _process(real) or process
        except Exception:
            pass
    return {
        "hwnd": int(hwnd),
        "pid": pid,
        "process": process,
        "title": _title(hwnd),
        "class": _class(hwnd),
        "rect": _rect(hwnd),
    }


def foreground():
    """window_info() for the window in front, or None."""
    return window_info(_user32.GetForegroundWindow())


def monitor_of(hwnd):
    """(left, top, right, bottom) of the monitor showing most of `hwnd`, or
    the nearest one when there's no window. None if Windows won't say."""
    info = wa.MONITORINFO()
    info.cbSize = ctypes.sizeof(wa.MONITORINFO)
    handle = _user32.MonitorFromWindow(hwnd or None, MONITOR_DEFAULTTONEAREST)
    if handle and _user32.GetMonitorInfoW(handle, ctypes.byref(info)):
        r = info.rcMonitor
        return (r.left, r.top, r.right, r.bottom)
    return None


def cursor():
    point = wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(point))
    return (point.x, point.y)
