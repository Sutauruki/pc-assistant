"""Window management -- focus, minimize, maximize, restore, arrange.

Goes in: commands/window_actions.py   (new file)

    "focus chrome"                  -> brings Chrome to the front
    "minimize everything"           -> clears the desktop
    "restore"                       -> puts back exactly what that minimized
    "maximize vs code"
    "put chrome and vs code side by side"
    "snap spotify to the right"
    "cascade my windows"
    "what's open"

ON pygetwindow AND pywinauto
----------------------------
Both were on the table and neither is used. Worth writing down why, because
the reason is the whole difficulty of this feature.

pygetwindow on Windows is a thin ctypes wrapper over user32 -- the same calls
commands/app_index.py already binds for close_app. It would add a dependency
to reach APIs we can already reach. More to the point, its .activate() is a
bare SetForegroundWindow, and a bare SetForegroundWindow *does not work* from
a background console process. Windows refuses it and flashes the taskbar
button instead. That is the single most common complaint about the library,
and working around it means dropping to ctypes anyway -- see _force_foreground
below, which is most of the real work in this module.

pywinauto is a UI automation framework: control trees, clicking buttons,
typing into other apps' text fields. It is a heavy dependency (comtypes,
pywin32) with a slow startup, and none of what it does is what "focus,
minimize, arrange" needs. If this ever grows into driving another app's
interface, that is the moment to add it.

What we get by staying on ctypes, beyond the dependency count: the same
ALIASES, the same fuzzy matcher and the same UWP unwrapping that open_app and
close_app already use, so "vs code" means the same thing to every command.

This module owns its own user32 handle rather than borrowing app_index's.
Setting .argtypes mutates a shared WinDLL's cached function objects, and
app_index's copy is what decides which processes get terminated -- not a
thing to reach into from a feature about moving windows around.
"""

import ctypes
import os
import time
from ctypes import wintypes

import psutil

from commands.app_index import (
    MATCH_FLOOR,
    _expand_alias,
    _normalize,
    _similarity,
    resolve,
)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# Focusing is not destructive -- the worst case is the wrong window comes
# forward and you say it again. So this is MATCH_FLOOR, the "safe to launch"
# threshold, rather than app_index's stricter "safe to kill" one.
WINDOW_MATCH_FLOOR = MATCH_FLOOR

# A process name is weaker evidence than a title: "Code.exe" is the same
# string for every VS Code window, so it can pick the app but never the right
# window of it. Scored slightly lower so a title match always wins a tie.
PROC_NAME_WEIGHT = 0.92

# How long to wait for a window to actually do what it was told. These are
# asynchronous requests to another process, so nothing is true the instant
# the call returns. Poll rather than sleep a flat amount: a responsive app
# lands in one tick and we move on.
SETTLE_TIMEOUT = 0.35
SETTLE_POLL = 0.02

# Placement is close enough if every edge is within this many pixels. Apps
# with a minimum size, fixed-size dialogs and anything with a custom frame
# will land near the target rather than on it, and reporting those as
# failures would be noise.
PLACE_TOLERANCE = 12

# Beyond this, tiles are too narrow to be worth looking at; extra windows are
# left where they are and the reply says how many.
MAX_TILED = 8

# Gap between tiles, in pixels. 0 matches Windows' own Snap.
TILE_GAP = 0

_ALL_WORDS = {
    "all", "everything", "all windows", "every window", "desktop",
    "the desktop", "all apps", "everything else", "them all",
}

# ---------------------------------------------------------------------------
# DPI
# ---------------------------------------------------------------------------
#
# Without this, a process on a scaled display (125% and 150% are the factory
# default on most Windows laptops) is lied to about coordinates: GetWindowRect
# hands back virtualised values and SetWindowPos scales what it is given, so
# a two-pane tile computed from the work area lands wrong by exactly the
# scaling factor. Everything looks fine on a 100% monitor and is visibly
# broken on a laptop screen.
#
# Process-wide and permanent, but mss already calls SetProcessDPIAware when it
# grabs a screenshot, so this makes an existing behaviour explicit and earlier
# rather than introducing a new one. Newest API first, falling back to the
# older ones so this works from Vista up.


def _set_dpi_aware():
    try:
        # PER_MONITOR_AWARE_V2 -- correct on mixed-DPI multi-monitor setups.
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        # An old Windows, or something already set it to a conflicting value
        # (the call fails once awareness is fixed). Neither is worth failing
        # startup over -- arrange is just less precise.
        pass


_set_dpi_aware()

# ---------------------------------------------------------------------------
# Win32 constants
# ---------------------------------------------------------------------------

SW_MAXIMIZE = 3
SW_MINIMIZE = 6
SW_SHOWMINNOACTIVE = 7
SW_RESTORE = 9

GWL_STYLE = -16
GWL_EXSTYLE = -20

WS_CHILD = 0x40000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
WS_EX_NOACTIVATE = 0x08000000

DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_NOSENDCHANGING = 0x0400
SWP_ASYNCWINDOWPOS = 0x4000

HWND_TOP = 0
MONITOR_DEFAULTTONEAREST = 2

VK_MENU = 0x12
KEYEVENTF_KEYUP = 0x0002


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", wintypes.DWORD),
    ]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

try:
    _dwmapi = ctypes.WinDLL("dwmapi")
except OSError:
    # Pre-Vista, or DWM disabled. Costs gap-free tiling and cloaked-window
    # filtering; everything else works.
    _dwmapi = None

_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

_user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
_user32.EnumWindows.restype = wintypes.BOOL
_user32.EnumChildWindows.argtypes = [wintypes.HWND, _WNDENUMPROC, wintypes.LPARAM]
_user32.EnumChildWindows.restype = wintypes.BOOL
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.IsWindowVisible.restype = wintypes.BOOL
_user32.IsWindow.argtypes = [wintypes.HWND]
_user32.IsWindow.restype = wintypes.BOOL
_user32.IsIconic.argtypes = [wintypes.HWND]
_user32.IsIconic.restype = wintypes.BOOL
_user32.IsZoomed.argtypes = [wintypes.HWND]
_user32.IsZoomed.restype = wintypes.BOOL
_user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
_user32.GetWindowTextLengthW.restype = ctypes.c_int
_user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetWindowTextW.restype = ctypes.c_int
_user32.GetWindowThreadProcessId.argtypes = [
    wintypes.HWND, ctypes.POINTER(wintypes.DWORD)
]
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.ShowWindow.restype = wintypes.BOOL
_user32.SetForegroundWindow.argtypes = [wintypes.HWND]
_user32.SetForegroundWindow.restype = wintypes.BOOL
_user32.GetForegroundWindow.argtypes = []
_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.BringWindowToTop.argtypes = [wintypes.HWND]
_user32.BringWindowToTop.restype = wintypes.BOOL
_user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
_user32.AttachThreadInput.restype = wintypes.BOOL
_user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_uint,
]
_user32.SetWindowPos.restype = wintypes.BOOL
_user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
_user32.GetWindowRect.restype = wintypes.BOOL
_user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
_user32.MonitorFromWindow.restype = wintypes.HANDLE
_user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFO)]
_user32.GetMonitorInfoW.restype = wintypes.BOOL
_user32.SystemParametersInfoW.argtypes = [
    wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT
]
_user32.SystemParametersInfoW.restype = wintypes.BOOL
_user32.keybd_event.argtypes = [
    wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.c_void_p
]
_user32.keybd_event.restype = None
_kernel32.GetCurrentThreadId.argtypes = []
_kernel32.GetCurrentThreadId.restype = wintypes.DWORD
_kernel32.GetConsoleWindow.argtypes = []
_kernel32.GetConsoleWindow.restype = wintypes.HWND

# GetWindowLongPtrW only exists on 64-bit; the 32-bit name is the fallback.
# Style bits fit in 32 bits either way, so either is correct here.
_get_window_long = getattr(_user32, "GetWindowLongPtrW", None) or _user32.GetWindowLongW
_get_window_long.argtypes = [wintypes.HWND, ctypes.c_int]
_get_window_long.restype = ctypes.c_ssize_t

if _dwmapi is not None:
    _dwmapi.DwmGetWindowAttribute.argtypes = [
        wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD
    ]
    _dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

class WindowInfo(object):
    __slots__ = ("hwnd", "pid", "title", "process")

    def __init__(self, hwnd, pid, title, process):
        self.hwnd = hwnd
        self.pid = pid
        self.title = title
        self.process = process

    def __repr__(self):
        return "<Window {} {!r}>".format(self.hwnd, self.title)


def _window_pid(hwnd):
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _window_thread(hwnd):
    return int(_user32.GetWindowThreadProcessId(hwnd, None))


def _real_owner_pid(hwnd, host_pid, names):
    """A Store app's visible window belongs to ApplicationFrameHost, not to the
    app. Its own CoreWindow is a child of the frame, so the first child with a
    different PID is the process we actually want.

    Same unwrapping app_index does before it kills anything -- here it only
    affects which process name a window is matched against, but getting it
    wrong would make every UWP app look identically named.
    """
    if names.get(host_pid, "") != "applicationframehost.exe":
        return host_pid

    found = []

    def callback(child, _lparam):
        pid = _window_pid(child)
        if pid and pid != host_pid:
            found.append(pid)
            return False
        return True

    _user32.EnumChildWindows(hwnd, _WNDENUMPROC(callback), 0)
    return found[0] if found else host_pid


def _is_cloaked(hwnd):
    """DWM-cloaked windows are 'visible' by the old API and invisible on
    screen: suspended Store apps, and every virtual desktop other than the one
    you are looking at. Without this filter, "minimize everything" reports
    minimizing eleven windows when four are on screen, and a grid arrange
    silently allocates tiles to ghosts.
    """
    if _dwmapi is None:
        return False
    value = ctypes.c_int(0)
    try:
        result = _dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_CLOAKED, ctypes.byref(value), ctypes.sizeof(value)
        )
    except OSError:
        return False
    return result == 0 and value.value != 0


def _is_manageable(hwnd, own_pid, own_console):
    if hwnd == own_console:
        return False
    if not _user32.IsWindowVisible(hwnd):
        return False
    if _is_cloaked(hwnd):
        return False

    style = _get_window_long(hwnd, GWL_STYLE)
    if style & WS_CHILD:
        return False

    ex_style = _get_window_long(hwnd, GWL_EXSTYLE)
    # Tool windows are palettes and overlays -- they are deliberately absent
    # from Alt-Tab, and they should be absent here for the same reason. The
    # APPWINDOW override exists because a few apps set both and mean it.
    if (ex_style & WS_EX_TOOLWINDOW) and not (ex_style & WS_EX_APPWINDOW):
        return False
    # WS_EX_NOACTIVATE is a window asking never to be focused: on-screen
    # keyboards, notification toasts, IME candidate popups.
    if ex_style & WS_EX_NOACTIVATE:
        return False

    if _window_pid(hwnd) == own_pid:
        return False

    rect = _window_rect(hwnd)
    if rect is None:
        return False
    # Zero and near-zero windows are message sinks and hidden helpers that
    # happen to carry a title.
    if (rect.right - rect.left) < 32 or (rect.bottom - rect.top) < 32:
        return False

    return True


def enumerate_windows():
    """Every top-level window a person would consider 'open'.

    Excludes our own console, so "minimize everything" doesn't hide the
    dashboard that is answering you.
    """
    own_pid = os.getpid()
    own_console = _kernel32.GetConsoleWindow()
    raw = []

    def callback(hwnd, _lparam):
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if title and _is_manageable(hwnd, own_pid, own_console):
            raw.append((hwnd, _window_pid(hwnd), title))
        return True

    _user32.EnumWindows(_WNDENUMPROC(callback), 0)

    # One psutil pass for the whole enumeration. Per-window lookups turned a
    # microsecond-scale API into something you could feel.
    names = {}
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            names[proc.info["pid"]] = (proc.info["name"] or "").lower()
        except Exception:
            continue

    windows = []
    for hwnd, pid, title in raw:
        real_pid = _real_owner_pid(hwnd, pid, names)
        windows.append(WindowInfo(hwnd, real_pid, title, names.get(real_pid, "")))
    return windows


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

_SEPARATORS = (" - ", " – ", " — ", " | ", " · ")


def _title_segments(title):
    """Titles are nearly always '<document> - <App Name>', so the app name is
    a tail segment rather than the whole string. Scoring the segments is what
    makes "focus chrome" match "Inbox (12) - Gmail - Google Chrome"."""
    segments = [_normalize(title)]
    for separator in _SEPARATORS:
        if separator in title:
            segments.extend(_normalize(part) for part in title.split(separator))
    return [s for s in segments if s]


def _queries_for(spoken):
    """Everything the spoken phrase could reasonably mean.

    Includes the catalog entry's own keys, which is what lets "focus vs code"
    reach a window titled "... - Visual Studio Code" -- the alias table maps
    the nickname, and the catalog supplies the full product name.
    """
    normalized = _normalize(spoken)
    queries = {normalized, _expand_alias(normalized)}

    try:
        entry, score = resolve(spoken)
    except Exception:
        entry, score = None, 0.0
    if entry is not None and score >= MATCH_FLOOR:
        queries |= set(entry.keys)
        if entry.path:
            stem = os.path.splitext(os.path.basename(entry.path))[0]
            queries.add(_normalize(stem))

    return set(q for q in queries if q)


def _score(window, queries):
    best = 0.0
    for q in queries:
        for segment in _title_segments(window.title):
            best = max(best, _similarity(q, segment, strict=True))
        if window.process:
            stem = _normalize(os.path.splitext(window.process)[0])
            if stem:
                best = max(
                    best, _similarity(q, stem, strict=True) * PROC_NAME_WEIGHT
                )
    return best


def find_windows(spoken, windows=None):
    """Matching windows, best first. Empty when nothing clears the floor."""
    windows = enumerate_windows() if windows is None else windows
    queries = _queries_for(spoken)
    if not queries:
        return []

    scored = [(_score(w, queries), w) for w in windows]
    hits = [(s, w) for s, w in scored if s >= WINDOW_MATCH_FLOOR]

    # Foreground first among equals: with six Chrome windows open, "focus
    # chrome" should mean the one you were last using.
    foreground = _user32.GetForegroundWindow()
    hits.sort(key=lambda pair: (pair[0], pair[1].hwnd == foreground), reverse=True)
    return [w for _s, w in hits]


def _is_all(target):
    return _normalize(target or "") in {_normalize(w) for w in _ALL_WORDS}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _window_rect(hwnd):
    rect = RECT()
    if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return rect


def _frame_padding(hwnd):
    """Windows 10 and 11 draw an invisible resize border outside the visible
    edge -- roughly 7px at 100% scaling. GetWindowRect includes it, so tiling
    two windows to exact halves leaves a ~14px gap down the middle and the
    outer edges hang off screen.

    DWM reports the real visual bounds separately; the difference is the
    padding to inflate a target rect by so the *visible* edges land where they
    were asked to.
    """
    if _dwmapi is None:
        return (0, 0, 0, 0)

    frame = RECT()
    try:
        result = _dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(frame), ctypes.sizeof(frame),
        )
    except OSError:
        return (0, 0, 0, 0)
    if result != 0:
        return (0, 0, 0, 0)

    outer = _window_rect(hwnd)
    if outer is None:
        return (0, 0, 0, 0)

    return (
        frame.left - outer.left,
        frame.top - outer.top,
        outer.right - frame.right,
        outer.bottom - frame.bottom,
    )


def _work_area(hwnd=None):
    """Usable desktop on the relevant monitor -- screen minus the taskbar.

    Tiling to the monitor rect instead puts the bottom row of windows behind
    the taskbar, and on a multi-monitor setup it moves everything onto the
    primary display.
    """
    if hwnd:
        monitor = _user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if monitor:
            info = MONITORINFO()
            info.cbSize = ctypes.sizeof(MONITORINFO)
            if _user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                work = info.rcWork
                return (work.left, work.top, work.right, work.bottom)

    rect = RECT()
    if _user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
        return (rect.left, rect.top, rect.right, rect.bottom)

    return (0, 0, _user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1))


def _split(area, cols, rows, gap=TILE_GAP):
    """Row-major tiles across `area`."""
    left, top, right, bottom = area
    width = (right - left - gap * (cols - 1)) // cols
    height = (bottom - top - gap * (rows - 1)) // rows

    tiles = []
    for row in range(rows):
        for col in range(cols):
            x = left + col * (width + gap)
            y = top + row * (height + gap)
            # Last column and row take the remainder so rounding never leaves
            # a one-pixel strip of desktop showing down the right edge.
            x2 = right if col == cols - 1 else x + width
            y2 = bottom if row == rows - 1 else y + height
            tiles.append((x, y, x2, y2))
    return tiles


def _grid_shape(count):
    if count <= 1:
        return 1, 1
    if count == 2:
        return 2, 1
    if count <= 4:
        return 2, 2
    if count <= 6:
        return 3, 2
    return 4, 2


def _half(area, side):
    left, top, right, bottom = area
    mid_x = left + (right - left) // 2
    mid_y = top + (bottom - top) // 2
    if side == "left":
        return (left, top, mid_x, bottom)
    if side == "right":
        return (mid_x, top, right, bottom)
    if side == "top":
        return (left, top, right, mid_y)
    if side == "bottom":
        return (left, mid_y, right, bottom)
    return area


def _centered(area, width_share=0.7, height_share=0.8):
    left, top, right, bottom = area
    width = int((right - left) * width_share)
    height = int((bottom - top) * height_share)
    x = left + ((right - left) - width) // 2
    y = top + ((bottom - top) - height) // 2
    return (x, y, x + width, y + height)


def _cascade_rects(area, count):
    left, top, right, bottom = area
    step = max(24, (bottom - top) // 24)
    width = int((right - left) * 0.68)
    height = int((bottom - top) * 0.72)

    # How many windows fit before the stack would walk off the bottom right
    # corner. At least 1, so the modulo below is always safe.
    span = min((right - left - width), (bottom - top - height))
    per_run = max(1, span // step)

    rects = []
    for i in range(count):
        offset = step * (i % per_run)  # wrap back to the corner and go again
        x = left + offset
        y = top + offset
        rects.append((x, y, min(x + width, right), min(y + height, bottom)))
    return rects


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _settle(check, timeout=SETTLE_TIMEOUT):
    """Waits for another process to act on what we asked.

    Nothing here is synchronous -- ShowWindow and SetWindowPos return long
    before the target app has redrawn -- so checking immediately would report
    failure on every well-behaved window.
    """
    deadline = time.monotonic() + timeout
    while True:
        if check():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(SETTLE_POLL)


def _nudge_input_state():
    """Taps ALT to unlock the foreground.

    Windows only lets the process that owns the most recent input event change
    the foreground window. A console assistant never owns it -- the user's
    input went to whatever they were typing in -- so SetForegroundWindow gets
    refused and the taskbar button flashes instead of the window coming up.

    A synthetic ALT press makes this process the last input source. ALT
    specifically because down-then-up with no other key is a no-op in every
    app: it opens and closes a menu bar in the ones that still have one, and
    is ignored everywhere else.
    """
    try:
        _user32.keybd_event(VK_MENU, 0, 0, None)
        _user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, None)
    except Exception:
        pass


def _force_foreground(hwnd):
    """Brings a window up, and reports honestly whether it worked."""
    if _user32.GetForegroundWindow() == hwnd:
        return True

    if _user32.IsIconic(hwnd):
        # SW_RESTORE, not SW_SHOWNORMAL: a window minimized from maximized
        # should come back maximized, which is what the user left it as.
        _user32.ShowWindow(hwnd, SW_RESTORE)

    _nudge_input_state()

    # Second lever. Sharing an input queue with the current foreground thread
    # makes Windows treat us as that thread for foreground purposes, which
    # covers the cases the ALT tap alone doesn't -- notably when the
    # foreground app is elevated and we are not.
    ours = _kernel32.GetCurrentThreadId()
    theirs = _window_thread(_user32.GetForegroundWindow())
    attached = False
    if theirs and theirs != ours:
        attached = bool(_user32.AttachThreadInput(ours, theirs, True))

    try:
        _user32.BringWindowToTop(hwnd)
        _user32.SetForegroundWindow(hwnd)
    finally:
        # In a finally because leaving two input queues attached is not a
        # local mistake: keyboard focus between those threads stays welded
        # together for as long as both live.
        if attached:
            _user32.AttachThreadInput(ours, theirs, False)

    return _settle(lambda: _user32.GetForegroundWindow() == hwnd)


def _place(hwnd, rect):
    """Moves and sizes one window. Returns True if it landed near enough."""
    if _user32.IsZoomed(hwnd):
        # A maximized window ignores SetWindowPos and snaps back the moment
        # it is touched. Restore first, then place.
        _user32.ShowWindow(hwnd, SW_RESTORE)
        _settle(lambda: not _user32.IsZoomed(hwnd), timeout=0.2)
    elif _user32.IsIconic(hwnd):
        _user32.ShowWindow(hwnd, SW_RESTORE)
        _settle(lambda: not _user32.IsIconic(hwnd), timeout=0.2)

    pad_l, pad_t, pad_r, pad_b = _frame_padding(hwnd)
    left = rect[0] - pad_l
    top = rect[1] - pad_t
    width = (rect[2] + pad_r) - left
    height = (rect[3] + pad_b) - top

    flags = (
        SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOSENDCHANGING
        # ASYNC because SetWindowPos otherwise blocks on the target window's
        # message loop. Arranging six windows where one has hung would freeze
        # the assistant along with it.
        | SWP_ASYNCWINDOWPOS
    )
    _user32.SetWindowPos(hwnd, HWND_TOP, left, top, width, height, flags)

    def landed():
        current = _window_rect(hwnd)
        if current is None:
            return False
        return (
            abs(current.left - left) <= PLACE_TOLERANCE
            and abs(current.top - top) <= PLACE_TOLERANCE
            and abs((current.right - current.left) - width) <= PLACE_TOLERANCE
            and abs((current.bottom - current.top) - height) <= PLACE_TOLERANCE
        )

    return _settle(landed)


def _label(window):
    """Shortest thing that still identifies the window when spoken aloud."""
    segments = [p.strip() for p in window.title.split(" - ") if p.strip()]
    if len(segments) > 1:
        return segments[-1]
    if len(window.title) > 48:
        return window.title[:45].rstrip() + "..."
    return window.title


def _joined(labels, limit=3):
    labels = list(labels)
    if len(labels) <= limit:
        if len(labels) <= 1:
            return "".join(labels)
        return ", ".join(labels[:-1]) + " and " + labels[-1]
    return ", ".join(labels[:limit]) + " and {} more".format(len(labels) - limit)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
#
# Every one returns a sentence. The wake loop scans results for "failed",
# "couldn't" and "didn't" to decide whether to spend a second round trip on a
# spoken correction -- so a failure has to use one of those words and a
# success must not.


# What the last minimize-all put away, so a bare "restore" can put back
# exactly that set instead of un-minimizing everything on the machine.
_last_minimized = []


def focus(target):
    if not target or not target.strip():
        return "Didn't catch which window to focus"

    matches = find_windows(target)
    if not matches:
        return "Couldn't find a window for \"{}\"".format(target)

    window = matches[0]
    if _force_foreground(window.hwnd):
        return "Focused {}".format(_label(window))

    # Worth saying rather than swallowing: a refused foreground change leaves
    # the taskbar button flashing, so there *is* something on screen to look
    # at, and the honest report explains it.
    return "Couldn't bring {} to the front -- its taskbar button is flashing".format(
        _label(window)
    )


def minimize(target):
    global _last_minimized

    windows = enumerate_windows()

    if _is_all(target):
        targets = [w for w in windows if not _user32.IsIconic(w.hwnd)]
        if not targets:
            return "Everything is already minimized"

        done = []
        for window in targets:
            # SHOWMINNOACTIVE rather than MINIMIZE for the bulk case: MINIMIZE
            # activates the next window in Z order, so minimizing ten windows
            # walks the focus through all ten on the way down.
            _user32.ShowWindow(window.hwnd, SW_SHOWMINNOACTIVE)
            done.append(window)

        settled = [w for w in done if _settle(
            lambda h=w.hwnd: bool(_user32.IsIconic(h)), timeout=0.2
        )]
        _last_minimized = settled

        if not settled:
            return "Couldn't minimize anything"
        if len(settled) < len(done):
            return "Minimized {} windows, {} wouldn't go".format(
                len(settled), len(done) - len(settled)
            )
        return "Minimized {} window{}".format(
            len(settled), "" if len(settled) == 1 else "s"
        )

    matches = find_windows(target, windows)
    if not matches:
        return "Couldn't find a window for \"{}\"".format(target)

    window = matches[0]
    _user32.ShowWindow(window.hwnd, SW_MINIMIZE)
    if _settle(lambda: bool(_user32.IsIconic(window.hwnd))):
        _last_minimized = [window]
        return "Minimized {}".format(_label(window))
    return "Couldn't minimize {}".format(_label(window))


def maximize(target):
    if _is_all(target):
        # Every window maximized means one window visible, which is not what
        # anyone wants -- read it as the arrange they probably meant.
        return arrange("grid")

    matches = find_windows(target)
    if not matches:
        return "Couldn't find a window for \"{}\"".format(target)

    window = matches[0]
    _user32.ShowWindow(window.hwnd, SW_MAXIMIZE)
    if _settle(lambda: bool(_user32.IsZoomed(window.hwnd))):
        _force_foreground(window.hwnd)
        return "Maximized {}".format(_label(window))
    return "Couldn't maximize {} -- it may be a fixed-size window".format(
        _label(window)
    )


def restore(target=None):
    """Un-minimizes. With no target, puts back the last minimize-all."""
    global _last_minimized

    if not target or _is_all(target):
        batch = [w for w in _last_minimized if _user32.IsWindow(w.hwnd)]
        if not batch and _is_all(target):
            batch = [w for w in enumerate_windows() if _user32.IsIconic(w.hwnd)]
        if not batch:
            return "Nothing to restore"

        for window in batch:
            _user32.ShowWindow(window.hwnd, SW_RESTORE)
        _last_minimized = []
        return "Restored {} window{}".format(
            len(batch), "" if len(batch) == 1 else "s"
        )

    matches = find_windows(target)
    if not matches:
        return "Couldn't find a window for \"{}\"".format(target)

    window = matches[0]
    _user32.ShowWindow(window.hwnd, SW_RESTORE)
    _force_foreground(window.hwnd)
    return "Restored {}".format(_label(window))


_HALVES = ("left", "right", "top", "bottom")

_LAYOUT_ALIASES = {
    "side by side": "side_by_side",
    "sidebyside": "side_by_side",
    "split": "side_by_side",
    "tile": "grid",
    "tiled": "grid",
    "stack": "cascade",
    "stacked": "cascade",
    "middle": "center",
    "centre": "center",
    "centered": "center",
}


def _resolve_targets(targets, windows):
    """Named targets in order, de-duplicated; all windows when none given."""
    if not targets:
        return windows[:MAX_TILED], []

    chosen, missing, seen = [], [], set()
    for name in targets:
        matches = find_windows(name, windows)
        picked = next((w for w in matches if w.hwnd not in seen), None)
        if picked is None:
            missing.append(name)
            continue
        seen.add(picked.hwnd)
        chosen.append(picked)
    return chosen, missing


def arrange(layout="grid", targets=None):
    layout = _LAYOUT_ALIASES.get(
        (layout or "grid").strip().lower(), (layout or "grid").strip().lower()
    )
    layout = layout.replace(" ", "_")

    windows = enumerate_windows()
    if not windows:
        return "Couldn't find any windows to arrange"

    chosen, missing = _resolve_targets(targets, windows)
    if not chosen:
        if missing:
            return "Couldn't find a window for {}".format(_joined(missing))
        return "Couldn't find any windows to arrange"

    overflow = 0
    if len(chosen) > MAX_TILED:
        overflow = len(chosen) - MAX_TILED
        chosen = chosen[:MAX_TILED]

    # Work area of the monitor the first target is on, so arranging on a
    # second screen stays on that screen.
    area = _work_area(chosen[0].hwnd)

    if layout in _HALVES:
        half = _half(area, layout)
        # "Put chrome and vs code on the left" means both, stacked in that
        # half -- not both on top of each other.
        rects = _split(half, 1, len(chosen)) if len(chosen) > 1 else [half]
    elif layout == "side_by_side":
        rects = _split(area, min(len(chosen), 3), 1)
    elif layout == "cascade":
        rects = _cascade_rects(area, len(chosen))
    elif layout == "center":
        rects = [_centered(area)] * len(chosen)
    else:
        cols, rows = _grid_shape(len(chosen))
        rects = _split(area, cols, rows)

    # side_by_side caps at three columns, so it can hand back fewer rects than
    # there are windows. Counting the difference as overflow keeps the reply
    # honest -- zip() would otherwise drop them without a word.
    if len(rects) < len(chosen):
        overflow += len(chosen) - len(rects)
        chosen = chosen[:len(rects)]

    placed, stubborn = [], []
    for window, rect in zip(chosen, rects):
        if _place(window.hwnd, rect):
            placed.append(window)
        else:
            stubborn.append(window)

    if placed and layout != "cascade":
        # Cascade already implies a Z order; re-focusing the first window
        # would drag it to the top and spoil the stack.
        _force_foreground(placed[0].hwnd)

    if not placed:
        return "Couldn't move {}".format(_joined([_label(w) for w in stubborn]))

    name = layout.replace("_", " ")
    msg = "Arranged {} window{} {}".format(
        len(placed), "" if len(placed) == 1 else "s", name
    )
    if stubborn:
        msg += ", but {} wouldn't move".format(
            _joined([_label(w) for w in stubborn])
        )
    if missing:
        msg += " (couldn't find {})".format(_joined(missing))
    if overflow:
        msg += " -- left {} more where they were".format(overflow)
    return msg


def list_windows(limit=8):
    windows = enumerate_windows()
    if not windows:
        return "Nothing open that I can see"

    labels, seen = [], set()
    for window in windows:
        label = _label(window)
        if label.lower() in seen:
            continue
        seen.add(label.lower())
        labels.append(label)

    shown = labels[:limit]
    msg = "Open: " + _joined(shown, limit=len(shown))
    if len(labels) > limit:
        msg += " -- {} more".format(len(labels) - limit)
    return msg


def explain(spoken, top=5):
    """Why a phrase matched what it did. Debugging aid, not spoken."""
    windows = enumerate_windows()
    queries = _queries_for(spoken)
    scored = sorted(
        ((_score(w, queries), w) for w in windows), reverse=True,
        key=lambda pair: pair[0],
    )

    lines = ["queries: {}".format(sorted(queries)),
             "floor:   {:.2f}".format(WINDOW_MATCH_FLOOR)]
    for score, window in scored[:top]:
        lines.append("  {:.2f}  {:<24} {}".format(
            score, window.process or "?", window.title[:60]
        ))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if not args:
        print(list_windows(limit=20))
        raise SystemExit(0)

    command = args[0].lower()
    rest = args[1:]

    if command == "explain":
        print(explain(" ".join(rest)))
    elif command == "focus":
        print(focus(" ".join(rest)))
    elif command == "minimize":
        print(minimize(" ".join(rest)))
    elif command == "maximize":
        print(maximize(" ".join(rest)))
    elif command == "restore":
        print(restore(" ".join(rest) or None))
    elif command == "arrange":
        print(arrange(rest[0] if rest else "grid", rest[1:] or None))
    else:
        print("usage: python -m commands.window_actions "
              "[focus|minimize|maximize|restore|arrange|explain] <target>")