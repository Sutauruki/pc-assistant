"""The agent's hands: real mouse and keyboard input, and your brakes.

Goes in: commands/agent_input.py   (new file)

INPUT
-----
Clicks, scrolls and keystrokes go through SendInput, exactly like a person's:
every app sees ordinary input, no app needs an API, nothing listens on a port.
Text is typed as Unicode characters (KEYEVENTF_UNICODE), so it doesn't depend
on your keyboard layout, and a newline is typed as Shift+Enter -- a new line in
a chat box, never "send". Pressing Enter on its own is a separate, checked step
(commands/pc_policy.risky_reason).

THE BRAKES (Guard)
------------------
A low-level keyboard hook watches the PHYSICAL keyboard while a task runs:
  Esc          -> stop the task immediately (and cancel any pending approval)
  Ctrl+Enter   -> approve the one risky step Jarvis is waiting on
  F8           -> the same, for keyboards where F-keys work without Fn
Windows marks synthetic keystrokes as injected (LLKHF_INJECTED), and the hook
ignores those -- so the agent pressing Esc to close a popup doesn't stop it,
and nothing the agent types, and no web page, can ever approve a risky step.
Only a finger on a key can.

While an approval is pending, the approval and stop keys are SWALLOWED: they
reach Jarvis and nothing else. Otherwise your Ctrl+Enter would also land in
the app in front -- where in Slack, Teams or a form it means "send" or
"submit" on its own.

Slamming the mouse into the top-left corner of the screen also stops it, and
commands/agent.py stops when the cursor has moved since its last action --
grabbing the mouse is taking over.
"""

import ctypes
import threading
import time
from ctypes import wintypes

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

ULONG_PTR = ctypes.c_size_t
LRESULT = ctypes.c_ssize_t

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
WHEEL_DELTA = 120

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_QUIT = 0x0012
LLKHF_INJECTED = 0x10
VK_ESCAPE = 0x1B
VK_F8 = 0x77
VK_RETURN = 0x0D
VK_CONTROL = 0x11

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77

# Tags our own input in dwExtraInfo -- handy when debugging with a hook.
_JARVIS_TAG = 0x4A415256


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
_user32.SendInput.restype = wintypes.UINT
_user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
_user32.SetCursorPos.restype = wintypes.BOOL
_user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
_user32.GetCursorPos.restype = wintypes.BOOL
_user32.GetSystemMetrics.argtypes = [ctypes.c_int]
_user32.GetSystemMetrics.restype = ctypes.c_int
_user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
_user32.SetWindowsHookExW.restype = wintypes.HHOOK
_user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
_user32.CallNextHookEx.restype = LRESULT
_user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
_user32.UnhookWindowsHookEx.restype = wintypes.BOOL
_user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
_user32.GetMessageW.restype = wintypes.BOOL
_user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.PostThreadMessageW.restype = wintypes.BOOL
_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_user32.GetAsyncKeyState.restype = ctypes.c_short
_kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
_kernel32.GetModuleHandleW.restype = wintypes.HMODULE
_kernel32.GetCurrentThreadId.argtypes = []
_kernel32.GetCurrentThreadId.restype = wintypes.DWORD


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

_VK = {
    "enter": 0x0D, "tab": 0x09, "escape": 0x1B, "backspace": 0x08,
    "delete": 0x2E, "space": 0x20, "up": 0x26, "down": 0x28, "left": 0x25,
    "right": 0x27, "home": 0x24, "end": 0x23, "pageup": 0x21,
    "pagedown": 0x22, "ctrl": 0x11, "shift": 0x10, "alt": 0x12,
    "plus": 0xBB, "minus": 0xBD,
}
_VK.update({"f{}".format(n): 0x6F + n for n in range(1, 13)})
_EXTENDED = {0x2E, 0x26, 0x28, 0x25, 0x27, 0x24, 0x23, 0x21, 0x22}

# Set while the agent is sending input, so the polling fallback (used only
# when the keyboard hook couldn't be installed) doesn't mistake our own Esc
# for yours.
injecting = threading.Event()


def _send(inputs):
    array = (INPUT * len(inputs))(*inputs)
    injecting.set()
    try:
        sent = _user32.SendInput(len(inputs), array, ctypes.sizeof(INPUT))
    finally:
        injecting.clear()
    if sent != len(inputs):
        raise OSError("Windows blocked the input (an elevated window may be in front)")


def _mouse(flags, data=0):
    return INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(
        0, 0, data & 0xFFFFFFFF, flags, 0, _JARVIS_TAG))


def _key(vk=0, scan=0, flags=0):
    if vk in _EXTENDED:
        flags |= KEYEVENTF_EXTENDEDKEY
    return INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(vk, scan, flags, 0, _JARVIS_TAG))


def move(x, y):
    if not _user32.SetCursorPos(int(x), int(y)):
        raise OSError("couldn't move the mouse")


def click(x, y, button="left", double=False):
    move(x, y)
    time.sleep(0.03)
    down, up = ((MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP) if button == "right"
                else (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP))
    for _ in range(2 if double else 1):
        _send([_mouse(down), _mouse(up)])
        time.sleep(0.05)


def scroll(x, y, clicks):
    """Positive clicks scroll up, negative down -- the wheel's convention."""
    move(x, y)
    time.sleep(0.02)
    _send([_mouse(MOUSEEVENTF_WHEEL, int(clicks) * WHEEL_DELTA)])


def press(combo):
    """'ctrl+l' -> Ctrl down, L down, L up, Ctrl up. `combo` must already be
    normalised and allowed by commands/pc_policy.key_reason."""
    codes = []
    for part in combo.split("+"):
        if part in _VK:
            codes.append(_VK[part])
        elif len(part) == 1 and part.isalnum():
            codes.append(ord(part.upper()))
        else:
            raise ValueError("unknown key {!r}".format(part))
    downs = [_key(vk) for vk in codes]
    ups = [_key(vk, flags=KEYEVENTF_KEYUP) for vk in reversed(codes)]
    _send(downs + ups)


def type_text(text, per_char=0.004):
    """Types `text` as Unicode. A newline is Shift+Enter (new line, never
    send); a tab is the Tab key. Characters outside the BMP go as their two
    UTF-16 halves, which is what Windows expects."""
    for ch in str(text):
        if ch == "\r":
            continue
        if ch == "\n":
            press("shift+enter")
        elif ch == "\t":
            press("tab")
        else:
            units = ch.encode("utf-16-le")
            events = []
            for i in range(0, len(units), 2):
                code = int.from_bytes(units[i:i + 2], "little")
                events.append(_key(0, code, KEYEVENTF_UNICODE))
                events.append(_key(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
            _send(events)
        if per_char:
            time.sleep(per_char)


def cursor():
    point = wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(point))
    return (point.x, point.y)


def corner():
    """Top-left of the whole desktop -- the 'stop' corner."""
    return (_user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            _user32.GetSystemMetrics(SM_YVIRTUALSCREEN))


# ---------------------------------------------------------------------------
# The brakes
# ---------------------------------------------------------------------------

class Stopped(Exception):
    """You stopped the task (Esc, mouse in the corner, or took the mouse)."""


class Guard(object):
    """Runs for the length of one task. Use as a context manager.

        with Guard() as guard:
            guard.check()                       # raises Stopped on the brakes
            ok = guard.wait_for_approval(30)    # True only on a physical F8
    """

    def __init__(self):
        self._stop = threading.Event()
        self._approve = threading.Event()
        self._pending = threading.Event()
        self._done = threading.Event()
        self._ready = threading.Event()
        self.reason = ""
        self.hooked = False
        self._hook = None
        self._proc = None
        self._thread = None
        self._thread_id = None
        self._corner_thread = None

    # -- lifecycle --------------------------------------------------------
    def __enter__(self):
        self._thread = threading.Thread(target=self._hook_loop, daemon=True)
        self._thread.start()
        self._ready.wait(2.0)
        self._corner_thread = threading.Thread(target=self._corner_loop, daemon=True)
        self._corner_thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        if self._thread_id:
            _user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread:
            self._thread.join(1.0)
        return False

    def stop(self, reason):
        if not self._stop.is_set():
            self.reason = reason
            self._stop.set()

    @property
    def stopped(self):
        return self._stop.is_set()

    def check(self):
        if self._stop.is_set():
            raise Stopped(self.reason or "stopped")

    # -- approval ---------------------------------------------------------
    def reset_approval(self):
        """Forget any earlier approval keypress and start swallowing the
        approval keys, so only one pressed from now on counts."""
        self._approve.clear()
        self._pending.set()
        if not self.hooked:
            _user32.GetAsyncKeyState(VK_F8)  # clear the "pressed since" bits
            _user32.GetAsyncKeyState(VK_RETURN)

    def _physical_approval_polled(self):
        """Fallback when the hook isn't installed: real key state. The agent
        is paused while it waits, so nothing is being injected right now --
        and it can never press F8 at all (not in pc_policy.ALLOWED_KEYS)."""
        if _user32.GetAsyncKeyState(VK_F8) & 0x8000:
            return True
        return bool(_user32.GetAsyncKeyState(VK_RETURN) & 0x8000
                    and _user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)

    def wait_for_approval(self, timeout, clear=True):
        """True only if a PHYSICAL Ctrl+Enter (or F8) arrives within
        `timeout` seconds -- or since reset_approval(), when clear=False.
        Physical Esc, timeout, or a stop all count as no."""
        if clear:
            self.reset_approval()
        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                if self._stop.is_set():
                    return False
                if self._approve.is_set():
                    return True
                if not self.hooked and self._physical_approval_polled():
                    return True
                time.sleep(0.05)
            return False
        finally:
            self._pending.clear()

    # -- watchers ---------------------------------------------------------
    def _hook_loop(self):
        self._thread_id = _kernel32.GetCurrentThreadId()

        def callback(n_code, w_param, l_param):
            swallow = False
            try:
                if n_code == 0 and w_param in (WM_KEYDOWN, WM_SYSKEYDOWN):
                    info = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                    if not info.flags & LLKHF_INJECTED:
                        pending = self._pending.is_set()
                        if info.vkCode == VK_ESCAPE:
                            self.stop("you pressed Esc")
                            swallow = pending
                        elif info.vkCode == VK_F8:
                            if pending:
                                self._approve.set()
                                swallow = True
                        elif info.vkCode == VK_RETURN and pending and \
                                _user32.GetAsyncKeyState(VK_CONTROL) & 0x8000:
                            self._approve.set()
                            swallow = True
            except Exception:
                pass
            if swallow:
                return 1  # handled: don't pass it on to the app in front
            return _user32.CallNextHookEx(None, n_code, w_param, l_param)

        self._proc = HOOKPROC(callback)  # keep a reference: Windows calls it
        self._hook = _user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._proc, _kernel32.GetModuleHandleW(None), 0)
        self.hooked = bool(self._hook)
        self._ready.set()
        if not self._hook:
            print("agent: keyboard hook unavailable; watching Esc by polling")
            return
        msg = wintypes.MSG()
        try:
            while _user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                pass
        finally:
            _user32.UnhookWindowsHookEx(self._hook)

    def _corner_loop(self):
        left, top = corner()
        if not self.hooked:
            _user32.GetAsyncKeyState(VK_ESCAPE)
        while not self._done.is_set() and not self._stop.is_set():
            x, y = cursor()
            if x <= left + 1 and y <= top + 1:
                self.stop("you moved the mouse to the corner")
            elif (not self.hooked and not injecting.is_set()
                  and _user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000):
                self.stop("you pressed Esc")
            time.sleep(0.05)
