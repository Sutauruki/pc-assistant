"""What's happening in a specific app -- and telling you when it's done.

Goes in: commands/app_watch.py   (new file)

    "is Claude done?"              -> status("claude", ...)
    "what's my terminal doing?"    -> status("terminal", ...)
    "tell me when Claude is done"  -> watch("claude")
    "stop watching"                -> stop()
    (Telegram) "show me Claude"    -> picture("claude"), sent as a photo

ONE WINDOW, NOT THE SCREEN
--------------------------
The app's window is found by name (window_actions.find_windows: "claude",
"vs code", "terminal", "chrome") and drawn on its own with PrintWindow, so
it works while the window is behind others, and nothing else on screen is
captured. A minimized window is restored without taking focus, captured,
and minimized again. Some GPU-drawn apps come out black; then the capture
falls back to the screen only when that window is in front -- otherwise
whatever covers it would be captured instead.

The window's text is read too (UI Automation, including a terminal's
buffer), because text beats pixels for terminals and editors. Both go to
Gemini with a status question, which answers in a sentence or two and
classifies the app as working / waiting for you / done / error / idle.

THE WATCHER
-----------
Every WATCH_INTERVAL seconds it draws a small grey thumbnail of the window
and compares it with the last one -- on the PC, milliseconds, no model
call. Only when the window has stopped changing for STILL_CHECKS checks in
a row is Gemini asked once whether it's finished. Done, waiting for you,
error or the window closing -> a notification on the PC and a Telegram
message if paired. Still working (a long silent build) -> it keeps
watching and asks again later. A 40-minute task costs a handful of calls.

PRIVACY
-------
Same rules as screen vision: windows commands/pc_policy.py calls private
(password managers, banking, private browsing) are refused, and password
boxes are blacked out. Text that looks like a key or token is replaced
before anything leaves the PC. A status answer never sends a picture to the
phone -- only "show me <app>" does, because you asked for one.
"""

import ctypes
import json
import re
import threading
import time

from commands import pc_policy

WATCH_INTERVAL = 30          # seconds between thumbnails
STILL_CHECKS = 2             # unchanged thumbnails before asking the model
RECHECK_AFTER = 4            # after "still working", this many more still checks
CHANGE_THRESHOLD = 0.8       # mean grey-level difference (0-255) that counts as change
MAX_WATCH_SECONDS = 4 * 3600
MAX_MODEL_CHECKS = 25        # per watch
MAX_WATCHES = 3
TEXT_TAIL = 4000             # characters of window text sent with the picture

PW_RENDERFULLCONTENT = 2
SW_SHOWNOACTIVATE = 4
SW_SHOWMINNOACTIVE = 7
DWMWA_EXTENDED_FRAME_BOUNDS = 9

STATES = ("working", "waiting", "done", "error", "idle", "unknown")
FINISHED = {"waiting", "done", "error", "idle"}


# ---------------------------------------------------------------------------
# Finding the window
# ---------------------------------------------------------------------------

def find(target):
    """window_info() dict for the best window matching `target`, or None."""
    from commands import desktop, window_actions as wa
    target = str(target or "").strip()
    if not target:
        return None
    try:
        hits = wa.find_windows(target)
    except Exception as e:
        print("app_watch: couldn't list windows ({})".format(e))
        return None
    # Jarvis's own console (titled with its name) when it runs inside
    # Windows Terminal -- "what's my terminal doing" means another one.
    from commands import identity
    hits = [w for w in hits if w.title.strip().lower() != identity.NAME.lower()]
    return desktop.window_info(hits[0].hwnd) if hits else None


def _label(window):
    """"Claude" from a window -- the app, not the whole title."""
    process = (window.get("process") or "").lower().replace(".exe", "")
    names = {"windowsterminal": "your terminal", "openconsole": "your terminal",
             "cmd": "Command Prompt", "powershell": "PowerShell", "pwsh": "PowerShell",
             "code": "VS Code", "chrome": "Chrome", "msedge": "Edge",
             "firefox": "Firefox", "claude": "Claude"}
    return names.get(process) or (process[:1].upper() + process[1:]) or "that app"


def is_terminal(window):
    process = (window.get("process") or "").lower()
    return process in {"windowsterminal.exe", "openconsole.exe", "cmd.exe",
                       "powershell.exe", "pwsh.exe", "conhost.exe", "wsl.exe",
                       "mintty.exe", "alacritty.exe", "wezterm-gui.exe"}


# ---------------------------------------------------------------------------
# Drawing one window
# ---------------------------------------------------------------------------

_gdi = {}


def _api():
    """Private ctypes bindings (handles are pointers on 64-bit Windows)."""
    if _gdi:
        return _gdi
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    H = ctypes.c_void_p
    user32.GetWindowDC.argtypes = [wintypes.HWND]
    user32.GetWindowDC.restype = H
    user32.ReleaseDC.argtypes = [wintypes.HWND, H]
    user32.PrintWindow.argtypes = [wintypes.HWND, H, wintypes.UINT]
    user32.PrintWindow.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.c_void_p]
    user32.GetForegroundWindow.restype = wintypes.HWND
    gdi32.CreateCompatibleDC.argtypes = [H]
    gdi32.CreateCompatibleDC.restype = H
    gdi32.CreateCompatibleBitmap.argtypes = [H, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = H
    gdi32.SelectObject.argtypes = [H, H]
    gdi32.SelectObject.restype = H
    gdi32.DeleteObject.argtypes = [H]
    gdi32.DeleteDC.argtypes = [H]
    gdi32.GetDIBits.argtypes = [H, H, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                                ctypes.c_void_p, wintypes.UINT]
    try:
        dwm = ctypes.WinDLL("dwmapi")
    except OSError:
        dwm = None
    _gdi.update(user32=user32, gdi32=gdi32, dwm=dwm)
    return _gdi


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32)]


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


def _window_rect(hwnd):
    rect = _RECT()
    if not _api()["user32"].GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return (rect.left, rect.top, rect.right, rect.bottom)


def _frame_rect(hwnd):
    """The visible frame, without Windows 10/11's invisible resize border."""
    dwm = _api()["dwm"]
    if dwm is not None:
        rect = _RECT()
        if dwm.DwmGetWindowAttribute(ctypes.c_void_p(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS,
                                     ctypes.byref(rect), ctypes.sizeof(rect)) == 0:
            return (rect.left, rect.top, rect.right, rect.bottom)
    return _window_rect(hwnd)


def _print_window(hwnd):
    """(PIL image, window rect) drawn by the window itself, or (None, rect)."""
    from PIL import Image
    api = _api()
    user32, gdi32 = api["user32"], api["gdi32"]
    rect = _window_rect(hwnd)
    if not rect:
        return None, None
    width, height = rect[2] - rect[0], rect[3] - rect[1]
    if width < 16 or height < 16:
        return None, rect
    wdc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(wdc)
    bmp = gdi32.CreateCompatibleBitmap(wdc, width, height)
    old = gdi32.SelectObject(mem, bmp)
    try:
        if not user32.PrintWindow(hwnd, mem, PW_RENDERFULLCONTENT):
            return None, rect
        header = _BITMAPINFOHEADER(biSize=ctypes.sizeof(_BITMAPINFOHEADER), biWidth=width,
                                   biHeight=-height, biPlanes=1, biBitCount=32)
        buf = ctypes.create_string_buffer(width * height * 4)
        if not gdi32.GetDIBits(mem, bmp, 0, height, buf, ctypes.byref(header), 0):
            return None, rect
        img = Image.frombuffer("RGB", (width, height), buf, "raw", "BGRX", 0, 1)
        return img.copy(), rect
    finally:
        gdi32.SelectObject(mem, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(hwnd, wdc)


def _is_blank(img):
    lo, hi = img.convert("L").getextrema()
    return hi < 16 or lo == hi


def capture(window, mask_passwords=True):
    """(PIL image of just this window, note) -- note is "" or why it's
    missing. Restores a minimized window without focusing it, and puts it
    back afterwards."""
    api = _api()
    user32 = api["user32"]
    hwnd = window["hwnd"]
    if not user32.IsWindow(hwnd):
        return None, "that window has closed"
    was_min = bool(user32.IsIconic(hwnd))
    try:
        if was_min:
            user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
            time.sleep(0.45)  # let it paint
        img, rect = _print_window(hwnd)
        if img is not None and _is_blank(img):
            img = None
        if img is None and rect and user32.GetForegroundWindow() == hwnd:
            # GPU-drawn and in front: the screen shows exactly this window.
            import mss
            from PIL import Image
            with mss.mss() as sct:
                shot = sct.grab({"left": rect[0], "top": rect[1],
                                 "width": rect[2] - rect[0], "height": rect[3] - rect[1]})
                img = Image.frombytes("RGB", shot.size, shot.rgb)
        if img is None:
            return None, "{} draws itself in a way I can't capture while it's behind other windows".format(_label(window))
        if mask_passwords:
            from commands import screen_actions, uia
            try:
                snap = uia.snapshot(hwnd, rect, max_items=0)
                screen_actions.mask(img, [e["rect"] for e in snap["elements"] if e["password"]], rect)
            except Exception:
                pass
        frame = _frame_rect(hwnd)
        if frame and rect:
            box = (max(0, frame[0] - rect[0]), max(0, frame[1] - rect[1]),
                   min(img.width, frame[2] - rect[0]), min(img.height, frame[3] - rect[1]))
            if box[2] - box[0] > 16 and box[3] - box[1] > 16:
                img = img.crop(box)
        return img, ""
    finally:
        if was_min:
            user32.ShowWindow(hwnd, SW_SHOWMINNOACTIVE)


def thumbnail(window):
    """Something to compare between checks, or None: a small grey picture
    of the window, or -- while it's minimized -- a hash of its text, so the
    watcher never flashes a minimized window back up every half minute.
    Neither leaves the PC."""
    try:
        if _api()["user32"].IsIconic(window["hwnd"]):
            text = window_text(window)
            return "text:{}".format(hash(text)) if text else None
        img, _ = capture(window, mask_passwords=False)
    except Exception:
        return None
    if img is None:
        return None
    return img.convert("L").resize((160, 100))


def changed(a, b):
    """How different two thumbnails are, 0-255 (text hashes: 0 or 255)."""
    if a is None or b is None or isinstance(a, str) != isinstance(b, str):
        return 255.0
    if isinstance(a, str):
        return 0.0 if a == b else 255.0
    from PIL import ImageChops, ImageStat
    return ImageStat.Stat(ImageChops.difference(a, b)).mean[0]


# ---------------------------------------------------------------------------
# Reading the window's text
# ---------------------------------------------------------------------------

_SECRET_TEXT = re.compile(
    r"(sk-[A-Za-z0-9_\-]{16,}|gsk_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z_\-]{30,}|xox[abp]-[A-Za-z0-9\-]{10,}|"
    r"\b\d{6,}:[A-Za-z0-9_\-]{30,}\b|"                       # Telegram bot tokens
    r"(?i:(?:api[_-]?key|token|secret|password|passwd)\s*[=:]\s*)\S+|"
    r"\b[A-Za-z0-9+/_\-]{40,}\b)")


def scrub(text):
    """Text with anything key- or token-shaped replaced."""
    return _SECRET_TEXT.sub("[hidden]", str(text or ""))


def window_text(window):
    """The window's text as UI Automation reports it: a terminal or editor
    buffer when there is one, plus visible labels. Tail-trimmed, scrubbed."""
    from commands import uia
    hwnd = window["hwnd"]
    parts = []
    try:
        api, mod = uia._uia()
        root = api.ElementFromHandle(hwnd)
        cond = api.CreatePropertyCondition(mod.UIA_IsTextPatternAvailablePropertyId, True)
        el = root.FindFirst(mod.TreeScope_Subtree, cond) if root is not None else None
        if el is not None:
            pattern = el.GetCurrentPattern(mod.UIA_TextPatternId)
            pattern = pattern.QueryInterface(mod.IUIAutomationTextPattern)
            text = pattern.DocumentRange.GetText(-1) or ""
            lines = [ln.rstrip() for ln in text.splitlines()]
            while lines and not lines[-1]:
                lines.pop()
            parts.append("\n".join(lines)[-TEXT_TAIL:])
    except Exception:
        pass
    try:
        snap = uia.snapshot(hwnd, None, max_items=40, max_text_chars=3000)
        if snap["text"]:
            parts.append(snap["text"][-2000:])
        buttons = [e["name"] for e in snap["elements"] if e["type"] == "button" and e["name"]]
        if buttons:
            parts.append("Buttons: " + ", ".join(buttons[:25]))
    except Exception:
        pass
    return scrub("\n\n".join(p for p in parts if p.strip()))


# ---------------------------------------------------------------------------
# Asking about it
# ---------------------------------------------------------------------------

_SYSTEM = """You look at one app window on the user's PC -- a screenshot plus the
text Windows reports for it -- and tell them what's happening in it, out
loud. Answer the question they asked. If it's an AI assistant or agent
(Claude, ChatGPT, Claude Code, Copilot), a terminal or a build: say whether
it's still working (a spinner, a Stop button, output still streaming, an
"esc to interrupt" hint), waiting for them (a permission prompt, a question,
a yes/no), finished (a final answer and an idle input box), or stuck on an
error -- and in a few words what it's doing or did. Be specific and accurate;
say what you actually see, and if you can't tell, say so. Never read out
code, long paths or keys.

Return JSON: {"state": one of "working", "waiting", "done", "error", "idle",
"unknown", "summary": one to three short spoken sentences, no markdown, no
lists}."""


def _system():
    try:
        from commands import persona
        return _SYSTEM + "\n\n" + persona.prompt_block()
    except Exception:
        return _SYSTEM


def _ask(window, img, text, question):
    """(state, summary) from Gemini."""
    from commands import screen_actions
    contents = []
    if img is not None:
        jpeg, _ = screen_actions.to_jpeg(img, max_size=1280, quality=70)
        contents.append(screen_actions.image_part(jpeg))
    contents.append("App: {} -- window title: {}\nQuestion: {}\n\nWindow text (may be "
                    "partial):\n{}".format(_label(window), scrub(window.get("title", ""))[:150],
                                           question, text or "(none readable)"))
    raw = "".join(screen_actions.gemini_stream(
        contents, screen_actions.gemini_config(_system(), json_mode=True)))
    try:
        data = json.loads(raw.strip().strip("`").replace("json\n", "", 1))
    except ValueError:
        return "unknown", raw.strip()[:400]
    state = str(data.get("state") or "unknown").lower()
    return (state if state in STATES else "unknown"), str(data.get("summary") or "").strip()


def _look(window, question):
    """Policy check, capture, text, model. Returns (state, summary)."""
    private = pc_policy.sensitive_reason(window)
    if private:
        return "unknown", "I'm not looking at that -- {}.".format(private)
    img, note = capture(window)
    text = window_text(window)
    if img is None and not text:
        return "unknown", "I couldn't see {}: {}.".format(_label(window), note)
    return _ask(window, img, text, question)


def status(target, question=""):
    """Spoken answer to "is Claude done?" / "what's my terminal doing?"."""
    window = find(target)
    if window is None:
        return "I can't find a {} window open.".format(target or "matching")
    question = (question or "").strip() or "What's happening in {} right now? Is it done?".format(target)
    try:
        _state, summary = _look(window, question)
    except Exception as e:
        return "I couldn't check {}: {}".format(_label(window), e)
    return summary or "I couldn't make out what {} is doing.".format(_label(window))


def picture(target):
    """(path to a JPEG of just that window, caption) or (None, why not).
    For Telegram's "show me <app>"; saved under Documents\\Jarvis\\Screenshots."""
    window = find(target)
    if window is None:
        return None, "I can't find a {} window open.".format(target)
    private = pc_policy.sensitive_reason(window)
    if private:
        return None, "Not sending that -- {}.".format(private)
    img, note = capture(window)
    if img is None:
        return None, "I couldn't capture {}: {}.".format(_label(window), note)
    from commands import paths, screen_actions
    jpeg, _ = screen_actions.to_jpeg(img, max_size=1600, quality=75)
    folder = paths.jarvis_dir("Screenshots")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "{} {}.jpg".format(_label(window).replace("your ", ""),
                                       time.strftime("%Y-%m-%d %H-%M-%S"))
    path.write_bytes(jpeg)
    return path, scrub(window.get("title", ""))[:200]


# ---------------------------------------------------------------------------
# Watching
# ---------------------------------------------------------------------------

_watches = {}            # hwnd -> {"label", "stop": Event, "thread"}
_watch_lock = threading.Lock()


def _announce(line):
    """Done/waiting/closed: a notification on the PC and a Telegram message."""
    print("app_watch: " + line)
    try:
        from commands import notify
        notify.notify("Heads up", line)
    except Exception:
        pass
    try:
        from listener import telegram_bot
        telegram_bot.send(line)
    except Exception:
        pass


def _name():
    try:
        from commands import recall
        p = recall.profile()
        return (p.get("call_me") or p.get("name") or "").strip()
    except Exception:
        return ""


def _finished_line(label, state, summary):
    who = _name()
    lead = {"done": "{} is done", "waiting": "{} needs you", "error": "{} hit a problem",
            "idle": "{} has stopped"}[state].format(label[:1].upper() + label[1:])
    return "{}{}. {}".format(lead, ", " + who if who else "", summary).strip()


def _watch_loop(window, stop):
    import comtypes
    comtypes.CoInitialize()
    label = _label(window)
    started = time.time()
    previous = thumbnail(window)
    still, need, checks, blank = 0, STILL_CHECKS, 0, 0
    try:
        while not stop.wait(WATCH_INTERVAL):
            if time.time() - started > MAX_WATCH_SECONDS:
                _announce("I stopped watching {} after four hours.".format(label))
                return
            if not _api()["user32"].IsWindow(window["hwnd"]):
                _announce("{}'s window closed while I was watching it.".format(
                    label[:1].upper() + label[1:]))
                return
            current = thumbnail(window)
            if current is None:
                blank += 1
                if blank == 20:  # ~10 minutes of nothing (PC locked?)
                    print("app_watch: can't see {} -- is the PC locked?".format(label))
                continue
            blank = 0
            if changed(previous, current) > CHANGE_THRESHOLD:
                still = 0
            else:
                still += 1
            previous = current
            if still < need:
                continue
            checks += 1
            try:
                state, summary = _look(window, "Is it finished, waiting for me, or still working?")
            except Exception as e:
                print("app_watch: couldn't check {} ({})".format(label, e))
                state, summary = "unknown", ""
            if state in FINISHED:
                _announce(_finished_line(label, state, summary))
                return
            if checks >= MAX_MODEL_CHECKS:
                _announce("I've checked {} {} times and it still looks busy, so I've "
                          "stopped watching. Last look: {}".format(label, checks, summary))
                return
            still, need = 0, RECHECK_AFTER  # still working: look again later
    finally:
        with _watch_lock:
            _watches.pop(window["hwnd"], None)
        comtypes.CoUninitialize()


def watch(target):
    """Starts watching `target`'s window. Returns what to say."""
    window = find(target)
    if window is None:
        return "I can't find a {} window open to watch.".format(target or "matching")
    private = pc_policy.sensitive_reason(window)
    if private:
        return "I won't watch that -- {}.".format(private)
    label = _label(window)
    with _watch_lock:
        if window["hwnd"] in _watches:
            return "I'm already watching {}.".format(label)
        if len(_watches) >= MAX_WATCHES:
            return "I'm already watching {} things -- say stop watching first.".format(MAX_WATCHES)
        stop = threading.Event()
        thread = threading.Thread(target=_watch_loop, args=(window, stop),
                                  name="watch-" + label, daemon=True)
        _watches[window["hwnd"]] = {"label": label, "stop": stop, "thread": thread}
    thread.start()
    try:
        from listener import telegram_bot
        how = "message your phone" if telegram_bot.paired() else "let you know"
    except Exception:
        how = "let you know"
    return "I'll keep an eye on {} and {} when it's done or needs you.".format(label, how)


def stop(target=""):
    """Stops one watch (by app name) or all of them. Returns what to say."""
    target = str(target or "").strip().lower()
    with _watch_lock:
        items = list(_watches.items())
    if not items:
        return "I'm not watching anything."
    chosen = [(h, w) for h, w in items
              if not target or target in w["label"].lower() or w["label"].lower() in target]
    if not chosen:
        return "I'm not watching {}.".format(target)
    for _hwnd, w in chosen:
        w["stop"].set()
    labels = [w["label"] for _h, w in chosen]
    return "Stopped watching {}.".format(" and ".join(labels))


def watching():
    with _watch_lock:
        return [w["label"] for w in _watches.values()]
