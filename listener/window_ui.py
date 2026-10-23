"""Jarvis's desktop window: a minimal, premium floating card.

Goes in: listener/window_ui.py

    compact    the resting state: just the orb, floating on a transparent,
               click-through background -- it still changes colour, breathes,
               swells with the voice and shows the thinking comet
    expanded   wordmark header, the orb, what you said, the reply, an input
               field -- and Approve / Cancel when Jarvis asks

WHEN IT OPENS
-------------
Compact unless there's a reason not to be:
  - Jarvis needs your approval   -> opens, and shrinks back ~1.5s after
                                    you decide
  - a result is meant to be read -> peek(): opens for a while (file lists,
    (or you said "show me...")      clipboard, command output), then
                                    shrinks unless you're using it
  - you ask                      -> show(): "maximize" / "show yourself"
                                    keeps it open until "minimize" (hide())
Double-click the orb, or "-" on the card, does the same by hand.

DESIGN
------
One canvas draws everything, so there are no stock widget borders anywhere
except the text field. The window is borderless; on Windows 11 the system
rounds its corners (DWM corner preference), elsewhere it's a clean square
card. The orb is an anti-aliased radial-gradient sphere with a soft glow,
rendered with Pillow (already a dependency) at a handful of sizes per state
and cached -- animating it is swapping one image, not redrawing shapes.
Idle, it breathes; speaking, it swells with the voice; thinking, three dots
orbit it.

LIGHT ON PURPOSE
----------------
Tkinter ships with Python. Orb images are rendered lazily and only for the
sizes actually shown (a few hundred KB per state). No state is kept here:
each frame reads listener/ui.py's snapshot(). Redraws run at ACTIVE_FPS while
recording / thinking / speaking and IDLE_FPS otherwise; text items are only
touched when the text changes.

THREADS
-------
Tk lives on its own daemon thread; nothing else touches it. Input goes to
console_input.submit(), so main.py treats it exactly like typing -- including
as a physical answer to a confirmation (a click is one).

SAFETY
------
Hidden while the computer agent works (set_agent_busy), so it never covers a
spot the agent clicks or appears in its screenshots; the agent also refuses
any window owned by the Jarvis process, so it can't press Approve itself.

    JARVIS_NO_WINDOW=1   don't open it
"""

import json
import math
import os
import threading
import time
from pathlib import Path

from commands import identity
from listener import console_input
from listener import ui

W, H = 340, 330            # expanded card
MINI = 70                  # compact orb window (transparent around the orb)
PEEK_SECONDS = 25          # how long a peek stays open
KEY = "#010203"            # colour made transparent in compact mode (Windows)
ACTIVE_FPS = 30
IDLE_FPS = 10              # enough for a smooth breath, cheap enough to ignore
REPLY_CHARS = 170
HEARD_CHARS = 80
LEVEL_STEPS = 16           # orb sizes per state (rendered lazily)
ORB_Y = 100                # orb centre in the expanded card

CARD = "#0c0e13"
EDGE = "#1d2230"
FIELD = "#151922"
TEXT = "#eef1f6"
SOFT = "#9aa3b5"
FAINT = "#5c6476"
ACCENT = "#5eead4"
ON_ACCENT = "#04201c"

# state -> (orb colour, label)
_STATES = {
    ui.IDLE: ("#6b7280", "Idle"),
    ui.LISTENING: ("#2dd4bf", "Listening"),
    ui.RECORDING: ("#fb7185", "Hearing you"),
    ui.THINKING: ("#fbbf24", "Thinking"),
    ui.SPEAKING: ("#60a5fa", "Speaking"),
}

_thread = None
_ready = threading.Event()
_busy = threading.Event()
_started = False
_instance = None
_requests = []             # ("show"|"hide"|"peek", arg), taken by the Tk thread
_requests_lock = threading.Lock()


def _request(kind, arg=None):
    with _requests_lock:
        _requests.append((kind, arg))


def show():
    """Open the card and keep it open until hide() -- "maximize", "show
    yourself". Safe from any thread."""
    _request("show")


def hide():
    """Back to the floating orb -- "minimize", "hide". Safe from any thread."""
    _request("hide")


def peek(seconds=PEEK_SECONDS):
    """Open for a while to show something worth reading, then shrink back on
    its own (unless you're hovering or typing). Safe from any thread."""
    _request("peek", seconds)


def enabled():
    return not os.environ.get("JARVIS_NO_WINDOW")


def start():
    """Opens the window on its own thread. True if it's up."""
    global _thread
    if not enabled() or _started:
        return _started
    try:
        import tkinter  # noqa: F401
        from PIL import ImageTk  # noqa: F401
    except ImportError as e:
        print("window: {} isn't available, staying in the terminal".format(e))
        return False
    _thread = threading.Thread(target=_run, daemon=True, name="jarvis-window")
    _thread.start()
    _ready.wait(5.0)
    return _started


def started():
    return _started


def set_agent_busy(busy):
    """Hide while the computer agent works; come back when it's done."""
    if busy:
        _busy.set()
    else:
        _busy.clear()


def _short(text, limit):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _spaced(text):
    """Letter-spaced caps for small labels (Tk has no tracking)."""
    return " ".join(text.upper())


def _rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _position_file():
    return Path(__file__).resolve().parent.parent / "data" / "window.json"


# ---------------------------------------------------------------------------
# The orb
# ---------------------------------------------------------------------------

class _Orbs(object):
    """Anti-aliased orb images, rendered on first use and cached."""

    def __init__(self):
        self._cache = {}

    def get(self, colour, radius, level, glow=True, bg=CARD):
        step = max(0, min(LEVEL_STEPS, int(round(level * LEVEL_STEPS))))
        key = (colour, radius, step, glow, bg)
        image = self._cache.get(key)
        if image is None:
            image = self._cache[key] = self._render(
                colour, radius, step / float(LEVEL_STEPS), glow, bg)
        return image

    @staticmethod
    def _render(colour, base, level, glow=True, bg=CARD):
        import numpy as np
        from PIL import Image, ImageTk

        r = base * (1.0 + 0.16 * level)              # sphere radius
        g = r + base * (0.45 + 0.75 * level)         # glow reach
        size = int(math.ceil(2 * base * 2.36)) + 4   # room for the biggest glow
        c = (size - 1) / 2.0
        y, x = np.mgrid[0:size, 0:size].astype(np.float32)
        d = np.hypot(x - c, y - c)

        card = np.array(_rgb(bg), np.float32)
        tint = np.array(_rgb(colour), np.float32)
        out = np.broadcast_to(card, (size, size, 3)).copy()

        # Glow: soft falloff outside the sphere. Not on the transparent
        # compact window: a half-transparent glow can't be shown there (only
        # the exact key colour turns transparent), so it would be a dark halo.
        if glow:
            t = np.clip(1.0 - (d - r) / max(g - r, 1.0), 0.0, 1.0)
            glow_a = (t ** 2.2) * (0.30 + 0.25 * level)
            glow_a[d < r] = 0.0
            out += (tint - out) * glow_a[..., None]

        # Sphere: lit from the upper left, deeper toward the rim.
        hx, hy = c - 0.32 * r, c - 0.38 * r
        hd = np.clip(np.hypot(x - hx, y - hy) / (1.35 * r), 0.0, 1.0)
        light = tint + (255.0 - tint) * 0.55
        deep = tint * 0.62
        sphere = light + (deep - light) * (hd ** 1.25)[..., None]
        rim = np.clip(r + 0.5 - d, 0.0, 1.0)          # analytic anti-aliasing
        out += (sphere - out) * rim[..., None]

        img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
        return ImageTk.PhotoImage(img)


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

class _Window(object):
    def __init__(self, root):
        import tkinter as tkm
        import tkinter.font as tkfont

        self.tkm = tkm
        self.root = root
        root.title(identity.NAME)
        root.overrideredirect(True)
        root.configure(bg=CARD)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.0)  # faded in once drawn
        except Exception:
            pass

        families = set(tkfont.families(root))
        face = next((f for f in ("Segoe UI Variable Text", "Segoe UI", "Inter",
                                 "Helvetica Neue", "DejaVu Sans") if f in families),
                    "TkDefaultFont")
        display = next((f for f in ("Segoe UI Variable Display", "Segoe UI Semibold", face)
                        if f in families), face)
        self.f_mark = (display, 9, "bold")
        self.f_label = (face, 8, "bold")
        self.f_heard = (face, 9)
        self.f_reply = (face, 11)
        self.f_button = (face, 9, "bold")
        self.f_entry = (face, 10)

        self.orbs = _Orbs()
        self.compact = True        # resting state: the floating orb
        self.pinned = False        # you asked for the card ("maximize")
        self.auto = None           # why it opened by itself: "confirm" / "peek"
        self.auto_until = 0.0
        self.hover = False
        self._asking = False
        self.orb_pos = None        # orb centre on screen -- the anchor for both sizes
        self.level = 0.0
        self.t0 = time.time()
        self.hidden_for_agent = False
        self._last_text = None
        self._drag = None
        self._save_pending = None
        self._placeholder = True
        self.entry = None

        self.canvas = tkm.Canvas(root, width=W, height=H, bg=CARD, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self._build()
        self._place()
        self._round_corners()

        root.bind("<Button-3>", self._menu)
        root.bind("<Enter>", lambda e: setattr(self, "hover", True))
        root.bind("<Leave>", lambda e: setattr(self, "hover", False))
        self._fade_in(0.0)

    # -- construction --------------------------------------------------------
    def _build(self):
        cv = self.canvas
        cv.delete("all")
        if self.entry is not None:
            self.entry.destroy()
            self.entry = None
        self.confirm_items = []
        self._last_text = None

        if self.compact:
            # Only the orb: everything in the key colour is see-through and
            # click-through on Windows, so the window itself disappears.
            self._transparent(True)
            cv.config(width=MINI, height=MINI)
            self.orb_xy = (MINI // 2, MINI // 2)
            self.orb_radius = 18
            self.orb = cv.create_image(*self.orb_xy)
            self.dots = [cv.create_oval(0, 0, 0, 0, outline="", fill="") for _ in range(3)]
            cv.bind("<ButtonPress-1>", self._drag_start)
            cv.bind("<B1-Motion>", self._drag_move)
            cv.bind("<Double-Button-1>", lambda e: self.expand(pin=True))
            return

        self._transparent(False)
        cv.config(width=W, height=H)
        cv.bind("<ButtonPress-1>", self._maybe_drag_start)
        cv.bind("<B1-Motion>", self._drag_move)
        cv.unbind("<Double-Button-1>")

        # Hairline frame (Windows 11 rounds the window itself to match).
        self._round_rect(1, 1, W - 2, H - 2, 10, outline=EDGE, fill="")

        # Header: wordmark, and "-" to shrink to the orb.
        cv.create_text(22, 22, text=_spaced(identity.NAME), anchor="w", fill=SOFT, font=self.f_mark)
        self.btn_min = cv.create_text(W - 24, 20, text="–", fill=FAINT,
                                      font=(self.f_mark[0], 13))
        cv.tag_bind(self.btn_min, "<Enter>", lambda e: cv.itemconfig(self.btn_min, fill=TEXT))
        cv.tag_bind(self.btn_min, "<Leave>", lambda e: cv.itemconfig(self.btn_min, fill=FAINT))
        cv.tag_bind(self.btn_min, "<Button-1>", lambda e: self.collapse())

        # Orb and state.
        self.orb_xy = (W // 2, ORB_Y)
        self.orb_radius = 30
        self.orb = cv.create_image(*self.orb_xy)
        cv.tag_bind(self.orb, "<Double-Button-1>", lambda e: self.collapse())
        self.dots = [cv.create_oval(0, 0, 0, 0, outline="", fill="") for _ in range(3)]
        self.state_text = cv.create_text(W // 2, 160, text="", fill=FAINT, font=self.f_label)

        # Conversation.
        self.heard_text = cv.create_text(24, 184, text="", anchor="nw", fill=SOFT,
                                         font=self.f_heard, width=W - 48)
        self.reply_text = cv.create_text(24, 206, text="", anchor="nw", fill=TEXT,
                                         font=self.f_reply, width=W - 48)

        # Input field with a round send button. While Jarvis is asking for a
        # confirmation, Approve / Cancel take its place (_show_confirm).
        fy = H - 54
        field = self._round_rect(16, fy, W - 58, fy + 38, 19, outline=EDGE, fill=FIELD)
        self.entry = self.tkm.Entry(self.root, bg=FIELD, fg=FAINT, insertbackground=TEXT,
                                    relief="flat", bd=0, highlightthickness=0,
                                    font=self.f_entry)
        self.entry.insert(0, "Ask {}…".format(identity.NAME))
        self._placeholder = True
        entry_item = cv.create_window(34, fy + 19, window=self.entry, anchor="w",
                                      width=W - 106, height=24)
        self.entry.bind("<FocusIn>", self._entry_focus)
        self.entry.bind("<FocusOut>", self._entry_blur)
        # A borderless window doesn't take keyboard focus on its own.
        self.entry.bind("<Button-1>", lambda e: (self.root.focus_force(), self.entry.focus_set()))
        self.entry.bind("<Return>", lambda e: self._submit())
        self.entry.bind("<Escape>", lambda e: self.canvas.focus_set())

        sx, sy = W - 37, fy + 19
        self.send_bg = cv.create_oval(sx - 17, sy - 17, sx + 17, sy + 17, fill=ACCENT, outline="")
        self.send_fg = cv.create_text(sx, sy - 1, text="↑", fill=ON_ACCENT,
                                      font=(self.f_mark[0], 12, "bold"))
        for item in (self.send_bg, self.send_fg):
            cv.tag_bind(item, "<Button-1>", lambda e: self._submit())
            cv.tag_bind(item, "<Enter>", lambda e: cv.itemconfig(self.send_bg, fill="#99f6e4"))
            cv.tag_bind(item, "<Leave>", lambda e: cv.itemconfig(self.send_bg, fill=ACCENT))
        self.field_items = [field, entry_item, self.send_bg, self.send_fg]

        self._pill(16, fy, (W - 44) // 2, 38, "Approve", ACCENT, ON_ACCENT, "yes")
        self._pill(W // 2 + 6, fy, (W - 44) // 2, 38, "Cancel", FIELD, TEXT, "no", outline=EDGE)
        self._show_confirm(False)

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
                  x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.canvas.create_polygon(points, smooth=True, splinesteps=24, **kw)

    def _pill(self, x, y, w, h, label, fill, fg, answer, outline=""):
        cv = self.canvas
        bg = self._round_rect(x, y, x + w, y + h, h // 2, fill=fill, outline=outline or fill)
        tx = cv.create_text(x + w // 2, y + h // 2, text=label, fill=fg, font=self.f_button)
        for item in (bg, tx):
            cv.tag_bind(item, "<Button-1>", lambda e: console_input.submit(answer))
        self.confirm_items += [bg, tx]

    def _show_confirm(self, show):
        for item in self.confirm_items:
            self.canvas.itemconfig(item, state="normal" if show else "hidden")
        for item in getattr(self, "field_items", []):
            self.canvas.itemconfig(item, state="hidden" if show else "normal")

    # -- window chrome -------------------------------------------------------
    def _round_corners(self):
        """Windows 11: native rounded corners for the borderless window."""
        if os.name != "nt":
            return
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            preference = ctypes.c_int(2)  # DWMWCP_ROUND
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 33, ctypes.byref(preference), ctypes.sizeof(preference))
        except Exception:
            pass  # Windows 10: square corners, still fine

    def _fade_in(self, alpha):
        try:
            self.root.attributes("-alpha", alpha)
        except Exception:
            return
        if alpha < 0.97:
            self.root.after(16, self._fade_in, min(0.97, alpha + 0.12))

    def _transparent(self, on):
        """Compact: the key colour becomes see-through (Windows; elsewhere the
        orb sits on a small dark square). Expanded: back to the solid card."""
        bg = KEY if on else CARD
        self.root.configure(bg=bg)
        self.canvas.config(bg=bg)
        try:
            self.root.attributes("-transparentcolor", KEY if on else "")
        except Exception:
            pass

    def _place(self):
        """Puts the window around the remembered orb position."""
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        if self.orb_pos is None:
            try:
                pos = json.loads(_position_file().read_text(encoding="utf-8"))
                if "orb_x" in pos:
                    self.orb_pos = (int(pos["orb_x"]), int(pos["orb_y"]))
                else:  # saved by the previous version: the card's corner
                    self.orb_pos = (int(pos["x"]) + W // 2, int(pos["y"]) + ORB_Y)
            except Exception:
                self.orb_pos = (sw - 60, sh - 120)
        ox, oy = self.orb_pos
        if self.compact:
            w, h, x, y = MINI, MINI, ox - MINI // 2, oy - MINI // 2
        else:
            w, h, x, y = W, H, ox - W // 2, oy - ORB_Y
        # Keep the whole card on screen even when the orb lives in a corner.
        x = min(max(0, x), max(0, sw - w))
        y = min(max(0, y), max(0, sh - h))
        self.root.geometry("{}x{}+{}+{}".format(w, h, x, y))

    def _remember_position(self):
        self._save_pending = None
        try:
            path = _position_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"orb_x": self.orb_pos[0], "orb_y": self.orb_pos[1]}),
                            encoding="utf-8")
        except Exception:
            pass

    def _maybe_drag_start(self, event):
        # Only the header drags the expanded card; the rest is for reading.
        self._drag = None
        if event.y <= 42 and event.x < W - 44:
            self._drag_start(event)

    def _drag_start(self, event):
        self._drag = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def _drag_move(self, event):
        if not self._drag:
            return
        x, y = event.x_root - self._drag[0], event.y_root - self._drag[1]
        self.root.geometry("+{}+{}".format(x, y))
        self.orb_pos = ((x + MINI // 2, y + MINI // 2) if self.compact
                        else (x + W // 2, y + ORB_Y))
        if self._save_pending:
            self.root.after_cancel(self._save_pending)
        self._save_pending = self.root.after(800, self._remember_position)

    def expand(self, pin=False, auto=None):
        """Open the card around the orb. pin: stays until collapse(); auto:
        why it opened by itself ("confirm" or "peek")."""
        if pin:
            self.pinned, self.auto = True, None
        elif auto and not self.pinned:
            self.auto = auto
        if self.compact:
            self.compact = False
            self._build()
            self._place()

    def collapse(self):
        """Back to the floating orb, wherever the card was opened from."""
        self.pinned, self.auto = False, None
        if not self.compact:
            if self.entry is not None:
                self.root.focus_set()
            self.compact = True
            self._build()
            self._place()

    def _in_use(self):
        """Hovering over the card, or typing in it -- don't shrink on them."""
        if self.hover:
            return True
        try:
            return (self.entry is not None and not self._placeholder
                    and bool(self.entry.get().strip()))
        except Exception:
            return False

    def _decide_size(self):
        """Requests from Jarvis, then the automatic open/close rules."""
        with _requests_lock:
            pending = list(_requests)
            del _requests[:]
        now = time.time()
        for kind, arg in pending:
            if kind == "show":
                self.expand(pin=True)
            elif kind == "hide":
                self.collapse()
            elif kind == "peek":
                if self.compact or self.auto == "peek":
                    self.expand(auto="peek")
                    self.auto_until = max(self.auto_until, now + float(arg or PEEK_SECONDS))

        from commands import confirm
        asking = confirm.is_pending()
        if asking and not self._asking:
            self.expand(auto="confirm")
        elif not asking and self._asking and self.auto == "confirm":
            # Decided: shrink back shortly, once the result has been seen.
            self.auto, self.auto_until = "peek", now + 1.5
        self._asking = asking

        if self.auto == "peek" and now > self.auto_until and not self._in_use():
            self.collapse()
        return asking

    def _menu(self, event):
        menu = self.tkm.Menu(self.root, tearoff=0, bg=FIELD, fg=TEXT, bd=0,
                             activebackground=EDGE, activeforeground=TEXT, relief="flat")
        menu.add_command(label="Expand" if self.compact else "Minimize",
                         command=(lambda: self.expand(pin=True)) if self.compact
                         else self.collapse)
        menu.add_command(label="Keep on top (toggle)", command=lambda: self.root.attributes(
            "-topmost", not self.root.attributes("-topmost")))
        menu.add_separator()
        menu.add_command(label="Quit {}".format(identity.NAME), command=self._quit)
        menu.tk_popup(event.x_root, event.y_root)

    def _quit(self):
        import _thread as threads
        threads.interrupt_main()  # same as Ctrl+C in the terminal

    # -- input -----------------------------------------------------------------
    def _entry_focus(self, _event=None):
        if self._placeholder:
            self.entry.delete(0, "end")
            self.entry.config(fg=TEXT)
            self._placeholder = False

    def _entry_blur(self, _event=None):
        if not self.entry.get():
            self.entry.insert(0, "Ask {}…".format(identity.NAME))
            self.entry.config(fg=FAINT)
            self._placeholder = True

    def _submit(self):
        if self.entry is None or self._placeholder:
            return
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, "end")
            console_input.submit(text)

    # -- frame -----------------------------------------------------------------
    def tick(self):
        if _busy.is_set():
            if not self.hidden_for_agent:
                self.hidden_for_agent = True
                self.root.withdraw()
            self.root.after(250, self.tick)
            return
        if self.hidden_for_agent:
            self.hidden_for_agent = False
            self.root.deiconify()

        asking = self._decide_size()
        snap = ui.snapshot()
        state = snap["state"]
        colour, label = _STATES.get(state, _STATES[ui.IDLE])
        now = time.time() - self.t0

        self.level = max(ui.take_window_level(), self.level * 0.78)
        active = state in (ui.RECORDING, ui.THINKING, ui.SPEAKING) or self.level > 0.02
        shown = self.level if active else 0.06 + 0.06 * math.sin(now * 2 * math.pi / 3.6)

        cv = self.canvas
        if self.compact:
            image = self.orbs.get(colour, self.orb_radius, shown, glow=False, bg=KEY)
        else:
            image = self.orbs.get(colour, self.orb_radius, shown)
        cv.itemconfig(self.orb, image=image)

        if state == ui.THINKING:
            cx, cy = self.orb_xy
            ring = self.orb_radius * 1.42
            for i, dot in enumerate(self.dots):
                a = now * 3.4 - i * 0.42          # a short comet: head and tail
                px, py = cx + ring * math.cos(a), cy + ring * math.sin(a)
                s = 3.4 - 0.9 * i
                cv.coords(dot, px - s, py - s, px + s, py + s)
                cv.itemconfig(dot, fill=colour)
        else:
            for dot in self.dots:
                cv.itemconfig(dot, fill="")

        if not self.compact:
            key = (state, snap["heard"], snap["reply"], asking)
            if key != self._last_text:
                self._last_text = key
                cv.itemconfig(self.state_text, text=_spaced(label))
                cv.itemconfig(self.heard_text, text=_short(snap["heard"], HEARD_CHARS))
                if snap["reply"]:
                    cv.itemconfig(self.reply_text, text=_short(snap["reply"], REPLY_CHARS),
                                  fill=TEXT)
                else:
                    cv.itemconfig(self.reply_text, fill=FAINT, text="" if snap["heard"] else
                                  "Say “{}”, or type below.".format(identity.wake_phrase()))
                # The reply moves up when there's no "you said" line above it.
                cv.coords(self.reply_text, 24, 206 if snap["heard"] else 186)
                self._show_confirm(asking)

        self.root.after(int(1000 / (ACTIVE_FPS if active else IDLE_FPS)), self.tick)


def _run():
    global _started
    try:
        import tkinter as tkm
        root = tkm.Tk()
        global _instance
        window = _instance = _Window(root)
        _started = True
        _ready.set()
        window.tick()
        root.mainloop()
    except Exception as e:
        print("window: couldn't open ({})".format(e))
    finally:
        _started = False
        _ready.set()
