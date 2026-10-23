"""Jarvis's desktop window: a minimal, premium floating card.

Goes in: listener/window_ui.py

    expanded   wordmark header, a glowing orb, what you said, the reply, an
               input field -- and Approve / Cancel when Jarvis asks
    compact    just the orb, floating; double-click it (or "-") to switch

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

from listener import console_input
from listener import ui

W, H = 340, 330            # expanded card
MINI = 84                  # compact orb tile
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

    def get(self, colour, radius, level):
        step = max(0, min(LEVEL_STEPS, int(round(level * LEVEL_STEPS))))
        key = (colour, radius, step)
        image = self._cache.get(key)
        if image is None:
            image = self._cache[key] = self._render(colour, radius, step / float(LEVEL_STEPS))
        return image

    @staticmethod
    def _render(colour, base, level):
        import numpy as np
        from PIL import Image, ImageTk

        r = base * (1.0 + 0.16 * level)              # sphere radius
        g = r + base * (0.45 + 0.75 * level)         # glow reach
        size = int(math.ceil(2 * base * 2.36)) + 4   # room for the biggest glow
        c = (size - 1) / 2.0
        y, x = np.mgrid[0:size, 0:size].astype(np.float32)
        d = np.hypot(x - c, y - c)

        card = np.array(_rgb(CARD), np.float32)
        tint = np.array(_rgb(colour), np.float32)
        out = np.broadcast_to(card, (size, size, 3)).copy()

        # Glow: soft falloff outside the sphere.
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
        root.title("Jarvis")
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
        self.compact = False
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
            self.root.geometry("{}x{}".format(MINI, MINI))
            cv.config(width=MINI, height=MINI)
            self._round_rect(1, 1, MINI - 2, MINI - 2, 10, outline=EDGE, fill="")
            self.orb_xy = (MINI // 2, MINI // 2)
            self.orb_radius = 18
            self.orb = cv.create_image(*self.orb_xy)
            self.dots = [cv.create_oval(0, 0, 0, 0, outline="", fill="") for _ in range(3)]
            cv.bind("<ButtonPress-1>", self._drag_start)
            cv.bind("<B1-Motion>", self._drag_move)
            cv.bind("<Double-Button-1>", lambda e: self.toggle_compact())
            return

        self.root.geometry("{}x{}".format(W, H))
        cv.config(width=W, height=H)
        cv.bind("<ButtonPress-1>", self._maybe_drag_start)
        cv.bind("<B1-Motion>", self._drag_move)
        cv.unbind("<Double-Button-1>")

        # Hairline frame (Windows 11 rounds the window itself to match).
        self._round_rect(1, 1, W - 2, H - 2, 10, outline=EDGE, fill="")

        # Header: wordmark, and "-" to shrink to the orb.
        cv.create_text(22, 22, text=_spaced("Jarvis"), anchor="w", fill=SOFT, font=self.f_mark)
        self.btn_min = cv.create_text(W - 24, 20, text="–", fill=FAINT,
                                      font=(self.f_mark[0], 13))
        cv.tag_bind(self.btn_min, "<Enter>", lambda e: cv.itemconfig(self.btn_min, fill=TEXT))
        cv.tag_bind(self.btn_min, "<Leave>", lambda e: cv.itemconfig(self.btn_min, fill=FAINT))
        cv.tag_bind(self.btn_min, "<Button-1>", lambda e: self.toggle_compact())

        # Orb and state.
        self.orb_xy = (W // 2, ORB_Y)
        self.orb_radius = 30
        self.orb = cv.create_image(*self.orb_xy)
        cv.tag_bind(self.orb, "<Double-Button-1>", lambda e: self.toggle_compact())
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
        self.entry.insert(0, "Ask Jarvis…")
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

    def _place(self):
        try:
            pos = json.loads(_position_file().read_text(encoding="utf-8"))
            x, y = int(pos["x"]), int(pos["y"])
        except Exception:
            x = self.root.winfo_screenwidth() - W - 24
            y = self.root.winfo_screenheight() - H - 72
        x = min(max(0, x), max(0, self.root.winfo_screenwidth() - 60))
        y = min(max(0, y), max(0, self.root.winfo_screenheight() - 60))
        self.root.geometry("+{}+{}".format(x, y))

    def _remember_position(self):
        self._save_pending = None
        if self.compact:
            return  # remember where the full card lives
        try:
            path = _position_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"x": self.root.winfo_x(), "y": self.root.winfo_y()}),
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
        self.root.geometry("+{}+{}".format(event.x_root - self._drag[0],
                                           event.y_root - self._drag[1]))
        if self._save_pending:
            self.root.after_cancel(self._save_pending)
        self._save_pending = self.root.after(800, self._remember_position)

    def toggle_compact(self):
        # Keep the orb in the same place on screen when switching sizes.
        x, y = self.root.winfo_x(), self.root.winfo_y()
        dx, dy = W // 2 - MINI // 2, ORB_Y - MINI // 2
        nx, ny = (x - dx, y - dy) if self.compact else (x + dx, y + dy)
        self.compact = not self.compact
        self._build()
        self.root.geometry("+{}+{}".format(max(0, nx), max(0, ny)))

    def _menu(self, event):
        menu = self.tkm.Menu(self.root, tearoff=0, bg=FIELD, fg=TEXT, bd=0,
                             activebackground=EDGE, activeforeground=TEXT, relief="flat")
        menu.add_command(label="Expand" if self.compact else "Compact",
                         command=self.toggle_compact)
        menu.add_command(label="Keep on top (toggle)", command=lambda: self.root.attributes(
            "-topmost", not self.root.attributes("-topmost")))
        menu.add_separator()
        menu.add_command(label="Quit Jarvis", command=self._quit)
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
            self.entry.insert(0, "Ask Jarvis…")
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

        snap = ui.snapshot()
        state = snap["state"]
        colour, label = _STATES.get(state, _STATES[ui.IDLE])
        now = time.time() - self.t0

        self.level = max(ui.take_window_level(), self.level * 0.78)
        active = state in (ui.RECORDING, ui.THINKING, ui.SPEAKING) or self.level > 0.02
        shown = self.level if active else 0.06 + 0.06 * math.sin(now * 2 * math.pi / 3.6)

        cv = self.canvas
        cv.itemconfig(self.orb, image=self.orbs.get(colour, self.orb_radius, shown))

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
            from commands import confirm
            asking = confirm.is_pending()
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
                                  "Say “Hey Jarvis”, or type below.")
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
