"""A small desktop window for Jarvis: status orb, what you said, the reply,
an input box, and Yes/No buttons when Jarvis asks.

Goes in: listener/window_ui.py   (new file)

LIGHT ON PURPOSE
----------------
Tkinter ships with Python -- nothing to install -- and this uses only a canvas
and a few labels: no images, no web engine. About 10-25 MB of RAM. It keeps no
state of its own: every frame it reads listener/ui.py's snapshot(), so the
terminal panel and this window can never disagree. It redraws ACTIVE_FPS times
a second while recording, thinking or speaking, and IDLE_FPS while waiting for
the wake word, so an idle Jarvis costs almost no CPU.

THREADS
-------
Tk runs on its own daemon thread and every Tk call happens on it. The rest of
Jarvis never touches Tk: it writes state into ui.py (as before), and text from
the input box goes into console_input's queue (console_input.submit), so
main.py handles it exactly like typing -- including as a physical answer to a
confirmation, which a click or a typed line is.

SAFETY
------
The window steps aside while the computer agent works (set_agent_busy): an
always-on-top window could otherwise sit over the spot the agent is about to
click, or end up in its screenshots. The agent also refuses to act in any
window belonging to Jarvis itself (commands/agent.py).

    JARVIS_NO_WINDOW=1   don't open it
"""

import json
import os
import threading
from pathlib import Path

from listener import console_input
from listener import ui

REPLY_CHARS = 170
ACTIVE_FPS = 20
IDLE_FPS = 4
WIDTH, HEIGHT = 380, 300

# Colours per state: (orb, glow, label)
_THEME = {
    ui.IDLE: ("#5b6373", "#2a2f3a", "Idle"),
    ui.LISTENING: ("#2dd4bf", "#123d3a", "Listening for “Hey Jarvis”"),
    ui.RECORDING: ("#f43f5e", "#4a1520", "Listening..."),
    ui.THINKING: ("#f59e0b", "#4a3410", "Thinking..."),
    ui.SPEAKING: ("#60a5fa", "#172a4a", "Speaking"),
}
BG, PANEL, TEXT, DIM, ACCENT = "#0f1218", "#171b23", "#e6e9ef", "#8a93a6", "#2dd4bf"

_thread = None
_ready = threading.Event()
_busy = threading.Event()
_started = False


def _short(text, limit):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _position_file():
    return Path(__file__).resolve().parent.parent / "data" / "window.json"


def enabled():
    return not os.environ.get("JARVIS_NO_WINDOW")


def start():
    """Opens the window on its own thread. True if it's up."""
    global _thread
    if not enabled() or _started:
        return _started
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("window: tkinter isn't available, staying in the terminal")
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


class _Window(object):
    def __init__(self, root):
        import tkinter as tkm
        self.root = root
        root.title("Jarvis")
        root.configure(bg=BG)
        root.attributes("-topmost", True)
        root.resizable(False, False)
        # Closing minimises; Jarvis keeps running (Quit is on the right-click menu).
        root.protocol("WM_DELETE_WINDOW", root.iconify)
        self._place()

        font = ("Segoe UI", 10)
        self.canvas = tkm.Canvas(root, width=WIDTH, height=110, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="x")
        cx, cy = WIDTH // 2, 55
        self.glow = self.canvas.create_oval(cx - 46, cy - 46, cx + 46, cy + 46, fill="", outline="")
        self.orb = self.canvas.create_oval(cx - 30, cy - 30, cx + 30, cy + 30,
                                           fill=_THEME[ui.IDLE][0], outline="")
        self.arc = self.canvas.create_arc(cx - 38, cy - 38, cx + 38, cy + 38, start=0, extent=0,
                                          style="arc", outline=ACCENT, width=3)
        self.state_label = tkm.Label(root, text="", bg=BG, fg=DIM, font=("Segoe UI", 9))
        self.state_label.pack()

        box = tkm.Frame(root, bg=PANEL, padx=10, pady=8)
        self.heard = tkm.Label(box, text="", bg=PANEL, fg=DIM, font=font, anchor="w",
                               justify="left", wraplength=WIDTH - 40)
        self.heard.pack(fill="x", anchor="n")
        self.reply = tkm.Label(box, text="", bg=PANEL, fg=TEXT, font=font, anchor="w",
                               justify="left", wraplength=WIDTH - 40)
        self.reply.pack(fill="x", anchor="n", pady=(4, 0))

        self.confirm_bar = tkm.Frame(root, bg=BG)
        btn = dict(font=("Segoe UI", 9, "bold"), relief="flat", bd=0, padx=14, pady=3,
                   cursor="hand2")
        tkm.Button(self.confirm_bar, text="Yes", bg=ACCENT, fg="#06201d", activebackground=ACCENT,
                   command=lambda: self._send("yes"), **btn).pack(side="left", padx=4)
        tkm.Button(self.confirm_bar, text="No", bg="#3a4150", fg=TEXT, activebackground="#3a4150",
                   command=lambda: self._send("no"), **btn).pack(side="left", padx=4)

        # Packed bottom-up before the text box, so the input row always keeps
        # its place and a long reply is what gets squeezed, never the box.
        self.input_row = tkm.Frame(root, bg=BG)
        self.input_row.pack(side="bottom", fill="x", padx=10, pady=(2, 10))
        self.entry = tkm.Entry(self.input_row, bg=PANEL, fg=TEXT, insertbackground=TEXT,
                               relief="flat", font=font, highlightthickness=1,
                               highlightbackground="#2a2f3a", highlightcolor=ACCENT)
        self.entry.pack(side="left", fill="x", expand=True, ipady=5)
        self.entry.bind("<Return>", lambda e: self._submit())
        tkm.Button(self.input_row, text="Send", bg=PANEL, fg=ACCENT, relief="flat", bd=0,
                   padx=10, activebackground=PANEL, command=self._submit, cursor="hand2",
                   highlightthickness=0, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(6, 0))

        box.pack(fill="both", expand=True, padx=10, pady=(6, 4))

        self.menu = tkm.Menu(root, tearoff=0)
        self.menu.add_command(label="Keep on top (toggle)", command=self._toggle_top)
        self.menu.add_command(label="Quit Jarvis", command=self._quit)
        root.bind("<Button-3>", lambda e: self.menu.tk_popup(e.x_root, e.y_root))
        root.bind("<Configure>", self._moved)

        self.level = 0.0
        self.spin = 0
        self.hidden_for_agent = False
        self._last = None
        self._save_pending = None

    # -- position ----------------------------------------------------------
    def _place(self):
        try:
            pos = json.loads(_position_file().read_text(encoding="utf-8"))
            x, y = int(pos["x"]), int(pos["y"])
        except Exception:
            # Bottom-right, above the taskbar.
            x = self.root.winfo_screenwidth() - WIDTH - 24
            y = self.root.winfo_screenheight() - HEIGHT - 90
        self.root.geometry("{}x{}+{}+{}".format(WIDTH, HEIGHT, max(0, x), max(0, y)))

    def _moved(self, event):
        # <Configure> fires for every widget and every pixel of a drag: save
        # once, a second after the window stops moving.
        if event.widget is not self.root:
            return
        if self._save_pending:
            self.root.after_cancel(self._save_pending)
        self._save_pending = self.root.after(1000, self._remember_position)

    def _remember_position(self):
        self._save_pending = None
        try:
            if self.root.state() != "normal":
                return
            path = _position_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"x": self.root.winfo_x(), "y": self.root.winfo_y()}),
                            encoding="utf-8")
        except Exception:
            pass

    # -- input -------------------------------------------------------------
    def _submit(self):
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, "end")
            self._send(text)

    def _send(self, text):
        console_input.submit(text)

    def _toggle_top(self):
        self.root.attributes("-topmost", not self.root.attributes("-topmost"))

    def _quit(self):
        import _thread as threads
        threads.interrupt_main()  # same as Ctrl+C in the terminal

    # -- frame -------------------------------------------------------------
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
        orb, glow, label = _THEME.get(state, _THEME[ui.IDLE])

        self.level = max(ui.take_window_level(), self.level * 0.75)
        active = state in (ui.RECORDING, ui.THINKING, ui.SPEAKING) or self.level > 0.02

        cx, cy = WIDTH // 2, 55
        r = 30 + 10 * self.level
        g = r + 8 + 16 * self.level
        self.canvas.coords(self.orb, cx - r, cy - r, cx + r, cy + r)
        self.canvas.coords(self.glow, cx - g, cy - g, cx + g, cy + g)
        self.canvas.itemconfig(self.orb, fill=orb)
        self.canvas.itemconfig(self.glow, fill=glow)
        if state == ui.THINKING:
            self.spin = (self.spin + 18) % 360
            self.canvas.itemconfig(self.arc, start=self.spin, extent=100, outline=orb)
        else:
            self.canvas.itemconfig(self.arc, extent=0)

        # Text only changes between turns: skip the relayout when nothing did.
        from commands import confirm
        asking = confirm.is_pending()
        key = (state, snap["heard"], snap["reply"], snap["timing"], asking)
        if key != self._last:
            self._last = key
            timing = snap["timing"]
            self.state_label.config(text=label + ("   ·   " + timing if timing else ""))
            # The window is a fixed size: long text is shortened here and read
            # in full in the terminal panel and log.txt.
            self.heard.config(text=("You: " + _short(snap["heard"], 90)) if snap["heard"] else "")
            self.reply.config(text=_short(snap["reply"], REPLY_CHARS)
                              or "Say “Hey Jarvis” or type below.")
            if asking:
                self.confirm_bar.pack(side="bottom", after=self.input_row, pady=(0, 4))
            else:
                self.confirm_bar.pack_forget()

        self.root.after(int(1000 / (ACTIVE_FPS if active else IDLE_FPS)), self.tick)


def _run():
    global _started
    try:
        import tkinter as tkm
        root = tkm.Tk()
        window = _Window(root)
        _started = True
        _ready.set()
        window.tick()
        root.mainloop()
    except Exception as e:
        print("window: couldn't open ({})".format(e))
    finally:
        _started = False
        _ready.set()
