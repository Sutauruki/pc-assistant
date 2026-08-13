"""Live terminal dashboard with an audio-synced waveform and a typed prompt.

Goes in: listener/ui.py   (replaces existing)

Draws a fixed panel at the bottom of the terminal -- state, waveform, what was
heard, what Jarvis said, and a prompt line you can type into. Log lines no
longer scroll above this panel (which used to shove it down the screen and
shred its borders); instead they are buffered into a history and shown as a
dim toast for a few seconds before the panel settles back to just the
conversation. The whole history is available on demand with Ctrl+L (see
toggle_history(), and Ctrl+L/arrows in listener/console_input.py).

The waveform is driven from two places:

  * listener/record.py pushes mic levels while you're speaking.
  * listener/audio_out.py pushes levels from the playback thread, one per
    slice it hands to the sound device.

Keeping the second one in sync takes two things. First, the levels are
computed at *write* time rather than when chunks arrive off the network: that
thread blocks on stream.write() once the device buffer is full, so it
self-throttles to real time instead of racing ahead with the download. Second,
those levels are pushed with a `delay` equal to the device latency, because
handing a slice to the sound card still happens some tens of milliseconds
before you hear it -- without that the bars visibly lead the voice.

The corollary, learned the hard way: push_level() runs on the thread feeding
the sound card, so it cannot be allowed to block. It takes its own small lock
rather than the render lock, which is held across the console write. See the
note in Dashboard.__init__.

THE PROMPT ROW(S)
------------------
set_input() draws whatever you are currently typing *inside* the panel, and
listener/console_input.py reads keystrokes without the console echoing them.
Shift+Enter (Windows only -- see console_input.py) can put more than one
line in there; _input_rows() draws one panel row per line, which the
existing variable-height redraw in _draw_locked() already handles without
changes -- it was never assuming a fixed row count to begin with.
Letting the terminal echo normally does not work here: the cursor parks on the
row below the panel, so a pressed Enter scrolls the screen by one line while
the panel's own `\x1b[{n}A` arithmetic still thinks it is n rows up. The panel
then walks down the screen one row per submitted line, leaving a trail of dead
borders. Owning the echo removes the whole class of problem -- nothing but
this module ever moves the cursor.

No new dependencies: ANSI escapes and a render thread. If stdout isn't a
terminal (piped to a file, running under a dumb console) the whole thing
switches itself off and log lines print normally. Set JARVIS_NO_UI=1 to force
that.
"""

import ctypes
import math
import os
import shutil
import sys
import threading
import time
from collections import deque

# --- States ----------------------------------------------------------------

IDLE = "IDLE"
LISTENING = "LISTENING"
RECORDING = "RECORDING"
THINKING = "THINKING"
SPEAKING = "SPEAKING"

# Label plus the accent colour the whole panel takes on in that state.
# Colours ported straight from the HTML mockup's `params` table (idle,
# listening, thinking, speaking) so the reskinned panel and the mockup
# read as the same design. RECORDING has no equivalent there -- it's this
# project's own state for "mic is actively capturing," distinct from
# LISTENING's "waiting for the wake word" -- so it keeps its own colour,
# an amber that sits clearly apart from the other four.
_STATE_STYLE = {
    IDLE:      ("IDLE",      (86, 94, 104)),
    LISTENING: ("LISTENING", (93, 202, 165)),
    RECORDING: ("RECORDING", (250, 170, 65)),
    THINKING:  ("THINKING",  (175, 157, 236)),
    SPEAKING:  ("SPEAKING",  (63, 212, 160)),
}

# --- Tunables --------------------------------------------------------------

FPS = 24
MAX_WIDTH = 84

# Ring size, in terminal columns. Rows are derived from this (see
# _ring_rows_count) so the shape stays round rather than egg-shaped --
# don't set rows independently.
RING_MIN_COLS = 8
RING_MAX_COLS = 16

# How fast the ring's audio-reactive swell falls back to its state
# baseline when no fresh level arrives. Same role DECAY played for the old
# bar wave.
DECAY = 0.72

# Height of the idle swell drawn while waiting on the model, on top of
# THINKING's own baseline ring amplitude. Low enough that it never reads
# as real audio.
THINK_AMPLITUDE = 0.22

# Ceiling on the playback delay line. A device reporting an absurd latency
# shouldn't be able to park levels a minute into the future.
MAX_DELAY_FRAMES = FPS  # one second

# Caret blink period, in frames per phase. Only blinks while there is text in
# the buffer -- see the note in _active_input_row.
CARET_FRAMES = FPS // 2

# RMS -> 0..1 mapping, in dBFS. Speech sits around -30dB, so a linear
# amplitude scale would keep the bars in the bottom fifth of the panel;
# the log mapping is what makes normal speech fill the height.
FLOOR_DB = -52.0
CEIL_DB = -8.0

BORDER_RGB = (70, 78, 92)
TEXT_RGB = (176, 184, 198)
DIM_RGB = (108, 116, 132)

# Info column values, ported from the mockup's "5dcaa5" accent -- the same
# green the STT/TTS/latency/sparkline/turns rows use in the HTML.
STATS_RGB = (93, 202, 165)

# Number of recent audio levels (in frames) the info column's sparkline
# shows at the ring's right. Same idea as the mockup's fixed "▁▂▁▃▂" row but
# live -- it scrolls with whatever the mic/speaker actually produced.
SPARK_WIDTH = 24

# How long the most recent log line stays visible as a dim toast inside the
# panel before it settles back to just the conversation. The panel itself
# never moves -- this is purely an overlay that clears on its own.
TOAST_SECONDS = 4.0

# Cap on buffered log lines kept for the history overlay (ctrl+L).
HISTORY_MAX = 200

_SPINNER = "◐◓◑◒"
_SPINNER_ASCII = "|/-\\"

# Half-block ring rendering. Two vertical "pixels" pack into one terminal
# row -- "top"/"bottom" each colour one half of the cell, "full" is both
# halves lit. See Dashboard._ring_rows()'s docstring for why this is what
# makes the ring round instead of egg-shaped.
_RING_UNICODE = {"top": "▀", "bottom": "▄", "full": "█"}
_RING_ASCII = {"top": "\"", "bottom": ",", "full": "#"}

# Sparkline glyphs for the info column, with an ASCII fallback for consoles
# that can't render the unicode half-blocks (paired with _RING_UNICODE via
# the same capability check in start()).
_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
_SPARK_ASCII = ".+*#@%"

_BOX_UNICODE = {
    "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
    "h": "─", "v": "│", "ml": "├", "mr": "┤",
    "dot": "●", "prompt": "❯", "reply": "◆", "sep": "·",
    "typed": "›", "caret": "▌",
}
_BOX_ASCII = {
    "tl": "+", "tr": "+", "bl": "+", "br": "+",
    "h": "-", "v": "|", "ml": "+", "mr": "+",
    "dot": "*", "prompt": ">", "reply": "*", "sep": "-",
    "typed": ">", "caret": "_",
}

# Four-stop colour cycle the ring sweeps through as it goes around --
# lifted directly from the HTML mockup's grad() function (teal -> blue ->
# deep blue -> violet).
_RING_GRADIENT = ((63, 212, 160), (42, 181, 208), (28, 157, 234), (127, 119, 221))

# Per-state ring motion, ported from the mockup's `params` table:
#   speed -- how fast the wobble travels around the ring
#   amp   -- baseline wobble depth (fraction of ring radius) with no audio
#   buzz  -- fast, small-amplitude jitter layered on top
#   reactivity -- how much extra `amp` the *real* current audio level
#                 (self._last_level) contributes on top of the baseline.
#                 This is the one thing with no equivalent in the mockup,
#                 which only ever animates against a canned sine -- here
#                 the ring genuinely swells with real mic/speaker level.
_RING_PARAMS = {
    IDLE:      dict(speed=0.2, amp=0.05, buzz=0.015, reactivity=0.10),
    LISTENING: dict(speed=0.6, amp=0.09, buzz=0.04,  reactivity=0.35),
    RECORDING: dict(speed=0.9, amp=0.13, buzz=0.05,  reactivity=0.55),
    THINKING:  dict(speed=1.1, amp=0.17, buzz=0.05,  reactivity=0.10),
    SPEAKING:  dict(speed=1.8, amp=0.20, buzz=0.12,  reactivity=0.60),
}


# --- Terminal capability ---------------------------------------------------

def _enable_windows_ansi():
    """Turns on VT processing for this console. Win10+ supports the escapes
    but doesn't honour them until asked."""
    if os.name != "nt":
        return True
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _can_encode(sample):
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        sample.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def level_from_rms(rms):
    """int16 RMS -> 0..1, log-scaled so speech uses the full panel height."""
    amp = max(float(rms), 1.0) / 32768.0
    db = 20.0 * math.log10(amp)
    return max(0.0, min(1.0, (db - FLOOR_DB) / (CEIL_DB - FLOOR_DB)))


def _fg(rgb):
    return "\x1b[38;2;{};{};{}m".format(*rgb)


def _mix(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


class _StreamProxy(object):
    """Routes stray print() calls through the dashboard.

    Every module in the project already prints -- executor announces what it
    did, tts explains failures. Rather than rewrite each call site, we take
    over sys.stdout so those lines land above the panel instead of tearing
    through the middle of it.

    stderr gets the same treatment, and it matters more: a traceback is a
    dozen lines written by whichever thread just died, straight into a
    terminal the render thread is repainting 24 times a second. The result
    is shredded -- half the traceback inside the panel, the panel's own
    borders scattered through the traceback.
    """

    def __init__(self, dashboard, original):
        self._dashboard = dashboard
        self._original = original
        self._partial = ""

    def write(self, text):
        if not text:
            return 0
        # print() arrives as two writes (the text, then the newline) and other
        # threads print too, so the partial-line buffer needs the same lock the
        # renderer uses or two threads can splice each other's lines together.
        with self._dashboard._lock:
            self._partial += text
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                self._dashboard.log(line)
        return len(text)

    def writelines(self, lines):
        # Not free via __getattr__: the original's writelines would go
        # straight to the terminal, past the panel.
        for line in lines:
            self.write(line)

    def flush(self):
        self._original.flush()

    def isatty(self):
        return self._original.isatty()

    def __getattr__(self, name):
        # Guard against recursing on _original itself, which happens if
        # anything touches an attribute before __init__ has bound it.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._original, name)


class Dashboard(object):
    def __init__(self):
        self._lock = threading.RLock()

        # Levels get their own lock, deliberately.
        #
        # _lock is held by the render thread for the whole of a frame --
        # including the write to the console, which on Windows is a slow,
        # synchronous few milliseconds. The playback thread pushes a level
        # just before each write to the sound card, so if those shared a lock
        # the audio would queue behind a terminal repaint tens of times a
        # second and you'd hear the reply in pieces. That is exactly what was
        # happening. This one is only ever held for a handful of instructions
        # and never across I/O.
        #
        # Ordering, where both are taken: _lock first, then _level_lock.
        # push_level takes _level_lock alone, so there's no cycle.
        self._level_lock = threading.Lock()

        self._stop = threading.Event()
        self._thread = None
        self._active = False

        self._state = IDLE
        self._pending = None
        self._delayed = []
        self._last_level = 0.0
        self._spin = 0
        self._tick = 0

        self._heard = ""
        self._reply = ""
        self._timing = ""

        # Info column (the mockup's rows to the right of the ring).
        self._stt = "gemini"
        self._tts = "groq · cloud"
        self._latency = "1.4s"
        self._turns = 0
        self._t0 = None
        self._spark = deque(maxlen=SPARK_WIDTH)

        # Log handling. Log lines no longer print above the panel and shove
        # it down the screen; they are buffered into _history and shown as a
        # transient toast that clears on its own, with a full-history overlay
        # available on demand (see toggle_history / history_scroll).
        self._history = deque(maxlen=HISTORY_MAX)
        self._toast = None
        self._toast_until = 0
        self._history_on = False
        self._history_scroll = 0

        # Prompt row. Off until console_input says it has a reader running --
        # showing a prompt you cannot type into would be a lie.
        self._input = ""
        self._input_enabled = False
        self._input_hint = "type a command"

        self._lines_drawn = 0
        self._last_frame = None

        # Fallback until start() captures the real one. sys.__stdout__ is None
        # under pythonw, hence the or.
        self._out = sys.__stdout__ or sys.stdout
        self._proxy = None
        self._err_proxy = None
        self._box = _BOX_UNICODE
        self._ring_chars = _RING_UNICODE
        self._spinner = _SPINNER
        self._ellipsis = "…"

    # --- lifecycle ---------------------------------------------------------

    def _write(self, text):
        """All terminal output goes through here. A closed or redirected
        stream during shutdown shouldn't take the assistant down with it."""
        try:
            self._out.write(text)
            self._out.flush()
        except Exception:
            pass

    def start(self):
        """Takes over the bottom of the terminal. No-op if stdout isn't one."""
        with self._lock:
            if self._active:
                return False

            stream = sys.stdout
            if stream is None or os.environ.get("JARVIS_NO_UI"):
                return False
            try:
                if not stream.isatty():
                    return False
            except Exception:
                return False
            if not _enable_windows_ansi():
                return False

            if not _can_encode(
                _BOX_UNICODE["tl"] + _RING_UNICODE["full"] + _SPINNER[0] + "…"
                + _BOX_UNICODE["typed"] + _BOX_UNICODE["caret"]
            ):
                self._box = _BOX_ASCII
                self._ring_chars = _RING_ASCII
                self._spinner = _SPINNER_ASCII
                self._ellipsis = "..."

            self._active = True
            self._stop.clear()
            self._t0 = time.time()
            self._out = stream          # whatever's really attached, not __stdout__
            self._proxy = _StreamProxy(self, stream)
            sys.stdout = self._proxy

            # Only take stderr if it's pointing at a terminal. Under
            # `python main.py 2> errors.log` it isn't, and diverting it into
            # the panel would empty the file the user asked for.
            #
            # No identity check against stdout: some consoles hand back the
            # same object for both, and skipping those left tracebacks
            # writing straight through the panel -- the exact thing this is
            # here to stop. Wrapping the same stream twice is harmless.
            err = sys.stderr
            if err is not None:
                try:
                    if err.isatty():
                        self._err_proxy = _StreamProxy(self, err)
                        sys.stderr = self._err_proxy
                except Exception:
                    self._err_proxy = None

            self._write("\x1b[?25l")  # hide cursor

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Restores the terminal, leaving the last frame in place."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        with self._lock:
            if not self._active:
                return
            self._active = False

            leftovers = []
            if self._proxy is not None:
                if sys.stdout is self._proxy:
                    sys.stdout = self._proxy._original
                leftovers.append(self._proxy)
            if self._err_proxy is not None:
                if sys.stderr is self._err_proxy:
                    sys.stderr = self._err_proxy._original
                leftovers.append(self._err_proxy)

            # Flush trailing text that never got its newline -- a bare
            # write() with no "\n" is still sitting in the partial buffer.
            for proxy in leftovers:
                if proxy._partial:
                    text, proxy._partial = proxy._partial, ""
                    self._write(text + "\n")

            self._proxy = None
            self._err_proxy = None

            # Forget the panel's position. A later start() would otherwise
            # emit a cursor-up for rows that are no longer on screen and
            # overwrite scrollback.
            self._lines_drawn = 0
            self._last_frame = None

            self._write("\x1b[?25h\n")  # show cursor

    # --- inputs ------------------------------------------------------------

    def is_active(self):
        return self._active

    def set_state(self, state):
        with self._lock:
            if state != self._state:
                self._state = state
                # Drop levels captured under the previous state -- a peak left
                # over from the mic shouldn't open the playback wave, and a
                # delayed playback level shouldn't land after we've gone back
                # to listening.
                with self._level_lock:
                    self._pending = None
                    del self._delayed[:]

    def push_level(self, level, delay=0.0):
        """0..1. Keeps the loudest value seen since the last rendered frame,
        so a burst between frames still registers instead of being averaged
        into nothing.

        `delay` holds the level back that many seconds before it's drawn.
        Playback uses it to cancel the device buffer: the write thread hands
        the sound card a slice some tens of milliseconds before you hear it,
        so an undelayed bar runs slightly ahead of the voice.

        Called from the playback thread between writes to the sound card, so
        it must never block on anything slow -- hence _level_lock rather than
        the render lock. See the note in __init__.
        """
        if not self._active:
            return

        level = 0.0 if level < 0.0 else (1.0 if level > 1.0 else float(level))
        slot = min(int(round(delay * FPS)), MAX_DELAY_FRAMES) if delay > 0 else 0

        with self._level_lock:
            if slot <= 0:
                if self._pending is None or level > self._pending:
                    self._pending = level
                return

            while len(self._delayed) < slot:
                self._delayed.append(0.0)
            index = slot - 1
            if level > self._delayed[index]:
                self._delayed[index] = level

    def push_rms(self, rms, delay=0.0):
        if not self._active:
            return
        self.push_level(level_from_rms(rms), delay=delay)

    def set_heard(self, text):
        with self._lock:
            self._heard = (text or "").strip()
            self._reply = ""
            self._timing = ""

    def set_reply(self, text):
        with self._lock:
            self._reply = (text or "").strip()

    def set_timing(self, text):
        with self._lock:
            self._timing = (text or "").strip()

    def set_engine(self, stt=None, tts=None):
        """Info column: the STT and TTS backend labels the mockup shows as
        "gemini" and "groq · cloud"."""
        if stt is None and tts is None:
            return
        with self._lock:
            if stt is not None:
                self._stt = stt
            if tts is not None:
                self._tts = tts

    def set_latency(self, text):
        with self._lock:
            if text:
                self._latency = (text or "").strip()

    def set_turns(self, count):
        with self._lock:
            self._turns = int(count or 0)

    # --- history overlay ---------------------------------------------------

    def toggle_history(self):
        """Toggles the full-log overlay (the history of everything that
        would once have scrolled above the panel). On entry the view is
        reset to the newest lines."""
        with self._lock:
            self._history_on = not self._history_on
            if self._history_on:
                self._history_scroll = 0
            self._last_frame = None

    def history_active(self):
        return self._history_on

    def history_room(self):
        """Rows available for the history list, roughly. Used to clamp the
        scroll offset; the render itself lets the existing too-tall check
        bail out rather than corrupting the cursor arithmetic."""
        return max(1, self._terminal_size()[1] - 5)

    def history_scroll(self, delta):
        """Scrolls the history overlay by `delta` (negative = up/older).
        Returns True if the overlay is still up, so the key reader knows
        whether to keep swallowing navigation keys."""
        with self._lock:
            if not self._history_on:
                return False
            count = len(self._history)
            top = max(0, count - self.history_room())
            self._history_scroll = max(
                0, min(top, self._history_scroll + int(delta))
            )
            self._last_frame = None
            return True

    def enable_input(self, enabled=True, hint=None):
        """Shows or hides the prompt row. console_input calls this once it
        knows it actually has a keystroke reader."""
        with self._lock:
            self._input_enabled = bool(enabled)
            if hint is not None:
                self._input_hint = hint
            if not enabled:
                self._input = ""

    def set_input(self, text):
        """The line currently being typed. Called from the reader thread on
        every keystroke -- cheap, and it never prints."""
        with self._lock:
            self._input = text or ""

    def log(self, line):
        """A log line. Kept out of the way instead of printing above the
        panel and shoving it down the screen (the old behaviour, which is
        what made the dashboard wander and shred).

        When the panel is live, the line goes into the history buffer and
        shows as a dim toast for TOAST_SECONDS, then settles back. The
        full history is available on demand -- see toggle_history(). When
        the UI isn't running (piped, JARVIS_NO_UI) there's no panel to
        protect, so it prints normally as plain stdout.
        """
        with self._lock:
            if not self._active:
                self._write(line + "\n")
                return

            self._history.append(line)
            # Each new line refreshes the window, so a burst of diagnostics
            # reads as one continuous status line rather than a flashing row.
            self._toast = line
            self._toast_until = self._tick + int(TOAST_SECONDS * FPS)
            self._last_frame = None

    # --- rendering ---------------------------------------------------------

    def _loop(self):
        interval = 1.0 / FPS
        while not self._stop.is_set():
            try:
                with self._lock:
                    self._advance_locked()
                    self._draw_locked()
            except Exception:
                # A render bug must never be the thing that kills the
                # assistant. Drop the frame and try again next tick.
                pass
            time.sleep(interval)

    def _advance_locked(self):
        """Scrolls the wave one column, at a fixed rate.

        Decoupling the scroll from the push rate is deliberate: the mic feeds
        us ~12 levels a second and playback ~24, and letting either drive the
        animation directly would make the wave visibly change speed depending
        on who was talking.
        """
        self._tick += 1

        # Anything held back by push_level(delay=...) comes due one frame at a
        # time, so the delay line advances with the render clock. Held only
        # long enough to lift the values out -- the playback thread is
        # blocked on this lock while we have it.
        with self._level_lock:
            if self._delayed:
                due = self._delayed.pop(0)
                if due > 0.0 and (self._pending is None or due > self._pending):
                    self._pending = due
            pending, self._pending = self._pending, None

        if pending is not None:
            level = pending
        elif self._state == THINKING:
            # Nothing is producing audio between the last word you said and
            # the first word of the reply, and a flat line there reads as a
            # hang. A slow travelling swell keeps it obviously alive without
            # pretending to be a signal.
            level = THINK_AMPLITUDE * (
                0.5 + 0.5 * math.sin(self._tick / 5.0)
            )
        else:
            level = self._last_level * DECAY
            if level < 0.004:
                level = 0.0

        self._last_level = level
        self._spark.append(level)
        self._spin = (self._spin + 1) % (len(self._spinner) * 4)

        # Let a stale toast fade back to just the conversation.
        if self._toast is not None and self._tick >= self._toast_until:
            self._toast = None
            self._last_frame = None

    def _ring_cols(self):
        """Ring width in terminal columns, scaled with the panel like the
        old _wave_width() was, capped so the ring neither dominates a wide
        terminal nor vanishes on a narrow one. Kept even so the two
        vertical half-block pixels per row split cleanly.
        """
        inner = self._panel_width() - 4
        cols = max(RING_MIN_COLS, min(RING_MAX_COLS, inner - 4))
        return cols if cols % 2 == 0 else cols - 1

    def _ring_rows_count(self, cols):
        """Half as many terminal rows as columns.

        Two vertical pixels pack into each row via the half-block trick, so
        the vertical pixel count is already `2 * rows`. A monospace cell is
        roughly twice as tall as it is wide, so halving that vertical step
        brings pixel width and pixel height back to about the same physical
        size -- without this the ring renders as an oval, not a circle.
        """
        return max(3, cols // 2)

    def _terminal_size(self):
        try:
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except Exception:
            return 80, 24

    def _panel_width(self):
        columns = self._terminal_size()[0]
        # Never wider than the terminal: a row that wraps adds a line the
        # cursor-up arithmetic doesn't know about, and the panel walks down
        # the screen one row per frame.
        return min(MAX_WIDTH, columns - 1)

    def _draw_locked(self):
        if not self._active:
            return

        frame = self._frame_locked()
        if frame is None:
            # Terminal too small to hold the panel. Clear whatever's left of
            # it so log output isn't fighting a half-drawn box.
            if self._lines_drawn:
                self._write(
                    "\x1b[{}A\x1b[J".format(self._lines_drawn)
                )
                self._lines_drawn = 0
                self._last_frame = None
            return
        if frame == self._last_frame:
            return

        out = []
        if self._lines_drawn:
            out.append("\x1b[{}A".format(self._lines_drawn))

        lines = frame.split("\n")
        for line in lines:
            out.append("\x1b[2K" + line + "\n")

        # Panel shrank (terminal resize) -- clear the orphaned rows below.
        if len(lines) < self._lines_drawn:
            out.append("\x1b[J")

        self._lines_drawn = len(lines)
        self._last_frame = frame
        self._write("".join(out))

    # --- frame construction ------------------------------------------------

    def _frame_locked(self):
        box = self._box
        width = self._panel_width()
        inner = width - 4

        # The header alone needs 8 columns of border/padding plus the title
        # and the longest status ("* RECORDING"). Below that the fill would
        # clamp, the row would overrun the panel, and a wrapped row throws off
        # the cursor-up count -- so draw nothing instead.
        if inner + 4 < 8 + len("J A R V I S") + len("* RECORDING") + 1:
            return None

        label, accent = _STATE_STYLE.get(self._state, _STATE_STYLE[IDLE])
        border = _fg(BORDER_RGB)
        reset = "\x1b[0m"

        if self._history_on:
            rows = self._history_rows(box, inner, border, reset, accent)
            if len(rows) + 2 > self._terminal_size()[1]:
                return None
            return "\n".join(rows)

        rows = []

        # --- header: ╭─ JARVIS ────────── ● SPEAKING ─╮
        marker = box["dot"]
        if self._state == THINKING:
            spinner = self._spinner
            marker = spinner[self._spin // 4 % len(spinner)]

        title = "J A R V I S"
        status = "{} {}".format(marker, label)

        # Every row must come out to exactly inner + 4 visible columns or the
        # right-hand border wanders as the state label changes length. The
        # header spends 8 on borders and padding; fill takes up the slack.
        fill = max(1, (inner + 4) - 8 - len(title) - len(status))

        rows.append(
            border + box["tl"] + box["h"] + " " + reset
            + _fg(TEXT_RGB) + title + reset
            + " " + border + box["h"] * fill + " " + reset
            + _fg(accent) + status + reset
            + border + " " + box["h"] + box["tr"] + reset
        )

        # --- ring + info column
        # Derived from `inner` rather than re-measuring the terminal, same
        # reasoning as the old wave: a resize between the two reads would
        # put the ring and the borders on different widths for a frame. The
        # ring sits left, the mockup's STT/TTS/latency/sparkline/turns
        # stats sit to its right.
        ring_cols = self._ring_cols()
        for row in self._ring_info_rows(ring_cols, accent, inner):
            rows.append(row)

        rows.append(
            border + box["ml"] + box["h"] * (inner + 2) + box["mr"] + reset
        )

        # --- transient toast: the most recent log line, which clears on its
        # own after TOAST_SECONDS (see _advance_locked). Dim so it never
        # competes with the actual conversation below it.
        if self._toast is not None:
            rows.append(self._text_row(
                box["sep"], self._toast, inner, DIM_RGB, DIM_RGB
            ))

        # --- transcript / reply / timing
        rows.append(self._text_row(
            box["prompt"], self._heard or "waiting for \"hey jarvis\"",
            inner, accent if self._heard else DIM_RGB,
            TEXT_RGB if self._heard else DIM_RGB,
        ))
        rows.append(self._text_row(
            box["reply"], self._reply, inner, accent, TEXT_RGB
        ))
        rows.append(self._text_row(
            " ", self._timing, inner, DIM_RGB, DIM_RGB
        ))

        # --- prompt
        if self._input_enabled:
            rows.extend(self._input_rows(inner, accent))

        rows.append(
            border + box["bl"] + box["h"] * (inner + 2) + box["br"] + reset
        )

        # A panel taller than the terminal cannot be drawn with relative
        # cursor moves -- the top rows scroll off, the cursor-up count
        # overshoots into scrollback, and the panel shreds itself. Two rows
        # spare so log output still has somewhere to land.
        if len(rows) + 2 > self._terminal_size()[1]:
            return None

        return "\n".join(rows)

    def _history_rows(self, box, inner, border, reset, accent):
        """The full-log overlay: every line that would once have scrolled
        above the panel, shown in place of the ring and conversation so it
        has the whole panel to itself. Opens with ctrl+L, closes the same
        way (or with Esc); ↑/↓/PageUp/PageDown scroll a window scrolled by
        history_scroll()."""
        title = "J A R V I S"
        status = "  H I S T O R Y  "
        fill = max(1, (inner + 4) - 8 - len(title) - len(status))
        rows = [
            border + box["tl"] + box["h"] + " " + reset
            + _fg(TEXT_RGB) + title + reset
            + " " + border + box["h"] * fill + " " + reset
            + _fg(accent) + status + reset
            + border + " " + box["h"] + box["tr"] + reset,
            self._text_row(
                box["sep"],
                "ctrl+L close · ↑/↓ or PageUp/PageDown scroll",
                inner, DIM_RGB, DIM_RGB,
            ),
            border + box["ml"] + box["h"] * (inner + 2) + box["mr"] + reset,
        ]

        lines = list(self._history)
        count = len(lines)
        room = self.history_room()
        start = max(0, count - max(1, room) - self._history_scroll)
        for ln in lines[start:]:
            rows.append(self._text_row(box["sep"], ln, inner, DIM_RGB, TEXT_RGB))

        if self._input_enabled:
            rows.extend(self._input_rows(inner, accent))

        rows.append(
            border + box["bl"] + box["h"] * (inner + 2) + box["br"] + reset
        )
        return rows

    def _ring_params(self):
        """Current (t, speed, amp, buzz) for the wobble formula below.

        `amp` is the state's baseline (_RING_PARAMS) plus a share of
        self._last_level -- the same value _advance_locked() already
        maintains for every state (decayed audio, real pushed levels, or
        THINKING's own idle swell). That's what makes the ring audio-
        reactive rather than a fixed per-state performance.
        """
        t = self._tick / float(FPS)
        p = _RING_PARAMS.get(self._state, _RING_PARAMS[IDLE])
        amp = p["amp"] + self._last_level * p["reactivity"]
        return t, p["speed"], amp, p["buzz"]

    def _ring_gradient(self, frac):
        """Position around the ring (0..1) -> an RGB colour, cycling
        through _RING_GRADIENT's four stops. Same shape as the mockup's
        grad() function."""
        stops = _RING_GRADIENT
        f = (frac % 1.0) * len(stops)
        i = int(f) % len(stops)
        return _mix(stops[i], stops[(i + 1) % len(stops)], f - int(f))

    def _half_block(self, top, bottom):
        """One terminal cell from two vertical pixels' colours (or None if
        unlit). Foreground-only -- no background codes -- so the ring
        draws as coloured strokes on whatever the terminal's own
        background is, same philosophy as the rest of the panel."""
        chars = self._ring_chars
        if top is not None and bottom is not None:
            return _fg(_mix(top, bottom, 0.5)) + chars["full"] + "\x1b[0m"
        if top is not None:
            return _fg(top) + chars["top"] + "\x1b[0m"
        if bottom is not None:
            return _fg(bottom) + chars["bottom"] + "\x1b[0m"
        return " "

    def _ring_rows(self, cols, accent):
        """The circular, audio-reactive centrepiece -- the terminal
        analogue of the HTML mockup's canvas blob ring.

        Every pixel's angle and distance from centre are tested against a
        wobbled ring radius, using the same three-harmonic shape the
        mockup's canvas script draws with (a slow orbit term, a faster
        ripple, and a buzz term) -- see _ring_params() for where the
        numbers come from. A pixel within ~1 unit of that radius is lit
        and coloured by its angle via _ring_gradient(); everything else is
        blank except a small solid core at dead centre, mirroring the
        little centre dot the mockup always draws regardless of state.

        Two vertical pixels pack into each terminal row -- see
        _ring_rows_count()'s docstring for why that specifically is what
        keeps this round instead of egg-shaped.
        """
        rows_count = self._ring_rows_count(cols)
        vpx = rows_count * 2
        t, speed, amp, buzz = self._ring_params()
        base_r = min(cols, vpx) * 0.5 * 0.62
        cx = (cols - 1) / 2.0
        cy = (vpx - 1) / 2.0
        core = _mix(accent, (255, 255, 255), 0.35)

        def lit(x, y):
            dx = x - cx
            dy = y - cy
            radius = math.hypot(dx, dy)
            if radius < 1.2:
                return core
            angle = math.atan2(dy, dx)
            wobble = (
                math.sin(angle * 3 + t * speed * 2) * amp
                + math.sin(angle * 5 - t * 3.1) * amp * 0.45
                + math.sin(t * 9 + angle * 2) * buzz
            )
            ring_r = base_r * (1.0 + wobble)
            if abs(radius - ring_r) > 1.05:
                return None
            return self._ring_gradient(angle / (2 * math.pi) + t * 0.05)

        out = []
        for r in range(rows_count):
            cells = [
                self._half_block(lit(c, r * 2), lit(c, r * 2 + 1))
                for c in range(cols)
            ]
            out.append("".join(cells))
        return out

    def _kv_line(self, label, value):
        """One label/value row for the info column -- dim label, then the
        accent-green value, mirroring the mockup's "STT  gemini" styling."""
        painted = (
            _fg(DIM_RGB) + label + "\x1b[0m"
            + _fg(STATS_RGB) + value + "\x1b[0m"
        )
        return painted, len(label) + len(value)

    def _spark_painted(self):
        """The mockup's live sparkline, grown from the recent audio levels
        (self._spark) into a column of ▁…█ blocks, all in the mockup green.
        Falls back to ASCII bars on consoles that can't show the blocks
        (same capability check that drives the ring's fallback)."""
        blocks = _SPARK_BLOCKS if self._ring_chars is _RING_UNICODE else _SPARK_ASCII
        vals = list(self._spark)
        if len(vals) < SPARK_WIDTH:
            vals = [0.0] * (SPARK_WIDTH - len(vals)) + vals
        vals = vals[-SPARK_WIDTH:]
        chars = "".join(blocks[min(len(blocks) - 1, int(v * (len(blocks) - 1) + 0.5))] for v in vals)
        painted = _fg(STATS_RGB) + chars + "\x1b[0m"
        return painted, len(chars)

    def _info_lines(self, inner):
        """The five mockup info rows, each as (painted, visible_len)."""
        uptime = 0
        if self._t0 is not None:
            uptime = int(time.time() - self._t0)
        minutes = uptime // 60
        up_str = "{}m".format(minutes) if minutes else "{}s".format(uptime)

        lines = [
            self._kv_line("STT ", self._stt),
            self._kv_line("TTS ", self._tts),
            self._kv_line("latency ", self._latency),
            self._spark_painted(),
            self._kv_line(
                "turns {} · up {}".format(self._turns, up_str), ""
            ),
        ]
        # Don't let an info line push past the right edge of the panel.
        room = inner - len(" ") - len("STT ")
        for i, (painted, vlen) in enumerate(lines):
            if vlen > room:
                lines[i] = (painted, room)
        return lines

    def _ring_info_rows(self, ring_cols, accent, inner):
        """The ring on the left with the info column centred vertically to
        its right -- the mockup's flex layout, one terminal row at a time."""
        ring_rows = self._ring_rows(ring_cols, accent)
        info = self._info_lines(inner)
        gap = 7
        start = (len(ring_rows) - len(info)) // 2
        if start < 0:
            start = 0
        border = _fg(BORDER_RGB)
        rows = []
        for i, ring_row in enumerate(ring_rows):
            if start <= i < start + len(info):
                painted, vlen = info[i - start]
            else:
                painted, vlen = "", 0
            combined = ring_row + " " * gap + painted
            visible = ring_cols + gap + vlen
            rows.append(self._pad_row(combined, visible, inner, border))
        return rows

    def _pad_row(self, painted, visible_len, inner, border, center=False):
        """Wraps one row of already-coloured content in the panel borders.

        Total visible width always works out to inner + 4, so the right border
        stays put no matter what the content does.
        """
        slack = max(0, inner - visible_len)
        left = 1 + (slack // 2 if center else 0)
        right = max(1, inner + 2 - left - visible_len)
        return (
            border + self._box["v"] + "\x1b[0m" + " " * left
            + painted
            + " " * right + border + self._box["v"] + "\x1b[0m"
        )

    def _text_row(self, glyph, text, inner, glyph_rgb, text_rgb):
        room = inner - 2
        text = text or ""
        if len(text) > room:
            cut = max(0, room - len(self._ellipsis))
            text = text[:cut].rstrip() + self._ellipsis

        plain = "{} {}".format(glyph, text)
        painted = (
            _fg(glyph_rgb) + glyph + "\x1b[0m "
            + _fg(text_rgb) + text + "\x1b[0m"
        )
        return self._pad_row(painted, len(plain), inner, _fg(BORDER_RGB))

    def _input_rows(self, inner, accent):
        """The line(s) being typed -- one row per line now that Shift+Enter
        (console_input.py) can put "\\n" in the buffer.

        Only the last line has a cursor on it in any meaningful sense (the
        buffer is append-only; there's no mid-buffer cursor to track), so
        only it keeps the original _input_row behaviour: the prompt glyph,
        the blinking caret, truncated from the *left* so the cursor stays
        on screen while you're still typing it. Finished lines above it are
        just text now -- rendered with _text_row's own left-to-right
        truncation, same as the heard/reply rows, since there's nothing
        moving on them to keep in view.

        A panel that ends up taller than the terminal already returns None
        from _frame_locked and skips the draw rather than corrupting the
        cursor-up arithmetic -- see the check at the end of that method --
        so an unusually long paste here degrades to "panel doesn't draw
        this frame" rather than anything worse.
        """
        box = self._box
        text = self._input

        if not text:
            return [self._text_row(
                box["typed"], self._input_hint, inner, DIM_RGB, DIM_RGB
            )]

        lines = text.split("\n")
        rows = [
            self._text_row(box["sep"], line, inner, DIM_RGB, TEXT_RGB)
            for line in lines[:-1]
        ]
        rows.append(self._active_input_row(lines[-1], inner, accent))
        return rows

    def _active_input_row(self, text, inner, accent):
        """One row for the line the cursor is actually on -- the caret, the
        prompt glyph, truncated from the left. Body unchanged from the old
        single-line _input_row; it just takes its text explicitly now that
        self._input can hold more than this one row's worth.

        The caret only blinks while there is text. A caret that blinks forever
        changes the frame twice a second, and _draw_locked skips the console
        write when the frame is unchanged -- so an always-blinking caret would
        quietly turn an idle panel from zero repaints per second into two, on
        a machine this project is trying to be gentle to.
        """
        box = self._box
        caret = box["caret"] if (self._tick // CARET_FRAMES) % 2 == 0 else " "

        # glyph + space + caret
        room = max(1, inner - 2 - len(caret))
        shown = text
        if len(shown) > room:
            keep = max(0, room - len(self._ellipsis))
            shown = self._ellipsis + shown[len(shown) - keep:]

        plain = "{} {}{}".format(box["typed"], shown, caret)
        painted = (
            _fg(accent) + box["typed"] + "\x1b[0m "
            + _fg(TEXT_RGB) + shown + "\x1b[0m"
            + _fg(accent) + caret + "\x1b[0m"
        )
        return self._pad_row(painted, len(plain), inner, _fg(BORDER_RGB))


# --- Module-level façade ---------------------------------------------------
#
# record.py, audio_out.py and console_input.py call straight into these.
# Keeping them as plain functions means those modules don't have to know
# whether a dashboard is running -- when it isn't, every call is a cheap
# no-op.

_dashboard = Dashboard()


def start():
    return _dashboard.start()


def stop():
    _dashboard.stop()


def set_state(state):
    _dashboard.set_state(state)


def push_level(level, delay=0.0):
    _dashboard.push_level(level, delay=delay)


def push_rms(rms, delay=0.0):
    _dashboard.push_rms(rms, delay=delay)


def set_heard(text):
    _dashboard.set_heard(text)


def set_reply(text):
    _dashboard.set_reply(text)


def set_timing(text):
    _dashboard.set_timing(text)


def set_engine(stt=None, tts=None):
    _dashboard.set_engine(stt=stt, tts=tts)


def set_latency(text):
    _dashboard.set_latency(text)


def set_turns(count):
    _dashboard.set_turns(count)


def toggle_history():
    _dashboard.toggle_history()


def history_active():
    return _dashboard.history_active()


def history_scroll(delta):
    return _dashboard.history_scroll(delta)


def enable_input(enabled=True, hint=None):
    _dashboard.enable_input(enabled, hint=hint)


def set_input(text):
    _dashboard.set_input(text)


def log(line):
    _dashboard.log(line)


def is_active():
    """True once the panel has taken over the terminal. Callers use this to
    drop banner prints that the panel already shows."""
    return _dashboard.is_active()


if __name__ == "__main__":
    # Walk the whole API so the panel can be eyeballed without the mic or
    # the API: every state, the toast behaviour for log lines, the history
    # overlay (auto-played, then left open for you to scroll), and a live
    # prompt driven by the real keystroke reader.
    import random

    if not start():
        print("dashboard inactive -- not a terminal, or JARVIS_NO_UI is set")
        raise SystemExit(1)

    ci = None

    try:
        enable_input(True)
        set_state(LISTENING)
        set_engine("gemini", "groq · cloud")
        set_turns(14)
        set_latency("1.4s")
        log("demo: JARVIS UI preview -- Ctrl+L = history, Ctrl+C = quit")
        for _ in range(int(FPS * 1.5)):
            push_level(random.uniform(0.0, 0.06))
            time.sleep(1.0 / FPS)

        # Prompt row: text appearing a character at a time, caret blinking.
        typed = "arrange chrome on the left"
        for i in range(len(typed) + 1):
            set_input(typed[:i])
            time.sleep(0.05)
        time.sleep(0.6)
        set_input("")

        set_state(RECORDING)
        set_heard("open spotify and take a screenshot")
        for i in range(int(FPS * 2)):
            push_level(abs(math.sin(i / 3.0)) * random.uniform(0.4, 1.0))
            time.sleep(1.0 / FPS)

        set_state(THINKING)
        time.sleep(1.5)  # the idle swell should keep moving here

        set_reply("Sure, opening Spotify and grabbing that screenshot.")
        set_state(SPEAKING)
        # These prints are captured by the dashboard's stdout proxy and show
        # up as the dim, self-clearing toast -- not as scrollback.
        print("Opened Spotify")
        print("Screenshot taken")
        for i in range(int(FPS * 3)):
            # delay= mirrors what audio_out does with the device latency
            push_level(
                abs(math.sin(i / 5.0)) * random.uniform(0.3, 0.95),
                delay=0.08,
            )
            time.sleep(1.0 / FPS)

        set_state(LISTENING)
        set_timing("stt 1.10s | reply 0.42s | first audio 0.09s")
        time.sleep(1.2)

        # Give the history overlay something to browse.
        for _ in range(5):
            log("demo: a thing Jarvis did (e.g. opened an app, grabbed output)")
        log("demo: history now has lines for you to scroll through")
        time.sleep(1.5)

        # Auto-play the overlay so you can see it open, scroll and close.
        log("demo: opening history overlay...")
        time.sleep(1.0)
        toggle_history()
        time.sleep(1.2)
        history_scroll(-2)   # a page up
        time.sleep(1.2)
        history_scroll(8)    # back toward the newest
        time.sleep(1.2)
        toggle_history()
        log("demo: overlay closed -- back to the panel")
        time.sleep(1.0)

        # Hand over to the real keystroke reader so you can drive it: type to
        # see the prompt, Ctrl+L for history, Ctrl+C to quit.
        try:
            import listener.console_input as ci
        except Exception:
            ci = None
        if ci is not None and ci.start():
            set_input("")
            set_state(LISTENING)
            log("demo: live now -- type anything, Ctrl+L = history, Ctrl+C = quit")
            end = None  # interactive until Ctrl+C
        else:
            ci = None
            log("demo: no key reader here -- will end in 5s (Ctrl+C sooner)")
            end = time.time() + 5.0

        while True:
            if ci is not None:
                line = ci.poll()
                if line:
                    set_heard(line)
                    set_reply('(demo) echoed "{}"'.format(line))
                    log("demo: submitted: {}".format(line))
            push_level(random.uniform(0.0, 0.04))
            if end is not None and time.time() >= end:
                break
            time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        pass
    finally:
        if ci is not None:
            try:
                ci.stop()
            except Exception:
                pass
        stop()