"""Live terminal dashboard with an audio-synced waveform and a typed prompt.

Goes in: listener/ui.py   (replaces existing)

Draws a fixed panel at the bottom of the terminal -- state, waveform, what was
heard, what Jarvis said, and a prompt line you can type into -- while ordinary
log output keeps scrolling above it.

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

# --- States ----------------------------------------------------------------

IDLE = "IDLE"
LISTENING = "LISTENING"
RECORDING = "RECORDING"
THINKING = "THINKING"
SPEAKING = "SPEAKING"

# Label plus the accent colour the whole panel takes on in that state.
_STATE_STYLE = {
    IDLE:      ("IDLE",      (95, 105, 120)),
    LISTENING: ("LISTENING", (60, 200, 205)),
    RECORDING: ("RECORDING", (250, 170, 65)),
    THINKING:  ("THINKING",  (170, 130, 255)),
    SPEAKING:  ("SPEAKING",  (95, 220, 145)),
}

# --- Tunables --------------------------------------------------------------

FPS = 24
MAX_WIDTH = 84
WAVE_ROWS = 5

# How fast a bar falls when no fresh audio arrives. Without this the wave
# freezes mid-stride on every pause instead of settling.
DECAY = 0.72

# Height of the idle swell drawn while waiting on the model. Low enough that
# it never reads as real audio.
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

_SPINNER = "◐◓◑◒"
_SPINNER_ASCII = "|/-\\"

_BLOCKS = " ▁▂▃▄▅▆▇█"
_BLOCKS_ASCII = " .:-=+*#@"

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
        self._wave = []
        self._pending = None
        self._delayed = []
        self._last_level = 0.0
        self._spin = 0
        self._tick = 0

        self._heard = ""
        self._reply = ""
        self._timing = ""

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
        self._blocks = _BLOCKS
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
                _BOX_UNICODE["tl"] + _BLOCKS[-1] + _SPINNER[0] + "…"
                + _BOX_UNICODE["typed"] + _BOX_UNICODE["caret"]
            ):
                self._box = _BOX_ASCII
                self._blocks = _BLOCKS_ASCII
                self._spinner = _SPINNER_ASCII
                self._ellipsis = "..."

            self._active = True
            self._stop.clear()
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
        """Prints above the panel."""
        with self._lock:
            if not self._active:
                self._write(line + "\n")
                return

            out = []
            if self._lines_drawn:
                out.append("\x1b[{}A".format(self._lines_drawn))
            out.append("\x1b[J")            # wipe panel and everything below
            out.append(line + "\n")
            self._lines_drawn = 0
            self._last_frame = None         # force a full repaint
            self._write("".join(out))
            self._draw_locked()

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

        width = self._wave_width()
        self._wave.append(level)
        if len(self._wave) > width:
            del self._wave[:len(self._wave) - width]

        self._spin = (self._spin + 1) % (len(self._spinner) * 4)

    def _wave_width(self):
        """Columns of history to keep. Matches what _frame_locked draws."""
        return max(8, self._panel_width() - 6)

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

        # --- waveform
        # Derived from `inner` rather than re-measuring the terminal: a resize
        # between the two reads would put the wave and the borders on
        # different widths for a frame.
        wave_width = max(8, inner - 2)
        for row in self._wave_rows(wave_width, accent):
            rows.append(
                self._pad_row(row, wave_width, inner, border, center=True)
            )

        rows.append(
            border + box["ml"] + box["h"] * (inner + 2) + box["mr"] + reset
        )

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

    def _wave_rows(self, width, accent):
        """Bars, tallest at the top, with an eight-level tip per cell.

        Mirroring around a centre line was the first instinct, but the block
        characters only fill upward -- there's no eight-step set that fills
        from the top of a cell -- so a mirrored wave ends up visibly coarser
        on the bottom half. Single-sided keeps the resolution even.
        """
        wave = self._wave
        pad = width - len(wave)
        values = ([0.0] * pad + wave) if pad > 0 else wave[-width:]

        blocks = self._blocks
        top = len(blocks) - 1
        base = _mix(accent, (18, 20, 26), 0.55)

        rows = []
        for row in range(WAVE_ROWS - 1, -1, -1):
            # Brighter toward the top of the panel, so peaks read as peaks.
            shade = _fg(_mix(base, accent, (row + 0.5) / WAVE_ROWS))
            cells = []
            for value in values:
                filled = value * WAVE_ROWS - row
                if filled <= 0.0:
                    cells.append(" ")
                elif filled >= 1.0:
                    cells.append(blocks[top])
                else:
                    cells.append(blocks[max(1, int(filled * top))])
            rows.append(shade + "".join(cells) + "\x1b[0m")
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
    # Fake a turn so the panel can be eyeballed without the mic or the API.
    import random

    if not start():
        print("dashboard inactive -- not a terminal, or JARVIS_NO_UI is set")
        raise SystemExit(1)

    try:
        enable_input(True)
        set_state(LISTENING)
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
        time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop()