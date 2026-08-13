"""Typed commands, read straight off the console without echo.

Goes in: listener/console_input.py   (new file)

Lets you type to Jarvis instead of saying "hey jarvis" out loud -- useful when
the room is loud, when the wake word won't trigger, and when you want to test
a command twenty times in a row without talking to your computer.

    console_input.start()
    ...
    text = console_input.poll()      # None if nothing has been submitted
    if text:
        handle_utterance_text(text)

WHY NOT input()
---------------
Three reasons, all of them things that were tried:

1. input() blocks a thread forever with no way to cancel it. On Ctrl+C the
   wake loop unwinds, main() runs its finally, and the process then sits
   there because a non-daemon reader is still parked inside a blocking read.
   Making it a daemon hides that but leaves the console in whatever state the
   read left it.

2. The terminal echoes what you type wherever the cursor happens to be, which
   is the row *below* the dashboard panel. Pressing Enter scrolls the screen
   by one line, but the panel's `\x1b[{n}A` still thinks it is n rows up, so
   the panel walks down the screen one row per line submitted and leaves a
   trail of dead borders behind it.

3. The prompt string input() writes is not part of a frame, so the render
   thread's next repaint erases it mid-word.

So this module owns the echo instead. Keystrokes are read one at a time with
no echo, the line under construction is handed to ui.set_input(), and the
panel draws it as one of its own rows. Nothing outside ui.py ever moves the
cursor, which is the invariant the whole dashboard rests on.

SHIFT+ENTER, AND WHY CTRL+J IS HERE TOO
----------------------------------------
Shift+Enter adds a line break to the buffer instead of submitting it --
standard multi-line-input behaviour. On Windows, _read_windows() reads the
*untranslated* key event straight off the Win32 console API
(ReadConsoleInputW), which carries the virtual key code and the
control-key-state bitmask a cooked read throws away, so the shift modifier
on VK_RETURN is visible and dispatched to _insert_newline().

That said, Shift+Enter alone isn't reliable everywhere: Windows Terminal
and VS Code's integrated terminal run child processes behind a pseudo
console (ConPTY), which reconstructs INPUT_RECORDs from an incoming VT
byte stream rather than handing over real hardware key events. Modern
Windows Terminal versions do forward full modifier state for this
(win32-input-mode), but not every terminal/version does, and when it
doesn't, the SHIFT_PRESSED bit silently reads as unset -- Shift+Enter then
just submits like a bare Enter, with nothing to indicate why.

So there's a second, independent trigger: Ctrl+J (linefeed, "\x0a"). Ctrl
plus a letter is a distinct virtual key + modifier combination, not a
modified Enter, so it isn't subject to the same VT-reconstruction gap --
it reads correctly through ConPTY as well as a native console. It's wired
in both readers:

  * _read_windows(): checked as VK 'J' + a CTRL bit, same as the
    Shift+Enter check right above it.
  * _read_posix(): Ctrl+J physically sends the raw byte 0x0a regardless of
    terminal, so as long as nothing on the way in remaps it to something
    else, _handle() can treat it as "insert newline" directly. The one
    thing that *does* remap it is covered below.

_read_posix() and Enter-vs-Ctrl+J
----------------------------------
A tty's ICRNL input flag translates an incoming "\r" to "\n" before a
reading process ever sees it, and tty.setcbreak() (used here to turn off
line buffering and echo) doesn't touch that flag -- it's normally still on
from whatever the shell set up. Left alone, that means plain Enter would
also show up as "\n", making it indistinguishable from an intentional
Ctrl+J. _read_posix() now clears ICRNL for the duration of the reader (see
its docstring), so Enter reliably arrives as "\r" and Ctrl+J as "\n" --
the same distinction _handle() relies on.

POLLED, NOT BLOCKING
--------------------
msvcrt.getwch() cannot be interrupted or given a timeout, so a thread sitting
inside one can never be asked to stop. kbhit() polling costs a syscall every
POLL_SECONDS and can exit whenever it's told to. At 40 polls a second that is
noise next to the 24fps repaint the panel is already doing.
"""

import ctypes  # cross-platform stdlib module; only ctypes.wintypes and
                # ctypes.WinDLL below are Windows-only, and those stay local
                # to the functions that need them.
import os
import queue
import sys
import threading
import time

from listener import ui

# How often to check for a keystroke. Below ~10ms you are burning CPU to no
# visible effect; above ~50ms typing starts to feel spongy.
POLL_SECONDS = 0.025

# Longest line accepted. Guards against a paste of something enormous -- the
# parser has to send it to the model, and the model has a context limit.
MAX_INPUT_CHARS = 500

HINT = "type a command (shift+enter or ctrl+j for a new line), or say \"hey jarvis\""

# --- control characters ----------------------------------------------------

_CTRL_C = "\x03"
_CTRL_D = "\x04"
_CTRL_J = "\x0a"  # Ctrl+J / linefeed -- the cross-terminal newline shortcut,
                  # see "SHIFT+ENTER, AND WHY CTRL+J IS HERE TOO" above.
_CTRL_L = "\x0c"  # Ctrl+L -- toggles the history overlay (listener/ui.py).
_CTRL_U = "\x15"
_CTRL_W = "\x17"
_BACKSPACE = ("\x08", "\x7f")
# "\n" used to live here too, back when Enter and Ctrl+J were
# indistinguishable on POSIX. Now that _read_posix() clears ICRNL, "\n"
# means an actual Ctrl+J and gets its own branch in _handle() below.
_ENTER = ("\r",)
_ESCAPE = "\x1b"
# Arrow / navigation virtual keys used to scroll the history overlay.
_VK_UP = 0x26
_VK_DOWN = 0x28
_VK_PAGEUP = 0x21
_VK_PAGEDOWN = 0x22

_queue = queue.Queue()
_stop = threading.Event()
_thread = None
_lock = threading.Lock()

_buffer = ""


def _publish():
    ui.set_input(_buffer)


def _submit():
    """Files the current line and clears the prompt."""
    global _buffer
    line = _buffer.strip()
    _buffer = ""
    _publish()
    if line:
        _queue.put(line)


def _backspace():
    global _buffer
    if _buffer:
        _buffer = _buffer[:-1]
        _publish()


def _insert_newline():
    """Shift+Enter. Same length guard as ordinary characters -- a multi-line
    paste shouldn't be able to bypass MAX_INPUT_CHARS just because most of
    its bulk is line breaks."""
    global _buffer
    if len(_buffer) < MAX_INPUT_CHARS:
        _buffer += "\n"
        _publish()


def _kill_word():
    """Ctrl+W. Drops trailing spaces first, then one word."""
    global _buffer
    trimmed = _buffer.rstrip()
    cut = trimmed.rfind(" ")
    _buffer = trimmed[:cut + 1] if cut != -1 else ""
    _publish()


def _clear():
    global _buffer
    if _buffer:
        _buffer = ""
        _publish()


def _interrupt_main():
    """Ctrl+C while the console is in raw mode.

    getwch() takes the keypress as data -- the console never generates the
    signal -- so the wake loop would sit in its blocking mic read forever
    while the user hammers a key that does nothing. Raise it there ourselves.
    """
    try:
        import _thread
        _thread.interrupt_main()
    except Exception:
        # Nothing survivable left to try, but the reader must not die holding
        # the terminal -- fall through and let stop() clean up.
        pass


def _handle(char):
    """One keystroke. Returns False to end the reader."""
    global _buffer

    if char in _ENTER:
        _submit()
        return True

    if char == _CTRL_J:
        _insert_newline()
        return True

    if char == _CTRL_L:
        ui.toggle_history()
        return True

    if char in _BACKSPACE:
        _backspace()
        return True

    if char == _CTRL_C:
        _interrupt_main()
        return False

    if char == _CTRL_D:
        # Only when the line is empty, same as a shell: mid-line it is a
        # forward-delete, which we don't implement, so ignore it there rather
        # than surprising someone into quitting.
        if not _buffer:
            _interrupt_main()
            return False
        return True

    if char == _CTRL_U:
        _clear()
        return True

    if char == _CTRL_W:
        _kill_word()
        return True

    if char == _ESCAPE:
        # Esc first closes the history overlay if it's open, so the muscle
        # memory of "Esc gets back to the normal view" works there too.
        if ui.history_active():
            ui.toggle_history()
        else:
            _clear()
        return True

    # Anything else in the control range is not text -- tabs, bells, stray
    # sequence remnants. Dropping them keeps the line printable, which
    # matters because the panel renders it with a fixed visible width.
    if char < " ":
        return True

    if len(_buffer) < MAX_INPUT_CHARS:
        _buffer += char
        _publish()
    return True


# --- platform readers ------------------------------------------------------

_VK_RETURN = 0x0D
_VK_J = 0x4A
_VK_L = 0x4C
_SHIFT_PRESSED = 0x0010  # dwControlKeyState bit; ignores caps/num lock etc.
_LEFT_CTRL_PRESSED = 0x0008
_RIGHT_CTRL_PRESSED = 0x0004
_CTRL_PRESSED = _LEFT_CTRL_PRESSED | _RIGHT_CTRL_PRESSED
_KEY_EVENT = 0x0001
_STD_INPUT_HANDLE = -10


def _console_input_record_type():
    """Builds the ctypes struct for a Win32 INPUT_RECORD (Windows console
    input event), sized and laid out to match the real one exactly.

    This has to be *exactly* right, not approximately right: ReadConsoleInputW
    fills the buffer using the real OS struct's size, which it gets from
    Windows itself, not from this Python definition -- if this struct came
    out smaller than the real one, the call would write past the end of
    whatever ctypes allocated for it. INPUT_RECORD is a WORD (EventType) then
    a union of five possible event records; only the key-event one is
    modelled here; but KEY_EVENT_RECORD is tied for the largest of the five
    (16 bytes), so a union containing only it still comes out the correct
    total size (20 bytes with the EventType field and its alignment padding),
    even though the other four variants are never read.
    """
    from ctypes import wintypes

    class _KeyEventRecord(ctypes.Structure):
        _fields_ = [
            ("bKeyDown", wintypes.BOOL),
            ("wRepeatCount", wintypes.WORD),
            ("wVirtualKeyCode", wintypes.WORD),
            ("wVirtualScanCode", wintypes.WORD),
            ("UnicodeChar", wintypes.WCHAR),
            ("dwControlKeyState", wintypes.DWORD),
        ]

    class _InputEvent(ctypes.Union):
        _fields_ = [("KeyEvent", _KeyEventRecord)]

    class _InputRecord(ctypes.Structure):
        _fields_ = [("EventType", wintypes.WORD), ("Event", _InputEvent)]

    return _InputRecord


def _read_windows():
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    InputRecord = _console_input_record_type()
    p_record = ctypes.POINTER(InputRecord)

    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.PeekConsoleInputW.restype = wintypes.BOOL
    kernel32.PeekConsoleInputW.argtypes = [
        wintypes.HANDLE, p_record, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.ReadConsoleInputW.restype = wintypes.BOOL
    kernel32.ReadConsoleInputW.argtypes = kernel32.PeekConsoleInputW.argtypes

    handle = kernel32.GetStdHandle(_STD_INPUT_HANDLE)
    record = InputRecord()
    count = wintypes.DWORD(0)

    while not _stop.is_set():
        # Peek first, same reason as the old kbhit() check: ReadConsoleInputW
        # alone blocks with no timeout, which is exactly what made input()
        # unusable here in the first place (see the module docstring).
        if (not kernel32.PeekConsoleInputW(
                handle, ctypes.byref(record), 1, ctypes.byref(count))
                or count.value == 0):
            time.sleep(POLL_SECONDS)
            continue

        if not kernel32.ReadConsoleInputW(
                handle, ctypes.byref(record), 1, ctypes.byref(count)):
            time.sleep(POLL_SECONDS)
            continue

        if record.EventType != _KEY_EVENT:
            continue  # resize, focus, mouse -- not ours

        key = record.Event.KeyEvent
        if not key.bKeyDown:
            continue  # the matching key-up for a press already handled

        if key.wVirtualKeyCode == _VK_RETURN:
            if key.dwControlKeyState & _SHIFT_PRESSED:
                _insert_newline()
            else:
                _submit()
            continue

        # Second, independent path to the same action -- see "SHIFT+ENTER,
        # AND WHY CTRL+J IS HERE TOO" in the module docstring for why
        # Shift+Enter alone isn't always enough under Windows Terminal/VS
        # Code. Checked by virtual key + modifier rather than UnicodeChar
        # so it doesn't depend on the console populating that field for a
        # Ctrl-modified key.
        if (key.wVirtualKeyCode == _VK_J
                and key.dwControlKeyState & _CTRL_PRESSED):
            _insert_newline()
            continue

        # Third pattern of the same kind as the Ctrl+J one above: Ctrl+L
        # toggles the history overlay. Checked by virtual key + modifier so
        # it survives the same ConPTY modifier-drop that Shift+Enter hits.
        if (key.wVirtualKeyCode == _VK_L
                and key.dwControlKeyState & _CTRL_PRESSED):
            ui.toggle_history()
            continue

        char = key.UnicodeChar
        if not char or char == "\x00":
            # A function/navigation/modifier key with no character of its
            # own -- arrows, Home/End, F-keys. The old reader needed a
            # second read to consume these; a raw key event already carries
            # everything in one, so there's nothing left to do but skip
            # them -- unless the history overlay is up, in which case the
            # navigation keys scroll it.
            if ui.history_active():
                vk = key.wVirtualKeyCode
                if vk == _VK_UP:
                    ui.history_scroll(-1)
                elif vk == _VK_DOWN:
                    ui.history_scroll(1)
                elif vk == _VK_PAGEUP:
                    ui.history_scroll(-10)
                elif vk == _VK_PAGEDOWN:
                    ui.history_scroll(10)
            continue

        if not _handle(char):
            return


def _read_posix():
    """Kept so this module imports and works over SSH or WSL.

    No Shift+Enter here -- a plain tty hands over a translated byte stream
    with no modifier-key channel at all (that's what _read_windows() reads
    around by going straight to the Win32 console API instead). Enter
    always submits on this path. Ctrl+J still works for a newline though --
    see "SHIFT+ENTER, AND WHY CTRL+J IS HERE TOO" in the module docstring.

    cbreak rather than full raw: it leaves output post-processing alone, so
    the panel's own writes still behave, while turning off line buffering and
    echo -- which is all we need. tty.setcbreak() doesn't touch input flags
    though, so ICRNL -- on by default on most ttys -- would still translate
    an incoming "\\r" (Enter) into "\\n" before we ever see it, making it
    indistinguishable from an intentional Ctrl+J. Clearing just that one
    input flag below (not going to tty.setraw()'s full raw mode, which would
    also touch output post-processing) keeps Enter as "\\r" and Ctrl+J as
    "\\n" -- exactly the distinction _handle() needs.
    """
    import select
    import termios
    import tty

    def _seq(timeout=0.05):
        """One more byte, or None if nothing arrives within `timeout`. Lets
        an escape sequence be read without blocking forever on a lone Esc."""
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            return None
        return os.read(fd, 1).decode("utf-8", "ignore")

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        mode = termios.tcgetattr(fd)
        mode[0] &= ~termios.ICRNL  # mode[0] is iflag
        termios.tcsetattr(fd, termios.TCSANOW, mode)
        while not _stop.is_set():
            ready, _, _ = select.select([fd], [], [], POLL_SECONDS)
            if not ready:
                continue
            char = os.read(fd, 1).decode("utf-8", "ignore")
            if not char:
                continue

            if char == _CTRL_L:
                ui.toggle_history()
                continue

            if char == _ESCAPE:
                # Escape sequences: arrows come as ESC [ A/B, PageUp/Down as
                # ESC [ 5~/6~. A lone Esc (no continuation in time) falls
                # through to the normal single-char handler.
                nxt = _seq()
                if nxt == "[":
                    third = _seq() or ""
                    if third in "ABCD":
                        if ui.history_active():
                            ui.history_scroll(-1 if third == "A" else 1)
                    elif third in "56":
                        _seq()  # consume the trailing '~'
                        if ui.history_active():
                            ui.history_scroll(-10 if third == "5" else 10)
                    continue
                if nxt is not None:
                    continue  # a multi-byte key we don't act on; drop the rest
                _handle(_ESCAPE)  # a bare Escape key
                continue

            if not _handle(char):
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _run():
    # A daemon thread that raises is silent by construction, and on POSIX one
    # that dies before its finally would leave the terminal in cbreak with no
    # echo -- which outlives the process and makes the user's shell unusable.
    try:
        if os.name == "nt":
            _read_windows()
        else:
            _read_posix()
    except Exception as e:
        print("typed input stopped: {}".format(e))
    finally:
        ui.enable_input(False)


# --- public API ------------------------------------------------------------

def available():
    """Whether a keystroke reader can run here at all.

    Needs a real console: under a pipe, a CI runner or an IDE's output pane
    there are no keystrokes to read, and msvcrt.kbhit() would spin forever
    against a handle that is not a console.
    """
    if os.environ.get("JARVIS_NO_TYPING"):
        return False
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return False
    except Exception:
        return False

    if os.name == "nt":
        try:
            import msvcrt  # noqa: F401
        except ImportError:
            return False
        return True

    try:
        import termios  # noqa: F401
        import tty      # noqa: F401
    except ImportError:
        return False
    return True


def start():
    """Begins reading keystrokes. Returns True if the reader is running.

    Safe to call when it can't work -- it reports False and the assistant
    stays voice-only, with no prompt row drawn.
    """
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return True
        if not available():
            return False

        _stop.clear()
        _thread = threading.Thread(target=_run, daemon=True)
        _thread.start()

    ui.enable_input(True, hint=HINT)
    return True


def stop():
    global _thread
    _stop.set()
    with _lock:
        thread = _thread
        _thread = None
    if thread is not None:
        # Just over one poll interval. The reader checks _stop between polls,
        # so this is the whole cost of shutting it down -- and on POSIX it is
        # what gives the finally time to put the terminal back.
        thread.join(timeout=POLL_SECONDS * 4)
    ui.enable_input(False)


def poll():
    """The next submitted line, or None. Never blocks."""
    try:
        return _queue.get_nowait()
    except queue.Empty:
        return None


def wait(timeout=None):
    """The next submitted line, blocking up to `timeout` seconds."""
    try:
        return _queue.get(timeout=timeout)
    except queue.Empty:
        return None


def drain():
    """Throws away anything typed but not yet picked up.

    Called after a turn: lines typed while Jarvis was talking are almost
    always someone waiting rather than someone queueing three commands, and
    running them all at once afterwards is a surprise nobody wants.
    """
    dropped = 0
    while True:
        try:
            _queue.get_nowait()
            dropped += 1
        except queue.Empty:
            return dropped


if __name__ == "__main__":
    # Type into the panel with no mic and no API. Ctrl+C to quit.
    if not ui.start():
        print("dashboard inactive -- not a terminal, or JARVIS_NO_UI is set")
        raise SystemExit(1)

    try:
        if not start():
            print("typed input unavailable here")
        ui.set_state(ui.LISTENING)
        while True:
            line = poll()
            if line:
                print("submitted: {!r}".format(line))
                ui.set_heard(line)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stop()
        ui.stop()