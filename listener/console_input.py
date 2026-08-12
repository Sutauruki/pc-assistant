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

SHIFT+ENTER
-----------
Adds a line break to the buffer instead of submitting it -- standard
multi-line-input behaviour. This is Windows-only, and has to be: a console
in its normal *cooked* mode translates a keypress to a character before
Python ever sees it, and Shift+Enter translates to exactly the same "\r" a
bare Enter does -- the modifier is thrown away before it reaches getwch() or
anything else reading translated characters. The only way to tell them apart
is to read the *untranslated* key event straight off the Win32 console API
(ReadConsoleInputW), which still carries the virtual key code and the
control-key-state bitmask. That's what _read_windows() does now, in place of
msvcrt.getwch() -- same POLL_SECONDS-interval polling via PeekConsoleInputW
first, same _handle() dispatch for every key that isn't Enter, just a
lower-level source of key data. _read_posix() is unchanged and can't gain
this: a plain POSIX tty has no equivalent channel, so Enter always submits
there. See _read_windows()'s docstring for the struct-layout details.

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

HINT = "type a command, or say \"hey jarvis\""

# --- control characters ----------------------------------------------------

_CTRL_C = "\x03"
_CTRL_D = "\x04"
_CTRL_U = "\x15"
_CTRL_W = "\x17"
_BACKSPACE = ("\x08", "\x7f")
_ENTER = ("\r", "\n")
_ESCAPE = "\x1b"

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
_SHIFT_PRESSED = 0x0010  # dwControlKeyState bit; ignores caps/num lock etc.
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

        char = key.UnicodeChar
        if not char or char == "\x00":
            # A function/navigation/modifier key with no character of its
            # own -- arrows, Home/End, F-keys. The old reader needed a
            # second read to consume these; a raw key event already carries
            # everything in one, so there's nothing left to do but skip it.
            continue

        if not _handle(char):
            return


def _read_posix():
    """Kept so this module imports and works over SSH or WSL.

    No Shift+Enter here -- a plain tty hands over a translated byte stream
    with no modifier-key channel at all (that's what _read_windows() reads
    around by going straight to the Win32 console API instead). Enter always
    submits on this path; there's nothing broken to fix, just a capability
    Windows happens to expose that POSIX ttys don't.

    cbreak rather than full raw: it leaves output post-processing alone, so
    the panel's own writes still behave, while turning off line buffering and
    echo -- which is all we need.
    """
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not _stop.is_set():
            ready, _, _ = select.select([fd], [], [], POLL_SECONDS)
            if not ready:
                continue
            char = os.read(fd, 1).decode("utf-8", "ignore")
            if not char:
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