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

POLLED, NOT BLOCKING
--------------------
msvcrt.getwch() cannot be interrupted or given a timeout, so a thread sitting
inside one can never be asked to stop. kbhit() polling costs a syscall every
POLL_SECONDS and can exit whenever it's told to. At 40 polls a second that is
noise next to the 24fps repaint the panel is already doing.
"""

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

# Windows sends arrows, function keys and the like as two reads: a null or
# 0xE0 lead byte, then a code. Both halves have to be consumed or the second
# one lands in the line as a stray letter.
_WIN_PREFIXES = ("\x00", "\xe0")

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

def _read_windows():
    import msvcrt

    while not _stop.is_set():
        if not msvcrt.kbhit():
            time.sleep(POLL_SECONDS)
            continue

        char = msvcrt.getwch()

        if char in _WIN_PREFIXES:
            # Second half of an extended key. Consume and discard -- arrows
            # and Home/End would need cursor tracking the panel doesn't do.
            if msvcrt.kbhit():
                msvcrt.getwch()
            continue

        if not _handle(char):
            return


def _read_posix():
    """Kept so this module imports and works over SSH or WSL.

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