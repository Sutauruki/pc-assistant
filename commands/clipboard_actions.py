"""Clipboard read and write.

Goes in: commands/clipboard_actions.py   (new file)

pyperclip when it's installed, a direct ctypes path when it isn't. The fallback
is here because pyperclip on Windows is itself a thin wrapper over the same
four user32/kernel32 calls, so reimplementing it is about forty lines and means
one less thing that has to be pip-installed before the feature works.

WHY READING ALOUD IS THE HARD PART
----------------------------------
Writing is trivial. Reading is not, because a clipboard holding four thousand
characters of copied web page will be handed to a text-to-speech model that
caps input at 200 -- so it gets truncated mid-word, and you hear a fragment
with no indication anything was cut. read() therefore returns two things: a
short spoken summary that says how much is there, and the full text, which
goes to the log and the dashboard where you can actually look at it.

Empty and whitespace-only are different answers. "Nothing's on the clipboard"
versus "it's just whitespace" saves you pasting to find out.
"""

import ctypes
import re
from ctypes import wintypes

try:
    import pyperclip as _pyperclip
    _HAVE_PYPERCLIP = True
except Exception:  # pragma: no cover - depends on the machine
    _pyperclip = None
    _HAVE_PYPERCLIP = False

# How much of the clipboard is spoken before it's summarised instead. Below
# Orpheus's 200-char cap so a short clipboard reads out whole.
SPEAK_LIMIT = 180

# Guard against pasting something enormous into a log line.
LOG_LIMIT = 4000

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_user32.OpenClipboard.argtypes = [wintypes.HWND]
_user32.OpenClipboard.restype = wintypes.BOOL
_user32.CloseClipboard.argtypes = []
_user32.CloseClipboard.restype = wintypes.BOOL
_user32.EmptyClipboard.argtypes = []
_user32.EmptyClipboard.restype = wintypes.BOOL
_user32.GetClipboardData.argtypes = [wintypes.UINT]
_user32.GetClipboardData.restype = wintypes.HANDLE
_user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
_user32.SetClipboardData.restype = wintypes.HANDLE
_user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
_user32.IsClipboardFormatAvailable.restype = wintypes.BOOL

_kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
_kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
_kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
_kernel32.GlobalLock.restype = wintypes.LPVOID
_kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
_kernel32.GlobalUnlock.restype = wintypes.BOOL
_kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
_kernel32.GlobalFree.restype = wintypes.HGLOBAL


class _Clipboard:
    """OpenClipboard fails while another process holds it -- and something
    always does, briefly, on a busy desktop. Retrying a handful of times over
    ~200ms turns a spurious failure into a short wait nobody notices."""

    def __enter__(self):
        for _ in range(12):
            if _user32.OpenClipboard(None):
                return self
            ctypes.windll.kernel32.Sleep(20)
        raise OSError("another program is holding the clipboard")

    def __exit__(self, *_exc):
        _user32.CloseClipboard()
        return False


def _native_get():
    with _Clipboard():
        if not _user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return ""
        handle = _user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        pointer = _kernel32.GlobalLock(handle)
        if not pointer:
            return ""
        try:
            return ctypes.c_wchar_p(pointer).value or ""
        finally:
            _kernel32.GlobalUnlock(handle)


def _native_set(text):
    data = ctypes.create_unicode_buffer(text)
    size = ctypes.sizeof(data)

    handle = _kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
    if not handle:
        raise OSError("couldn't allocate clipboard memory")

    pointer = _kernel32.GlobalLock(handle)
    if not pointer:
        _kernel32.GlobalFree(handle)
        raise OSError("couldn't lock clipboard memory")
    ctypes.memmove(pointer, data, size)
    _kernel32.GlobalUnlock(handle)

    try:
        with _Clipboard():
            _user32.EmptyClipboard()
            if not _user32.SetClipboardData(CF_UNICODETEXT, handle):
                raise OSError(ctypes.get_last_error())
        # Ownership passed to the system on success -- freeing it here would
        # hand every later paste a dangling handle.
        handle = None
    finally:
        if handle:
            _kernel32.GlobalFree(handle)


def get_text():
    """Clipboard contents as a string. Raises only if both paths fail."""
    if _HAVE_PYPERCLIP:
        try:
            return _pyperclip.paste() or ""
        except Exception as e:
            print("clipboard: pyperclip read failed ({}), using ctypes".format(e))
    return _native_get()


def set_text(text):
    if _HAVE_PYPERCLIP:
        try:
            _pyperclip.copy(text)
            return
        except Exception as e:
            print("clipboard: pyperclip write failed ({}), using ctypes".format(e))
    _native_set(text)


def _describe(text):
    """A one-line summary of something too long to read out."""
    words = len(text.split())
    lines = text.count("\n") + 1
    shape = "{} words".format(words) if words > 1 else "{} characters".format(len(text))
    if lines > 3:
        shape += " across {} lines".format(lines)

    # A leading fragment is far more useful than a bare count -- it's usually
    # enough to recognise what you copied.
    head = " ".join(text.split())[:90].strip()
    return shape, head


def read():
    """Speaks a summary, logs the whole thing."""
    try:
        text = get_text()
    except Exception as e:
        return "Couldn't read the clipboard: {}".format(e)

    if not text:
        return "There's nothing on the clipboard."
    if not text.strip():
        return "The clipboard only has whitespace on it."

    print("Clipboard ({} chars):".format(len(text)))
    print(text[:LOG_LIMIT] + ("..." if len(text) > LOG_LIMIT else ""))

    flat = " ".join(text.split())
    if len(flat) <= SPEAK_LIMIT:
        return "Clipboard says: {}".format(flat)

    shape, head = _describe(text)
    return "The clipboard has {}, starting: {}...".format(shape, head)


def write(text):
    if text is None:
        return "I didn't catch what to copy."

    text = str(text)
    if not text:
        return "I didn't catch what to copy."

    try:
        set_text(text)
    except Exception as e:
        return "Couldn't write to the clipboard: {}".format(e)

    flat = " ".join(text.split())
    shown = flat if len(flat) <= 60 else flat[:57] + "..."
    return "Copied: {}".format(shown)


def clear():
    try:
        set_text("")
        return "Cleared the clipboard"
    except Exception as e:
        return "Couldn't clear the clipboard: {}".format(e)


def append(text):
    """Adds to what's already there rather than replacing it."""
    if not text:
        return "I didn't catch what to add."
    try:
        existing = get_text()
    except Exception as e:
        return "Couldn't read the clipboard: {}".format(e)

    joiner = "" if not existing or existing.endswith(("\n", " ")) else "\n"
    return write(existing + joiner + str(text))


_URL_RE = re.compile(r"https?://\S+")


def open_link():
    """Opens a URL sitting on the clipboard.

    Small feature, disproportionately useful -- copying a link and saying "open
    that" is faster than finding the window you copied it from.
    """
    try:
        text = get_text()
    except Exception as e:
        return "Couldn't read the clipboard: {}".format(e)

    match = _URL_RE.search(text or "")
    if not match:
        return "There's no link on the clipboard."

    url = match.group(0).rstrip(').,"\'')
    try:
        import webbrowser
        webbrowser.open(url)
        return "Opening {}".format(url[:60])
    except Exception as e:
        return "Couldn't open that link: {}".format(e)


if __name__ == "__main__":
    # python -m commands.clipboard_actions
    print("pyperclip available:", _HAVE_PYPERCLIP)
    print(read())
    original = ""
    try:
        original = get_text()
    except Exception:
        pass
    print(write("jarvis clipboard test"))
    print(read())
    if original:
        set_text(original)
        print("(restored what was there before)")