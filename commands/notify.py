"""Toast notifications.

Goes in: commands/notify.py   (new file)

A note on the tool you named, because I did not use it as the primary path.

win10toast is effectively unmaintained. Its last release predates Python 3.10,
and the failure mode people hit is `WNDPROC return value cannot be converted to
LRESULT` -- ctypes got stricter about the window procedure's return type and
the library never adjusted. It also spins up its own hidden window and, in the
common non-threaded call, blocks for the full duration of the toast, which on
the wake loop's thread would freeze the assistant for five seconds per
notification.

So the design is a chain rather than a dependency, tried in order:

    1. win11toast       -- actively maintained, real WinRT toasts, non-blocking
    2. windows_toasts   -- same idea, different author, also maintained
    3. win10toast       -- YOUR pick, kept in place, with the LRESULT crash caught
    4. plyer            -- cross-platform, already common in Python installs
    5. PowerShell       -- no dependency at all, raw WinRT via the shell

Whatever is installed gets used. If nothing is, the PowerShell path still puts
a real toast on screen, so `pip install` is optional for this feature. The
chosen backend is decided once and remembered, because probing five of them per
notification would be slower than the notification.

    pip install win11toast        <- the one I'd recommend

confirm() builds on win11toast specifically -- it is the only backend that can
hand a button click or typed answer back to the process (see "Interactive
confirm" below). Everywhere else it degrades to a static toast that points the
user at the terminal, where the confirmation stays answerable the normal way.

WHY IT NEVER RUNS ON THE CALLING THREAD
---------------------------------------
Every backend here either blocks for the toast duration or talks to WinRT,
which wants an apartment-threaded COM context. Both are reasons to keep it off
the wake loop, so the actual show happens on a short-lived daemon thread and
the caller gets its result string immediately.
"""

import asyncio
import os
import subprocess
import threading

from commands import identity

APP_NAME = identity.NAME

# Windows truncates well before this, but trimming ourselves means the cut
# lands on a word rather than mid-syllable.
MAX_TITLE = 64
MAX_BODY = 200

DEFAULT_DURATION = "short"   # "short" ~5s, "long" ~25s

_backend = None
_backend_lock = threading.Lock()


def _trim(text, limit):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut if len(cut) > limit * 0.6 else text[:limit]).rstrip() + "..."


# --- Backends --------------------------------------------------------------
#
# Each returns True on success. None of them raise -- _pick() decides which one
# works by calling it, so an exception here would just be caught and discarded.

def _try_win11toast(title, body, duration):
    from win11toast import notify  # noqa: PLC0415 - probed lazily on purpose
    notify(title, body, app_id=APP_NAME, duration=duration)
    return True


def _try_windows_toasts(title, body, duration):
    import windows_toasts  # noqa: PLC0415

    toaster = windows_toasts.WindowsToaster(APP_NAME)
    # The class was renamed between 1.x and 2.x; accept either.
    factory = getattr(windows_toasts, "Toast", None) or windows_toasts.ToastText2
    toast = factory()
    try:
        toast.text_fields = [title, body]
    except AttributeError:
        toast.SetHeadline(title)
        toast.SetBody(body)

    duration_enum = getattr(windows_toasts, "ToastDuration", None)
    if duration_enum is not None:
        toast.duration = (duration_enum.Long if duration == "long"
                          else duration_enum.Short)

    toaster.show_toast(toast)
    return True


def _try_win10toast(title, body, duration):
    from win10toast import ToastNotifier  # noqa: PLC0415

    # threaded=True matters twice over: it keeps the 5-second toast off this
    # thread, and it is the configuration the library is least broken in.
    ToastNotifier().show_toast(
        title,
        body,
        duration=25 if duration == "long" else 5,
        threaded=True,
    )
    return True


def _try_plyer(title, body, duration):
    from plyer import notification  # noqa: PLC0415

    notification.notify(
        title=title,
        message=body,
        app_name=APP_NAME,
        timeout=25 if duration == "long" else 5,
    )
    return True


_PS_TEMPLATE = """
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType=WindowsRuntime] > $null

$template = @"
<toast duration="{duration}">
  <visual><binding template="ToastText02">
    <text id="1">{title}</text>
    <text id="2">{body}</text>
  </binding></visual>
</toast>
"@

$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($template)
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("{app}").Show($toast)
"""


def _xml_escape(text):
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _try_powershell(title, body, duration):
    """No pip install required. Slowest of the five -- PowerShell startup is
    roughly half a second -- but it is the reason this feature works on a
    machine with nothing installed."""
    script = _PS_TEMPLATE.format(
        title=_xml_escape(title),
        body=_xml_escape(body),
        duration="long" if duration == "long" else "short",
        app=_xml_escape(APP_NAME),
    )

    flags = 0
    startupinfo = None
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive",
         "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=flags,
        startupinfo=startupinfo,
    )
    if result.returncode != 0:
        raise OSError((result.stderr or "powershell failed").strip()[:200])
    return True


_BACKENDS = (
    ("win11toast", _try_win11toast),
    ("windows_toasts", _try_windows_toasts),
    ("win10toast", _try_win10toast),
    ("plyer", _try_plyer),
    ("powershell", _try_powershell),
)


def _show(title, body, duration):
    """Runs the chain, remembering whichever backend worked."""
    global _backend

    with _backend_lock:
        chosen = _backend

    if chosen is not None:
        name, fn = chosen
        try:
            return fn(title, body, duration), name
        except Exception as e:
            # A backend that worked once and then broke -- an unplugged
            # notification service, a package upgraded underneath us. Forget it
            # and fall through to a fresh probe rather than failing forever.
            print("notify: {} stopped working ({}), re-probing".format(name, e))
            with _backend_lock:
                _backend = None

    errors = []
    for name, fn in _BACKENDS:
        try:
            fn(title, body, duration)
        except ImportError:
            continue  # not installed; not worth reporting
        except Exception as e:
            errors.append("{}: {}".format(name, str(e).splitlines()[0][:120]))
            continue
        with _backend_lock:
            _backend = (name, fn)
        print("notify: using {}".format(name))
        return True, name

    if errors:
        print("notify: every backend failed --")
        for line in errors:
            print("  " + line)
    return False, None


def notify(title, message=None, duration=DEFAULT_DURATION, blocking=False):
    """Puts a toast on screen. Returns a result string.

    Returns before the toast is shown by default. The caller is a voice turn;
    making it wait on a shell round trip so it can report a failure nobody
    asked about is the wrong trade.
    """
    title = _trim(title or APP_NAME, MAX_TITLE)
    body = _trim(message or "", MAX_BODY)

    if not title and not body:
        return "I didn't catch what to put in the notification."

    duration = "long" if str(duration).lower().startswith("l") else "short"

    if blocking:
        ok, name = _show(title, body, duration)
        if not ok:
            return "Couldn't show a notification -- no working toast backend."
        print("notify [{}]: {} -- {}".format(name, title, body))
        return "Notification shown"

    def run():
        ok, name = _show(title, body, duration)
        if ok:
            print("notify [{}]: {} -- {}".format(name, title, body))
        else:
            print("notify: couldn't show \"{}\"".format(title))

    threading.Thread(target=run, daemon=True).start()
    return "Notification sent"


# --- Interactive confirm ---------------------------------------------------
#
# A confirmation toast with quick-reply controls, WhatsApp style: [Yes]/[No]
# buttons plus a text box you can type a full answer into. Only one backend
# (win11toast) can hand an answer back in-process, so anything else falls back
# to a plain long toast that points the user at the terminal/voice -- where
# the confirmation is still pending and answerable the normal way. The win11
# path registers a background activation handler that runs on_answered(text)
# on a background thread when the toast is answered.

# Same app id as every other Jarvis toast. A made-up id ("Jarvis.Confirmation")
# that was never registered is one Windows can silently refuse to show.
_INTERACTIVE_APP_ID = APP_NAME


def _can_encode(text):
    """True if `text` can go into toast XML: valid Unicode with no control
    characters besides newline and tab. Anything else falls back to the plain
    toast. (This was called but never defined -- the NameError took the whole
    turn down and, with it, the pending confirmation.)"""
    try:
        str(text).encode("utf-16", "strict")
    except UnicodeError:
        return False
    return all(ch in "\n\t" or ord(ch) >= 32 for ch in str(text))


def confirm(title, question, on_answered):
    """Shows an answerable confirmation notification.

    question:   the full question, e.g. "Should I permanently delete Downloads/
                temp.txt?"
    on_answered: called with the answer text (a button click or whatever was
                 typed into the toast) on a background thread. May be called
                 more than once if a self-similar toast is answered twice.

    Returns a result string. Non-blocking either way.
    """
    if _interactive_confirm(title, question, on_answered):
        return "Confirmation shown"

    notify(
        title,
        question + "  (reply by typing in the terminal, or say your answer.)",
        duration="long",
    )
    return "Confirmation shown"


def _interactive_confirm(title, question, on_answered):
    """Returns True if an interactive toast is on screen and will deliver the
    answer. False means there's no backend that can -- caller shows a static
    toast instead."""
    try:
        import win11toast  # noqa: PLC0415 - probed lazily on purpose
    except ImportError:
        return False
    if not _can_encode(question):
        return False

    def runner():
        try:
            _win11toast_driver(title, question, on_answered)
        except Exception as e:
            print("notify.confirm: interactive toast failed ({})".format(e))

    threading.Thread(target=runner, daemon=True).start()
    return True


def _win11toast_driver(title, question, on_answered):
    """win11toast's toast() is a coroutine that waits on activation, so it gets
    its own event loop on the worker thread. The click callback fires on that
    loop; we forward straight to on_answered."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_win11toast_ask(title, question, on_answered))
    finally:
        loop.close()


async def _win11toast_ask(title, question, on_answered):
    import win11toast  # noqa: PLC0415

    def on_click(args):
        # Prefer free text from the input box; buttons fall back to the label.
        user = args.get("user_input") or {}
        raw = ""
        for value in user.values():
            if value:
                raw = str(value).strip()
                break
        if not raw:
            argument = args.get("arguments", "").lower()
            if "yes" in argument:
                raw = "yes"
            elif "no" in argument or "cancel" in argument:
                raw = "no"
        if raw:
            on_answered(raw)

    await win11toast.toast(
        title,
        question,
        on_click=on_click,
        input="reply",
        buttons=["Yes", "No"],
        duration="long",
        app_id=_INTERACTIVE_APP_ID,
    )


def available():
    """Names of the backends importable on this machine. Diagnostic only."""
    found = []
    for name, _fn in _BACKENDS:
        if name == "powershell":
            found.append(name)
            continue
        try:
            __import__(name)
            found.append(name)
        except Exception:
            pass
    return found


if __name__ == "__main__":
    # python -m commands.notify
    print("importable backends:", ", ".join(available()) or "(none)")
    print(notify(APP_NAME, "Toast test -- if you can see this, it works.",
                 blocking=True))