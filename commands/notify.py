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

WHY IT NEVER RUNS ON THE CALLING THREAD
---------------------------------------
Every backend here either blocks for the toast duration or talks to WinRT,
which wants an apartment-threaded COM context. Both are reasons to keep it off
the wake loop, so the actual show happens on a short-lived daemon thread and
the caller gets its result string immediately.
"""

import os
import subprocess
import threading

APP_NAME = "Jarvis"

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
    print(notify("Jarvis", "Toast test -- if you can see this, it works.",
                 blocking=True))