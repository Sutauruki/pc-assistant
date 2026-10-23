"""What Jarvis may look at, and what it may touch -- the security policy.

Goes in: commands/pc_policy.py   (new file)

Everything that sees the screen (commands/screen_actions.py) or drives the
mouse and keyboard (commands/agent.py) asks this module first. The rules live
here, in plain Python with no Windows calls, so they are small enough to review
in one sitting and testable anywhere. Nothing in here is decided by a model:
a model can propose an action, only this file can allow it.

THE FOUR QUESTIONS
------------------
sensitive_reason(window)  -> never screenshot it, never send it anywhere.
                             Password managers, banking, anything listed in
                             JARVIS_SENSITIVE.
blocked_reason(window)    -> may be looked at, never clicked or typed into.
                             Shells, the Run dialog, registry, Task Manager,
                             Windows Security, Settings, installers, UAC and
                             credential prompts -- the places one bad keystroke
                             turns into a compromised machine.
risky_reason(...)         -> allowed only after a PHYSICAL keypress from you.
                             Send, post, submit, pay, delete, install, download,
                             share, and Enter in a chat box (that is "send" on
                             WhatsApp and most messengers).
text/key/url checks       -> things no task ever needs: typing your API keys,
                             Win-key shortcuts, file:// or javascript: URLs.

Windows are matched on the PROCESS (from the OS), not just the title, so a web
page titled "Notepad" can't make a terminal look harmless, and a terminal can't
be renamed into something that isn't blocked.
"""

import os
import re

# --- Never look -------------------------------------------------------------

# Process names (lower-case, no path) whose windows are never captured.
SENSITIVE_PROCESSES = {
    "1password.exe", "bitwarden.exe", "keepass.exe", "keepassxc.exe",
    "lastpass.exe", "dashlane.exe", "keeper.exe", "keeperpasswordmanager.exe",
    "nordpass.exe", "protonpass.exe", "roboform.exe", "enpass.exe",
    "credentialuibroker.exe", "consent.exe", "logonui.exe",
}

# Words that, in a window title, mean "this is private". Matched as whole
# words, case-insensitively. JARVIS_SENSITIVE adds your own (comma-separated):
# your bank's name, your broker, your health portal.
SENSITIVE_TITLE_WORDS = {
    "bank", "banking", "1password", "bitwarden", "keepass", "keepassxc",
    "lastpass", "dashlane", "nordpass", "proton pass", "roboform", "enpass",
    "password manager", "passwords", "online banking", "paypal", "revolut",
    "crypto wallet", "metamask", "seed phrase", "recovery phrase",
    "private key", "authenticator",
}


def _extra_sensitive():
    raw = os.getenv("JARVIS_SENSITIVE", "")
    return {w.strip().lower() for w in raw.split(",") if w.strip()}


def _has_word(text, word):
    return re.search(r"(?<![a-z0-9]){}(?![a-z0-9])".format(re.escape(word)),
                     text) is not None


def sensitive_reason(window):
    """Why this window must never be captured, or None.

    `window` is a dict with at least "process" and "title" (as from
    commands/desktop.foreground()). Missing information counts as sensitive:
    if we can't tell what it is, we don't send a picture of it anywhere.
    """
    if not window:
        return "I couldn't tell which window is in front"
    process = (window.get("process") or "").lower()
    title = (window.get("title") or "").lower()
    if process in SENSITIVE_PROCESSES:
        return "{} is a private app".format(window.get("process"))
    for word in SENSITIVE_TITLE_WORDS | _extra_sensitive():
        if _has_word(title, word) or _has_word(process, word):
            return "that window looks private ({})".format(word)
    return None


# --- Never touch ------------------------------------------------------------

BLOCKED_PROCESSES = {
    # shells and terminals -- including the one Jarvis itself runs in
    "cmd.exe", "powershell.exe", "pwsh.exe", "powershell_ise.exe",
    "windowsterminal.exe", "openconsole.exe", "conhost.exe", "wsl.exe",
    "bash.exe", "wt.exe", "mintty.exe", "alacritty.exe", "wezterm-gui.exe",
    # system administration
    "regedit.exe", "mmc.exe", "taskmgr.exe", "systemsettings.exe",
    "sechealthui.exe", "msconfig.exe", "control.exe", "services.exe",
    "gpedit.exe", "secpol.exe", "compmgmt.exe", "eventvwr.exe",
    "resmon.exe", "perfmon.exe", "diskmgmt.exe", "netplwiz.exe",
    "rstrui.exe", "bcdedit.exe", "certmgr.exe",
    # script hosts and installers
    "wscript.exe", "cscript.exe", "mshta.exe", "msiexec.exe",
    "rundll32.exe", "regsvr32.exe", "installer.exe", "setup.exe",
    "appinstaller.exe",
} | SENSITIVE_PROCESSES

# Browsers: their window title is whatever the page says, so title words
# below are not applied to them -- a tab about "PowerShell" isn't a shell.
# Real shells are caught by process name above, which a page can't change.
BROWSERS = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe",
            "opera.exe", "vivaldi.exe", "arc.exe"}

# Title words that mark an installer or system dialog even when the process
# is something generic.
BLOCKED_TITLE_WORDS = {
    "user account control", "windows security", "command prompt",
    "powershell", "terminal", "registry editor", "task manager",
    "task scheduler", "group policy", "setup wizard", "install wizard",
}


def blocked_reason(window):
    """Why the agent may not click or type in this window, or None."""
    if not window:
        return "I couldn't tell which window is in front"
    process = (window.get("process") or "").lower()
    title = (window.get("title") or "").lower()
    cls = window.get("class") or ""
    if not process:
        return "I couldn't tell which app that window belongs to"
    if process in BLOCKED_PROCESSES:
        return "{} is a protected system app".format(window.get("process"))
    # The Run dialog belongs to explorer.exe, which can't be blocked as a
    # whole without blocking File Explorer, so it is matched on its own.
    if process == "explorer.exe" and cls == "#32770" and title.strip() == "run":
        return "the Run dialog is protected"
    if process in BROWSERS:
        return None
    for word in BLOCKED_TITLE_WORDS:
        if _has_word(title, word):
            return "that window looks like a protected system tool ({})".format(word)
    if re.search(r"\b(setup|installer|install)\b.*\.(exe|msi)\b", title):
        return "that looks like an installer"
    return None


_BLOCKED_APP_NAMES = (
    "cmd", "command prompt", "powershell", "pwsh", "terminal", "regedit",
    "registry", "task manager", "taskmgr", "task scheduler", "settings",
    "windows security", "defender", "control panel", "msconfig",
    "group policy", "gpedit", "services", "event viewer", "wsl", "bash",
    "run dialog", "mmc", "system configuration", "uac",
)


def blocked_app_name(name):
    """Why open_app may not launch `name`, or None. Spoken names, so this
    matches loosely: "power shell", "cmd", "the registry editor"."""
    raw = str(name or "")
    if re.search(r"[\\/:]", raw) or re.search(
            r"\.(exe|bat|cmd|com|ps1|psm1|vbs|vbe|js|jse|wsf|wsh|msi|msp|hta|"
            r"scr|lnk|reg|cpl|jar|py|pyw)\b", raw.lower()):
        return "I only open installed apps by name, not files or paths"
    spoken = " ".join(raw.lower().replace("-", " ").split())
    squashed = spoken.replace(" ", "")
    for bad in _BLOCKED_APP_NAMES:
        if _has_word(spoken, bad) or bad.replace(" ", "") in squashed.split():
            return "{} is a protected system app".format(name)
        if squashed == bad.replace(" ", ""):
            return "{} is a protected system app".format(name)
    return None


_RUNNABLE = (".bat", ".cmd", ".com", ".ps1", ".vbs", ".vbe", ".js", ".jse",
             ".wsf", ".wsh", ".msi", ".msp", ".hta", ".scr", ".reg", ".cpl")


def blocked_app_entry(name, path):
    """Why the app a spoken name RESOLVED to may not be launched, or None.
    Checked after the catalog lookup, because fuzzy matching can turn a
    harmless-sounding name into a shell or a built-in admin tool."""
    target = re.split(r"[\\/]", str(path or "").strip().strip('"'))[-1].lower()
    if target in BLOCKED_PROCESSES or target.endswith(_RUNNABLE):
        return "{} is a protected system app".format(name or target)
    return blocked_app_name(name)


# --- Needs your keypress ----------------------------------------------------

# Words on a button or link that mean "this changes something you can't take
# back". Matched as whole words in the element's name.
RISKY_WORDS = (
    "send", "post", "publish", "submit", "pay", "buy", "purchase", "order",
    "checkout", "check out", "place order", "confirm", "delete", "remove",
    "uninstall", "install", "download", "run", "share", "transfer", "tweet",
    "reply", "comment", "upload", "attach", "accept", "allow", "grant",
    "approve", "authorize", "authorise", "sign", "agree", "subscribe",
    "unsubscribe", "donate", "book", "reserve", "apply", "save password",
    "forward", "invite", "follow", "unfollow", "block", "report", "leave",
    "empty trash", "empty recycle bin", "factory reset", "sign out", "log out",
    "call", "video call", "voice call",
)

# A focused box whose name says it's for searching or addresses: Enter there
# navigates or searches, it doesn't send anything to anyone.
_SAFE_ENTER_FIELDS = ("search", "address", "url", "find", "filter", "go to",
                      "location", "omnibox")


def _risky_word(name):
    lowered = (name or "").lower()
    for word in RISKY_WORDS:
        if _has_word(lowered, word):
            return word
    return None


def risky_reason(kind, element=None, keys=None, focused=None, model_flag=False):
    """Why this step needs your physical OK, or None.

    kind:       the agent action ("click", "press", "type", ...)
    element:    the target element dict (name, type), if any
    keys:       for "press", the normalised key combo ("enter", "ctrl+enter")
    focused:    the element that has keyboard focus right now, if known
    model_flag: the model's own opinion -- it can ADD a confirmation, never
                remove one this function asks for
    """
    if kind in ("click", "double_click", "right_click", "click_xy"):
        if element and _risky_word(element.get("name")):
            return 'clicking "{}"'.format(element.get("name"))
        if kind == "click_xy":
            # A raw coordinate has no name to check, so it can't be shown to
            # be harmless.
            return "clicking a spot I can't identify by name"
    if kind == "press" and keys:
        combo = normalise_keys(keys)
        if combo.split("+")[-1] == "enter":
            field = ((focused or {}).get("name") or "").lower()
            ftype = ((focused or {}).get("type") or "").lower()
            if not any(w in field for w in _SAFE_ENTER_FIELDS) or ftype == "document":
                return "pressing Enter here may send or submit"
        if combo in ("delete", "shift+delete"):
            return "pressing Delete"
    if model_flag:
        return "this step changes something"
    return None


# --- Things no task needs ---------------------------------------------------

# Keys and combos the agent may press. Anything else is refused -- notably the
# Windows key (Win+R is the Run dialog, Win+X the admin menu), Alt+F4 on the
# wrong window, and Ctrl+Shift+Esc (Task Manager).
ALLOWED_KEYS = {
    "enter", "tab", "shift+tab", "escape", "backspace", "delete", "space",
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
    "ctrl+a", "ctrl+c", "ctrl+v", "ctrl+x", "ctrl+z", "ctrl+y", "ctrl+f",
    "ctrl+l", "ctrl+t", "ctrl+w", "ctrl+tab", "ctrl+shift+tab", "ctrl+enter",
    "ctrl+home", "ctrl+end", "ctrl+backspace", "ctrl+left", "ctrl+right",
    "ctrl+plus", "ctrl+minus", "ctrl+0", "ctrl+r", "f5", "f3", "f6",
    "alt+left", "alt+right", "shift+enter", "shift+up", "shift+down",
    "shift+left", "shift+right", "shift+home", "shift+end",
}

_KEY_ALIASES = {"return": "enter", "esc": "escape", "control": "ctrl",
                "del": "delete", "pgup": "pageup", "pgdn": "pagedown",
                "page up": "pageup", "page down": "pagedown", "=": "plus",
                "-": "minus", "windows": "win", "super": "win", "cmd": "win",
                "meta": "win", "option": "alt"}


def normalise_keys(keys):
    """'Ctrl + L' -> 'ctrl+l'. Modifiers first, in a fixed order."""
    raw = str(keys or "").lower().replace(" + ", "+")
    parts = [_KEY_ALIASES.get(p.strip(), p.strip()) for p in raw.split("+")]
    parts = [p for p in parts if p]
    mods = [m for m in ("ctrl", "shift", "alt", "win") if m in parts]
    rest = [p for p in parts if p not in ("ctrl", "shift", "alt", "win")]
    return "+".join(mods + rest)


def key_reason(keys):
    """Why this key combo is refused, or None."""
    combo = normalise_keys(keys)
    if "win" in combo.split("+"):
        return "Windows-key shortcuts are off limits"
    if combo not in ALLOWED_KEYS:
        return "{} isn't a key I'm allowed to press".format(combo or "that")
    return None


_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD)$", re.I)


def _secrets():
    values = []
    for name, value in os.environ.items():
        if _SECRET_ENV.search(name) and value and len(value.strip()) >= 8:
            values.append(value.strip())
    return values


def text_reason(text):
    """Why the agent may not type this text, or None."""
    text = str(text or "")
    if len(text) > 4000:
        return "that's too much text to type in one go"
    for secret in _secrets():
        if secret in text:
            return "that text contains one of your API keys or passwords"
    lowered = text.lower()
    if re.search(r"(^|[\s;&|])(powershell|pwsh|cmd)(\.exe)?\s+[-/]", lowered):
        return "that text looks like a shell command"
    return None


_LOCKED_URL = ("file:", "javascript:", "data:", "vbscript:", "chrome:",
               "edge:", "about:", "ms-settings:", "search-ms:", "shell:")
_LOCAL_HOST = re.compile(
    r"^(localhost|127\.|0\.0\.0\.0|\[?::1\]?|10\.|192\.168\.|169\.254\.|"
    r"172\.(1[6-9]|2\d|3[01])\.)", re.I)


def url_reason(url):
    """Why open_url may not open this, or None. http(s) to public hosts only:
    no local files, no scripts, no settings deep links, no LAN devices."""
    url = str(url or "").strip()
    lowered = url.lower()
    if lowered.startswith(_LOCKED_URL):
        return "that kind of link is off limits"
    match = re.match(r"^https?://([^/\s?#]+)", lowered)
    if not match:
        return "I only open http or https web addresses"
    authority = match.group(1)
    if "@" in authority:
        return "that address hides its real destination"
    host = authority.split(":")[0] if not authority.startswith("[") else authority
    if _LOCAL_HOST.match(host):
        return "I don't open addresses on this computer or your local network"
    return None


def mask(text, keep=40):
    """For the audit log: enough to recognise, never the whole thing, and
    nothing at all if it contains a secret."""
    text = str(text or "")
    if any(s in text for s in _secrets()):
        return "<{} chars, contains a secret>".format(len(text))
    if len(text) <= keep:
        return text
    return "{}... <{} chars>".format(text[:keep], len(text))
