"""System-wide app catalog -- find, launch and close arbitrary applications.

Goes in: commands/app_index.py   (new file)

Replaces the hand-maintained APP_PATHS / PROCESS_NAMES dicts with a catalog
built from what is actually installed on this machine.

Three sources, merged:

  1. Start Menu .lnk shortcuts -- the shortcut name is what a person actually
     says out loud, and it covers essentially every Win32 app.
  2. Get-StartApps -- name + AppID. Covers Store/UWP apps, which have no
     shortcut and no directly launchable .exe.
  3. App Paths registry -- the canonical launch commands Windows itself
     resolves from the Run box, so "chrome", "msedge", "code" work as typed.

A full disk scan for .exe files was deliberately not used: it takes tens of
seconds and fills the catalog with uninstallers, crash handlers and updaters
that are worse than useless as voice targets.

Sources 1 and 3 are pure Python -- os.walk plus winreg -- and take a few tens
of milliseconds. The first cut of this used PowerShell and WScript.Shell COM
to read each shortcut's TargetPath, and it blew a 45 second timeout: resolving
a .lnk through the shell tries to reach its target, so a single shortcut
pointing at a disconnected network share or an unplugged drive stalls on an
SMB timeout, and a Start Menu has hundreds of shortcuts. The .lnk binary
format carries the target path inline, so we read the bytes ourselves instead
and never touch the filesystem the shortcut points at.

Only Get-StartApps still needs PowerShell, and a failure there is non-fatal --
it costs the Store apps, not the catalog.

The scan runs once, caches to JSON, and reloads in the background at startup.
Resolution at request time is a dict lookup plus fuzzy matching over a few
hundred short strings -- well under a millisecond, so it never lands on the
latency path.

Closing is the harder half. Friendly names are not process names (Calculator
runs as CalculatorApp.exe, VS Code as Code.exe, a Store app's window belongs
to ApplicationFrameHost.exe), so close() tries four strategies in order of
confidence and refuses to kill anything it isn't reasonably sure about.
"""

import base64
import ctypes
import difflib
import json
import os
import re
import struct
import subprocess
import threading
import time
import winreg

from ctypes import wintypes

import psutil

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "app_index.json")
CACHE_MAX_AGE_DAYS = 7

# How long a command will wait for a cold catalog before giving up. The
# catalog is warmed at startup, so in practice this only matters if the very
# first spoken command lands within a second or two of launch.
CATALOG_WAIT_SECONDS = 6.0

SCAN_TIMEOUT_SECONDS = 20

# Shortcuts deeper than this are almost always vendor sub-folders of extras
# rather than apps, and every level costs a directory walk.
MAX_LNK_DEPTH = 5

# Below this, report no match instead of launching the nearest string. Getting
# this wrong is cheap but annoying -- "open discord" should not launch Disk
# Cleanup because it was the closest thing installed.
MATCH_FLOOR = 0.62

# Killing the wrong process is much worse than launching the wrong app, so
# the bar for matching a *running* process is higher.
PROC_MATCH_FLOOR = 0.74

# After launching we watch the process table for a few seconds to learn which
# PID the app actually became. ShellExecute doesn't tell us, and UWP apps are
# started by explorer rather than by us, so there's nothing to return.
ADOPT_WINDOW_SECONDS = 4.0
ADOPT_POLL_SECONDS = 0.4

# Ask windows to close before terminating, so editors and browsers get to save
# their state. Set False for instant, unconditional termination.
GRACEFUL_CLOSE = True
GRACEFUL_TIMEOUT = 1.2

# How long to wait after terminate() before escalating to kill(), and after
# kill() before accepting that the process is beyond us. Both are polled, so
# they cost nothing when the process dies immediately -- which is the norm.
ESCALATE_TIMEOUT = 1.0
KILL_TIMEOUT = 0.5
DEATH_POLL_SECONDS = 0.05

CREATE_NO_WINDOW = 0x08000000
WM_CLOSE = 0x0010

# Spoken shorthand -> what the catalog is likely to call it. Only for cases
# where fuzzy matching genuinely can't get there ("vs code" shares almost no
# characters with "Visual Studio Code" once normalized).
ALIASES = {
    "vs code": "visual studio code",
    "vscode": "visual studio code",
    "code": "visual studio code",
    "chrome": "google chrome",
    "browser": "google chrome",
    "edge": "microsoft edge",
    "word": "microsoft word",
    "excel": "microsoft excel",
    "powerpoint": "microsoft powerpoint",
    "outlook": "microsoft outlook",
    "teams": "microsoft teams",
    "explorer": "file explorer",
    "files": "file explorer",
    "file manager": "file explorer",
    "my computer": "file explorer",
    "cmd": "command prompt",
    "command line": "command prompt",
    "terminal": "windows terminal",
    "calc": "calculator",
    "photoshop": "adobe photoshop",
    "premiere": "adobe premiere pro",
    "vlc": "vlc media player",
    "task manager": "task manager",
    "settings": "settings",
    "notepad plus plus": "notepad++",
    "notepad pp": "notepad++",
    "sublime": "sublime text",
    "intellij": "intellij idea",
    "postman": "postman",
    "docker": "docker desktop",
}

# Last-resort launch commands. These are built into Windows and always work,
# so if the catalog somehow missed one we can still honour the request.
FALLBACK_COMMANDS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "file explorer": "explorer.exe",
    "task manager": "taskmgr.exe",
    "control panel": "control.exe",
    "command prompt": "cmd.exe",
    "windows powershell": "powershell.exe",
    "registry editor": "regedit.exe",
    "character map": "charmap.exe",
    "snipping tool": "snippingtool.exe",
}

# Start Menu folders are full of things that are not the app. Launching an
# uninstaller by voice would be a genuinely bad day.
_JUNK_TOKENS = (
    "uninstall", "readme", "read me", "release notes", "documentation",
    "website", "web site", "changelog", "change log", "license", "licence",
    "manual", "troubleshoot", "repair", "installer", "crash report",
    "safe mode", "verbose logging",
)

# Never terminate these, whatever the match score says.
_PROTECTED = {
    "system", "system idle process", "registry", "memory compression",
    "smss.exe", "csrss.exe", "wininit.exe", "winlogon.exe", "services.exe",
    "lsass.exe", "svchost.exe", "explorer.exe", "dwm.exe", "fontdrvhost.exe",
    "audiodg.exe", "spoolsv.exe", "conhost.exe", "ctfmon.exe", "sihost.exe",
    "taskhostw.exe", "dllhost.exe", "runtimebroker.exe", "searchhost.exe",
    "searchapp.exe", "searchindexer.exe", "startmenuexperiencehost.exe",
    "shellexperiencehost.exe", "textinputhost.exe", "applicationframehost.exe",
    "backgroundtaskhost.exe", "wudfhost.exe", "lsaiso.exe", "logonui.exe",
    "python.exe", "pythonw.exe", "py.exe",
}

_NOISE_TOKENS = {"the", "a", "an", "app", "application", "program", "software"}

# ---------------------------------------------------------------------------
# Win32 window enumeration (ctypes -- no extra dependency, microseconds)
# ---------------------------------------------------------------------------

_user32 = ctypes.WinDLL("user32", use_last_error=True)

_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

_user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
_user32.EnumWindows.restype = wintypes.BOOL
_user32.EnumChildWindows.argtypes = [wintypes.HWND, _WNDENUMPROC, wintypes.LPARAM]
_user32.EnumChildWindows.restype = wintypes.BOOL
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.IsWindowVisible.restype = wintypes.BOOL
_user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
_user32.GetWindowTextLengthW.restype = ctypes.c_int
_user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetWindowTextW.restype = ctypes.c_int
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_user32.PostMessageW.argtypes = [wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM]
_user32.PostMessageW.restype = wintypes.BOOL


def _window_pid(hwnd):
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _real_owner_pid(hwnd, host_pid):
    """A Store app's visible window belongs to ApplicationFrameHost, not to
    the app -- killing that PID would take down every UWP window on the
    desktop. The app's own CoreWindow is a child of the frame, so the first
    child with a different PID is the process we actually want."""
    try:
        name = psutil.Process(host_pid).name().lower()
    except Exception:
        return host_pid

    if name != "applicationframehost.exe":
        return host_pid

    found = []

    def callback(child, _lparam):
        pid = _window_pid(child)
        if pid and pid != host_pid:
            found.append(pid)
            return False  # stop enumerating
        return True

    proc = _WNDENUMPROC(callback)
    _user32.EnumChildWindows(hwnd, proc, 0)
    return found[0] if found else host_pid


def _iter_windows():
    """[(hwnd, pid, title)] for visible top-level windows that have a title."""
    results = []

    def callback(hwnd, _lparam):
        if not _user32.IsWindowVisible(hwnd):
            return True
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if title:
            results.append((hwnd, _window_pid(hwnd), title))
        return True

    proc = _WNDENUMPROC(callback)
    _user32.EnumWindows(proc, 0)
    return results


# ---------------------------------------------------------------------------
# Normalisation and similarity
# ---------------------------------------------------------------------------

def _normalize(text):
    """Lowercase, drop punctuation and filler, collapse whitespace.

    '+' and '#' survive so that Notepad++ and C# stay distinguishable.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\(.*?\)", " ", text)  # "Chrome (64-bit)"
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#]+", " ", text)
    tokens = [t for t in text.split() if t]
    kept = [t for t in tokens if t not in _NOISE_TOKENS]
    return " ".join(kept if kept else tokens)


def _split_camel(text):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text or "")


def _similarity(query, candidate, strict=False):
    """0..1 over two already-normalised strings.

    Hand-rolled rather than pure difflib because the shapes that matter here
    are prefixes and containment -- "spotify" inside "Spotify Music", "vlc"
    at the front of "VLC media player" -- and a raw character-ratio scores
    those far too low next to coincidental letter overlap.

    strict=True drops the character-ratio fallback entirely and only accepts
    structural matches. Use it when the result decides whether to kill a
    process: difflib rates "code" against "node" at 0.75, which is enough to
    clear the threshold and take out node.exe on "close vs code".
    """
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0

    ratio = float(len(query)) / float(len(candidate))

    if len(query) >= 3 and candidate.startswith(query):
        return 0.86 + 0.12 * min(ratio, 1.0)
    if len(candidate) >= 3 and query.startswith(candidate):
        return 0.82 + 0.12 * min(1.0 / ratio, 1.0)
    if len(query) >= 4 and query in candidate:
        return 0.74 + 0.12 * min(ratio, 1.0)

    q_tokens = set(query.split())
    c_tokens = set(candidate.split())
    if q_tokens and q_tokens <= c_tokens:
        return 0.80
    if c_tokens and c_tokens <= q_tokens:
        return 0.76

    overlap = q_tokens & c_tokens
    token_score = 0.0
    if overlap:
        token_score = 0.55 + 0.20 * (len(overlap) / float(len(q_tokens | c_tokens)))

    if strict:
        return token_score

    return max(token_score, difflib.SequenceMatcher(None, query, candidate).ratio())


def _expand_alias(query):
    """Returns the alias target, or the query unchanged."""
    return ALIASES.get(query, query)


# ---------------------------------------------------------------------------
# Catalog entries
# ---------------------------------------------------------------------------

class AppEntry(object):
    __slots__ = ("name", "path", "args", "appid", "sources", "keys")

    def __init__(self, name, path="", args="", appid="", source=""):
        self.name = name or ""
        self.path = path or ""
        self.args = args or ""
        self.appid = appid or ""
        self.sources = set([source] if source else [])
        self.keys = set()
        self._rebuild_keys()

    def _rebuild_keys(self):
        keys = set()
        norm = _normalize(self.name)
        if norm:
            keys.add(norm)
            expanded = _expand_alias(norm)
            if expanded != norm:
                keys.add(expanded)

        if self.path:
            base = os.path.splitext(os.path.basename(self.path))[0]
            stem = _normalize(_split_camel(base))
            if stem:
                keys.add(stem)

        for key in _appid_keys(self.appid):
            keys.add(key)

        self.keys = set(k for k in keys if k)

    def merge(self, other):
        if not self.path and other.path:
            self.path = other.path
            self.args = other.args
        if not self.appid and other.appid:
            self.appid = other.appid
        self.sources |= other.sources
        self._rebuild_keys()
        self.keys |= other.keys

    def exe_names(self):
        """Lowercase .exe basenames this app is likely to run as."""
        names = set()
        if self.path:
            names.add(os.path.basename(self.path).lower())
        return names

    def launchable(self):
        return bool(self.path or self.appid)

    def to_dict(self):
        return {
            "name": self.name,
            "path": self.path,
            "args": self.args,
            "appid": self.appid,
            "sources": sorted(self.sources),
        }

    def __repr__(self):
        return "AppEntry({!r})".format(self.name)


def _appid_keys(appid):
    """Pulls speakable names out of a UWP AppID.

    'SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify' carries the app's name
    twice -- in the package family name and after the '!' -- and neither is
    the friendly name, so both go in as extra match keys.
    """
    if not appid:
        return []

    keys = []
    head, _, tail = appid.partition("!")

    if tail:
        cleaned = _normalize(_split_camel(tail))
        if cleaned and cleaned != "app":
            keys.append(cleaned)

    family = head.split("_")[0]
    for segment in family.split("."):
        cleaned = _normalize(_split_camel(segment))
        if cleaned and cleaned not in ("microsoft", "windows", "app"):
            keys.append(cleaned)

    return keys


def _is_junk(name):
    lowered = (name or "").lower()
    return any(token in lowered for token in _JUNK_TOKENS)


# ---------------------------------------------------------------------------
# Scanning: Start Menu shortcuts
# ---------------------------------------------------------------------------

# Shell link format, [MS-SHLLINK]. Only the few fields we need are decoded.
_LNK_MAGIC = b"L\x00\x00\x00"
_LNK_HAS_LINK_TARGET_IDLIST = 0x01
_LNK_HAS_LINK_INFO = 0x02
_LNK_HAS_NAME = 0x04
_LNK_HAS_REL_PATH = 0x08
_LNK_HAS_WORKING_DIR = 0x10
_LNK_HAS_ARGUMENTS = 0x20
_LNK_HAS_ICON_LOCATION = 0x40
_LNK_IS_UNICODE = 0x80
_LNK_HEADER_SIZE = 0x4C


def _read_lnk(path):
    """Extracts (target, arguments) from a .lnk without going through COM.

    The shell's own CreateShortcut().TargetPath resolves the link, which means
    touching whatever it points at -- and a shortcut aimed at a disconnected
    network share blocks on an SMB timeout for seconds. Multiply by a few
    hundred shortcuts and the scan never finishes. The path is stored inline in
    the file, so reading the bytes is both instant and unable to hang.

    Returns (None, "") for anything unparseable; a malformed shortcut just
    means one missing app, not a failed scan.
    """
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return None, ""

    if len(data) < _LNK_HEADER_SIZE or data[:4] != _LNK_MAGIC:
        return None, ""

    flags = struct.unpack_from("<I", data, 20)[0]
    pos = _LNK_HEADER_SIZE

    # LinkTargetIDList: a length-prefixed blob we skip wholesale.
    if flags & _LNK_HAS_LINK_TARGET_IDLIST:
        if pos + 2 > len(data):
            return None, ""
        pos += 2 + struct.unpack_from("<H", data, pos)[0]

    target = None

    # LinkInfo holds the local base path -- the fully-qualified target.
    if flags & _LNK_HAS_LINK_INFO:
        if pos + 4 > len(data):
            return None, ""
        info_size = struct.unpack_from("<I", data, pos)[0]
        # 0x1C is the minimum LinkInfo header: everything up to and including
        # CommonPathSuffixOffset at +24.
        if info_size >= 0x1C and pos + info_size <= len(data):
            header_size = struct.unpack_from("<I", data, pos + 4)[0]
            base_offset = struct.unpack_from("<I", data, pos + 16)[0]
            suffix_offset = struct.unpack_from("<I", data, pos + 24)[0]

            # Header >= 0x24 means the unicode offsets are present, and those
            # are the ones that survive non-ASCII install paths.
            if header_size >= 0x24 and info_size >= 0x24:
                base_unicode = struct.unpack_from("<I", data, pos + 28)[0]
                suffix_unicode = struct.unpack_from("<I", data, pos + 32)[0]
                if base_unicode:
                    target = _read_sz(data, pos + base_unicode, True)
                    if target:
                        target += _read_sz(
                            data, pos + suffix_unicode, True
                        ) or ""

            if not target and base_offset:
                target = _read_sz(data, pos + base_offset, False)
                if target:
                    # The path is split in two: LocalBasePath carries the
                    # directory, CommonPathSuffix the filename. Usually the
                    # suffix is empty and the base is complete, but not always.
                    target += _read_sz(data, pos + suffix_offset, False) or ""

        pos += info_size

    unicode_strings = bool(flags & _LNK_IS_UNICODE)
    arguments = ""

    # StringData: counted (not terminated) strings, in a fixed order.
    for bit in (
        _LNK_HAS_NAME,
        _LNK_HAS_REL_PATH,
        _LNK_HAS_WORKING_DIR,
        _LNK_HAS_ARGUMENTS,
        _LNK_HAS_ICON_LOCATION,
    ):
        if not flags & bit:
            continue
        if pos + 2 > len(data):
            break

        count = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        size = count * 2 if unicode_strings else count
        raw = data[pos:pos + size]
        pos += size

        value = (
            raw.decode("utf-16-le", "replace")
            if unicode_strings
            else raw.decode("mbcs", "replace")
        )

        if bit == _LNK_HAS_ARGUMENTS:
            arguments = value.strip()
        elif bit == _LNK_HAS_REL_PATH and not target:
            # No LinkInfo (portable/relocatable shortcut) -- resolve the
            # relative path against the shortcut's own folder.
            candidate = os.path.normpath(
                os.path.join(os.path.dirname(path), value)
            )
            if _path_exists(candidate):
                target = candidate

    return target, arguments


def _read_sz(data, offset, unicode_string):
    """Null-terminated string at an absolute offset."""
    if offset <= 0 or offset >= len(data):
        return None

    if unicode_string:
        end = offset
        while end + 1 < len(data) and data[end:end + 2] != b"\x00\x00":
            end += 2
        return data[offset:end].decode("utf-16-le", "replace")

    end = data.find(b"\x00", offset)
    if end == -1:
        end = len(data)
    return data[offset:end].decode("mbcs", "replace")


def _start_menu_roots():
    roots = []
    for var, tail in (
        ("ProgramData", r"Microsoft\Windows\Start Menu\Programs"),
        ("APPDATA", r"Microsoft\Windows\Start Menu\Programs"),
    ):
        base = os.environ.get(var)
        if base:
            roots.append(os.path.join(base, tail))
    return roots


def _scan_start_menu():
    rows = []
    for root in _start_menu_roots():
        if not os.path.isdir(root):
            continue

        root_depth = root.rstrip("\\").count("\\")
        for folder, dirs, files in os.walk(root):
            if folder.count("\\") - root_depth >= MAX_LNK_DEPTH:
                dirs[:] = []

            for filename in files:
                if not filename.lower().endswith(".lnk"):
                    continue

                name = os.path.splitext(filename)[0]
                if _is_junk(name):
                    continue

                full = os.path.join(folder, filename)
                target, arguments = _read_lnk(full)
                if not target or not target.lower().endswith(".exe"):
                    continue

                rows.append({
                    "name": name,
                    "path": target,
                    "args": arguments,
                    "appid": "",
                    "source": "startmenu",
                })
    return rows


# ---------------------------------------------------------------------------
# Scanning: App Paths registry
# ---------------------------------------------------------------------------

_APP_PATHS_KEYS = (
    (winreg.HKEY_LOCAL_MACHINE,
     r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths", 0),
    (winreg.HKEY_LOCAL_MACHINE,
     r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
     winreg.KEY_WOW64_32KEY),
    (winreg.HKEY_CURRENT_USER,
     r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths", 0),
)


def _scan_app_paths():
    rows = []
    for hive, subkey, extra in _APP_PATHS_KEYS:
        try:
            root = winreg.OpenKey(
                hive, subkey, 0, winreg.KEY_READ | extra
            )
        except OSError:
            continue

        with root:
            for index in range(winreg.QueryInfoKey(root)[0]):
                try:
                    child_name = winreg.EnumKey(root, index)
                    with winreg.OpenKey(root, child_name) as child:
                        value = winreg.QueryValue(child, None)
                except OSError:
                    continue

                if not value:
                    continue

                name = os.path.splitext(child_name)[0]
                if _is_junk(name):
                    continue

                rows.append({
                    "name": name,
                    "path": value.strip().strip('"'),
                    "args": "",
                    "appid": "",
                    "source": "apppaths",
                })
    return rows


# ---------------------------------------------------------------------------
# Scanning: Store / UWP apps
# ---------------------------------------------------------------------------

# Get-StartApps is the only piece that still needs PowerShell -- UWP AppIDs
# aren't exposed anywhere cheaper. It's also the only piece allowed to fail
# without failing the scan: losing it costs the Store apps, not the catalog.
_STARTAPPS_SCRIPT = (
    "$ErrorActionPreference='SilentlyContinue';"
    "$ProgressPreference='SilentlyContinue';"
    "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
    "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress -Depth 2"
)


def _scan_start_apps():
    encoded = base64.b64encode(
        _STARTAPPS_SCRIPT.encode("utf-16-le")
    ).decode("ascii")

    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-EncodedCommand", encoded,
            ],
            capture_output=True,
            timeout=SCAN_TIMEOUT_SECONDS,
            creationflags=CREATE_NO_WINDOW,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        print("app index: Get-StartApps skipped ({})".format(e))
        return []

    raw = completed.stdout.decode("utf-8", errors="replace").strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except ValueError:
        return []

    if isinstance(parsed, dict):  # ConvertTo-Json unwraps single results
        parsed = [parsed]

    rows = []
    for item in parsed:
        name = (item.get("Name") or "").strip()
        appid = (item.get("AppID") or "").strip()
        if not name or not appid or _is_junk(name):
            continue
        rows.append({
            "name": name,
            "path": "",
            "args": "",
            "appid": appid,
            "source": "startapps",
        })
    return rows


def _collect_rows():
    """All three sources. Any one failing costs its own apps, nothing more."""
    rows = []
    timings = []

    for label, scan in (
        ("start menu", _scan_start_menu),
        ("app paths", _scan_app_paths),
        ("start apps", _scan_start_apps),
    ):
        started = time.perf_counter()
        try:
            found = scan()
        except Exception as e:
            print("app index: {} scan failed ({})".format(label, e))
            found = []
        rows.extend(found)
        timings.append("{} {}/{:.1f}s".format(
            label, len(found), time.perf_counter() - started
        ))

    print("app index: " + ", ".join(timings))
    return rows


def _build_catalog(rows):
    """Merges raw rows into deduplicated entries keyed by normalised name."""
    catalog = {}

    for row in rows:
        name = (row.get("name") or "").strip()
        if not name or _is_junk(name):
            continue

        path = (row.get("path") or "").strip()
        if path and not _path_exists(path):
            # Stale shortcut pointing at something uninstalled. Keep going --
            # an AppID entry for the same app may still be valid.
            path = ""

        entry = AppEntry(
            name=name,
            path=path,
            args=(row.get("args") or "").strip(),
            appid=(row.get("appid") or "").strip(),
            source=(row.get("source") or "").strip(),
        )
        if not entry.launchable():
            continue

        key = _normalize(name)
        if not key:
            continue

        existing = catalog.get(key)
        if existing is None:
            catalog[key] = entry
        else:
            existing.merge(entry)

    return catalog


def _path_exists(path):
    """os.path.exists that can't hang.

    Same trap as the COM shortcut resolver: a stale target on a disconnected
    UNC share blocks until the SMB layer gives up. Local paths are checked
    normally; network paths are taken on trust and fail later at launch, which
    is recoverable in a way that a frozen scan isn't.
    """
    if path.startswith("\\\\"):
        return True
    try:
        return os.path.exists(path)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Cache + lifecycle
# ---------------------------------------------------------------------------

_catalog = {}
_catalog_lock = threading.RLock()
_catalog_ready = threading.Event()
_scan_lock = threading.Lock()
_preloaded = False
_launched = {}          # normalised app key -> set of PIDs we started
_launched_lock = threading.Lock()


def _load_cache():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None, 0.0

    rows = payload.get("apps")
    if not isinstance(rows, list) or not rows:
        return None, 0.0

    return rows, float(payload.get("scanned_at", 0.0))


def _save_cache(catalog):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        payload = {
            "version": 1,
            "scanned_at": time.time(),
            "apps": [entry.to_dict() for entry in catalog.values()],
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
    except OSError as e:
        print("app index: couldn't write cache ({})".format(e))


def _install(catalog):
    global _catalog
    with _catalog_lock:
        _catalog = catalog
    _catalog_ready.set()


def _scan_now(quiet=False):
    """Rescans and installs the result. Returns the number of apps found.

    Never raises. This runs on a daemon thread at startup, and an escape
    there would both dump a traceback into the terminal and leave
    _catalog_ready unset -- so every later lookup would sit through the full
    CATALOG_WAIT_SECONDS before falling back.
    """
    with _scan_lock:
        started = time.perf_counter()
        try:
            rows = _collect_rows()
        except Exception as e:
            print("app index: scan failed ({})".format(e))
            _catalog_ready.set()  # unblock waiters; they'll get fallbacks
            return 0

        try:
            catalog = _build_catalog(rows)
        except Exception as e:
            print("app index: building the catalog failed ({})".format(e))
            _catalog_ready.set()
            return 0

        if not catalog:
            print("app index: scan found nothing usable")
            _catalog_ready.set()
            return 0

        _install(catalog)

        try:
            _save_cache(catalog)
        except Exception as e:
            # Cache is an optimisation. Losing it costs one slow startup.
            print("app index: couldn't cache the scan ({})".format(e))

        if not quiet:
            print("app index: {} apps in {:.1f}s".format(
                len(catalog), time.perf_counter() - started
            ))
        return len(catalog)


def _ensure_catalog():
    """Serve the cache immediately if we have one; rescan when it's stale.

    Also a daemon thread target, so it swallows too -- see _scan_now.
    """
    try:
        rows, scanned_at = _load_cache()
    except Exception as e:
        print("app index: couldn't read the cache ({})".format(e))
        rows, scanned_at = None, 0

    if rows:
        try:
            catalog = _build_catalog(rows)
        except Exception as e:
            print("app index: cached catalog unusable ({})".format(e))
            catalog = None

        if catalog:
            _install(catalog)
            age_days = (time.time() - scanned_at) / 86400.0
            if age_days <= CACHE_MAX_AGE_DAYS:
                return
            # Stale but usable -- keep serving it while a fresh scan runs.
            threading.Thread(
                target=_scan_now, kwargs={"quiet": True}, daemon=True
            ).start()
            return

    _scan_now()


def preload():
    """Warms the catalog on a background thread. Safe to call more than once.

    Called at import time from executor.py so the scan overlaps with model
    loading rather than with the user's first command.
    """
    global _preloaded
    with _catalog_lock:
        if _preloaded:
            return None
        _preloaded = True

    thread = threading.Thread(target=_ensure_catalog, daemon=True)
    thread.start()
    return thread


def rescan():
    """Forces a fresh scan. Wired to the 'refresh_apps' voice action."""
    count = _scan_now()
    if count:
        return "Found {} apps".format(count)
    return "Couldn't refresh the app list"


def get_catalog(timeout=CATALOG_WAIT_SECONDS):
    """Returns the catalog, waiting briefly if the first scan is still going."""
    if not _catalog_ready.is_set():
        preload()
        _catalog_ready.wait(timeout)
    with _catalog_lock:
        return _catalog


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve(spoken):
    """Spoken app name -> (AppEntry or None, score).

    Returns (None, score) below MATCH_FLOOR so the caller can honestly say it
    didn't find the app rather than launching whatever was nearest.
    """
    query = _normalize(spoken)
    if not query:
        return None, 0.0

    catalog = get_catalog()
    queries = [query]
    expanded = _expand_alias(query)
    if expanded != query:
        queries.append(expanded)

    best_entry = None
    best_score = 0.0

    for entry in catalog.values():
        score = 0.0
        for candidate in entry.keys:
            for q in queries:
                score = max(score, _similarity(q, candidate))
                if score >= 1.0:
                    break
            if score >= 1.0:
                break

        # A tie between two equally-named entries goes to the one we can also
        # close later -- a real .exe path gives us a process name to match.
        if score > best_score or (
            score == best_score
            and best_entry is not None
            and entry.path
            and not best_entry.path
        ):
            best_entry, best_score = entry, score

    if best_score < MATCH_FLOOR:
        return None, best_score

    return best_entry, best_score


def _fallback_entry(spoken):
    """Built-in Windows commands, for the case where the catalog missed."""
    query = _expand_alias(_normalize(spoken))

    best_name, best_score = None, 0.0
    for name in FALLBACK_COMMANDS:
        score = _similarity(query, _normalize(name))
        if score > best_score:
            best_name, best_score = name, score

    if best_name is None or best_score < MATCH_FLOOR:
        return None

    return AppEntry(
        name=best_name, path=FALLBACK_COMMANDS[best_name], source="builtin"
    )


# ---------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------

def _snapshot_pids():
    # Called just before launching, outside open_app's try -- so a raise here
    # would abort the launch itself. An empty snapshot only costs precision
    # in the adopt pass, never the launch.
    try:
        return set(proc.pid for proc in psutil.process_iter())
    except Exception:
        return set()


def _remember(key, pids):
    if not pids:
        return
    with _launched_lock:
        _launched.setdefault(key, set()).update(pids)


def _adopt_new_processes(key, entry, before):
    """Learns which PID the app became, after the fact.

    ShellExecute doesn't hand back a PID, and UWP apps are started by explorer
    rather than by us, so there is nothing to return at launch time. Polling
    the process table for a few seconds afterwards -- on a daemon thread, off
    the latency path -- catches the new PIDs so close() can be precise later.
    """
    expected = entry.exe_names()
    name_keys = entry.keys
    deadline = time.monotonic() + ADOPT_WINDOW_SECONDS
    seen = set(before)

    while time.monotonic() < deadline:
        time.sleep(ADOPT_POLL_SECONDS)
        adopted = set()

        try:
            processes = list(psutil.process_iter(["pid", "name"]))
        except Exception:
            continue

        for proc in processes:
            try:
                pid = proc.info["pid"]
                name = (proc.info["name"] or "").lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue

            if pid in seen:
                continue
            seen.add(pid)

            if not name or name in _PROTECTED:
                continue

            if name in expected:
                adopted.add(pid)
                continue

            stem = _normalize(_split_camel(os.path.splitext(name)[0]))
            if any(_similarity(k, stem, strict=True) >= PROC_MATCH_FLOOR
                   for k in name_keys):
                adopted.add(pid)

        _remember(key, adopted)


def _track_key(entry, spoken):
    """PID bookkeeping is keyed on the *resolved* name, not the spoken one.

    Otherwise "open vs code" then "close visual studio code" would file and
    look up under different keys and the tracked PID would be missed.
    """
    if entry is not None and entry.name:
        return _normalize(entry.name)
    return _normalize(spoken)


def open_app(spoken):
    """Launches the named app. Returns a result string for the spoken reply."""
    entry, score = resolve(spoken)
    if entry is None:
        entry = _fallback_entry(spoken)

    if entry is None:
        return "Couldn't find an app called '{}'".format(spoken)

    key = _track_key(entry, spoken)
    before = _snapshot_pids()

    try:
        if entry.path:
            _start_path(entry)
        else:
            _start_appid(entry)
    except Exception as e:
        return "Failed to open {}: {}".format(entry.name, e)

    threading.Thread(
        target=_adopt_new_processes,
        args=(key, entry, before),
        daemon=True,
    ).start()

    return "Opened {}".format(entry.name)


def _start_path(entry):
    """ShellExecute rather than Popen: it detaches properly, so the app
    doesn't die with our console or catch our Ctrl+C, and it handles elevation
    prompts and file associations the way double-clicking would."""
    if entry.args:
        try:
            os.startfile(entry.path, arguments=entry.args)
            return
        except TypeError:
            pass  # Python < 3.10 has no arguments= ; launch bare
    os.startfile(entry.path)


def _start_appid(entry):
    """Store/UWP apps have no exe to launch. The AppsFolder shell namespace is
    the supported way in, and it works for Win32 AppIDs too."""
    subprocess.Popen(
        ["explorer.exe", "shell:AppsFolder\\{}".format(entry.appid)],
        creationflags=CREATE_NO_WINDOW,
    )


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------

def _self_pids():
    pids = set([os.getpid()])
    try:
        for parent in psutil.Process().parents():
            pids.add(parent.pid)
    except Exception:
        pass
    return pids


def _safe_to_kill(proc, protected_pids):
    try:
        if proc.pid in protected_pids or proc.pid <= 4:
            return False
        return (proc.name() or "").lower() not in _PROTECTED
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False


def _tracked_pids(key):
    with _launched_lock:
        pids = set(_launched.get(key, ()))

    alive = set()
    for pid in pids:
        try:
            proc = psutil.Process(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        # _is_running, not proc.is_running() -- the bare call raises
        # AccessDenied on an elevated PID, which used to throw away this
        # whole set and drop us onto the fuzzier strategies below.
        if _is_running(proc):
            alive.add(pid)

    with _launched_lock:
        if key in _launched:
            _launched[key] = alive
    return alive


def _pids_by_exe_name(entry):
    """Exact process-name match against the catalog entry's executable."""
    if entry is None:
        return set()

    expected = entry.exe_names()
    if not expected:
        return set()

    found = set()
    for proc in psutil.process_iter(["pid", "name"]):
        name = (proc.info["name"] or "").lower()
        if name and name in expected:
            found.add(proc.info["pid"])
    return found


def _pids_by_fuzzy_name(spoken, entry):
    """Fuzzy match on the running process names themselves.

    Covers apps we didn't launch and whose catalog entry has no path -- the
    process name is often close enough on its own (Discord.exe, Slack.exe).
    """
    queries = set([_expand_alias(_normalize(spoken)), _normalize(spoken)])
    if entry is not None:
        queries |= entry.keys
    queries = set(q for q in queries if q)

    found = set()
    for proc in psutil.process_iter(["pid", "name"]):
        name = (proc.info["name"] or "").lower()
        if not name or name in _PROTECTED:
            continue
        stem = _normalize(_split_camel(os.path.splitext(name)[0]))
        if not stem:
            continue
        if any(_similarity(q, stem, strict=True) >= PROC_MATCH_FLOOR
               for q in queries):
            found.add(proc.info["pid"])
    return found


def _pids_by_window_title(spoken, entry):
    """Last resort: match what's written in the title bar.

    This is what rescues Store apps, whose process name resembles neither the
    friendly name nor anything the user would say, and windowed apps that run
    inside a generic host binary.
    """
    queries = set([_expand_alias(_normalize(spoken)), _normalize(spoken)])
    if entry is not None:
        queries |= entry.keys
    queries = set(q for q in queries if q)

    found = set()
    for hwnd, pid, title in _iter_windows():
        norm_title = _normalize(title)
        if not norm_title:
            continue

        # Titles are usually "<document> - <App Name>", so score the tail
        # segments too rather than the whole string.
        segments = [norm_title]
        for separator in (" - ", " – ", " | "):
            if separator in title:
                segments.extend(
                    _normalize(part) for part in title.split(separator)
                )

        hit = any(
            _similarity(q, segment, strict=True) >= PROC_MATCH_FLOOR
            for q in queries
            for segment in segments
            if segment
        )
        if hit:
            found.add(_real_owner_pid(hwnd, pid))

    return found


def _request_window_close(pids):
    """Asks each window to close itself. Returns how many actually accepted.

    PostMessageW's return value matters here. UIPI blocks messages sent from
    a medium-integrity process to a high-integrity window, and it does it
    silently -- the call just returns FALSE. Counting those as posted meant
    burning the whole graceful timeout waiting on a WM_CLOSE that was never
    delivered, every single time an elevated app was targeted.
    """
    posted = 0
    pidset = set(pids)
    for hwnd, pid, _title in _iter_windows():
        if pid not in pidset:
            continue
        try:
            if _user32.PostMessageW(hwnd, WM_CLOSE, 0, 0):
                posted += 1
        except OSError:
            continue
    return posted


def _is_running(proc):
    """True if the process is still there.

    Unknown counts as *alive*. Guessing the other way is how you end up
    announcing "Closed Task Manager" to someone staring at Task Manager.
    """
    try:
        return proc.is_running()
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError):
        return True


def _wait_gone(procs, timeout):
    """Waits for processes to die. Returns the ones still standing.

    This is psutil.wait_procs' job, but wait_procs is not survivable: its
    inner check_gone catches TimeoutExpired and NoSuchProcess and nothing
    else, so one process we're not allowed to open -- Task Manager, anything
    launched as admin -- raises AccessDenied straight out of the call and
    takes the entire close path down with it. Polling ourselves costs a few
    milliseconds and cannot be derailed by a single PID.
    """
    alive = list(procs)
    deadline = time.monotonic() + timeout
    while True:
        alive = [proc for proc in alive if _is_running(proc)]
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(DEATH_POLL_SECONDS)


def _terminate(pids):
    """Closes gracefully first, then escalates.

    Returns (killed, survived): how many processes actually went away, and
    how many shrugged us off. Every psutil call here is guarded -- one
    untouchable PID must not abort the rest of the set, and it must not
    escape into the caller's thread either.
    """
    protected_pids = _self_pids()

    roots = []
    for pid in pids:
        try:
            proc = psutil.Process(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        if _safe_to_kill(proc, protected_pids):
            roots.append(proc)

    if not roots:
        return 0, 0

    # Browsers and editors fan out into helper processes; taking only the
    # parent leaves orphans behind.
    targets = {}
    for proc in roots:
        targets[proc.pid] = proc
        try:
            for child in proc.children(recursive=True):
                if _safe_to_kill(child, protected_pids):
                    targets[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue

    procs = list(targets.values())
    attempted = len(procs)

    if GRACEFUL_CLOSE and _request_window_close(targets.keys()):
        procs = _wait_gone(procs, GRACEFUL_TIMEOUT)

    for proc in procs:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue

    survivors = _wait_gone(procs, ESCALATE_TIMEOUT)
    for proc in survivors:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue

    remaining = _wait_gone(survivors, KILL_TIMEOUT)

    # Count what died, not what we aimed at. The old version returned the
    # size of the target set, so a close that was refused outright still
    # reported success.
    return attempted - len(remaining), len(remaining)


def close_app(spoken):
    """Closes the named app. Returns a result string for the spoken reply.

    Four strategies, most confident first. The first one that finds anything
    wins -- we don't union them, because a broad fuzzy sweep alongside a
    precise hit is how you accidentally kill the wrong thing.
    """
    entry, _score = resolve(spoken)
    if entry is None:
        entry = _fallback_entry(spoken)

    key = _track_key(entry, spoken)
    label = entry.name if entry is not None else spoken

    strategies = (
        ("tracked", lambda: _tracked_pids(key)),
        ("exe name", lambda: _pids_by_exe_name(entry)),
        ("process name", lambda: _pids_by_fuzzy_name(spoken, entry)),
        ("window title", lambda: _pids_by_window_title(spoken, entry)),
    )

    for how, find in strategies:
        try:
            pids = find()
        except Exception as e:
            print("app index: {} lookup failed ({})".format(how, e))
            continue

        if not pids:
            continue

        try:
            killed, survived = _terminate(pids)
        except Exception as e:
            print("app index: terminate failed ({})".format(e))
            return "Failed to close {}".format(label)

        if killed and not survived:
            with _launched_lock:
                _launched.pop(key, None)
            return "Closed {}".format(label)

        if killed:
            # Partial: some of the set went, some refused. Don't drop the
            # tracked PIDs -- the survivors are still ours, and forgetting
            # them costs us the most precise strategy on the next attempt.
            #
            # Worded with "Couldn't" deliberately: main.py decides whether to
            # speak a correction by scanning these strings for it.
            return (
                "Closed part of {}, but couldn't close the rest"
                " -- needs admin rights".format(label)
            )

        if survived:
            # We found it and were refused. Task Manager, anything started as
            # administrator, and most services land here: a medium-integrity
            # process cannot terminate a high-integrity one, no matter how
            # politely it asks.
            #
            # Stop rather than fall through. The remaining strategies are
            # fuzzier than this one, and killing the wrong thing after failing
            # on the right thing is the worst outcome available.
            return "Couldn't close {}, it's running as administrator".format(
                label
            )

    return "Couldn't find '{}' running".format(spoken)


# ---------------------------------------------------------------------------
# Introspection helpers (handy from a REPL while tuning)
# ---------------------------------------------------------------------------

def list_apps(limit=None):
    names = sorted(entry.name for entry in get_catalog().values())
    return names[:limit] if limit else names


def explain(spoken, top=5):
    """Prints the ranked candidates for a spoken name. Debugging aid."""
    query = _normalize(spoken)
    expanded = _expand_alias(query)
    scored = []
    for entry in get_catalog().values():
        score = max(
            [_similarity(q, k) for k in entry.keys for q in (query, expanded)]
            or [0.0]
        )
        scored.append((score, entry.name, sorted(entry.keys)))
    scored.sort(reverse=True)

    print("query: {!r} -> {!r}".format(spoken, expanded))
    for score, name, keys in scored[:top]:
        flag = "  " if score >= MATCH_FLOOR else "x "
        print("{}{:.3f}  {:<40} {}".format(flag, score, name[:40], keys))
    return scored[:top]


if __name__ == "__main__":
    import sys

    _ensure_catalog()
    if len(sys.argv) > 1:
        explain(" ".join(sys.argv[1:]))
    else:
        apps = list_apps()
        print("{} apps indexed\n".format(len(apps)))
        for app in apps:
            print("  " + app)