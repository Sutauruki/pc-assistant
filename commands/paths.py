"""Spoken paths in, real paths out -- plus the one gate that decides what is
allowed to be touched.

Goes in: commands/paths.py   (new file)

Everything that reads, writes, moves or deletes a file goes through here.
Nothing else in the project should be building paths by hand, and nothing else
should be deciding what is safe, because two safety checks that disagree are
worse than one.

WHY THIS ISN'T JUST os.path.expanduser
--------------------------------------
`~/Desktop` is wrong on a large fraction of Windows machines. OneDrive Backup
redirects Desktop, Documents and Pictures into `%USERPROFILE%\\OneDrive\\...`
and leaves the old folders behind, sometimes empty, sometimes stale. A voice
assistant that tidies the wrong Desktop is worse than one that can't find it,
so the real locations come from SHGetKnownFolderPath, which is what Explorer
itself asks.

THE SAFETY POLICY
-----------------
You chose "anywhere but system dirs". That is implemented as a denylist, not
an allowlist, so the assistant can reach a projects folder on D: or a scratch
directory without anyone editing a config. What it cannot reach:

  Windows, Program Files, Program Files (x86), ProgramData, AppData,
  $Recycle.Bin, System Volume Information, Recovery, other users' profiles,
  and any drive root *as the target of an operation*.

Two details that matter more than the list itself:

  - Checks run against the *resolved* path as well as the literal one. A
    junction or symlink pointing into C:\\Windows would otherwise walk straight
    past a string comparison, and Windows is full of them.
  - The check is operation-aware. Listing your profile root is fine; deleting
    it is not. Read and write are different questions and get different
    answers.
"""

import ctypes
import fnmatch
import os
import re
from ctypes import wintypes
from pathlib import Path

# Set False if you ever want the assistant to reach into AppData -- cache
# folders live there and clearing one is a reasonable thing to ask for. It is
# on by default because AppData is also where every program keeps the state it
# cannot survive losing.
BLOCK_APPDATA = True

# Refuse to touch the assistant's own source tree. Cheap insurance against a
# mis-parsed "delete the old files in this folder" while you're sitting in it.
PROTECT_SELF = True

# Caps, so a mis-parse can't turn into a ten-minute walk of an entire drive.
MAX_GLOB_RESULTS = 400
MAX_SEARCH_RESULTS = 40
MAX_SEARCH_DEPTH = 4
MAX_SEARCH_DIRS = 4000

# Above either of these a delete stops and asks first. Both are cheap to
# measure because the walk gives up as soon as one is exceeded.
BIG_DELETE_FILES = 40
BIG_DELETE_BYTES = 250 * 1024 * 1024


class Unsafe(Exception):
    """Raised when a path is refused. The message is spoken to the user, so
    write it as a sentence, not as a stack trace."""


# --- Known folders ---------------------------------------------------------

class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid(text):
    text = text.strip("{}")
    parts = text.split("-")
    out = _GUID()
    out.Data1 = int(parts[0], 16)
    out.Data2 = int(parts[1], 16)
    out.Data3 = int(parts[2], 16)
    tail = parts[3] + parts[4]
    for i in range(8):
        out.Data4[i] = int(tail[i * 2:i * 2 + 2], 16)
    return out


_KNOWN_GUIDS = {
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
    "home": "{5E6C858F-0E22-4760-9AFE-EA3317B67173}",
}

# Fallbacks used only if the shell call fails outright -- wrong on a
# OneDrive-redirected machine, which is exactly why they're the fallback.
_KNOWN_FALLBACK = {
    "desktop": "~/Desktop",
    "documents": "~/Documents",
    "downloads": "~/Downloads",
    "pictures": "~/Pictures",
    "music": "~/Music",
    "videos": "~/Videos",
    "home": "~",
}

_folders = None
_folders_error = None


def _shell_known_folder(guid_text):
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    ole32 = ctypes.WinDLL("ole32", use_last_error=True)

    shell32.SHGetKnownFolderPath.argtypes = [
        ctypes.POINTER(_GUID),
        wintypes.DWORD,
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_wchar_p),
    ]
    shell32.SHGetKnownFolderPath.restype = ctypes.c_long
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoTaskMemFree.restype = None

    out = ctypes.c_wchar_p()
    rid = _guid(guid_text)
    # KF_FLAG_DEFAULT (0). Deliberately not KF_FLAG_CREATE -- we want to know
    # where a folder *is*, not to bring one into existence as a side effect of
    # someone saying its name.
    hr = shell32.SHGetKnownFolderPath(ctypes.byref(rid), 0, None, ctypes.byref(out))
    if hr != 0 or not out.value:
        return None
    try:
        return str(out.value)
    finally:
        ole32.CoTaskMemFree(out)


def known_folders():
    """name -> Path for the six user folders plus home. Cached; built once."""
    global _folders, _folders_error
    if _folders is not None:
        return _folders

    found = {}
    for name, guid_text in _KNOWN_GUIDS.items():
        value = None
        try:
            value = _shell_known_folder(guid_text)
        except Exception as e:  # non-Windows, or a shell that won't answer
            if _folders_error is None:
                _folders_error = str(e)
        if not value:
            value = os.path.expanduser(_KNOWN_FALLBACK[name])
        found[name] = Path(value)

    # Handy extras that don't have a GUID worth the ceremony.
    found["temp"] = Path(os.environ.get("TEMP") or os.path.expanduser("~"))
    _folders = found
    return _folders


# Spoken names that mean one of the above. Kept small on purpose -- the model
# is told to emit these words, so this is a normaliser, not a guessing game.
_ROOT_ALIASES = {
    "desktop": "desktop",
    "document": "documents",
    "documents": "documents",
    "docs": "documents",
    "doc": "documents",
    "download": "downloads",
    "downloads": "downloads",
    "picture": "pictures",
    "pictures": "pictures",
    "photos": "pictures",
    "photo": "pictures",
    "images": "pictures",
    "music": "music",
    "songs": "music",
    "audio": "music",
    "video": "videos",
    "videos": "videos",
    "movies": "videos",
    "home": "home",
    "user folder": "home",
    "profile": "home",
    "temp": "temp",
    "tmp": "temp",
}

_FILLER_LEADING = ("my ", "the ", "your ")
_FILLER_TRAILING = (" folder", " directory", " dir")

# "delete idea.txt in this location" resolved to nothing, because "this
# location" is not a folder name -- so the whole spec failed to resolve and the
# turn fell back to chat. These phrases all mean the directory Jarvis is
# running in, which for a terminal-launched assistant is the directory the user
# is looking at.
#
# Deliberately narrow. "here" and "this folder" are unambiguous; a bare "this"
# is not, and mapping it would turn "delete this" into a working-directory
# operation on a mis-hear.
_HERE_ALIASES = (
    "this location", "this folder", "this directory", "this dir",
    "current location", "current folder", "current directory", "current dir",
    "the current directory", "the current folder", "working directory",
    "the working directory", "here", "right here", "in here", "where i am",
    "cwd", "this path",
)


def _means_here(segment):
    """True if this path component is a spoken way of saying "the folder I'm
    in". Compared after filler stripping, so "the current folder" and "current"
    both land here."""
    probe = " ".join((segment or "").strip().lower().split())
    probe = probe.strip(".,!?")
    return probe in _HERE_ALIASES


def _strip_filler(segment):
    out = segment.strip()
    lowered = out.lower()
    changed = True
    while changed:
        changed = False
        for word in _FILLER_LEADING:
            if lowered.startswith(word):
                out = out[len(word):].strip()
                lowered = out.lower()
                changed = True
        for word in _FILLER_TRAILING:
            if lowered.endswith(word):
                out = out[:-len(word)].strip()
                lowered = out.lower()
                changed = True
    return out


# --- Protected locations ---------------------------------------------------

def _env_path(name):
    value = os.environ.get(name)
    return Path(value) if value else None


def _system_roots():
    roots = []
    for name in ("SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)",
                 "ProgramW6432", "ProgramData"):
        path = _env_path(name)
        if path is not None:
            roots.append(path)

    if BLOCK_APPDATA:
        for name in ("LOCALAPPDATA", "APPDATA"):
            path = _env_path(name)
            if path is not None:
                roots.append(path)

    drive = os.environ.get("SystemDrive") or "C:"
    for leaf in ("$Recycle.Bin", "System Volume Information", "Recovery",
                 "Boot", "PerfLogs"):
        roots.append(Path(drive + os.sep + leaf))

    return roots


_system_roots_cache = None


def system_roots():
    global _system_roots_cache
    if _system_roots_cache is None:
        _system_roots_cache = _system_roots()
    return _system_roots_cache


def _self_root():
    # commands/paths.py -> commands -> repo root
    return Path(__file__).resolve().parent.parent


def _is_under(path, parent):
    try:
        path.relative_to(parent)
        return True
    except (ValueError, OSError):
        return False


def _is_drive_root(path):
    try:
        return path.parent == path
    except Exception:
        return False


def _candidates(path):
    """The literal path and its resolved form. Both get checked, because a
    junction is a perfectly ordinary-looking string."""
    seen = [path]
    try:
        real = path.resolve(strict=False)
        if real != path:
            seen.append(real)
    except (OSError, RuntimeError):
        pass
    return seen


def _other_user_profile(path):
    """True if this lives in someone else's profile folder."""
    users = _env_path("SystemDrive")
    users = Path((users or Path("C:")).anchor) / "Users"
    if not _is_under(path, users):
        return False
    mine = known_folders()["home"]
    return not _is_under(path, mine) and path != mine


def protection_reason(path, op="write"):
    """None if allowed, otherwise a spoken-English reason it isn't.

    op is "read", "write" or "delete". Read is permitted in a few places
    write is not -- you can list your profile root, you cannot delete it.
    """
    path = Path(path)

    for candidate in _candidates(path):
        for root in system_roots():
            if candidate == root or _is_under(candidate, root):
                return "{} is a protected system location".format(root.name or root)

        if _other_user_profile(candidate):
            return "that belongs to another user account"

        if op != "read":
            if _is_drive_root(candidate):
                return "that's a whole drive"

            users_root = Path(candidate.anchor) / "Users"
            if candidate == users_root:
                return "that's the folder every user account lives in"

            folders = known_folders()
            if candidate == folders["home"]:
                return "that's your entire user folder"

            if PROTECT_SELF and _is_under(candidate, _self_root()):
                return "that's my own program folder"

    return None


def check(path, op="write"):
    """Raises Unsafe if the operation isn't allowed. Returns the Path."""
    path = Path(path)
    reason = protection_reason(path, op)
    if reason:
        raise Unsafe(reason)
    return path


# --- New-file destinations -------------------------------------------------
#
# The Tier-6 generators (docs, slides, sheets, pdf, website) all create files,
# so they all need the same two answers: where is this allowed to go, and what
# name won't clobber an existing file. Grouped here so the safety policy lives
# in one place instead of being re-derived by half a dozen modules.

def output_path(spec, extension="", create=True):
    """A write-safe destination Path for a brand-new file.

    `spec` may be a full path, a folder, or a filename. `extension` is appended
    when the resolved path has no suffix yet -- "called Q3 report" should
    become "Q3 report.docx", not an extensionless file. When the name already
    exists a numeric "(2)" suffix is added rather than overwriting, which is
    the same refusal file_actions.mv makes: an overwrite here is a silent
    delete, and a document generator must never be that.

    Raises paths.Unsafe when the target is a protected location.
    """
    path = Path(spec)
    if extension and not path.suffix:
        path = path.with_name(path.name + extension)

    check(path, "write")

    if path.exists():
        stem, suffix, parent = path.stem, path.suffix, path.parent
        for n in range(2, 1000):
            candidate = parent / "{stem} ({n}){suffix}".format(stem=stem, n=n, suffix=suffix)
            if not candidate.exists():
                path = candidate
                break

    if create:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    return path


# --- Resolution ------------------------------------------------------------

_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_GLOB_CHARS = "*?["


def has_glob(text):
    return any(ch in text for ch in _GLOB_CHARS)


def _normalise_spec(spec):
    spec = (spec or "").strip().strip('"').strip("'")
    spec = os.path.expandvars(spec)
    spec = spec.replace("/", os.sep)
    # Speech-to-text loves inserting a space after a colon.
    spec = re.sub(r"^([A-Za-z]):\s+\\", r"\1:\\", spec)
    return spec


normalise = _normalise_spec  # public name; callers outside this module use it


def resolve(spec, base=None):
    """Spoken or written path -> absolute Path. Does not touch the disk.

    Accepts absolute paths, UNC paths, ~, %VARS%, and folder names spoken the
    way people speak them: "downloads", "my downloads folder",
    "desktop/old stuff", "documents\\taxes\\2025".
    """
    spec = _normalise_spec(spec)
    if not spec:
        return Path(base) if base else known_folders()["home"]

    # "in this location" on its own, or as the folder half of a path.
    if _means_here(spec):
        return Path(base) if base else Path.cwd()

    if spec.startswith("~"):
        return Path(os.path.expanduser(spec))

    if spec.startswith("\\\\") or _DRIVE_RE.match(spec):
        return Path(spec)

    # A bare drive letter: "D:" means D:\
    if re.fullmatch(r"[A-Za-z]:", spec):
        return Path(spec + os.sep)

    parts = [p for p in spec.split(os.sep) if p.strip()]
    if not parts:
        return Path(base) if base else known_folders()["home"]

    head = _strip_filler(parts[0]).lower()
    folders = known_folders()

    if head in _ROOT_ALIASES:
        root = folders[_ROOT_ALIASES[head]]
        rest = parts[1:]
    elif _means_here(parts[0]) or _means_here(head):
        # "this location/idea.txt" -- the head names the working directory and
        # the rest is relative to it.
        root = Path(base) if base else Path.cwd()
        rest = parts[1:]
    elif base is not None:
        root = Path(base)
        rest = parts
    else:
        # No recognised root and no base: treat it as relative to home. The
        # caller can still fall back to search() when this doesn't exist.
        root = folders["home"]
        rest = parts

    for part in rest:
        root = root / part
    return root


def resolve_targets(spec, base=None, must_exist=True):
    """Like resolve(), but expands globs and always returns a list.

    Empty list means nothing matched -- which the caller should report as
    "couldn't find", not as an error.
    """
    spec = _normalise_spec(spec)
    if not spec:
        return []

    if not has_glob(spec):
        path = resolve(spec, base=base)
        if not must_exist or path.exists():
            return [path]
        return []

    # Split the pattern off the last non-glob directory component so we glob
    # from a real folder rather than walking from the drive root.
    parts = spec.split(os.sep)
    fixed, pattern = [], []
    for part in parts:
        if pattern or has_glob(part):
            pattern.append(part)
        else:
            fixed.append(part)

    root = resolve(os.sep.join(fixed), base=base) if fixed else (
        Path(base) if base else known_folders()["home"]
    )
    if not root.is_dir():
        return []

    found = []
    try:
        for match in root.glob(os.sep.join(pattern).replace(os.sep, "/")):
            found.append(match)
            if len(found) >= MAX_GLOB_RESULTS:
                break
    except (OSError, ValueError, NotImplementedError):
        return []

    found.sort(key=lambda p: (p.is_dir(), p.name.lower()))
    return found


_SEARCH_ROOT_ORDER = ("desktop", "downloads", "documents", "pictures",
                      "videos", "music")


def search(query, roots=None, limit=MAX_SEARCH_RESULTS):
    """Shallow, bounded search for a file or folder by name.

    Exists for the very common "delete the file called notes.txt" where the
    speaker never says where it is. Deliberately shallow and capped -- this is
    a convenience, not an indexer, and an unbounded walk of a home directory
    on a spinning disk is a ten-second silence.
    """
    query = (query or "").strip().lower()
    if not query:
        return []

    folders = known_folders()
    if roots is None:
        roots = [folders[name] for name in _SEARCH_ROOT_ORDER]
    else:
        roots = [Path(r) for r in roots]

    pattern = query if has_glob(query) else "*{}*".format(query)
    hits, seen, visited = [], set(), 0

    for root in roots:
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _e: None):
            visited += 1
            if visited > MAX_SEARCH_DIRS:
                return hits

            here = Path(dirpath)
            if len(here.parts) - base_depth >= MAX_SEARCH_DEPTH:
                dirnames[:] = []
            # Don't descend into anything we'd refuse to act on anyway.
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".")
                and protection_reason(here / d, "read") is None
            ]

            for name in list(dirnames) + filenames:
                if fnmatch.fnmatch(name.lower(), pattern):
                    candidate = here / name
                    key = str(candidate).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    hits.append(candidate)
                    if len(hits) >= limit:
                        return hits

    return hits


# --- Reporting helpers -----------------------------------------------------

def display(path):
    """Short, speakable form of a path.

    Reads back "Downloads\\report.pdf" instead of the full C:\\Users\\... , which
    is both faster to say and easier to recognise.
    """
    path = Path(path)
    folders = known_folders()
    for name in ("desktop", "downloads", "documents", "pictures", "videos",
                 "music", "temp", "home"):
        root = folders.get(name)
        if root is None:
            continue
        if path == root:
            return name.capitalize()
        if _is_under(path, root):
            rest = path.relative_to(root)
            label = "Home" if name == "home" else name.capitalize()
            return "{}\\{}".format(label, rest)
    return str(path)


def human_size(count):
    step = float(count)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if step < 1024 or unit == "TB":
            if unit == "bytes":
                return "{:.0f} bytes".format(step)
            return "{:.1f} {}".format(step, unit)
        step /= 1024.0
    return "{:.1f} TB".format(step)


def weigh(path, file_cap=BIG_DELETE_FILES, byte_cap=BIG_DELETE_BYTES):
    """(files, bytes, exceeded) for a path, giving up as soon as a cap is hit.

    The early exit is the whole point: this runs before every delete, and a
    full recursive size of a large folder would be a noticeable pause on a
    command that is supposed to feel instant.
    """
    path = Path(path)
    if path.is_file():
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return 1, size, size > byte_cap

    files = 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, onerror=lambda _e: None):
        for name in filenames:
            files += 1
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
            if files > file_cap or total > byte_cap:
                return files, total, True
    return files, total, False


if __name__ == "__main__":
    # python -m commands.paths   -- sanity-check resolution on this machine.
    print("Known folders:")
    for key, value in sorted(known_folders().items()):
        print("  {:<10} {}".format(key, value))

    print("\nProtected roots:")
    for root in system_roots():
        print("  {}".format(root))

    print("\nResolution:")
    for probe in ("downloads", "my downloads folder", "desktop/old stuff",
                  "~/Documents", "C:\\Windows\\System32", "D:", "downloads/*.pdf"):
        try:
            shown = resolve_targets(probe, must_exist=False) if has_glob(probe) \
                else [resolve(probe)]
            for item in shown[:3] or [None]:
                note = ""
                if item is not None:
                    reason = protection_reason(item, "delete")
                    note = "  <- REFUSED: {}".format(reason) if reason else ""
                print("  {:<24} -> {}{}".format(probe, item, note))
        except Exception as exc:
            print("  {:<24} -> error: {}".format(probe, exc))