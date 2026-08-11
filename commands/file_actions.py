"""Move, delete, organize.

Goes in: commands/file_actions.py   (new file)

Every path here goes through commands/paths.py, which knows where Desktop
really is on a OneDrive machine and which locations are off limits. Nothing in
this file builds a path itself.

DELETES GO TO THE RECYCLE BIN
-----------------------------
send2trash, never os.remove. That single choice is what makes the whole feature
survivable: a voice assistant will eventually mis-hear something, and the
difference between a bad minute and a bad week is whether the file is still
sitting in the bin. If send2trash isn't installed the fallback moves the file
into a dated folder under your profile rather than deleting it -- still
recoverable, and it says so out loud instead of pretending it went to the bin.

TWO GATES, AND THEY DO DIFFERENT JOBS
-------------------------------------
The path gate refuses system locations outright. The size gate is softer: a
delete over BIG_DELETE_FILES files or BIG_DELETE_BYTES parks itself in
commands/confirm.py and asks first. That threshold is where "delete the
screenshots in Downloads" stops being a small tidy-up and starts being
something you'd want to have heard read back.

WHAT COMES BACK
---------------
Result strings, and the wording is load-bearing. main.py scans results for
"failed", "couldn't" and "didn't" to decide whether to spend a second model
round trip on a spoken correction. So every failure here contains one of those
words and no success does -- "Moved 3 files", not "Moved 3 files, didn't move
1", which would read as a total failure.
"""

import os
import shutil
import time
from pathlib import Path

from commands import confirm
from commands import paths

# --- Recycle bin -----------------------------------------------------------

try:
    from send2trash import send2trash as _send2trash
    _HAVE_SEND2TRASH = True
except Exception:  # pragma: no cover - depends on the machine
    _send2trash = None
    _HAVE_SEND2TRASH = False


def _fallback_trash(path):
    """No send2trash: move it somewhere recoverable instead of unlinking it.

    Deliberately not a hand-rolled SHFileOperation. That call needs a packed
    struct whose layout differs between 32- and 64-bit Python, and getting it
    subtly wrong deletes without undo -- exactly the failure this whole design
    exists to avoid. A dated folder is boring and correct.
    """
    bin_root = paths.known_folders()["home"] / ".jarvis-trash" / time.strftime("%Y-%m-%d")
    bin_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(_unique(bin_root / path.name)))


def _trash(path):
    if _HAVE_SEND2TRASH:
        _send2trash(str(path))
        return True
    _fallback_trash(path)
    return False


# --- Shared helpers --------------------------------------------------------

ORGANIZE_CONFIRM_AT = 150
LIST_LIMIT = 40
SPOKEN_LIST_LIMIT = 8

_BUCKETS = (
    ("Images", ("jpg", "jpeg", "png", "gif", "bmp", "webp", "heic", "heif",
                "tiff", "tif", "svg", "ico", "raw", "cr2", "nef", "dng", "avif")),
    ("Documents", ("pdf", "doc", "docx", "txt", "rtf", "odt", "md", "tex",
                   "epub", "mobi", "pages")),
    ("Spreadsheets", ("xls", "xlsx", "xlsm", "csv", "tsv", "ods", "numbers")),
    ("Presentations", ("ppt", "pptx", "odp", "key")),
    ("Audio", ("mp3", "wav", "flac", "aac", "ogg", "oga", "m4a", "wma", "opus",
               "aiff", "mid", "midi")),
    ("Video", ("mp4", "mkv", "avi", "mov", "wmv", "flv", "webm", "m4v", "mpg",
               "mpeg", "3gp")),
    ("Archives", ("zip", "rar", "7z", "tar", "gz", "bz2", "xz", "zst", "iso",
                  "cab")),
    ("Installers", ("exe", "msi", "msix", "appx", "msixbundle", "appxbundle")),
    ("Code", ("py", "js", "mjs", "cjs", "ts", "jsx", "tsx", "html", "htm",
              "css", "scss", "json", "java", "c", "cpp", "cc", "h", "hpp",
              "cs", "go", "rs", "rb", "php", "swift", "kt", "sh", "ps1",
              "bat", "cmd", "sql", "yml", "yaml", "toml", "ini", "cfg", "xml",
              "ipynb", "r", "lua", "pl", "vb")),
    ("Fonts", ("ttf", "otf", "woff", "woff2", "fon")),
    ("Shortcuts", ("lnk", "url", "webloc")),
)

_BUCKET_BY_EXT = {}
for _name, _exts in _BUCKETS:
    for _ext in _exts:
        _BUCKET_BY_EXT[_ext] = _name

_BUCKET_NAMES = {name for name, _ in _BUCKETS} | {"Other"}


def _bucket(path):
    return _BUCKET_BY_EXT.get(path.suffix.lower().lstrip("."), "Other")


def _unique(dest):
    """A free name next to `dest`. Never overwrite -- an overwrite is a delete
    with no recycle bin, which is the one outcome this module refuses to have."""
    dest = Path(dest)
    if not dest.exists():
        return dest

    stem, suffix, parent = dest.stem, dest.suffix, dest.parent
    for n in range(2, 1000):
        candidate = parent / "{} ({}){}".format(stem, n, suffix)
        if not candidate.exists():
            return candidate
    return parent / "{} ({}){}".format(stem, int(time.time()), suffix)


def _plural(count, word):
    return "{} {}{}".format(count, word, "" if count == 1 else "s")


def _gather(spec, op="write"):
    """Spoken target -> ([Path], error_message_or_None).

    Order matters. Exact/glob resolution first, then the working directory,
    then a name search as the last resort -- and a search that finds several
    things reports them rather than picking, because guessing which
    "report.pdf" you meant is how files get lost.
    """
    spec = (spec or "").strip()
    if not spec:
        return [], "I didn't catch which file you meant."

    found = paths.resolve_targets(spec, must_exist=True)

    if not found:
        # Not where the spoken path pointed. Try the folder Jarvis is running
        # in before searching: "delete idea.txt" said while sitting in a
        # project folder means the one right there, and the plain search only
        # covers Desktop, Downloads and the other known folders.
        found = paths.resolve_targets(spec, base=Path.cwd(), must_exist=True)

    if not found:
        # Nothing at that path. Try the name on its own before giving up.
        leaf = Path(paths.normalise(spec)).name or spec
        hits = paths.search(leaf, limit=6)
        if len(hits) == 1:
            found = hits
        elif len(hits) > 1:
            names = ", ".join(paths.display(h) for h in hits[:4])
            return [], "I found {} things called that -- {}. Which one?".format(
                len(hits), names
            )
        else:
            return [], "Couldn't find {}.".format(spec)

    allowed, refused = [], None
    for item in found:
        reason = paths.protection_reason(item, op)
        if reason:
            refused = refused or reason
            continue
        allowed.append(item)

    if not allowed:
        return [], "Couldn't do that -- {}.".format(refused)
    return allowed, None


# --- Actions ---------------------------------------------------------------

def move(source, destination, copy=False):
    """Move (or copy) one or more files. Globs allowed on the source."""
    verb = "copy" if copy else "move"
    targets, error = _gather(source, op="write")
    if error:
        return error

    dest_spec = (destination or "").strip()
    if not dest_spec:
        return "I didn't catch where to {} that.".format(verb)

    dest = paths.resolve(dest_spec)
    try:
        paths.check(dest, "write")
    except paths.Unsafe as e:
        return "Couldn't {} there -- {}.".format(verb, e)

    # Work out whether `dest` names a folder or a new filename. Existing
    # folder wins; after that, a suffix means a filename and no suffix means
    # a folder we're being asked to create.
    into_folder = dest.is_dir() or len(targets) > 1 or not dest.suffix

    try:
        if into_folder:
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return "Couldn't create {}: {}".format(paths.display(dest), e)

    done, failed = 0, []
    for item in targets:
        try:
            final = _unique(dest / item.name) if into_folder else _unique(dest)
            if copy:
                if item.is_dir():
                    shutil.copytree(str(item), str(final))
                else:
                    shutil.copy2(str(item), str(final))
            else:
                shutil.move(str(item), str(final))
            done += 1
        except Exception as e:
            failed.append("{} ({})".format(item.name, e))

    where = paths.display(dest if into_folder else dest.parent)
    if done and not failed:
        if done == 1:
            return "{} {} to {}".format(
                "Copied" if copy else "Moved", targets[0].name, where
            )
        return "{} {} to {}".format(
            "Copied" if copy else "Moved", _plural(done, "item"), where
        )
    if done and failed:
        return "{} {} to {}, but couldn't {} {}".format(
            "Copied" if copy else "Moved", _plural(done, "item"), where,
            verb, _plural(len(failed), "other")
        )
    return "Couldn't {} that: {}".format(verb, failed[0] if failed else "unknown error")


def copy(source, destination):
    return move(source, destination, copy=True)


def rename(source, new_name):
    targets, error = _gather(source, op="write")
    if error:
        return error
    if len(targets) > 1:
        return "Couldn't rename that -- {} things matched.".format(len(targets))

    item = targets[0]
    new_name = (new_name or "").strip().strip('"').strip("'")
    if not new_name:
        return "I didn't catch the new name."
    if any(ch in new_name for ch in '\\/:*?"<>|'):
        return "Couldn't use that name -- it has characters Windows won't allow."

    # Keep the old extension when the new name doesn't carry one. "rename this
    # to invoice" should not quietly turn a .pdf into an extensionless file.
    target = Path(new_name)
    if not target.suffix and item.suffix:
        new_name = new_name + item.suffix

    try:
        final = _unique(item.parent / new_name)
        item.rename(final)
        return "Renamed to {}".format(final.name)
    except Exception as e:
        return "Couldn't rename that: {}".format(e)


def _do_delete(targets):
    """The part that actually bins things. Split out so confirm.py can hold a
    reference to it without re-running resolution -- resolving twice would let
    the folder change between the question and the answer."""
    done, failed = 0, []
    for item in targets:
        try:
            _trash(item)
            done += 1
        except Exception as e:
            failed.append("{} ({})".format(item.name, e))

    where = "the recycle bin" if _HAVE_SEND2TRASH else "a dated folder in your home directory (send2trash isn't installed)"

    if done and not failed:
        what = targets[0].name if done == 1 else _plural(done, "item")
        return "Sent {} to {}".format(what, where)
    if done and failed:
        return "Sent {} to {}, but couldn't remove {}".format(
            _plural(done, "item"), where, _plural(len(failed), "other")
        )
    return "Couldn't delete that: {}".format(failed[0] if failed else "unknown error")


def delete(target):
    """Recycle bin, always -- and always after asking.

    The first cut of this only asked when paths.weigh() said the delete was
    large, or when a folder held more than one file. That made "delete the
    test.txt file from my desktop" a silent, immediate delete, which is not
    what "confirm the rest" means. A misheard filename is at least as likely
    as a mis-sized one, and the recycle bin is an undo you have to know to go
    looking for.

    So every delete asks now. The size probe is still worth running: it makes
    the question specific ("that's 340 files, about 1.2 GB") instead of vague,
    and being told what you're about to lose is most of the value of asking.
    """
    targets, error = _gather(target, op="delete")
    if error:
        return error

    total_files, total_bytes, over = 0, 0, False
    for item in targets:
        files, size, exceeded = paths.weigh(item)
        total_files += files
        total_bytes += size
        over = over or exceeded

    what = (targets[0].name if len(targets) == 1
            else _plural(len(targets), "item"))

    if over or total_files > 1:
        # Big enough that the count and size are the point of the question.
        detail = "That's {}, about {}".format(
            _plural(total_files, "file"), paths.human_size(total_bytes)
        )
        question = "{} in {}. Send it to the recycle bin?".format(detail, what)
    else:
        question = "Send {} to the recycle bin?".format(what)

    return confirm.request(
        question,
        lambda: _do_delete(targets),
        label="delete {}".format(what),
    )


def _do_organize(folder, entries, plan):
    moved, failed = 0, []
    for item in entries:
        bucket = plan[item]
        try:
            destination = folder / bucket
            destination.mkdir(exist_ok=True)
            shutil.move(str(item), str(_unique(destination / item.name)))
            moved += 1
        except Exception as e:
            failed.append("{} ({})".format(item.name, e))

    used = len({plan[item] for item in entries})
    if moved and not failed:
        return "Organized {} into {} in {}".format(
            _plural(moved, "file"), _plural(used, "folder"), paths.display(folder)
        )
    if moved and failed:
        return "Organized {}, but couldn't move {}".format(
            _plural(moved, "file"), _plural(len(failed), "other")
        )
    return "Couldn't organize that: {}".format(failed[0] if failed else "nothing moved")


def organize(target, by="type"):
    """Sort the top-level files of a folder into subfolders.

    Top level only, and files only. Recursing would rearrange project trees
    someone deliberately structured, and moving folders into buckets makes a
    mess that is genuinely tedious to undo.
    """
    spec = (target or "").strip()
    folder = paths.resolve(spec) if spec else paths.known_folders()["downloads"]

    try:
        paths.check(folder, "write")
    except paths.Unsafe as e:
        return "Couldn't organize that -- {}.".format(e)

    if not folder.is_dir():
        return "Couldn't find the folder {}.".format(paths.display(folder))

    by = (by or "type").strip().lower()
    if by in ("date", "month", "time", "when", "age"):
        by = "date"
    else:
        by = "type"

    entries, plan = [], {}
    try:
        for item in folder.iterdir():
            if item.is_dir() or item.name.startswith("."):
                continue
            if by == "type":
                bucket = _bucket(item)
            else:
                try:
                    bucket = time.strftime("%Y-%m", time.localtime(item.stat().st_mtime))
                except OSError:
                    bucket = "Unknown"
            plan[item] = bucket
            entries.append(item)
    except OSError as e:
        return "Couldn't read {}: {}".format(paths.display(folder), e)

    if not entries:
        return "Nothing loose to organize in {} -- it's already tidy.".format(
            paths.display(folder)
        )

    # A folder whose files are all one kind gains nothing from a single
    # subfolder, and burying them is worse than leaving them alone.
    if by == "type" and len({plan[e] for e in entries}) < 2:
        return "Everything in {} is the same kind of file, so there's nothing to split up.".format(
            paths.display(folder)
        )

    if len(entries) > ORGANIZE_CONFIRM_AT:
        buckets = len({plan[e] for e in entries})
        question = "That's {} in {}, sorted into {}. Go ahead?".format(
            _plural(len(entries), "file"), paths.display(folder),
            _plural(buckets, "folder")
        )
        return confirm.request(
            question,
            lambda: _do_organize(folder, entries, plan),
            label="organize {}".format(paths.display(folder)),
        )

    return _do_organize(folder, entries, plan)


def make_folder(target):
    spec = (target or "").strip()
    if not spec:
        return "I didn't catch what to call the new folder."

    folder = paths.resolve(spec)
    try:
        paths.check(folder, "write")
    except paths.Unsafe as e:
        return "Couldn't create that -- {}.".format(e)

    if folder.exists():
        return "{} already exists.".format(paths.display(folder))

    try:
        folder.mkdir(parents=True)
        return "Created {}".format(paths.display(folder))
    except Exception as e:
        return "Couldn't create that folder: {}".format(e)


def list_files(target=None, limit=LIST_LIMIT):
    """A speakable summary of what's in a folder.

    Speakable is the constraint. Reading forty filenames aloud is useless, so
    this leads with the counts and names only the first handful.
    """
    spec = (target or "").strip()
    folder = paths.resolve(spec) if spec else paths.known_folders()["downloads"]

    try:
        paths.check(folder, "read")
    except paths.Unsafe as e:
        return "Couldn't look in there -- {}.".format(e)

    if not folder.is_dir():
        return "Couldn't find the folder {}.".format(paths.display(folder))

    try:
        entries = sorted(
            folder.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except OSError as e:
        return "Couldn't read {}: {}".format(paths.display(folder), e)

    if not entries:
        return "{} is empty.".format(paths.display(folder))

    folders = [e for e in entries if e.is_dir()]
    files = [e for e in entries if not e.is_dir()]

    parts = []
    if folders:
        parts.append(_plural(len(folders), "folder"))
    if files:
        parts.append(_plural(len(files), "file"))

    head = ", ".join(e.name for e in entries[:SPOKEN_LIST_LIMIT])
    more = len(entries) - SPOKEN_LIST_LIMIT
    if more > 0:
        head += ", and {} more".format(more)

    # Full list to the log, short list to the voice.
    print("Contents of {}:".format(folder))
    for entry in entries[:limit]:
        print("  {}{}".format(entry.name, os.sep if entry.is_dir() else ""))

    return "{} has {}: {}".format(paths.display(folder), " and ".join(parts), head)


def find(query, where=None):
    roots = None
    if where:
        root = paths.resolve(where)
        if root.is_dir():
            roots = [root]

    hits = paths.search(query, roots=roots)
    if not hits:
        return "Couldn't find anything called {}.".format(query)

    print("Matches for {!r}:".format(query))
    for hit in hits:
        print("  {}".format(hit))

    if len(hits) == 1:
        return "Found {} in {}".format(hits[0].name, paths.display(hits[0].parent))

    names = ", ".join(paths.display(h) for h in hits[:4])
    extra = " and {} more".format(len(hits) - 4) if len(hits) > 4 else ""
    return "Found {} matches: {}{}".format(len(hits), names, extra)


def open_path(target):
    targets, error = _gather(target, op="read")
    if error:
        return error

    item = targets[0]
    try:
        os.startfile(str(item))  # noqa: S606 - the whole point of the action
        return "Opening {}".format(item.name)
    except Exception as e:
        return "Couldn't open {}: {}".format(item.name, e)


if __name__ == "__main__":
    # python -m commands.file_actions  -- read-only probes, nothing is moved.
    print(list_files("downloads"))
    print(list_files("desktop"))
    print(find("screenshot"))
    print("send2trash available:", _HAVE_SEND2TRASH)