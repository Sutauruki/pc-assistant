"""Prompt to a structured Word document.

Goes in: commands/docs.py   (new file)

     .env
     GEMINI_API_KEY=sk-...

The rest of the run is pure python-docx: the model supplies the outline as
JSON -- title, intro, sections with headings, paragraphs and bullet lists --
and this module renders it. Nothing here guesses at prose.

PATHS GO THROUGH paths.py
-------------------------
The target may be a spoken "called Q3 report in documents" or nothing at all.
Either way the final path goes through paths.output_path, which checks the
write safety policy before one byte is written. A doc the user asked for is
still a file the assistant was told to create, so it gets the same protection
as everything else -- and the "couldn't" wording below is what lets main.py
know the turn failed.
"""

import os
import re
import time
from pathlib import Path

from commands import genbrain
from commands import paths

_FILE_EXT = ".docx"
_SHAPE = (
    "a JSON object with exactly these keys: title (string), intro (string), "
    "and sections (array of objects, each with keys heading (string) and "
    "paragraphs (array of strings))."
)


def make_docx(prompt, target=None):
    """Prompt -> one .docx on disk. Returns a spoken result string."""
    prompt = (prompt or "").strip()
    if not prompt:
        return "I didn't catch what the document should say."

    try:
        content = genbrain.structured(_SHAPE, prompt)
    except Exception as e:
        msg = "Couldn't draft the document: {}".format(e)
        print(msg)
        return msg

    title = str(content.get("title") or "Document").strip()
    intro = str(content.get("intro") or "").strip()
    sections = content.get("sections") or []

    try:
        import docx
    except ImportError:
        return "Couldn't create a Word document -- python-docx isn't installed."

    filepath = destination_path(target, title)
    try:
        document = docx.Document()
        document.add_heading(title, level=0)
        if intro:
            document.add_paragraph(intro)
        for section in sections:
            heading = str(section.get("heading") or "").strip()
            if heading:
                document.add_heading(heading, level=1)
            for paragraph in section.get("paragraphs") or []:
                text = str(paragraph).strip()
                if text:
                    document.add_paragraph(text)
        document.save(str(filepath))
    except Exception as e:
        msg = "Couldn't write the document: {}".format(e)
        print(msg)
        return msg

    try:
        import os
        os.startfile(str(filepath))  # noqa: S606 - showing the user their doc
        note = " and opened it"
    except Exception:
        note = ""

    out = "Made {} in {}{}.".format(filepath.name, paths.display(filepath.parent), note)
    print(out)
    return out


def _slug(title):
    safe = "".join(ch if ch.isalnum() or ch in " -_" else " " for ch in title)
    return " ".join(safe.split())


# One folder per kind under Documents/Jarvis (paths.jarvis_dir).
_KIND_FOLDERS = {".docx": "Documents", ".pptx": "Presentations",
                 ".xlsx": "Spreadsheets", ".pdf": "PDFs"}

# "Q3 report in documents", "notes on my desktop", "in downloads"
_IN_FOLDER = re.compile(
    r"^(?:(?P<name>.*?)\s+)?(?:in|into|inside|on|to|under)\s+(?P<folder>.+)$", re.I)
_CALLED = re.compile(r"^(?:called|named|titled|name it|call it)\s+", re.I)


def _folder_like(spec):
    """The folder a spoken spec names, or None if it isn't one we can trust:
    a known folder (desktop, documents, downloads...), an absolute path, or a
    folder that already exists."""
    if not spec:
        return None
    resolved = paths.resolve(spec)
    head = paths._strip_filler(paths.normalise(spec).split(os.sep)[0]).lower()
    if head in paths._ROOT_ALIASES or os.path.isabs(paths.normalise(spec)) or resolved.is_dir():
        return resolved
    return None


def destination_path(target, title, extension=_FILE_EXT):
    """Where a generated file (doc / deck / sheet / pdf) should land.

    The organised default is Documents/Jarvis/<Kind>/<date> <title>.<ext>:
    one folder per kind, names that sort by date. When the user named the
    file, their name is used as is; when they named a folder ("in
    documents", "on my desktop", a full path), it goes there. A bare name is
    never resolved against the home folder -- that's how files used to end up
    loose in C:\\Users\\<you>.
    """
    default_dir = paths.jarvis_dir(_KIND_FOLDERS.get(extension, "Documents"))
    spec = _CALLED.sub("", (target or "").strip().strip(".")).strip()

    name, folder = spec, None
    match = _IN_FOLDER.match(spec) if spec else None
    if match:
        candidate = _folder_like(match.group("folder"))
        if candidate is not None:
            name, folder = (match.group("name") or "").strip(), candidate
            name = _CALLED.sub("", name).strip()

    base = folder or default_dir
    if name:
        spoken = paths.normalise(name)
        head = paths._strip_filler(spoken).lower()
        if os.sep not in spoken and head in paths._ROOT_ALIASES:
            base = paths.resolve(spoken)                  # just a folder: "desktop"
        elif os.path.isabs(spoken) or os.sep in spoken:
            # A path. Known roots ("desktop/notes") resolve as usual; anything
            # else is relative to the Jarvis folder, never the home folder.
            return paths.output_path(paths.resolve(spoken, base=base), extension)
        else:
            if spoken.lower().endswith(extension):
                spoken = spoken[:-len(extension)]
            stem = _slug(spoken)
            if stem:
                return paths.output_path(base / stem, extension)

    stem = "{} {}".format(time.strftime("%Y-%m-%d"), _slug(title or "document") or "document")
    return paths.output_path(base / stem, extension)


if __name__ == "__main__":
    # python -m commands.docs  "a one page memo about the Q3 numbers"
    import sys

    probe = " ".join(sys.argv[1:]) or "a short memo about the Q3 revenue numbers"
    print(make_docx(probe, "demo_memo"))