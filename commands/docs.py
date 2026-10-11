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


def destination_path(target, title, extension=_FILE_EXT):
    """Where a generated file (doc / deck / sheet / pdf) should land.

    Shared convention across the generator modules: a spoken target like "called
    Q3 report in documents" or a partial path is resolved by paths.py like any
    other location; failing that the file lands in Documents/Jarvis under a safe
    slug of the title. `extension` is that module's own suffix.
    """
    spec = (target or "").strip()
    base = None

    if spec:
        resolved = paths.resolve(spec)
        if resolved.suffix:
            return paths.output_path(resolved, extension)
        # No extension = a folder or a bare name. If it resolves to something
        # directory-like, use it; otherwise treat it as a new filename.
        if resolved.is_dir():
            base = resolved
        elif resolved.parent != resolved:
            return paths.output_path(resolved, extension)

    if base is None:
        base = Path(paths.known_folders()["documents"]) / "Jarvis"

    stem = _slug(title or "document") or "document"
    return paths.output_path(base / stem, extension)


if __name__ == "__main__":
    # python -m commands.docs  "a one page memo about the Q3 numbers"
    import sys

    probe = " ".join(sys.argv[1:]) or "a short memo about the Q3 revenue numbers"
    print(make_docx(probe, "demo_memo"))