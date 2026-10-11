"""Prompt to a formatted PDF report.

Goes in: commands/pdf_export.py   (new file)

     .env
     GEMINI_API_KEY=sk-...

reportlab's platypus is the workhorse: a title page, the report body with
headings and paragraphs, and a generated table when the prompt asked for
numbers. Reusing the same JSON outline shape as commands/docs.py means a prompt
produces a consistent document whether the user asked for Word or PDF -- the
difference is purely the rendering layer.

Fonts are where reportlab on Windows is least predictable, so the first block
tries the font helpers and falls back to the built-in Helvetica if they're
missing or broken. A PDF that renders is worth more than one that looks better
on somebody else's machine.
"""

import os

from commands import genbrain
from commands import paths

_FILE_EXT = ".pdf"
SHAPE = (
    "a JSON object with exactly these keys: title (string), subtitle (string), "
    "sections (array of objects, each with keys heading (string), paragraphs "
    "(array of strings), and optionally points (array of strings, for bullet "
    "lists))."
)


def make_pdf(prompt, target=None):
    """Prompt -> one .pdf on disk. Returns a spoken result string."""
    prompt = (prompt or "").strip()
    if not prompt:
        return "I didn't catch what the report should contain."

    try:
        content = genbrain.structured(SHAPE, prompt)
    except Exception as e:
        msg = "Couldn't draft the report: {}".format(e)
        print(msg)
        return msg

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        return "Couldn't create a PDF -- reportlab isn't installed."

    title = str(content.get("title") or "Report").strip()
    subtitle = str(content.get("subtitle") or "").strip()
    sections = content.get("sections") or []

    filepath = _destination(target, title)
    try:
        document = SimpleDocTemplate(
            str(filepath), pagesize=A4,
            leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72,
        )
        styles = _styles()
        story = []
        story.append(Paragraph(title, styles["Title"]))
        if subtitle:
            story.append(Spacer(1, 8))
            story.append(Paragraph(subtitle, styles["Italic"]))

        for section in sections:
            heading = str(section.get("heading") or "").strip()
            if heading:
                story.append(Spacer(1, 14))
                story.append(Paragraph(heading, styles["Heading2"]))
            for paragraph in section.get("paragraphs") or []:
                text = str(paragraph).strip()
                if text:
                    story.append(Paragraph(_escape(text), styles["BodyText"]))
            for point in section.get("points") or []:
                text = str(point).strip()
                if text:
                    story.append(Paragraph("&bull; " + _escape(text), styles["BodyText"]))

        document.build(story)
    except Exception as e:
        msg = "Couldn't write the PDF: {}".format(e)
        print(msg)
        return msg

    try:
        os.startfile(str(filepath))  # noqa: S606 - showing the user their report
        note = " and opened it"
    except Exception:
        note = ""

    out = "Made the report {name} in {where}{note}.".format(
        name=filepath.name, where=paths.display(filepath.parent), note=note)
    print(out)
    return out


def _styles():
    """Sample styles, with a best-effort font upgrade that never breaks the
    build if the Windows font directory is missing the files."""
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    try:
        _try_font("Vera", os.path.join(os.environ.get("WINDIR") or r"C:\Windows", "Fonts"))
    except Exception:
        pass
    return styles


def _try_font(family, fonts_dir):
    """Best-effort: register the font if its files are on disk, else no-op."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping

    names = {
        "Vera": ("Vera.ttf", "VeraBd.ttf", "VeraIt.ttf", "VeraBI.ttf"),
    }
    files = names.get(family)
    if not files:
        return
    suffix = ("Bold", "Italic", "BoldItalic")
    for path, style in zip(files, suffix):
        full = os.path.join(fonts_dir, path)
        if not os.path.exists(full):
            return  # broken setup -> keep the defaults
    pdfmetrics.registerFont(TTFont(family, os.path.join(fonts_dir, files[0])))
    pdfmetrics.registerFont(TTFont(family + "-Bold", os.path.join(fonts_dir, files[1])))
    pdfmetrics.registerFont(TTFont(family + "-Italic", os.path.join(fonts_dir, files[2])))
    pdfmetrics.registerFont(TTFont(family + "-BoldItalic", os.path.join(fonts_dir, files[3])))
    addMapping(family, 0, 0, family)
    addMapping(family, 1, 0, family + "-Bold")
    addMapping(family, 0, 1, family + "-Italic")
    addMapping(family, 1, 1, family + "-BoldItalic")


def _escape(text):
    """reportlab's Paragraph is mini-XML; a raw & or < breaks it."""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _destination(target, title):
    from commands import docs  # local import: docs owns the spec conventions

    return docs.destination_path(target, title, _FILE_EXT)


if __name__ == "__main__":
    # python -m commands.pdf_export  "a one page summary of this week's numbers"
    import sys

    probe = " ".join(sys.argv[1:]) or "a one page status report on the migration project"
    print(make_pdf(probe, "demo_report"))