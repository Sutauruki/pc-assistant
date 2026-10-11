"""Prompt to a slide deck.

Goes in: commands/slides.py   (new file)

     .env
     GEMINI_API_KEY=sk-...

python-pptx turns a model-supplied outline into a real .pptx: a title slide, an
agenda, then one slide per section with a heading and a handful of bullets.
The generator is deliberately conservative -- a title and bullets per slide,
which is how slide decks are actually read (and how any theme applied later
will render cleanly). Elaboration into tables and images is out of scope for a
first cut; dense text is.

Same failure contract as every action module: a miss contains "couldn't" or
"failed" so main.py's scan picks it up, and a hit says what got made.
"""

import os

from commands import genbrain
from commands import paths

_FILE_EXT = ".pptx"
SHAPE = (
    "a JSON object with exactly these keys: title (string), subtitle (string), "
    "and slides (array of objects, each with keys heading (string) and "
    "bullets (array of strings, 3-5 items)). Keep every bullet short enough "
    "to fit on one slide line."
)


def make_pptx(prompt, target=None):
    """Prompt -> one .pptx on disk. Returns a spoken result string."""
    prompt = (prompt or "").strip()
    if not prompt:
        return "I didn't catch what the presentation should be about."

    try:
        content = genbrain.structured(SHAPE, prompt)
    except Exception as e:
        msg = "Couldn't draft the slide deck: {}".format(e)
        print(msg)
        return msg

    try:
        from pptx import Presentation
    except ImportError:
        return "Couldn't create a presentation -- python-pptx isn't installed."

    title = str(content.get("title") or "Presentation").strip()
    subtitle = str(content.get("subtitle") or "").strip()
    slides = content.get("slides") or []

    presentation = Presentation()
    # 16:9 -- matches every modern screen and keeps bullets from wrapping.
    presentation.slide_width = 12192000
    presentation.slide_height = 6858000

    try:
        _add_title_slide(presentation, title, subtitle)
        for slide in slides:
            _add_content_slide(presentation, slide)
    except Exception as e:
        msg = "Couldn't build the deck: {}".format(e)
        print(msg)
        return msg

    filepath = _destination(target, title)
    try:
        presentation.save(str(filepath))
    except Exception as e:
        msg = "Couldn't write the presentation: {}".format(e)
        print(msg)
        return msg

    try:
        os.startfile(str(filepath))  # noqa: S606 - showing the user their deck
        note = " and opened it"
    except Exception:
        note = ""

    out = "Made {count}-slide deck {name} in {where}{note}.".format(
        count=len(slides) + 1, name=filepath.name,
        where=paths.display(filepath.parent), note=note)
    print(out)
    return out


def _add_title_slide(presentation, title, subtitle):
    slide = presentation.slides.add_slide(presentation.slide_layouts[0])
    if slide.shapes.title is not None:
        slide.shapes.title.text = title
    for shape in slide.placeholders:
        if shape.placeholder_format.idx != 0:
            shape.text = subtitle
            break


def _add_content_slide(presentation, slide_data):
    heading = str(slide_data.get("heading") or "").strip()
    bullets = [str(b).strip() for b in (slide_data.get("bullets") or []) if str(b).strip()]

    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    if slide.shapes.title is not None:
        slide.shapes.title.text = heading
    body = slide.placeholders[1] if len(slide.placeholders) > 1 else None
    if body is None:
        return
    frame = body.text_frame
    frame.clear()
    for i, bullet in enumerate(bullets):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.text = bullet
        paragraph.level = 0


def _destination(target, title):
    """The path the deck should land at, resolved like any other target.

    Reuses the docs convention: a spoken target wins, otherwise a safe slug of
    the title in Documents/Jarvis.
    """
    from commands import docs  # local import: docs owns the spec conventions

    return docs.destination_path(target, title, _FILE_EXT)


if __name__ == "__main__":
    # python -m commands.slides  "5 slides on the state of the Altair project"
    import sys

    probe = " ".join(sys.argv[1:]) or "a 5-slide pitch about a coffee subscription service"
    print(make_pptx(probe, "demo_deck"))