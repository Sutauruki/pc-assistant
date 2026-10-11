"""Read a file and tell the user what is in it.

Goes in: commands/file_analysis.py   (new file)

     .env
     GEMINI_API_KEY=sk-...       (summarising answers)

What each format contributes:
  PDF     -> extracted text (PyPDF2), summarized with the question.
  CSV     -> pandas shape + columns + first rows, summarized.
  Excel   -> each sheet's shape + a preview of the first sheet, summarized.
  audio   -> duration / channels / sample rate via pydub, and the perceptible
             answer is about the file's shape, not its speech content. Real
             speech transcription is a separate pipeline, far out of scope for
             a "what is this file" read.

EVERYTHING IS A READ
--------------------
Unlike the document generators, nothing here writes to disk. That means no
paths.output_path, no "created", no open-with -- just a spoken answer, which is
why this action (like web_search and youtube_video) joins SPOKEN_RESULTS in
main.py: the result string IS what the user asked for.
"""

import os
import warnings
from pathlib import Path

from commands import genbrain
from commands import paths

# pydub warns on import that ffmpeg is missing. For header-only reads (duration,
# channels, sample rate) that warning is noise -- pydub reads the metadata
# without ffmpeg, and this module never transcodes audio.
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)

AUDIO_EXTENSIONS = ("wav", "mp3", "flac", "aac", "ogg", "m4a", "opus", "wma", "aiff")
TEXT_EXTENSIONS = ("txt", "md", "log", "json", "csv", "tsv")


def analyze_file(target="", question=""):
    """Return a spoken analysis of `target`. Never writes anything."""
    spec = (target or "").strip()
    if not spec:
        return "I didn't catch which file you want me to look at."

    targets, error = _gather(spec)
    if error:
        return error
    item = targets[0]

    suffix = item.suffix.lower().lstrip(".")
    try:
        if suffix == "pdf":
            summary, detail = _read_pdf(item)
        elif suffix in ("csv", "tsv"):
            summary, detail = _read_delimited(item, suffix == "tsv")
        elif suffix in ("xlsx", "xls", "xlsm"):
            summary, detail = _read_excel(item)
        elif suffix in AUDIO_EXTENSIONS:
            summary, detail = _read_audio(item)
        elif suffix in TEXT_EXTENSIONS:
            summary, detail = _read_text(item)
        else:
            return ("I can read PDF, spreadsheet, CSV, audio and text files -- "
                    "{} isn't one of those.".format(item.name))
    except Exception as e:
        msg = "Couldn't read {}: {}".format(item.name, e)
        print(msg)
        return msg

    if question:
        question = (question or "").strip()
        try:
            answer = genbrain.summarise(
                "{}\n\n{}".format(summary, detail),
                question and "QUESTION: {}".format(question),
            )
            return answer
        except Exception as e:
            return "Couldn't answer about {}: {}".format(item.name, e)

    return summary


def _gather(spec):
    """Resolve the spoken target to exactly one readable file."""
    found = paths.resolve_targets(spec, must_exist=True)
    if not found:
        leaf = Path(paths.normalise(spec)).name or spec
        hits = paths.search(leaf, limit=6)
        if len(hits) == 1:
            found = hits
        elif len(hits) > 1:
            names = ", ".join(paths.display(h) for h in hits[:4])
            return [], "I found {} things called that -- {}. Which one?".format(
                len(hits), names)
        else:
            return [], "Couldn't find {}.".format(spec)

    files = [f for f in found if f.is_file()]
    if not files:
        return [], "That's a folder, not a file I can read."
    if len(files) > 1:
        names = ", ".join(f.name for f in files[:4])
        return [], "That matched {} files -- {}.".format(len(files), names)
    return files, None


def _read_pdf(path):
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise RuntimeError("PyPDF2 isn't installed")

    reader = PdfReader(str(path))
    pages = len(reader.pages)
    text = []
    for page_num, page in enumerate(reader.pages):
        chunk = (page.extract_text() or "").strip()
        if chunk:
            text.append("(page {}) {}".format(page_num + 1, chunk[:600]))
        if len(" ".join(text)) > 8000:
            break
    body = " ".join(text)
    if not body.strip():
        raise RuntimeError("it looks like a scanned PDF with no extractable text")
    return (
        "{} is a PDF with {} pages.".format(path.name, pages),
        body,
    )


def _read_delimited(path, tsv=False):
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas isn't installed")

    sep = "\t" if tsv else ","
    frame = pd.read_csv(path, sep=sep, nrows=200)
    columns = list(frame.columns)
    rows = len(frame)
    preview = frame.head(min(5, rows)).to_string(index=False)
    return (
        "{} is a {} file with {} {} and {} rows.".format(
            path.name, "tab-separated" if tsv else "CSV",
            len(columns), "column" if len(columns) == 1 else "columns", rows),
        "Columns: {}\nFirst rows:\n{}".format(", ".join(map(str, columns)), preview),
    )


def _read_excel(path):
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas isn't installed")

    sheets = pd.read_excel(path, sheet_name=None, nrows=100)
    parts = []
    for name, frame in sheets.items():
        parts.append("{}: {} rows, {}".format(
            name, len(frame), ", ".join(map(str, frame.columns))))
    preview = list(sheets.values())[0].head(5).to_string(index=False)
    return (
        "{} is a workbook with {} sheets ({}).".format(
            path.name, len(sheets), "; ".join(sheets.keys())),
        "Sheets:\n{}\nFirst rows:\n{}".format("\n".join(parts), preview),
    )


def _read_audio(path):
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError("pydub isn't installed")

    segment = AudioSegment.from_file(str(path))
    seconds = len(segment) / 1000.0
    channels = segment.channels
    rate = segment.frame_rate
    size = os.path.getsize(path)

    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    duration = "{} minutes, {} seconds".format(minutes, secs) if minutes else "{} seconds".format(secs)
    channels = "{} channel{}".format(channels, "" if channels == 1 else "s")
    return (
        "{} is audio, {} long, {}, sampled at {} hertz, about {}.".format(
            path.name, duration, channels, rate, paths.human_size(size)),
        "pydub read this file directly from its header; it was not transcribed.",
    )


def _read_text(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = len(text.splitlines())
    excerpt = text[:8000]
    if len(text) > 8000:
        excerpt += "\n... (truncated)"
    return (
        "{} is a text file with {} lines.".format(path.name, lines),
        excerpt,
    )


if __name__ == "__main__":
    # python -m commands.file_analysis  <path> ["question"]
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "idea.txt"
    question = " ".join(sys.argv[2:])
    print(analyze_file(path, question))