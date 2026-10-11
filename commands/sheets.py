"""Prompt to a spreadsheet workbook.

Goes in: commands/sheets.py   (new file)

     .env
     GEMINI_API_KEY=sk-...

openpyxl renders a model-supplied workbook: one or more sheets, each with a
name, a header row and data rows. The model decides the columns and rows that
answer the prompt -- a product catalog, a budget, a golf-scores tracker -- and
this module types them into a real .xlsx with a bold header and reasonable
column widths. The first tip is used as the sheet name.

Numbers stay numbers, text stays text: the model is told to return strings so
openpyxl gets clean values, and nothing here tries to infer types it could
guess wrong about.
"""

import os

from commands import genbrain
from commands import paths

_FILE_EXT = ".xlsx"
SHAPE = (
    "a JSON object with exactly these keys: title (string), and sheets (array "
    "of objects, each with keys name (string, max 30 chars), columns (array of "
    "strings), and rows (array of arrays of strings). Every row must have one "
    "entry per column. Values are always strings -- numbers as digits, never "
    "formulas."
)


def make_xlsx(prompt, target=None):
    """Prompt -> one .xlsx on disk. Returns a spoken result string."""
    prompt = (prompt or "").strip()
    if not prompt:
        return "I didn't catch what the spreadsheet should contain."

    try:
        content = genbrain.structured(SHAPE, prompt)
    except Exception as e:
        msg = "Couldn't draft the spreadsheet: {}".format(e)
        print(msg)
        return msg

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        return "Couldn't create a spreadsheet -- openpyxl isn't installed."

    title = str(content.get("title") or "Spreadsheet").strip()
    sheets = content.get("sheets") or []

    try:
        workbook = Workbook()
        workbook.remove(workbook.active)  # drop the default sheet
        for sheet_data in sheets:
            _add_sheet(workbook, sheet_data, Font)
        if not workbook.sheetnames:
            return "Couldn't build the spreadsheet -- it had no sheets at all."
    except Exception as e:
        msg = "Couldn't build the workbook: {}".format(e)
        print(msg)
        return msg

    filepath = _destination(target, title)
    try:
        workbook.save(str(filepath))
    except Exception as e:
        msg = "Couldn't write the spreadsheet: {}".format(e)
        print(msg)
        return msg

    try:
        os.startfile(str(filepath))  # noqa: S606 - showing the user their sheet
        note = " and opened it"
    except Exception:
        note = ""

    out = "Made {name} with {count} {word} in {where}{note}.".format(
        name=filepath.name, count=len(sheets), word="sheet" if len(sheets) == 1 else "sheets",
        where=paths.display(filepath.parent), note=note)
    print(out)
    return out


def _add_sheet(workbook, sheet_data, Font):
    name = str(sheet_data.get("name") or "Sheet").strip().replace("/", "-")[:30]
    columns = [str(c).strip() for c in (sheet_data.get("columns") or []) if str(c).strip()]
    rows = sheet_data.get("rows") or []

    ws = workbook.create_sheet(title=name or "Sheet")
    for col_index, header in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_index, value=header)
        cell.font = Font(bold=True)

    for row_index, row in enumerate(rows, start=2):
        for col_index, value in enumerate(columns, start=1):
            raw = ""
            if isinstance(row, (list, tuple)) and len(row) >= col_index:
                raw = row[col_index - 1]
            elif isinstance(row, dict):
                raw = row.get(columns[col_index - 1], "")
            cell = ws.cell(row=row_index, column=col_index, value=str(raw))

    # Column widths scoped to the actual content, capped so a 400-char cell
    # doesn't turn the sheet into a scroll marathon.
    for col_index in range(1, len(columns) + 1):
        longest = len(columns[col_index - 1])
        for row in rows:
            if isinstance(row, (list, tuple)) and len(row) >= col_index:
                value = str(row[col_index - 1])
            elif isinstance(row, dict):
                value = str(row.get(columns[col_index - 1], ""))
            else:
                value = ""
            longest = max(longest, len(value))
        ws.column_dimensions[_colletter(col_index)].width = min(60, longest + 2)


def _colletter(index):
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _destination(target, title):
    from commands import docs  # local import: docs owns the spec conventions

    return docs.destination_path(target, title, _FILE_EXT)


if __name__ == "__main__":
    # python -m commands.sheets  "a monthly budget with income and expenses"
    import sys

    probe = " ".join(sys.argv[1:]) or "a small product catalog with name, price and stock"
    print(make_xlsx(probe, "demo_sheet"))