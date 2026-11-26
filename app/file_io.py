"""Reading uploaded feedback files (CSV / Excel) and simple checks. No Streamlit code here, so it can be tested."""
import io
import re

import pandas as pd

MAX_BYTES = 10 * 1024 * 1024      # 10 MB
MAX_ROWS = 5000
KHMER = re.compile(r"[ក-៿]")
TEXT_COLUMN_HINTS = ("text", "review", "feedback", "comment", "message", "មតិ")


class FileProblem(Exception):
    """A problem with the uploaded file, explained in plain words for the user."""


def read_feedback_file(file_name, data):
    """Reads a .csv, .xlsx or .xls file (given as bytes) into a DataFrame of strings."""
    if len(data) > MAX_BYTES:
        raise FileProblem(f"The file is {len(data) / 1e6:.1f} MB. The limit is 10 MB.")
    name = file_name.lower()
    try:
        if name.endswith(".csv"):
            try:
                table = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", dtype=str, keep_default_na=False)
            except UnicodeDecodeError:
                raise FileProblem("The CSV file is not saved as UTF-8, so Khmer letters cannot be read. "
                                  "In Excel use File → Save As → 'CSV UTF-8 (Comma delimited)'.")
        elif name.endswith(".xlsx"):
            table = pd.read_excel(io.BytesIO(data), engine="openpyxl", dtype=str).fillna("")
        elif name.endswith(".xls"):
            table = pd.read_excel(io.BytesIO(data), engine="xlrd", dtype=str).fillna("")
        else:
            raise FileProblem("Please upload a .csv, .xlsx or .xls file.")
    except FileProblem:
        raise
    except Exception as error:
        raise FileProblem(f"The file could not be read ({type(error).__name__}). Is it a valid CSV or Excel file?")
    if len(table) == 0:
        raise FileProblem("The file has no rows.")
    if len(table) > MAX_ROWS:
        raise FileProblem(f"The file has {len(table):,} rows. The limit is {MAX_ROWS:,} rows.")
    table.columns = [str(column) for column in table.columns]
    return table


def guess_text_column(table):
    """Index of the column that most likely holds the feedback text: among the columns whose name looks like
    text/review/feedback/comment, the one with the longest texts (so "review_id" loses against "review").
    If no name matches, simply the column with the longest texts."""
    columns = list(table.columns)
    average_length = {c: table[c].astype(str).str.len().mean() for c in columns}
    named = [c for c in columns if any(hint in c.lower() for hint in TEXT_COLUMN_HINTS)]
    candidates = named or columns
    return columns.index(max(candidates, key=lambda c: average_length[c]))


def guess_date_column(table, text_column=None):
    """Name of a column that looks like dates (by its name, or because most of its values are dates), else "(none)"."""
    import warnings
    for column in table.columns:
        if column == text_column:
            continue
        values = table[column].astype(str).str.strip()
        values = values[values != ""].head(200)
        if len(values) == 0:
            continue
        named = any(hint in column.lower() for hint in ("date", "time", "day", "created", "ថ្ងៃ"))
        if not named and values.str.fullmatch(r"\d+(\.\d+)?").mean() > 0.5:
            continue                                   # plain numbers (ids, scores) are not dates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        if parsed.notna().mean() >= (0.5 if named else 0.9):
            return column
    return "(none)"


def has_khmer(text):
    return bool(KHMER.search(str(text)))
