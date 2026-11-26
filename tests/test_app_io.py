"""Reading uploaded files with Khmer text: CSV (UTF-8 and UTF-8 with BOM) and Excel."""
import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import file_io  # noqa: E402

REVIEWS = ["អ្នកបើកបរមិនសូវគោរពអ្នកដំណើរទេ", "សេវាកម្មល្អណាស់ 👍", "The bus was late"]
TABLE = pd.DataFrame({"id": ["1", "2", "3"], "Customer Feedback": REVIEWS, "branch": ["PP", "SR", "PP"]})


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_csv_utf8_and_bom(encoding):
    data = TABLE.to_csv(index=False).encode(encoding)
    table = file_io.read_feedback_file("feedback.csv", data)
    assert list(table.columns) == ["id", "Customer Feedback", "branch"]     # no BOM left in the first column name
    assert table["Customer Feedback"].tolist() == REVIEWS


def test_excel_xlsx():
    buffer = io.BytesIO()
    TABLE.to_excel(buffer, index=False, engine="openpyxl")
    table = file_io.read_feedback_file("feedback.XLSX", buffer.getvalue())
    assert table["Customer Feedback"].tolist() == REVIEWS


def test_not_utf8_csv_gives_clear_message():
    data = "id,text\n1,caf\xe9 tr\xe8s bien\n".encode("cp1252")
    with pytest.raises(file_io.FileProblem, match="UTF-8"):
        file_io.read_feedback_file("feedback.csv", data)


def test_wrong_file_type_and_limits():
    with pytest.raises(file_io.FileProblem, match="csv"):
        file_io.read_feedback_file("feedback.txt", b"hello")
    too_many = pd.DataFrame({"text": ["ល្អ"] * (file_io.MAX_ROWS + 1)}).to_csv(index=False).encode("utf-8")
    with pytest.raises(file_io.FileProblem, match="rows"):
        file_io.read_feedback_file("feedback.csv", too_many)


def test_guess_text_column_and_khmer_check():
    assert file_io.guess_text_column(TABLE) == 1
    # "review_id" also contains "review", but the real text column has the longer texts
    demo_like = pd.DataFrame({"review_id": ["12", "345"], "review": REVIEWS[:2]})
    assert file_io.guess_text_column(demo_like) == 1
    # no matching name → the column with the longest texts
    assert file_io.guess_text_column(pd.DataFrame({"a": ["1", "2"], "b": REVIEWS[:2]})) == 1
    assert file_io.has_khmer(REVIEWS[0]) and not file_io.has_khmer(REVIEWS[2])
