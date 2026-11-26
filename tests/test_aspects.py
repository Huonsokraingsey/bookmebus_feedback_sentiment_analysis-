"""Keyword-based aspect matching and date-column guessing (web app, Stage 2)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import aspects  # noqa: E402
import file_io  # noqa: E402


def test_khmer_and_english_keywords():
    assert "Seat & comfort" in aspects.find_aspects("កៅអីខូច")                 # Khmer: matched anywhere
    assert "Driver & driving" in aspects.find_aspects("The driver was rude")
    assert "Staff & service" in aspects.find_aspects("The driver was rude")
    assert aspects.find_aspects("ឡានចេញយឺតពីរម៉ោង")[0] == "Punctuality & waiting"


def test_english_keywords_are_whole_words():
    assert "Air conditioning & temperature" in aspects.find_aspects("The AC was broken")
    assert "Air conditioning & temperature" not in aspects.find_aspects("I came back by bus")   # "ac" inside "back"


def test_aspect_table_counts_and_shares():
    texts = ["កៅអីខូច", "កៅអីស្រួល", "អ្នកបើកបរល្អ"]
    table = aspects.aspect_table(texts, ["negative", "positive", "positive"]).set_index("aspect")
    assert table.loc["Seat & comfort", "reviews"] == 2
    assert table.loc["Seat & comfort", "% negative"] == 50.0
    assert table.loc["Driver & driving", "% positive"] == 100.0
    assert table.loc["Price & money", "reviews"] == 0


def test_guess_date_column():
    table = pd.DataFrame({"id": ["1", "2", "3"], "review": ["ល្អ", "អន់", "ល្អណាស់"],
                          "submitted": ["2024-01-05", "2024-02-11", "2024-03-02"]})
    assert file_io.guess_date_column(table, "review") == "submitted"
    assert file_io.guess_date_column(table[["id", "review"]], "review") == "(none)"
