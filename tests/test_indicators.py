"""Word statistics on a tiny hand-made example where the right answers are known."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import indicators  # noqa: E402

TOKENS = [
    ["seat", "bad"],             # negative
    ["seat", "good"],            # positive
    ["seat", "bad", "seat"],     # negative ("seat" twice, but counted once for this review)
    ["driver", "good"],          # positive
    ["driver", "good", "x"],     # positive ("x" is a single character → hidden)
]
SENTIMENTS = ["negative", "positive", "negative", "positive", "positive"]


def table(min_reviews=1, hidden=frozenset()):
    return indicators.word_table(TOKENS, SENTIMENTS, set(hidden), min_reviews=min_reviews).set_index("word")


def test_counts_once_per_review():
    t = table()
    assert t.loc["seat", "reviews"] == 3
    assert t.loc["good", "reviews"] == 3
    assert t.loc["bad", "reviews"] == 2
    assert t.loc["driver", "reviews"] == 2
    assert "x" not in t.index                       # single characters are hidden


def test_percentages():
    t = table()
    assert t.loc["seat", "% negative"] == pytest.approx(66.7)
    assert t.loc["seat", "% positive"] == pytest.approx(33.3)
    assert t.loc["good", "% positive"] == 100.0
    assert t.loc["bad", "% negative"] == 100.0
    assert (t["% positive"] + t["% uncertain"] + t["% negative"]).round(0).eq(100).all()


def test_indicator_score_signs_and_value():
    t = table()
    assert t.loc["bad", "indicator score"] > 0       # only in negative reviews
    assert t.loc["seat", "indicator score"] > 0      # mostly negative
    assert t.loc["good", "indicator score"] < 0      # only in positive reviews
    assert t.loc["bad", "indicator score"] > t.loc["seat", "indicator score"]
    # seat: in 2 of 2 negative and 1 of 3 positive reviews, add-0.5 smoothing
    expected = math.log(((2 + 0.5) / (0 + 0.5)) / ((1 + 0.5) / (2 + 0.5)))
    assert t.loc["seat", "indicator score"] == pytest.approx(expected, abs=1e-3)


def test_log_odds_is_symmetric():
    assert indicators.log_odds(3, 10, 1, 8) == pytest.approx(-indicators.log_odds(1, 8, 3, 10))
    assert indicators.log_odds(2, 10, 2, 10) == pytest.approx(0)


def test_min_reviews_and_hidden_words():
    assert set(table(min_reviews=3).index) == {"seat", "good"}
    assert "seat" not in table(hidden={"seat"}).index


def test_negation_hidden_from_display_only():
    hidden = indicators.hidden_words()
    for word in ["មិន", "ទេ", "not", "no", "the", "ខ្ញុំ"]:
        assert word in hidden
    assert "ល្អ" not in hidden and "កៅអី" not in hidden
