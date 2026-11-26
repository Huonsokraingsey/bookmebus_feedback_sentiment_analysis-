"""src/preprocess.py must give exactly the same tokens as notebook 01 did (so the web app matches training)."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.preprocess import preprocess_text  # noqa: E402


def test_same_tokens_as_notebook():
    examples = pd.read_csv(os.path.join(ROOT, "data", "processed", "preprocess_examples.csv"))
    for text, expected in zip(examples["text"], examples["tokens"]):
        assert " ".join(preprocess_text(text)) == expected


def test_negation_and_emojis_are_kept():
    tokens = preprocess_text("អ្នកបើកបរមិនសូវគោរពអ្នកដំណើរទេ 👍 not good")
    for word in ["មិន", "ទេ", "👍", "not"]:
        assert word in tokens


def test_repeats_punctuation_numbers_urls():
    tokens = preprocess_text("sooooo good!!! 2024 ១២ https://bookmebus.com")
    assert tokens == ["sooo", "good"]
