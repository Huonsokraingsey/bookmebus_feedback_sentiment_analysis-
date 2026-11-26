"""Word statistics for the dashboard: how often a word is mentioned, and how strongly it is linked to
negative or positive feedback (smoothed log-odds ratio).

Display only: function words, negation words and single characters are HIDDEN FROM THE CHARTS here,
but the sentiment model still sees every word, including negation (មិន, ទេ, not, …), when it predicts.
"""
import math
import os

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

APP_DIR = os.path.dirname(os.path.abspath(__file__))
NEGATION_WORDS = {"មិន", "ទេ", "គ្មាន", "កុំ", "ពុំ", "not", "no"}
SENTIMENTS = ["positive", "uncertain", "negative"]


def load_khmer_stopwords(path=os.path.join(APP_DIR, "stopwords_km.txt")):
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def hidden_words():
    """Words hidden from the word charts (display only)."""
    return set(ENGLISH_STOP_WORDS) | load_khmer_stopwords() | NEGATION_WORDS


def review_words(tokens, hidden):
    """The set of words of one review that we show in charts: each word counts ONCE per review."""
    return {word for word in tokens if word not in hidden and len(word) > 1 and not word.isdigit()}


def log_odds(neg_with, neg_total, pos_with, pos_total, smoothing=0.5):
    """Smoothed log-odds ratio of a word appearing in negative vs. positive reviews.
    > 0: more typical of negative reviews, < 0: more typical of positive reviews."""
    neg_odds = (neg_with + smoothing) / (neg_total - neg_with + smoothing)
    pos_odds = (pos_with + smoothing) / (pos_total - pos_with + smoothing)
    return math.log(neg_odds / pos_odds)


def word_table(token_lists, sentiments, hidden, min_reviews=3):
    """One row per word: in how many reviews it appears, the sentiment split of those reviews, and the indicator score.
    Only words that appear in at least `min_reviews` reviews."""
    sentiments = list(sentiments)
    word_sets = [review_words(tokens, hidden) for tokens in token_lists]
    neg_total = sum(s == "negative" for s in sentiments)
    pos_total = sum(s == "positive" for s in sentiments)

    counts = {}
    for words, sentiment in zip(word_sets, sentiments):
        for word in words:
            counts.setdefault(word, {"positive": 0, "uncertain": 0, "negative": 0})[sentiment] += 1

    rows = []
    for word, c in counts.items():
        n = c["positive"] + c["uncertain"] + c["negative"]
        if n < min_reviews:
            continue
        rows.append({
            "word": word, "reviews": n,
            "% positive": round(100 * c["positive"] / n, 1),
            "% uncertain": round(100 * c["uncertain"] / n, 1),
            "% negative": round(100 * c["negative"] / n, 1),
            "indicator score": round(log_odds(c["negative"], neg_total, c["positive"], pos_total), 3),
        })
    columns = ["word", "reviews", "% positive", "% uncertain", "% negative", "indicator score"]
    return pd.DataFrame(rows, columns=columns).sort_values(["reviews", "word"], ascending=[False, True]).reset_index(drop=True)
