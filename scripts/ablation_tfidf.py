"""Small ablation studies for the presentation (validation set only – the test sets are used once, in notebooks 02–04).

1. Pre-processing impact: word TF-IDF + Logistic Regression (C = 10, the best C in notebook 02) trained with different
   pre-processing choices; validation macro-F1.
2. Class weights: Logistic Regression with class_weight="balanced" vs. no class weights (word + char, C = 10).

Writes results/metrics/ablation.json and results/figures/ablation_preprocessing.png.
Run after notebook 01:   python scripts/ablation_tfidf.py
"""
import json
import os
import re
import sys
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from khmernltk import word_tokenize  # noqa: E402
from src.preprocess import preprocess_text  # noqa: E402

SEED = 45
NEGATION = {"មិន", "ទេ", "គ្មាន", "កុំ", "ពុំ", "not", "no"}
with open(os.path.join(ROOT, "app", "stopwords_km.txt"), encoding="utf-8") as f:
    KHMER_STOPWORDS = {line.strip() for line in f if line.strip() and not line.startswith("#")}

train = pd.read_csv(os.path.join(ROOT, "data", "processed", "train.csv"))
val = pd.read_csv(os.path.join(ROOT, "data", "processed", "val.csv"))


def raw_segmented(text):
    """Only khmer-nltk segmentation: no normalization, no punctuation/number removal, no lowercasing."""
    return [t for t in word_tokenize(str(text)) if t.strip()]


VARIANTS = {
    "Our pre-processing (segmentation, cleaning, negation kept)": lambda d: d["tokens"].str.split(),
    "Without negation words (មិន, ទេ, គ្មាន, not …)": lambda d: d["tokens"].str.split().apply(
        lambda t: [w for w in t if w not in NEGATION]),
    "Stopwords removed (function words + negation)": lambda d: d["tokens"].str.split().apply(
        lambda t: [w for w in t if w not in KHMER_STOPWORDS | NEGATION]),
    "Segmentation only (no cleaning / normalization)": lambda d: d["text_km"].apply(raw_segmented),
    "No word segmentation (split on spaces only)": lambda d: d["text_km"].apply(
        lambda s: unicodedata.normalize("NFC", str(s)).lower().split()),
}


def word_tfidf_f1(train_tokens, val_tokens):
    vec = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None, lowercase=False,
                          ngram_range=(1, 2), min_df=2)
    model = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=SEED)
    model.fit(vec.fit_transform(train_tokens), train["label_id"])
    return f1_score(val["label_id"], model.predict(vec.transform(val_tokens)), average="macro"), len(vec.vocabulary_)


results = {"preprocessing": [], "class_weights": []}
for name, make_tokens in VARIANTS.items():
    f1, vocab = word_tfidf_f1(make_tokens(train), make_tokens(val))
    results["preprocessing"].append({"variant": name, "val_macro_f1": round(f1, 4), "features": vocab})
    print(f"{f1:.3f}  ({vocab:6d} features)  {name}")

# class weights (best setting of notebook 02: word + char, C = 10)
word = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2), min_df=2)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
X_train = hstack([word.fit_transform(train["tokens"]), char.fit_transform(train["text_km"])]).tocsr()
X_val = hstack([word.transform(val["tokens"]), char.transform(val["text_km"])]).tocsr()
for weights in ["balanced", None]:
    model = LogisticRegression(C=10, class_weight=weights, max_iter=2000, random_state=SEED).fit(X_train, train["label_id"])
    pred = model.predict(X_val)
    f1s = f1_score(val["label_id"], pred, average=None)
    results["class_weights"].append({"class_weight": str(weights), "val_macro_f1": round(f1_score(val["label_id"], pred, average="macro"), 4),
                                     "val_f1_negative": round(f1s[0], 4), "val_f1_positive": round(f1s[1], 4)})
    print(f"class_weight={weights}: {results['class_weights'][-1]}")
results["note"] = (f"Validation set only ({len(val)} reviews). Training set: {int((train.label_id == 0).sum())} negative / "
                   f"{int((train.label_id == 1).sum())} positive.")

with open(os.path.join(ROOT, "results", "metrics", "ablation.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

table = pd.DataFrame(results["preprocessing"])
plt.rcParams["font.family"] = ["Khmer OS", "Khmer UI", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(9, 3.8))
colors = ["#1565c0"] + ["#90a4ae"] * (len(table) - 1)
ax.barh(table["variant"][::-1], table["val_macro_f1"][::-1], color=colors[::-1])
for i, v in enumerate(table["val_macro_f1"][::-1]):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=10)
ax.set_xlim(min(table["val_macro_f1"]) - 0.05, 1.0)
ax.set_xlabel("Validation macro-F1 (word TF-IDF + Logistic Regression)")
ax.set_title("Pre-processing impact")
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "results", "figures", "ablation_preprocessing.png"), dpi=150)
print("wrote results/metrics/ablation.json and results/figures/ablation_preprocessing.png")
