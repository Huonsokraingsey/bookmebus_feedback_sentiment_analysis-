"""Loads the models trained in Part B and predicts sentiment in batches.

The app only USES the trained models: nothing is retrained, and the evaluation numbers shown in the app are read
from results/metrics/ (written by the notebooks).
"""
import json
import os
import re
import sys
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.preprocess import preprocess_text  # noqa: E402  (same code as notebook 01)

TFIDF_PATH = os.path.join(ROOT, "models", "tfidf_logreg.joblib")
XLMR_DIR = os.path.join(ROOT, "models", "xlmr_best")
METRICS_DIR = os.path.join(ROOT, "results", "metrics")

LABEL_NAMES = ["negative", "positive"]   # the models are 2-class (see README: neutral was dropped)
UNCERTAIN_BELOW = 0.6                    # top probability below this → shown as "uncertain"
ZERO_WIDTH = re.compile("[​‌‍﻿]")


def load_metrics():
    """{model key: metrics dict} for every model that has a results/metrics/<key>.json file."""
    metrics = {}
    for key in ["tfidf_logreg", "bilstm", "xlmr"]:
        path = os.path.join(METRICS_DIR, f"{key}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                metrics[key] = json.load(f)
    return metrics


def tfidf_available():
    return os.path.exists(TFIDF_PATH)


def xlmr_available():
    return os.path.exists(os.path.join(XLMR_DIR, "config.json"))


def tokenize(texts):
    """khmer-nltk tokens of every text (the same preprocessing as training)."""
    return [preprocess_text(text) for text in texts]


# ---------------------------------------------------------------- TF-IDF + Logistic Regression
def load_tfidf():
    import joblib
    if not os.path.exists(TFIDF_PATH):
        raise FileNotFoundError("models/tfidf_logreg.joblib is missing: run notebooks/02_statistical_ml_tfidf_logreg.ipynb")
    return joblib.load(TFIDF_PATH)


def predict_tfidf(bundle, texts, token_lists):
    """Returns (probabilities of shape (n, 2), top-3 words per review that pushed the prediction)."""
    from scipy.sparse import hstack
    joined = [" ".join(tokens) for tokens in token_lists]
    word = bundle["word_vectorizer"].transform(joined)        # word features use the tokens (like notebook 02)
    char = bundle["char_vectorizer"].transform(list(texts))   # char features use the raw Khmer text (like notebook 02)
    features = {"word": word, "char": char, "word + char": None}[bundle["features"]]
    if features is None:
        features = hstack([word, char]).tocsr()
    model = bundle["model"]
    probs = model.predict_proba(features)

    # Explanation: contribution of a word = its Logistic Regression weight × its TF-IDF value in this review.
    # Only word features are used (character pieces are hard to read).
    top_words = [[] for _ in texts]
    if bundle["features"] in ("word", "word + char"):
        n_word = word.shape[1]
        weights = model.coef_[0][:n_word]
        names = bundle["word_vectorizer"].get_feature_names_out()
        contributions = word.multiply(weights).tocsr()
        for i in range(len(texts)):
            row = contributions.getrow(i)
            if row.nnz == 0:
                continue
            idx, values = row.indices, row.data
            if probs[i, 1] >= 0.5:      # predicted positive → words with the largest positive push
                order = [j for j in np.argsort(-values) if values[j] > 0]
            else:                       # predicted negative → words with the largest negative push
                order = [j for j in np.argsort(values) if values[j] < 0]
            top_words[i] = [names[idx[j]] for j in order[:3]]
    return probs, top_words


def explain_tfidf(bundle, tokens):
    """Contribution of every word feature (word or word pair) of ONE review: weight × TF-IDF value.
    > 0 pushes towards positive, < 0 towards negative. Only the word part of the model (it also uses character pieces)."""
    if bundle["features"] not in ("word", "word + char"):
        return {}
    word = bundle["word_vectorizer"].transform([" ".join(tokens)])
    weights = bundle["model"].coef_[0][:word.shape[1]]
    names = bundle["word_vectorizer"].get_feature_names_out()
    row = word.multiply(weights).tocsr()
    return {names[j]: float(value) for j, value in zip(row.indices, row.data)}


# ---------------------------------------------------------------- XLM-R
def load_xlmr():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not xlmr_available():
        raise FileNotFoundError("models/xlmr_best/ is missing: run notebooks/04_transformer_xlmr.ipynb")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(XLMR_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(XLMR_DIR).to(device).eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


def normalize_text(text):
    """Same normalization as notebook 04: Unicode NFC + remove zero-width characters."""
    return ZERO_WIDTH.sub("", unicodedata.normalize("NFC", str(text)))


def predict_xlmr(bundle, texts, max_length=128):
    import torch
    inputs = bundle["tokenizer"]([normalize_text(t) for t in texts], padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(bundle["device"])
    with torch.no_grad():
        logits = bundle["model"](**inputs).logits.float()
    return torch.softmax(logits, dim=1).cpu().numpy()


# ---------------------------------------------------------------- shared
def to_sentiment(probs):
    """Predicted class per review, or "uncertain" when the model is not sure (top probability < 0.6)."""
    confidence = probs.max(axis=1)
    labels = np.array(LABEL_NAMES)[probs.argmax(axis=1)]
    return np.where(confidence < UNCERTAIN_BELOW, "uncertain", labels), confidence
