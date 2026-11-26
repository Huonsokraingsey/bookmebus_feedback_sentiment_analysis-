"""Figures for the presentation: results/figures/pipeline.png and results/figures/05_error_categories.png.

The error-category counts are the output of notebooks/05_comparison_and_error_analysis.ipynb (section 6.2, final run
with gold labels): 16 XLM-R errors, read and categorized by hand.
Run:  python scripts/make_slide_figures.py
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")

# ---------------------------------------------------------------- pipeline
steps = [
    ("1. Data", "BookMeBus reviews\n4,114 rows\nEnglish + Khmer translation", "#e3f2fd"),
    ("2. Labeling", "Silver labels from text (AI)\n+ gold labels (2 humans,\nκ = 0.902) for the Khmer test", "#e8f5e9"),
    ("3. Pre-processing", "NFC, clean, collapse repeats\nkhmer-nltk segmentation\nkeep negation + emojis", "#fff3e0"),
    ("4. Splits", "Train / val / test on translations\n(80 / 10 / 10)\n+ test_native: real Khmer", "#f3e5f5"),
    ("5. Three models", "TF-IDF + Logistic Regression\nBiLSTM + fastText\nXLM-R (fine-tuned)", "#e0f7fa"),
    ("6. Evaluation", "Macro-F1, bootstrap CIs\ntranslation gap\nerror analysis", "#fce4ec"),
    ("7. Web app (add-on)", "Upload feedback → dashboard\nor analyse one review", "#eeeeee"),
]
fig, ax = plt.subplots(figsize=(16, 4.2))
ax.set_xlim(0, len(steps) * 2.3); ax.set_ylim(0, 3); ax.axis("off")
for i, (title, text, color) in enumerate(steps):
    x = i * 2.3 + 0.1
    ax.add_patch(FancyBboxPatch((x, 0.45), 1.95, 2.1, boxstyle="round,pad=0.05", fc=color, ec="#455a64", lw=1.2))
    ax.text(x + 0.975, 2.25, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(x + 0.975, 1.3, text, ha="center", va="center", fontsize=9.2, linespacing=1.5)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + 2.0, 1.5), (x + 2.35, 1.5), arrowstyle="-|>", mutation_scale=18, color="#455a64"))
ax.set_title("Project pipeline: Khmer sentiment analysis of BookMeBus reviews", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "pipeline.png"), dpi=160)
plt.close()

# ---------------------------------------------------------------- error categories (notebook 05, section 6.2)
categories = {"mixed sentiment": 6, "indirect / polite complaint": 5, "very short review": 2,
              "possible labeling error (silver)": 2, "factual complaint (no opinion words)": 1}
fig, ax = plt.subplots(figsize=(8, 3.6))
names, counts = list(categories)[::-1], list(categories.values())[::-1]
ax.barh(names, counts, color=["#90a4ae", "#90a4ae", "#90a4ae", "#c62828", "#c62828"])
for i, c in enumerate(counts):
    ax.text(c + 0.1, i, str(c), va="center", fontsize=11)
ax.set_xlabel("Number of XLM-R errors (16 in total: 7 real Khmer + 9 translated)")
ax.set_title("Why does the best model (XLM-R) still make mistakes?")
ax.set_xlim(0, 7.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "05_error_categories.png"), dpi=160)
print("wrote results/figures/pipeline.png and results/figures/05_error_categories.png")
