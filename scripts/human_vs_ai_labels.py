"""How well do the AI (silver) labels agree with a human? (Part A quality check)

Compares data/labels/human_review_sample.csv (150 random reviews, 50 per AI label, labeled by a team member who did not
see the AI label) with the silver labels in data/labels/bookmebus_labeled.csv.
Writes data/labels/human_review_results.md.   Run:  python scripts/human_vs_ai_labels.py
"""
import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["negative", "neutral", "positive"]

sample = pd.read_csv(os.path.join(ROOT, "data", "labels", "human_review_sample.csv"), dtype=str,
                     keep_default_na=False, encoding="utf-8-sig")
silver = pd.read_csv(os.path.join(ROOT, "data", "labels", "bookmebus_labeled.csv"), dtype=str, keep_default_na=False)
data = sample.merge(silver[["row_id", "label"]].rename(columns={"label": "ai"}), on="row_id")

annotators = [c for c in ["annotator_A", "annotator_B"]
              if data[c].str.strip().str.lower().isin(LABELS).all()]
lines = ["# Human check of the AI (silver) labels", "",
         f"150 random reviews (50 per AI label, shuffled), labeled by team members **without seeing the AI label**. "
         f"Annotators who filled in every row: {', '.join(annotators) or 'none'}.", ""]
for annotator in annotators:
    human = data[annotator].str.strip().str.lower()
    agree = (human == data["ai"]).mean()
    kappa = cohen_kappa_score(human, data["ai"])
    two_class = data["ai"].isin(["positive", "negative"]) & human.isin(["positive", "negative"])
    agree2 = (human[two_class] == data["ai"][two_class]).mean()
    table = pd.crosstab(pd.Series(data["ai"], name="AI label"), pd.Series(human, name=f"human ({annotator})")) \
        .reindex(index=LABELS, columns=LABELS, fill_value=0)
    per_class = {label: (human[data["ai"] == label] == label).mean() for label in LABELS}
    lines += [f"## {annotator} vs. AI", "",
              f"- Agreement (3 classes): **{agree:.1%}**, Cohen's kappa **{kappa:.3f}**",
              f"- Agreement when both say positive or negative (the 2 classes the models use): "
              f"**{agree2:.1%}** ({int(two_class.sum())} reviews)",
              "- Share of each AI label that the human confirmed: "
              + ", ".join(f"{label} {value:.0%}" for label, value in per_class.items()), "",
              table.to_markdown(), ""]
    print(f"{annotator}: agreement {agree:.1%}, kappa {kappa:.3f}, 2-class agreement {agree2:.1%} "
          f"on {int(two_class.sum())} reviews, per class {({k: round(v, 3) for k, v in per_class.items()})}")
    print(table)
if len(annotators) == 2:
    a, b = (data[c].str.strip().str.lower() for c in annotators)
    lines += [f"Agreement between the two annotators: {(a == b).mean():.1%}, kappa {cohen_kappa_score(a, b):.3f}", ""]

with open(os.path.join(ROOT, "data", "labels", "human_review_results.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("wrote data/labels/human_review_results.md")
