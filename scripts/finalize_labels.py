"""Part A - FINAL step: merge silver labels, write the outputs, run consistency checks.

Inputs  : data/bookmebus_reviews_khmer.csv (never modified)
          data/labels/labels_progress.csv  (first pass, incl. pilot + auto-excluded rows)
          data/labels/labels_review.csv    (blind consistency review of borderline rows; wins)
Outputs : data/labels/bookmebus_labeled.csv
          data/labels/label_report.md
          data/labels/human_review_sample.csv
          data/labels/gold_test_annotation_A.csv, gold_test_annotation_B.csv
"""
import os
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SEED = 45
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(ROOT, "data", "labels")
KHMER = re.compile(r"[ក-៿]")
LABELS = ["positive", "neutral", "negative"]


def has_khmer(s):
    return isinstance(s, str) and bool(KHMER.search(s))


def norm_text(s):
    """Lower-case, drop punctuation/spaces: used only to find duplicate texts."""
    return re.sub(r"[^\wក-៿]+", " ", str(s).lower()).strip()


# ---------------------------------------------------------------- merge
data = pd.read_csv(os.path.join(ROOT, "data", "bookmebus_reviews_khmer.csv"))
data.insert(0, "row_id", range(len(data)))
first = pd.read_csv(os.path.join(LAB, "labels_progress.csv"), dtype=str, keep_default_na=False)
first["row_id"] = first.row_id.astype(int)
review = pd.read_csv(os.path.join(LAB, "labels_review.csv"), dtype=str, keep_default_na=False)
review["row_id"] = review.row_id.astype(int)
review_set = pd.read_csv(os.path.join(LAB, "work", "review_set.csv"))

assert first.row_id.is_unique and len(first) == len(data), "first pass must cover every row once"
assert review.row_id.is_unique, "a row was reviewed twice"
missing = set(review_set.row_id) - set(review.row_id)
assert not missing, f"{len(missing)} rows of the review set were not reviewed"

cols = ["label", "confidence", "mixed", "exclude", "exclude_reason", "translation_issue", "note"]
final = first.set_index("row_id")[cols].copy()
rv = review.set_index("row_id")
final.loc[rv.index, cols] = rv[cols]
final = final.reset_index()

out = data.merge(final, on="row_id", how="left")
out["is_original_khmer"] = out.text.apply(has_khmer).map({True: "yes", False: "no"})
# translation_issue: automatic (no Khmer letters in Translation) OR flagged by a labeler/reviewer
auto_tr = ~out.Translation.apply(has_khmer)
out["translation_issue"] = ((out.translation_issue == "yes") | auto_tr).map({True: "yes", False: "no"})
out.loc[out.exclude == "yes", ["label", "confidence", "mixed"]] = ["", "", "no"]
out = out[["text", "score", "Translation", "row_id", "label", "confidence", "mixed", "exclude",
           "exclude_reason", "is_original_khmer", "translation_issue", "note"]]

# ---------------------------------------------------------------- consistency checks
print("=== Consistency checks ===")
has_label = out.label.isin(LABELS)
excluded = out.exclude == "yes"
c1 = (has_label ^ excluded).all()
print(f"[{'OK' if c1 else 'FAIL'}] every row has exactly one of (label, exclude=yes)")
c2 = out.loc[~excluded, "label"].isin(LABELS).all() and set(out.label.unique()) <= set(LABELS + [""])
print(f"[{'OK' if c2 else 'FAIL'}] labels use only positive / neutral / negative")
out["_norm"] = out.text.apply(norm_text)
dup = out[(out._norm != "") & (out._norm != "nan") & out._norm.duplicated(keep=False)]
conflicts = dup.groupby("_norm").label.nunique()
conflicts = conflicts[conflicts > 1]
c3 = len(conflicts) == 0
print(f"[{'OK' if c3 else 'FAIL'}] duplicate texts have identical labels "
      f"({dup._norm.nunique()} duplicate groups, {len(conflicts)} conflicts)")
if not c3:
    print(dup[dup._norm.isin(conflicts.index)][["row_id", "text", "label"]].to_string())
c4 = out.loc[out.mixed == "yes", "label"].eq("neutral").all()
print(f"[{'OK' if c4 else 'FAIL'}] mixed = yes only on neutral rows (rule R9)")
c5 = out[["text", "score", "Translation"]].equals(
    pd.read_csv(os.path.join(ROOT, "data", "bookmebus_reviews_khmer.csv"))[["text", "score", "Translation"]])
print(f"[{'OK' if c5 else 'FAIL'}] original columns unchanged")
out = out.drop(columns="_norm")
if not all([c1, c2, c3, c4, c5]):
    sys.exit("consistency checks failed - nothing written")

counts = out.loc[~excluded, "label"].value_counts().reindex(LABELS)
print("\nFinal label counts:", counts.to_dict(), "| excluded:", int(excluded.sum()))

# ---------------------------------------------------------------- output 1
out.to_csv(os.path.join(LAB, "bookmebus_labeled.csv"), index=False, encoding="utf-8")

# ---------------------------------------------------------------- output 4: human review sample
lab = out[~excluded]
sample = (lab.groupby("label", group_keys=False).sample(50, random_state=SEED)
          .sample(frac=1, random_state=SEED))
sample = sample[["row_id", "text", "Translation"]].assign(annotator_A="", annotator_B="")
sample.to_csv(os.path.join(LAB, "human_review_sample.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- output 5: gold annotation files
native = lab[lab.is_original_khmer == "yes"].sample(frac=1, random_state=SEED)
gold = native[["row_id", "text"]].assign(label="", is_usable="", notes="")
for who in ("A", "B"):
    gold.to_csv(os.path.join(LAB, f"gold_test_annotation_{who}.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- output 3: label report
def pct_table(s):
    c = s.value_counts().reindex(LABELS).fillna(0).astype(int)
    return pd.DataFrame({"count": c, "percent": (100 * c / c.sum()).round(1)})


def md(df):
    return df.to_markdown()


bins = pd.cut(lab.score, [-0.01, 1.9, 3.9, 5.0], labels=["0–1.9", "2–3.9", "4–5"])
cross = pd.crosstab(lab.label, bins).reindex(LABELS)

# strongest disagreements: negative reviews with the highest scores, positive with the lowest
neg_hi = lab[lab.label == "negative"].sort_values(["score", "row_id"], ascending=[False, True]).head(15)
pos_lo = lab[lab.label == "positive"].sort_values(["score", "row_id"], ascending=[True, True]).head(15)
dis = pd.concat([neg_hi, pos_lo])[["row_id", "score", "label", "confidence", "text"]]
dis["text"] = dis.text.str.replace(r"\s+", " ", regex=True).str.replace("|", "/")

# first pass vs blind review (quality signal for the borderline rows)
cmp_ = review.merge(first[["row_id", "label", "exclude"]], on="row_id", suffixes=("_review", "_first"))
cmp_["first"] = cmp_.label_first.where(cmp_.exclude_first != "yes", "excluded")
cmp_["review"] = cmp_.label_review.where(cmp_.exclude_review != "yes", "excluded")
agree = (cmp_["first"] == cmp_["review"]).mean()
cmp_tab = pd.crosstab(cmp_["first"], cmp_["review"])
part = cmp_.merge(review_set, on="row_id").groupby("part").apply(
    lambda g: round(100 * (g["first"] == g["review"]).mean(), 1), include_groups=False)

tr_flagged = out[(out.translation_issue == "yes")][["row_id", "text", "Translation", "note"]]
excl = out[excluded][["row_id", "text", "exclude_reason"]]

report = f"""# Label Report – BookMeBus silver labels

Labels were generated by **Claude (Anthropic)** by reading each review, following
`LABELING_GUIDELINES.md`. They are **silver labels**, not human gold labels. All numbers below are
computed by `scripts/finalize_labels.py`.

## 1. Overall label distribution ({len(lab)} labeled reviews)

{md(pct_table(lab.label))}

- Excluded rows: **{int(excluded.sum())}**
- Low-confidence labels: **{int((lab.confidence == 'low').sum())}** · medium: {int((lab.confidence == 'medium').sum())} · high: {int((lab.confidence == 'high').sum())}
- Mixed (balanced neutral) reviews: **{int((lab.mixed == 'yes').sum())}**
- Rows with `translation_issue = yes`: **{int((out.translation_issue == 'yes').sum())}**

## 2. Khmer-original reviews ({len(lab[lab.is_original_khmer == 'yes'])} labeled)

{md(pct_table(lab.loc[lab.is_original_khmer == 'yes', 'label']))}

## 3. New label vs. score bins (analysis only – the score was never used for labeling)

{md(cross)}

## 4. The 30 strongest disagreements between label and score

The 15 negative reviews with the highest scores and the 15 positive reviews with the lowest scores.

{dis.to_markdown(index=False)}

## 5. Consistency review of borderline labels

After the first pass ({len(first)} rows, 6 labeling batches running in parallel plus the pilot), all
borderline rows ({len(review_set)}: first-pass neutral, mixed, low-confidence, excluded, very short, or
containing a contrast word such as "but") were re-labeled **blind** by one reviewer per text length,
using the extra rules R1–R9. The review label is the final label.

- Agreement between first pass and blind review: **{100 * agree:.1f}%**
  (short texts: {part.get('short', float('nan'))}%, long texts: {part.get('long', float('nan'))}%)

First pass (rows) vs. review (columns):

{md(cmp_tab)}

## 6. Excluded rows

{excl.to_markdown(index=False)}

## 7. Rows with a translation issue

{tr_flagged.to_markdown(index=False)}
"""
with open(os.path.join(LAB, "label_report.md"), "w", encoding="utf-8") as f:
    f.write(report)

print(f"\nWrote bookmebus_labeled.csv ({len(out)} rows), label_report.md, human_review_sample.csv "
      f"({len(sample)} rows), gold_test_annotation_A/B.csv ({len(gold)} rows each)")
print("First pass vs review agreement on borderline rows:", f"{100 * agree:.1f}%", part.to_dict())
