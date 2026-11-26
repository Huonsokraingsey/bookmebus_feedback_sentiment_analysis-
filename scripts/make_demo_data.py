"""Builds data/demo/sample_feedback.csv for the web app's "Try sample data" button.

Only TEST-set reviews are used (test_native + test_translated), never training or validation reviews,
and the label columns are removed. Run after notebook 01:  python scripts/make_demo_data.py
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 45

parts = []
for test_set in ["test_native", "test_translated"]:
    data = pd.read_csv(os.path.join(ROOT, "data", "processed", f"{test_set}.csv"))
    parts.append(data[["row_id", "text_km"]])

demo = pd.concat(parts).sample(frac=1, random_state=SEED)          # mix the two test sets
demo = demo.rename(columns={"row_id": "review_id", "text_km": "review"})

os.makedirs(os.path.join(ROOT, "data", "demo"), exist_ok=True)
out = os.path.join(ROOT, "data", "demo", "sample_feedback.csv")
demo.to_csv(out, index=False, encoding="utf-8-sig")
print(f"Wrote {len(demo)} test-set reviews (no labels) to {out}")
