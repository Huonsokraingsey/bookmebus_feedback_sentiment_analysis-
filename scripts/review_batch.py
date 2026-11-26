"""Consistency review of borderline silver labels (Part A).

One reviewer re-labels every borderline review **blind** (the first-pass label is not shown),
using LABELING_GUIDELINES.md including rules R1-R9. The reviewer's label becomes the final label.

Borderline = first-pass label neutral, mixed = yes, confidence low, excluded by a labeler,
very short text (<= 5 words), or the text contains a contrast word (but / except / however /
ប៉ុន្តែ / តែ ...). Keywords are only used to SELECT reviews to re-read, never to decide labels.

Usage:
  python scripts/review_batch.py init
  python scripts/review_batch.py show --part long --n 100
  python scripts/review_batch.py append --part long --file review.json
  python scripts/review_batch.py status
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_batch import ROOT, FileLock, load_data, load_progress  # noqa: E402

REVIEW_SET = os.path.join(ROOT, "data", "labels", "work", "review_set.csv")
REVIEW = os.path.join(ROOT, "data", "labels", "labels_review.csv")
CONTRAST = (r"\b(?:but|however|except|although|though|only|apart from|other than|otherwise|"
            r"aside from|besides|yet)\b|ប៉ុន្តែ|តែ|លើកលែង|ទោះ")


def cmd_init(args):
    df = load_data()
    p = load_progress()
    p["row_id"] = p.row_id.astype(int)
    m = p.merge(df, on="row_id")
    text = m.text.fillna("")
    agent_excluded = (m.exclude == "yes") & (m.batch.str.startswith("s"))
    pick = ((m.label == "neutral") | (m.mixed == "yes") | (m.confidence == "low") | agent_excluded
            | (text.str.split().str.len() <= 5) & (m.batch != "inspect")
            | text.str.contains(CONTRAST, case=False, regex=True))
    # part "short" (<= 40 characters; Khmer has no spaces, so characters not words) and part "long"
    # go to two reviewers, so each rule is applied by one person
    m["part"] = (text.str.len() <= 40).map({True: "short", False: "long"})
    sel = m.loc[pick, ["row_id", "part"]].sample(frac=1, random_state=45)
    sel.to_csv(REVIEW_SET, index=False)
    print(f"review set: {len(sel)} rows", sel.part.value_counts().to_dict())


def done_ids():
    if not os.path.exists(REVIEW):
        return set()
    return set(pd.read_csv(REVIEW).row_id.astype(int))


def cmd_show(args):
    df = load_data()
    rs = pd.read_csv(REVIEW_SET)
    ids = rs.loc[rs.part == args.part, "row_id"].tolist()
    done = done_ids()
    todo = [i for i in ids if i not in done]
    print(f"# {len(todo)} rows left; showing {min(args.n, len(todo))}")
    for rid in todo[:args.n]:
        r = df.loc[rid]
        tr = r.Translation if str(r.Translation).strip() != str(r.text).strip() else "(same as text)"
        print(f"\n[{rid}]\nTEXT: {r.text}\nTRAN: {tr}")


def cmd_append(args):
    with open(args.file, encoding="utf-8") as f:
        items = json.load(f)
    rs = pd.read_csv(REVIEW_SET)
    allowed = set(rs.loc[rs.part == args.part, "row_id"])
    done = done_ids()
    errors, rows = [], []
    for it in items:
        rid = int(it["row_id"])
        lab, exc = (it.get("label") or "").strip(), it.get("exclude", "no")
        if rid not in allowed:
            errors.append(f"{rid}: not in review set"); continue
        if rid in done:
            errors.append(f"{rid}: already reviewed")
        if exc == "yes" and (lab or not it.get("exclude_reason")):
            errors.append(f"{rid}: excluded rows need empty label and exclude_reason")
        if exc == "no" and lab not in ("positive", "neutral", "negative"):
            errors.append(f"{rid}: bad label {lab!r}")
        if exc == "no" and it.get("confidence") not in ("high", "medium", "low"):
            errors.append(f"{rid}: bad confidence")
        mixed = it.get("mixed", "no")
        if mixed not in ("yes", "no") or (mixed == "yes" and lab != "neutral"):
            errors.append(f"{rid}: mixed must be yes/no and yes only for neutral (rule R9)")
        if it.get("translation_issue", "no") not in ("yes", "no"):
            errors.append(f"{rid}: translation_issue must be yes/no")
        rows.append(dict(row_id=rid, label=lab, confidence=it.get("confidence", "") if exc == "no" else "",
                         mixed=mixed, exclude=exc, exclude_reason=it.get("exclude_reason", "") if exc == "yes" else "",
                         translation_issue=it.get("translation_issue", "no"), note=it.get("note", ""),
                         reason=it.get("reason", ""), rule=it.get("rule", "")))
    if len({r["row_id"] for r in rows}) != len(rows):
        errors.append("duplicate row_id inside the batch")
    if errors:
        print("REJECTED, nothing written:\n  " + "\n  ".join(errors)); sys.exit(1)
    with FileLock():
        if done_ids() & {r["row_id"] for r in rows}:
            print("REJECTED: some rows were already reviewed"); sys.exit(1)
        out = pd.DataFrame(rows)
        out.to_csv(REVIEW, mode="a", header=not os.path.exists(REVIEW), index=False, encoding="utf-8")
    left = len(allowed - done_ids())
    print(f"OK: appended {len(rows)}; part {args.part} has {left} rows left")


def cmd_status(args):
    rs = pd.read_csv(REVIEW_SET)
    rs["done"] = rs.row_id.isin(done_ids())
    print(rs.groupby("part").done.agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    a = sub.add_parser("show"); a.add_argument("--part", choices=["short", "long"], required=True)
    a.add_argument("--n", type=int, default=100)
    a = sub.add_parser("append"); a.add_argument("--part", choices=["short", "long"], required=True)
    a.add_argument("--file", required=True)
    sub.add_parser("status")
    args = p.parse_args()
    {"init": cmd_init, "show": cmd_show, "append": cmd_append, "status": cmd_status}[args.cmd](args)
