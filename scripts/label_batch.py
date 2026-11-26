"""Helper for Part A labeling (silver labels written by Claude).

The labels themselves are decided by reading each review; this script only
  * splits the unlabeled rows into shards,
  * shows the next batch of a shard (text + Translation, never the score),
  * validates a labeled batch and appends it to data/labels/labels_progress.csv.

Usage:
  python scripts/label_batch.py init --shards 6
  python scripts/label_batch.py show --shard 1 --n 100
  python scripts/label_batch.py append --shard 1 --file batch.json
  python scripts/label_batch.py status
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "bookmebus_reviews_khmer.csv")
PROGRESS = os.path.join(ROOT, "data", "labels", "labels_progress.csv")
SHARDS = os.path.join(ROOT, "data", "labels", "work", "shards.csv")
LOCK = PROGRESS + ".lock"

KHMER = re.compile(r"[ក-៿]")
COLUMNS = ["row_id", "label", "confidence", "mixed", "exclude", "exclude_reason",
           "is_original_khmer", "translation_issue", "note", "reason", "batch"]
AUTO_EXCLUDE = {  # found during INSPECT
    507: "missing text (Translation is a placeholder 'none')",
    790: "missing text (Translation is a placeholder 'none')",
    3009: "missing text (Translation is a placeholder 'none')",
    442: "original text corrupted by Excel (#NAME?)",
    3280: "original text corrupted by Excel (#NAME?)",
    1062: "number-only text ('250')",
}


def load_data():
    df = pd.read_csv(DATA)
    df.insert(0, "row_id", range(len(df)))
    return df


def has_khmer(s):
    return isinstance(s, str) and bool(KHMER.search(s))


def load_progress():
    if not os.path.exists(PROGRESS):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(PROGRESS, dtype=str, keep_default_na=False)


class FileLock:
    """Very small cross-process lock (several labelers append to one file)."""
    def __enter__(self):
        for _ in range(600):
            try:
                self.fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                time.sleep(0.1)
        raise RuntimeError("could not get lock " + LOCK)

    def __exit__(self, *exc):
        os.close(self.fd)
        os.remove(LOCK)


def cmd_init(args):
    df = load_data()
    with FileLock():
        prog = load_progress()
        done = set(prog.row_id.astype(int))
        rows = []
        for rid, reason in AUTO_EXCLUDE.items():
            if rid in done:
                continue
            r = df.loc[rid]
            rows.append(dict(row_id=rid, label="", confidence="", mixed="no", exclude="yes",
                             exclude_reason=reason,
                             is_original_khmer="no",
                             translation_issue="yes" if rid in (507, 790, 3009, 1062) else "no",
                             note="", reason="auto-excluded at INSPECT", batch="inspect"))
        if rows:
            prog = pd.concat([prog, pd.DataFrame(rows)], ignore_index=True)
            prog[COLUMNS].to_csv(PROGRESS, index=False, encoding="utf-8")
        done = set(prog.row_id.astype(int))
    todo = df.loc[~df.row_id.isin(done), "row_id"].sample(frac=1, random_state=45).tolist()
    shard = [i % args.shards + 1 for i in range(len(todo))]
    os.makedirs(os.path.dirname(SHARDS), exist_ok=True)
    pd.DataFrame({"row_id": todo, "shard": shard}).sort_values(["shard", "row_id"]).to_csv(SHARDS, index=False)
    print(f"{len(rows)} rows auto-excluded; {len(todo)} rows split into {args.shards} shards")


def shard_rows(k):
    s = pd.read_csv(SHARDS)
    return s.loc[s.shard == k, "row_id"].tolist()


def cmd_show(args):
    df = load_data()
    done = set(load_progress().row_id.astype(int))
    todo = [r for r in shard_rows(args.shard) if r not in done]
    print(f"# shard {args.shard}: {len(todo)} rows left; showing {min(args.n, len(todo))}")
    for rid in todo[:args.n]:
        r = df.loc[rid]
        tr = r.Translation if str(r.Translation).strip() != str(r.text).strip() else "(same as text)"
        print(f"\n[{rid}]\nTEXT: {r.text}\nTRAN: {tr}")


def cmd_append(args):
    df = load_data()
    with open(args.file, encoding="utf-8") as f:
        items = json.load(f)
    mine = set(shard_rows(args.shard))
    errors, rows = [], []
    for it in items:
        rid = int(it["row_id"])
        lab = (it.get("label") or "").strip()
        exc = it.get("exclude", "no")
        if rid not in mine:
            errors.append(f"{rid}: not in shard {args.shard}")
            continue
        if exc not in ("yes", "no"):
            errors.append(f"{rid}: exclude must be yes/no")
        if exc == "yes" and (lab or not it.get("exclude_reason")):
            errors.append(f"{rid}: excluded rows need empty label and an exclude_reason")
        if exc == "no" and lab not in ("positive", "neutral", "negative"):
            errors.append(f"{rid}: bad label {lab!r}")
        if exc == "no" and it.get("confidence") not in ("high", "medium", "low"):
            errors.append(f"{rid}: bad confidence")
        if it.get("mixed", "no") not in ("yes", "no"):
            errors.append(f"{rid}: mixed must be yes/no")
        if it.get("translation_issue", "no") not in ("yes", "no"):
            errors.append(f"{rid}: translation_issue must be yes/no")
        r = df.loc[rid]
        auto_tr = not has_khmer(r.Translation)
        rows.append(dict(row_id=rid, label=lab, confidence=it.get("confidence", "") if exc == "no" else "",
                         mixed=it.get("mixed", "no"), exclude=exc,
                         exclude_reason=it.get("exclude_reason", "") if exc == "yes" else "",
                         is_original_khmer="yes" if has_khmer(r.text) else "no",
                         translation_issue="yes" if (auto_tr or it.get("translation_issue") == "yes") else "no",
                         note=it.get("note", ""), reason=it.get("reason", ""), batch=""))
    if len({r["row_id"] for r in rows}) != len(rows):
        errors.append("duplicate row_id inside the batch")
    if errors:
        print("REJECTED, nothing written:\n  " + "\n  ".join(errors))
        sys.exit(1)
    with FileLock():
        prog = load_progress()
        already = set(prog.row_id.astype(int)) & {r["row_id"] for r in rows}
        if already:
            print(f"REJECTED, already labeled: {sorted(already)}")
            sys.exit(1)
        n_batch = prog.loc[prog.batch.str.startswith(f"s{args.shard}-"), "batch"].nunique()
        tag = f"s{args.shard}-b{n_batch + 1:02d}"
        for r in rows:
            r["batch"] = tag
        prog = pd.concat([prog, pd.DataFrame(rows).astype(str)], ignore_index=True)
        prog[COLUMNS].to_csv(PROGRESS, index=False, encoding="utf-8")
    left = len([x for x in mine if x not in set(prog.row_id.astype(int))])
    print(f"OK: appended {len(rows)} rows as {tag}; shard {args.shard} has {left} rows left")


def cmd_status(args):
    prog = load_progress()
    s = pd.read_csv(SHARDS)
    done = set(prog.row_id.astype(int))
    s["done"] = s.row_id.isin(done)
    print("labeled rows:", len(prog), "| label counts:", prog.label.replace("", "(excluded)").value_counts().to_dict())
    print(s.groupby("shard").done.agg(["sum", "count"]).rename(columns={"sum": "done", "count": "total"}).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init"); a.add_argument("--shards", type=int, default=6)
    a = sub.add_parser("show"); a.add_argument("--shard", type=int, required=True); a.add_argument("--n", type=int, default=100)
    a = sub.add_parser("append"); a.add_argument("--shard", type=int, required=True); a.add_argument("--file", required=True)
    sub.add_parser("status")
    args = p.parse_args()
    {"init": cmd_init, "show": cmd_show, "append": cmd_append, "status": cmd_status}[args.cmd](args)
