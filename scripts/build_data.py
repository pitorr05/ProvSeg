"""Run every loader -> data/processed/<dataset>.jsonl (+ stats).

Usage: python scripts/build_data.py --hf ../data/hf --refs ../refs --out ../data/processed
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from provseg.data.loaders import apt_eval, beemo, editlens, pasted, semeval8c, seqxgpt  # noqa: E402
from provseg.data.schema import write_jsonl  # noqa: E402

LOADERS = {
    "seqxgpt": (seqxgpt.load, "refs"),
    "semeval8c": (semeval8c.load, "hf"),
    "pasted": (pasted.load, "hf"),
    "beemo": (beemo.load, "hf"),
    "apt_eval": (apt_eval.load, "hf"),
    "editlens": (editlens.load, "refs"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="../data/hf")
    ap.add_argument("--refs", default="../refs")
    ap.add_argument("--out", default="../data/processed")
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0, help="max records per dataset (smoke tests)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = a.only.split(",") if a.only else list(LOADERS)
    for name in names:
        fn, kind = LOADERS[name]
        root = a.hf if kind == "hf" else a.refs
        by_ds = collections.defaultdict(list)
        try:
            for i, r in enumerate(fn(root)):
                if a.limit and i >= a.limit:
                    break
                r.meta["uid"] = f"{r.dataset}:{len(by_ds[r.dataset])}"
                by_ds[r.dataset].append(r)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        for ds, recs in by_ds.items():
            n = write_jsonl(recs, os.path.join(a.out, f"{ds}.jsonl"))
            sp = collections.Counter(r.split for r in recs)
            ai = sum(r.ai_ratio for r in recs) / max(n, 1)
            nw = sum(len(r.words) for r in recs) / max(n, 1)
            mixed = sum(0 < r.ai_ratio < 1 for r in recs)
            print(f"{ds:12s} n={n:6d} words/doc={nw:6.1f} mean_ai={ai:.2f} mixed={mixed:6d} splits={dict(sp)}")
        if not by_ds:
            print(f"[empty] {name}")


if __name__ == "__main__":
    main()
