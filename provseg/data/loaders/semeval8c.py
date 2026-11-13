"""SemEval-2024 Task 8 subtask C (= M4GT-Bench task 3): human prefix then machine continuation.

HF mirror: d0rj/SemEval2024-task8, subtaskC/{train,dev,test}-*.parquet
Fields: id, text, label (index of the first machine word, whitespace split).
The official split is kept (train / dev / test); 20% of the official dev is
moved to "cal" (by id hash) for conformal calibration.
"""
from __future__ import annotations

import glob
import os
from typing import Iterator

import pandas as pd

from ..schema import A, H, Record, assign_split


def load(root: str) -> Iterator[Record]:
    base = os.path.join(root, "d0rj__SemEval2024-task8", "subtaskC")
    for split in ("train", "dev", "test"):
        for p in glob.glob(os.path.join(base, f"{split}-*.parquet")):
            df = pd.read_parquet(p)
            for _, r in df.iterrows():
                words = str(r["text"]).split()
                b = int(r["label"]) if "label" in r and pd.notna(r["label"]) else -1
                if b < 0 or len(words) < 8:
                    continue
                b = min(b, len(words))
                doc_id = f"semeval8c:{r['id']}"
                s = split
                if split == "dev" and assign_split(doc_id, (0.0, 0.8, 0.2, 0.0)) == "cal":
                    s = "cal"
                yield Record(doc_id=doc_id, dataset="semeval8c", words=words,
                             labels=[H] * b + [A] * (len(words) - b), split=s,
                             generator=str(r.get("model", "")), domain=str(r.get("source", "")),
                             edit_level="continuation")
