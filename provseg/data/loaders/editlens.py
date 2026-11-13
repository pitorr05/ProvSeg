"""EditLens (ICLR 2026) data: AI edits applied to human source texts.

Local CSVs ship with the repo: refs/EditLens/data/{val,test,test_llama,...}.csv
Columns include text, source_text, text_type, model, split, and precomputed
document scores of baselines (binoculars_score, fastdetectgpt_score,
editlens_*_score, pangram_*) which we reuse as document-level baselines.

Provenance: source_text is human (H); the edited text was produced by the LLM
(model column) -> words copied from the source keep H, new words become A.
Rows whose text_type marks pure human / pure AI get uniform labels.
License: CC BY-NC-SA 4.0 (research use only; cite EditLens).
"""
from __future__ import annotations

import os
from typing import Iterator

import pandas as pd

from ..diff_provenance import chain_edits, edit_ratio
from ..schema import A, H, Record, assign_split

SCORE_COLS = ["binoculars_score", "fastdetectgpt_score", "editlens_roberta-large_score",
              "editlens_Llama-3.2-3B_score", "pangram_v3.2_score"]


def load(root: str, files=("val.csv", "test.csv")) -> Iterator[Record]:
    base = os.path.join(root, "EditLens", "data")
    for fn in files:
        p = os.path.join(base, fn)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        for k, r in df.iterrows():
            text = str(r["text"])
            src = r.get("source_text")
            ttype = str(r.get("text_type", ""))
            sid = str(r.get("source_id", r.get("text_id", k)))
            doc_id = f"editlens:{sid}"
            if ttype == "human_written":
                w = text.split()
                words, labels, er, versions, edited = w, [H] * len(w), 0.0, [], [0] * len(w)
            elif ttype == "ai_edited" and isinstance(src, str) and src.strip():
                words, labels, edited = chain_edits([src, text], [H, A])
                er, versions = edit_ratio(src, text), [src]
            else:
                w = text.split()
                words, labels, er, versions, edited = w, [A] * len(w), 1.0, [], [0] * len(w)  # fresh generation: no edit step
            if len(words) < 8:
                continue
            yield Record(doc_id=doc_id, dataset="editlens", words=words, labels=labels, edited=edited,
                         # official val.csv -> train/dev/cal (version-supervised); test.csv held out
                         split="test" if fn.startswith("test") else assign_split(doc_id, (0.7, 0.1, 0.2, 0.0)),
                         generator=str(r.get("model", "")), domain=str(r.get("source", "")),
                         edit_level=ttype, versions=versions,
                         meta={"edit_ratio": er, "file": fn,
                               "doc_scores": {c: float(r[c]) for c in SCORE_COLS if c in r and pd.notna(r[c])}})
