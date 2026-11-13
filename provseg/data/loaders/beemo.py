"""Beemo (toloka/beemo, NAACL 2025): machine outputs + expert human edits.

Per prompt we emit (all sharing doc_id = prompt_id so they share a split):
  * human_output            -> all H
  * model_output            -> all A
  * model_output -> human_edits  -> retained provenance via diff (AI->Human edit),
    versions=[model_output] for the records track, edit_level = edit-ratio bucket.
"""
from __future__ import annotations

import glob
import os
from typing import Iterator

import pandas as pd

from ..diff_provenance import chain_edits, edit_ratio
from ..schema import A, H, Record, assign_split


def _bucket(r: float) -> str:
    for hi in (0.1, 0.25, 0.5, 0.75):
        if r <= hi:
            return f"<= {hi:.2f}"
    return "> 0.75"


def load(root: str) -> Iterator[Record]:
    files = glob.glob(os.path.join(root, "toloka__beemo", "data", "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files])
    for _, r in df.iterrows():
        doc_id = f"beemo:{r['prompt_id']}"
        split = assign_split(doc_id)
        base = dict(doc_id=doc_id, dataset="beemo", split=split,
                    generator=str(r["model"]).split("/")[-1], domain=str(r["category"]))
        mo, ho, he = (str(r[c] or "") for c in ("model_output", "human_output", "human_edits"))
        if len(ho.split()) >= 8:
            w = ho.split()
            yield Record(words=w, labels=[H] * len(w), edit_level="human", edited=[0] * len(w), **{**base, "generator": "human"})
        if len(mo.split()) >= 8:
            w = mo.split()
            yield Record(words=w, labels=[A] * len(w), edit_level="machine", edited=[0] * len(w), **base)
        if len(he.split()) >= 8 and len(mo.split()) >= 8:
            words, labels, edited = chain_edits([mo, he], [A, H])
            er = edit_ratio(mo, he)
            yield Record(words=words, labels=labels, edited=edited, edit_level=_bucket(er), versions=[mo],
                         meta={"edit_ratio": er, "id": int(r["id"])}, **base)
