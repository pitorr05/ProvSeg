"""APT-Eval (Saha & Feizi 2025, "Almost AI, Almost Human"): human texts polished by LLMs
at controlled degrees (extreme_minor ... major) and percentages.

HF: smksaha/apt-eval  original.csv (300 pure human) + merged_apt_eval_dataset.csv (14,650 polished).
The polished rows do not carry the original id, so each is matched to its source
human text by maximal word-set Jaccard within the same domain (checked: the
match is unambiguous; margin logged in meta). Provenance via diff: H source ->
A polisher. APT-Eval has no training split: every record is test (the
polish-degree curve), except a 30% cal slice by source id.
"""
from __future__ import annotations

import os
from typing import Iterator

import pandas as pd

from ..diff_provenance import chain_edits, edit_ratio
from ..schema import A, H, Record, assign_split


def _jac(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def load(root: str) -> Iterator[Record]:
    base = os.path.join(root, "smksaha__apt-eval")
    orig = pd.read_csv(os.path.join(base, "original.csv"))
    pol = pd.read_csv(os.path.join(base, "merged_apt_eval_dataset.csv"))
    origs = [(str(r["id"]), str(r["domain"]), str(r["generation"])) for _, r in orig.iterrows()]
    sets = [set(t.lower().split()) for _, _, t in origs]

    def split_of(oid):
        return "cal" if assign_split(f"apt:{oid}", (0.0, 0.0, 0.3, 0.7)) == "cal" else "test"

    for (oid, dom, text) in origs:
        w = text.split()
        if len(w) >= 8:
            yield Record(doc_id=f"apt:{oid}", dataset="apt_eval", words=w, labels=[H] * len(w), edited=[0] * len(w),
                         split=split_of(oid), generator="human", domain=dom, edit_level="none")
    for _, r in pol.iterrows():
        text = str(r["generation"])
        s = set(text.lower().split())
        scores = sorted(((_jac(s, sets[i]), i) for i, o in enumerate(origs) if o[1] == str(r["domain"])), reverse=True)
        if not scores:
            continue
        (best, i), second = scores[0], (scores[1][0] if len(scores) > 1 else 0.0)
        if best - second < 0.05:   # ambiguous source match (<1% of rows): drop rather than mislabel
            continue
        oid, _, src = origs[i]
        words, labels, edited = chain_edits([src, text], [H, A])
        if len(words) < 8:
            continue
        level = str(r["polishing_degree"]) if str(r["polish_type"]) == "degree-based" else f"{r['polishing_percent']}%"
        yield Record(doc_id=f"apt:{oid}", dataset="apt_eval", words=words, labels=labels, edited=edited,
                     split=split_of(oid), generator=str(r["polisher"]), domain=str(r["domain"]),
                     edit_level=level, versions=[src],
                     meta={"polish_type": str(r["polish_type"]), "match_jacc": best, "match_margin": best - second,
                           "edit_ratio": edit_ratio(src, text)})
