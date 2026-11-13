"""PASTED (Findings ACL 2024): human/machine documents with some sentences AI-paraphrased.

Format (eval-OOD/test.json and the in-distribution classification split):
  {"index", "text": "s1 </s> s2 </s> ...", "label": "[0,1,...]" per sentence (1 = paraphrased),
   "src": "<domain>_<human|machine_...>", "is_para"}
Provenance: a sentence is A if it was paraphrased by the LLM, or if the
source document itself was machine-written; otherwise H.
"""
from __future__ import annotations

import ast
import glob
import json
import os
from typing import Iterator

from ..schema import A, H, Record, assign_split


def _rows(path):
    with open(path, encoding="utf8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            yield from json.load(f)
        else:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def load_file(path: str, dataset: str, fixed_split: str | None = None) -> Iterator[Record]:
    for d in _rows(path):
        sents = [s.strip() for s in d["text"].split("</s>")]
        lab = d["label"]
        lab = ast.literal_eval(lab) if isinstance(lab, str) else lab
        if not isinstance(lab, list) or len(lab) != len(sents):
            continue
        src = str(d.get("src", ""))
        machine_src = "human" not in src
        words, labels, edited = [], [], []
        for s, l in zip(sents, lab):
            w = s.split()
            words += w
            labels += [A if (int(l) == 1 or machine_src) else H] * len(w)
            edited += [int(l)] * len(w)
        if len(words) < 8:
            continue
        doc_id = f"{dataset}:{d['index']}"
        split = fixed_split or assign_split(doc_id)
        if split == "dev" and assign_split(doc_id, (0.0, 0.8, 0.2, 0.0)) == "cal":
            split = "cal"            # 20% of the official dev -> conformal calibration
        domain = src.split("_")[0]
        yield Record(doc_id=doc_id, dataset=dataset, words=words, labels=labels, edited=edited,
                     split=split,
                     generator="machine-src" if machine_src else "human-src",
                     domain=domain, edit_level="paraphrase",
                     meta={"src": src, "n_para": int(sum(int(x) for x in lab))})


def load(root: str) -> Iterator[Record]:
    base = os.path.join(root, "linzw__PASTED", "x")
    # in-distribution (classification config) if unpacked
    for split in ("train", "dev", "test"):
        for p in glob.glob(os.path.join(base, "classification", "**", f"{split}.json"), recursive=True):
            yield from load_file(p, "pasted", fixed_split={"dev": "dev"}.get(split, split))
    # OOD test (different paraphraser) -> always test
    for p in glob.glob(os.path.join(base, "eval-OOD", "**", "test.json"), recursive=True):
        yield from load_file(p, "pasted_ood", fixed_split="test")
