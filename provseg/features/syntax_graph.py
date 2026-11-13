"""Dependency graph per record (PRDetect-style: spaCy edge token -> head), at whitespace-word level.

PRDetect (Li et al., Findings NAACL 2025; PRDetect/building_graph.py) builds edges
``[word.i, word.head.i]`` from spaCy and runs a document-level GCN. We keep the
same graph but map spaCy tokens onto our whitespace words (char offsets), so a
GCN can propagate evidence *per word* instead of mean-pooling the document.

Output: <out>/<dataset>_dep.npz with one int32 array (E, 2) of undirected word
edges (i < j, deduplicated, no self loops) per record uid.

Usage: python -m provseg.features.syntax_graph --data ../data/processed --out ../data/feats [--only ds1,ds2]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from tqdm import tqdm


def word_char_spans(words):
    spans, pos = [], 0
    for w in words:
        spans.append((pos, pos + len(w)))
        pos += len(w) + 1
    return spans


def doc_to_word_edges(doc, words) -> np.ndarray:
    spans = word_char_spans(words)
    starts = np.array([s for s, _ in spans])
    # token -> word index by char offset
    tok2w = np.searchsorted(starts, [t.idx for t in doc], side="right") - 1
    edges = set()
    for t in doc:
        a, b = int(tok2w[t.i]), int(tok2w[t.head.i])
        if a != b and a >= 0 and b >= 0:
            edges.add((min(a, b), max(a, b)))
    return np.array(sorted(edges), dtype=np.int32).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--out", default="../data/feats")
    ap.add_argument("--only", default="")
    ap.add_argument("--splits", default="", help="comma list; default all")
    ap.add_argument("--n_process", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.max_length = 2_000_000
    os.makedirs(a.out, exist_ok=True)
    names = a.only.split(",") if a.only else [f[:-6] for f in sorted(os.listdir(a.data)) if f.endswith(".jsonl")]
    keep = set(a.splits.split(",")) if a.splits else None
    for ds in names:
        recs = []
        with open(os.path.join(a.data, f"{ds}.jsonl"), encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                if keep and r["split"] not in keep:
                    continue
                recs.append((r["meta"]["uid"], r["words"]))
                if a.limit and len(recs) >= a.limit:
                    break
        out = {}
        texts = (" ".join(w) for _, w in recs)
        for (uid, words), doc in tqdm(zip(recs, nlp.pipe(texts, batch_size=64, n_process=a.n_process)),
                                      total=len(recs), desc=ds, mininterval=30):
            out[uid] = doc_to_word_edges(doc, words)
        np.savez_compressed(os.path.join(a.out, f"{ds}_dep.npz"), **out)
        print(f"{ds}: {len(out)} graphs, mean edges/word = "
              f"{np.mean([len(e) / max(len(w), 1) for (u, w), e in zip(recs, out.values())]):.2f}")


if __name__ == "__main__":
    main()
