"""SeqXGPT-Bench: human prefix (``prompt_len`` chars) followed by an AI continuation.

Files: refs/SeqXGPT/SeqXGPT/dataset/SeqXGPT-Bench/en_{gen}_lines.jsonl
Each line: {"text", "prompt_len", "label"}; ``label`` names the generator (or "human").
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Iterator

from ..schema import A, H, Record, assign_split, char_labels_to_words

GENS = ["gpt2", "gpt3", "gptj", "gptneo", "llama", "human"]


def load(root: str) -> Iterator[Record]:
    bench = os.path.join(root, "SeqXGPT", "SeqXGPT", "dataset", "SeqXGPT-Bench")
    for gen in GENS:
        path = os.path.join(bench, f"en_{gen}_lines.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf8") as f:
            for k, line in enumerate(f):
                d = json.loads(line)
                text, plen = d["text"], int(d.get("prompt_len", 0))
                if gen == "human":
                    plen = len(text)
                char_lab = [H] * min(plen, len(text)) + [A] * max(len(text) - plen, 0)
                words, labels = char_labels_to_words(text, char_lab)
                if len(words) < 8:
                    continue
                # SeqXGPT shares human prefixes across generators -> id by prefix
                doc_id = "seqx:" + hashlib.md5(text[:plen][:300].encode()).hexdigest()[:16]
                yield Record(doc_id=doc_id, dataset="seqxgpt", words=words, labels=labels,
                             split=assign_split(doc_id), generator=gen,
                             meta={"label_raw": d.get("label"), "line": k})
