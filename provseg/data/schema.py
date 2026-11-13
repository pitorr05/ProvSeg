"""Unified word-level provenance record shared by every loader.

Labels are per whitespace word: 0 = human (H), 1 = AI (A), -1 = ignore.
Every split is assigned by ``doc_id`` hash so all versions of one source
document land in the same partition (no leakage).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator

H, A, IGNORE = 0, 1, -1
SPLITS = ("train", "dev", "cal", "test")


@dataclass
class Record:
    doc_id: str                 # id of the *source* document (shared across versions)
    dataset: str
    words: list[str]
    labels: list[int]           # per word, H/A/IGNORE
    split: str = ""
    generator: str = ""
    domain: str = ""
    edit_level: str = ""        # e.g. polish degree, #rounds; "" if n/a
    versions: list[str] = field(default_factory=list)  # earlier versions (records track)
    meta: dict = field(default_factory=dict)
    # per word: 1 = new in the last edit step, 0 = copied, -1 = unknown; [] = no version info.
    # Privileged supervision (train only) for the edit-trace channel; never a test-time input.
    edited: list[int] = field(default_factory=list)

    def __post_init__(self):
        if len(self.words) != len(self.labels):
            raise ValueError(f"{self.doc_id}: {len(self.words)} words vs {len(self.labels)} labels")
        if self.edited and len(self.edited) != len(self.words):
            raise ValueError(f"{self.doc_id}: edited length mismatch")

    @property
    def ai_ratio(self) -> float:
        lab = [l for l in self.labels if l != IGNORE]
        return sum(lab) / max(len(lab), 1)


def assign_split(doc_id: str, ratios=(0.7, 0.1, 0.1, 0.1), salt: str = "provseg") -> str:
    h = int(hashlib.md5(f"{salt}:{doc_id}".encode()).hexdigest(), 16) % 10_000 / 10_000
    acc = 0.0
    for name, r in zip(SPLITS, ratios):
        acc += r
        if h < acc:
            return name
    return SPLITS[-1]


def char_labels_to_words(text: str, char_labels: list[int]) -> tuple[list[str], list[int]]:
    """Whitespace-tokenise ``text``; a word's label is the majority of its char labels."""
    words, labels = [], []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        j = i
        while j < n and not text[j].isspace():
            j += 1
        if j > i:
            seg = [c for c in char_labels[i:j] if c != IGNORE]
            words.append(text[i:j])
            labels.append(IGNORE if not seg else int(sum(seg) * 2 >= len(seg)))
        i = j
    return words, labels


def write_jsonl(records: Iterable[Record], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[Record]:
    with open(path, encoding="utf8") as f:
        for line in f:
            yield Record(**json.loads(line))
