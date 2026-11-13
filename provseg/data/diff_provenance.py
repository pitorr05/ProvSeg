"""Retained-provenance labels propagated through a chain of versions.

Given versions v_0 -> v_1 -> ... -> v_K with the author of each step, every
word of the final version inherits the author of the most recent step that
wrote it; words copied unchanged keep their previous label. Alignment is a
word-level ``difflib`` diff (deterministic, auditable, no model involved).
"""
from __future__ import annotations

import difflib
from typing import Sequence

from .schema import IGNORE


def _norm(w: str) -> str:
    return w.lower().strip(".,;:!?\"'()[]")


def propagate(prev_words: Sequence[str], prev_labels: Sequence[int],
              new_words: Sequence[str], editor: int) -> list[int]:
    """Labels for ``new_words``: copied words keep ``prev_labels``, others get ``editor``."""
    sm = difflib.SequenceMatcher(a=[_norm(w) for w in prev_words],
                                 b=[_norm(w) for w in new_words], autojunk=False)
    out = [editor] * len(new_words)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out[j1:j2] = prev_labels[i1:i2]
    return out


def chain(versions: Sequence[str], authors: Sequence[int]) -> tuple[list[str], list[int]]:
    """``versions[k]`` was written/edited by ``authors[k]``; returns final words + labels."""
    assert len(versions) == len(authors) and versions
    words = versions[0].split()
    labels = [authors[0]] * len(words)
    for text, who in zip(versions[1:], authors[1:]):
        new = text.split()
        labels = propagate(words, labels, new, who)
        words = new
    return words, labels


def copied_mask(prev_words: Sequence[str], new_words: Sequence[str]) -> list[int]:
    """1 for words of ``new_words`` copied unchanged from ``prev_words``."""
    sm = difflib.SequenceMatcher(a=[_norm(w) for w in prev_words],
                                 b=[_norm(w) for w in new_words], autojunk=False)
    out = [0] * len(new_words)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out[j1:j2] = [1] * (j2 - j1)
    return out


def chain_edits(versions: Sequence[str], authors: Sequence[int]) -> tuple[list[str], list[int], list[int]]:
    """Like ``chain`` but also returns ``edited`` (1 = not copied from the immediately previous version)."""
    words, labels = chain(versions, authors)
    if len(versions) == 1:
        return words, labels, [-1] * len(words)
    copied = copied_mask(versions[-2].split(), words)
    return words, labels, [1 - c for c in copied]


def edit_ratio(prev: str, new: str) -> float:
    """Fraction of words in ``new`` not copied from ``prev`` (edit magnitude)."""
    a, b = prev.split(), new.split()
    if not b:
        return 0.0
    sm = difflib.SequenceMatcher(a=[_norm(w) for w in a], b=[_norm(w) for w in b], autojunk=False)
    kept = sum(j2 - j1 for t, i1, i2, j1, j2 in sm.get_opcodes() if t == "equal")
    return 1.0 - kept / len(b)


def mask_short_runs(labels: list[int], min_run: int = 1) -> list[int]:
    """Optionally ignore isolated label flips shorter than ``min_run`` (punctuation noise)."""
    if min_run <= 1:
        return labels
    out = list(labels)
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < min_run and 0 < i and j < len(out):
            out[i:j] = [IGNORE] * (j - i)
        i = j
    return out
