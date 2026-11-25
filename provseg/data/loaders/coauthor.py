"""CoAuthor (Lee, Liang & Yang, CHI 2022): real human-GPT-3 co-writing sessions, keystroke-level logs.

Download: https://cs.stanford.edu/~minalee/zip/chi2022-coauthor-v1.0.zip (from https://coauthor.stanford.edu),
extracted to <hf_root>/coauthor/x/coauthor-v1.0/*.jsonl (one session per file, 1,447 sessions).

Each event carries a Quill delta (`textDelta.ops`: retain / insert / delete). We replay the session and keep
the author of every character:
  * the initial prompt (`system-initialize` currentDoc) is human-written            -> H
  * `text-insert` with eventSource == "user" (typing, pasting by the writer)        -> H
  * `text-insert` with eventSource == "api" (an accepted GPT-3 suggestion)          -> A
  * `text-delete` removes characters whatever their author
Characters edited after insertion keep their *final* author (a GPT-3 word retyped by the user becomes H).
Words take the majority label of their characters (schema.char_labels_to_words). This is the real
co-authoring benchmark also used by the 2026 change-point segmentation work (WCP).
Splits are assigned per session id (no session appears in two splits).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Iterator

from ..schema import A, H, Record, assign_split, char_labels_to_words


def replay(events):
    """Return (final_text, per-char author list)."""
    doc, auth = [], []
    for e in events:
        name, src = e.get("eventName"), e.get("eventSource")
        if name == "system-initialize":
            init = e.get("currentDoc") or ""
            doc, auth = list(init), [H] * len(init)
            continue
        delta = e.get("textDelta")
        if not isinstance(delta, dict) or "ops" not in delta:
            continue
        who = A if (name == "text-insert" and src == "api") else H
        pos = 0
        for op in delta["ops"]:
            if "retain" in op:
                pos += int(op["retain"])
            elif "insert" in op:
                s = op["insert"] if isinstance(op["insert"], str) else ""
                doc[pos:pos] = list(s)
                auth[pos:pos] = [who] * len(s)
                pos += len(s)
            elif "delete" in op:
                n = int(op["delete"])
                del doc[pos:pos + n]
                del auth[pos:pos + n]
    return "".join(doc), auth


def load(root: str) -> Iterator[Record]:
    files = sorted(glob.glob(os.path.join(root, "coauthor", "x", "coauthor-v1.0", "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"CoAuthor sessions not found under {root}/coauthor/x/coauthor-v1.0 "
                                "(run setup_server.sh, which downloads them)")
    for f in files:
        sid = os.path.basename(f)[:-6]
        with open(f, encoding="utf8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        text, auth = replay(events)
        words, labels = char_labels_to_words(text, auth)
        if len(words) < 30:
            continue
        doc_id = f"coauthor:{sid}"
        n_ai = sum(1 for e in events if e.get("eventName") == "text-insert" and e.get("eventSource") == "api")
        yield Record(doc_id=doc_id, dataset="coauthor", words=words, labels=labels, split=assign_split(doc_id),
                     generator="gpt-3", edit_level=f"accepted_suggestions={n_ai}",
                     meta={"session": sid, "n_accepted": n_ai})
