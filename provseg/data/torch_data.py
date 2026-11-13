"""Torch dataset: records -> word windows -> sub-token batches
(+ optional word features, dependency edges, privileged ``edited`` labels)."""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import IGNORE


def cap_key(uid: str) -> str:
    return hashlib.md5(f"cap:{uid}".encode()).hexdigest()


def cap_subset(records, cap):
    """Deterministic subset (same docs for every model and for feature extraction)."""
    if not cap or len(records) <= cap:
        return records
    keep = set(sorted(cap_key(r["meta"]["uid"]) for r in records)[:cap])
    return [r for r in records if cap_key(r["meta"]["uid"]) in keep]


def load_records(paths, splits=None, limit=0, cap=0, seed=0):
    """cap: max records per file (deterministic uid-hash subset) to balance huge sets such as PASTED."""
    out = []
    for p in paths:
        rs = []
        with open(p, encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                if splits and r["split"] not in splits:
                    continue
                rs.append(r)
                if limit and len(rs) >= limit:
                    break
        out += cap_subset(rs, cap)
    return out


class FeatureStore:
    """uid -> (n_words, F) array from .npz files, z-normalised with train stats (NaN -> train mean)."""

    def __init__(self, npz_paths, feat_idx=None):
        self.arrs = {}
        for p in npz_paths:
            z = np.load(p)
            for k in z.files:
                self.arrs[k] = z[k]
        self.feat_idx = feat_idx
        self.mu = self.sd = None

    def fit_norm(self, uids):
        X = np.concatenate([self.arrs[u] for u in uids if u in self.arrs])
        X = X[:, self.feat_idx] if self.feat_idx is not None else X
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
        return self

    def get(self, uid, n_words):
        a = self.arrs.get(uid)
        if a is None or len(a) != n_words:
            raise KeyError(f"features missing/misaligned for {uid}")
        a = a[:, self.feat_idx] if self.feat_idx is not None else a
        a = (a - self.mu) / self.sd
        return np.nan_to_num(a, nan=0.0).astype(np.float32)


def load_dep(feat_dir):
    """uid -> (E,2) int edges from every <ds>_dep.npz."""
    dep = {}
    for p in glob.glob(os.path.join(feat_dir, "*_dep.npz")):
        z = np.load(p)
        dep.update({k: z[k] for k in z.files})
    return dep


class WindowDataset(Dataset):
    """Splits each record into consecutive windows of <= ``win`` words (no overlap)."""

    def __init__(self, records, tokenizer, feats: FeatureStore | None = None, dep: dict | None = None,
                 win: int = 160, max_len: int = 512):
        self.items = []
        self.tok, self.feats, self.max_len = tokenizer, feats, max_len
        self.use_dep = dep is not None
        for ri, r in enumerate(records):
            n = len(r["words"])
            uid = r["meta"]["uid"]
            F = feats.get(uid, n) if feats is not None else None
            E = dep.get(uid) if dep is not None else None
            ed = r.get("edited") or [IGNORE] * n
            for s in range(0, n, win):
                e = None
                if E is not None and len(E):
                    m = (E[:, 0] >= s) & (E[:, 1] < s + win)
                    e = E[m] - s
                self.items.append((ri, s, r["words"][s:s + win], r["labels"][s:s + win],
                                   None if F is None else F[s:s + win], e, ed[s:s + win]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def collate(self, batch):
        words = [b[2] for b in batch]
        W = max(len(w) for w in words)
        B = len(batch)
        out = {"rec_idx": torch.tensor([b[0] for b in batch]), "start": torch.tensor([b[1] for b in batch])}
        labels = torch.full((B, W), IGNORE, dtype=torch.long)
        edited = torch.full((B, W), IGNORE, dtype=torch.long)
        wmask = torch.zeros((B, W), dtype=torch.bool)
        for k, b in enumerate(batch):
            labels[k, :len(b[3])] = torch.tensor(b[3])
            edited[k, :len(b[6])] = torch.tensor(b[6])
            wmask[k, :len(b[3])] = True
        if self.tok is not None:
            enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt")
            widx = torch.full(enc["input_ids"].shape, -1, dtype=torch.long)
            for k in range(B):
                for t, w in enumerate(enc.word_ids(k)):
                    if w is not None:
                        widx[k, t] = w
            for k in range(B):  # words lost to truncation (rare with win=160)
                seen = set(widx[k][widx[k] >= 0].tolist())
                for w in range(len(words[k])):
                    if w not in seen:
                        wmask[k, w] = False
                        labels[k, w] = IGNORE
                n_ok = int(wmask[k].long().cumprod(0).sum())   # keep masks left-contiguous for the CRF
                wmask[k, n_ok:] = False
            out.update(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], word_index=widx)
        out["labels"], out["word_mask"], out["edited"] = labels, wmask, edited
        if self.feats is not None:
            Fd = batch[0][4].shape[1]
            f = torch.zeros((B, W, Fd))
            for k, b in enumerate(batch):
                f[k, :len(b[4])] = torch.from_numpy(b[4])
            out["feats"] = f
        if self.use_dep:
            es = [torch.cat([torch.full((len(b[5]), 1), k), torch.from_numpy(b[5]).long()], 1)
                  for k, b in enumerate(batch) if b[5] is not None and len(b[5])]
            out["edges"] = torch.cat(es) if es else torch.zeros((0, 3), dtype=torch.long)
        return out
