"""One trainer for the whole ladder (S0-S4) and the supervised baselines.

  --arch seqxgpt      S0 / baseline: the *official* SeqXGPT class (ModelWiseTransformerClassifier),
                      4 LM log-prob channels (--feat_kind lp4)
  --arch seqx         S1-S4: SeqXNet (same computation, any #channels, optional --context encoder,
                      --evidence_dropout, --shapley)
  --arch seqroberta   baseline: official Seq-RoBERTa class (RoBERTa + BMES tags), from the SeqXGPT repo
  --arch sentroberta  baseline: official Sent-RoBERTa class (one label per sentence)

Training follows SeqXGPT: BMES tags, token cross-entropy, AdamW(betas .9/.98, wd .1), linear warm-up
10%, BMES-constrained Viterbi decoding. The checkpoint with the best dev word macro-F1 is kept.

Writes <out>/<name>/pred_<ds>.pkl: {"__meta__": {...}, uid: {"pred": (n,) 0/1, "prob": (n,2), "split"}}
and metrics_<ds>.json (test split; for --multiclass also the 6-way SeqXGPT protocol).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baselines.seqxgpt import (ModelWiseTransformerClassifier, SentRoBERTaClassifier,  # noqa: E402
                               SeqRoBERTaClassifier)
from provseg.data.torch_data import FeatureStore, WindowDataset, cap_subset, load_records  # noqa: E402
from provseg.eval.metrics import all_metrics, sentence_metrics, sentence_spans  # noqa: E402
from provseg.features.token_features import FEAT_NAMES  # noqa: E402
from provseg.models.seqx import (SeqXNet, bmes_id2label, channel_shapley, class_probs,  # noqa: E402
                                 segment_starts, to_bmes)

SEQX_CLASSES = ["gpt2", "gptneo", "gptj", "llama", "gpt3", "human"]   # SeqXGPT-Bench, 6-way protocol
LP4_NAMES = ["nll_gpt2", "nll_gptneo", "nll_gptj", "nll_llama"]


def amp(a, dev):
    """bf16 autocast on CUDA when --bf16 (losses, probabilities and Viterbi are computed in fp32)."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(getattr(a, "bf16", False)) and dev == "cuda")


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- data
def with_classes(recs, classes, multiclass):
    """Replace word labels (0 H / 1 A / -1) by class ids."""
    out = []
    hum = classes.index("human")
    for r in recs:
        r = dict(r)
        if multiclass:
            gen = classes.index(r["generator"]) if r["generator"] in classes else None
            r["labels"] = [hum if l == 0 else (gen if l == 1 and gen is not None else -1) for l in r["labels"]]
        else:
            r["labels"] = [hum if l == 0 else (1 - hum if l == 1 else -1) for l in r["labels"]]
        out.append(r)
    return out


def feature_store(a, ds_list):
    if a.arch in ("seqroberta", "sentroberta"):
        return None, []
    if a.feat_kind == "lp4":
        paths, idx, names = [os.path.join(a.feats, f"{d}_lp4.npz") for d in ds_list], None, LP4_NAMES
    else:
        cols = [int(c) for c in a.feat_cols.split(",")]
        paths, idx, names = [os.path.join(a.feats, f"{d}.npz") for d in ds_list], cols, [FEAT_NAMES[c] for c in cols]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing feature files {missing}: run the feats stage of scripts/run_ladder.sh first")
    return FeatureStore(paths, idx), names


def tag_tensor(labels, wmask, seq_len=None):
    """(B, W) class ids -> (B, W') BMES tag ids, -1 on padding / ignored words."""
    B, W = labels.shape
    L = seq_len or W
    out = torch.full((B, L), -1, dtype=torch.long)
    for k in range(B):
        n = int(wmask[k].sum())
        out[k, :n] = torch.tensor(to_bmes(labels[k, :n].tolist()))
    return out


def pad_to(x, L, value=0.0):
    if x.shape[1] >= L:
        return x[:, :L]
    pad = list(x.shape)
    pad[1] = L - x.shape[1]
    return torch.cat([x, x.new_full(pad, value)], 1)


# ---------------------------------------------------------------- models
def build(a, n_classes, n_channels):
    id2label = bmes_id2label(a.classes)
    if a.arch == "seqxgpt":
        assert n_channels == 4, "official SeqXGPT expects exactly 4 LM channels (--feat_kind lp4)"
        return ModelWiseTransformerClassifier(id2labels=id2label, seq_len=a.win)
    if a.arch == "seqx":
        return SeqXNet(id2label, seq_len=a.win, n_channels=n_channels, context=a.context or None,
                       boundary=a.boundary_head)
    if a.arch == "seqroberta":
        return SeqRoBERTaClassifier(a.encoder, id2label)
    if a.arch == "sentroberta":
        return SentRoBERTaClassifier(a.encoder, class_num=n_classes)
    raise ValueError(a.arch)


def token_tags(batch, tags):
    """Word tags -> sub-token tags on each word's first sub-token (Seq-RoBERTa labelling)."""
    widx = batch["word_index"]
    out = torch.full(widx.shape, -1, dtype=torch.long)
    prev = torch.cat([torch.full((widx.shape[0], 1), -2), widx[:, :-1]], 1)
    first = (widx >= 0) & (widx != prev)
    b, t = first.nonzero(as_tuple=True)
    w = widx[b, t]
    ok = w < tags.shape[1]
    out[b[ok], t[ok]] = tags[b[ok], w[ok]]
    return out, first


def step_loss(model, a, batch, dev):
    tags = tag_tensor(batch["labels"], batch["word_mask"], a.win if a.arch in ("seqxgpt", "seqx") else None)
    if a.arch == "seqxgpt":
        x = pad_to(batch["feats"], a.win).to(dev)
        return model(x, tags.to(dev))["loss"]
    if a.arch == "seqx":
        x = pad_to(batch["feats"], a.win).to(dev)
        mask = pad_to(batch["word_mask"], a.win, False).to(dev)
        ctx = None
        if model.enc is not None:
            ctx = pad_to(model.context_states(batch["input_ids"].to(dev), batch["attention_mask"].to(dev),
                                              batch["word_index"].to(dev), batch["word_mask"].shape[1]), a.win)
        keep = None
        if a.evidence_dropout > 0:
            keep = (torch.rand(x.shape[0], model.n_players, device=dev) >= a.evidence_dropout).float()
        logits = model(x, mask, ctx, keep)
        tags = tags.to(dev)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tags.reshape(-1), ignore_index=-1)
        if model.boundary:                       # Boundary Curator: rare positives -> pos_weight
            st = segment_starts(tags)
            m = st >= 0
            loss = loss + a.lam_bnd * F.binary_cross_entropy_with_logits(
                model.last_boundary[m], st[m].float(), pos_weight=torch.tensor(a.bnd_pos_weight, device=dev))
        return loss
    if a.arch == "seqroberta":
        tt, _ = token_tags(batch, tags)
        return model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev), tt.to(dev))["loss"]
    raise ValueError(a.arch)


@torch.no_grad()
def word_logits(model, a, batch, dev):
    """(logits (B, W, T), tag paths (B, W)) for a window batch."""
    W = batch["word_mask"].shape[1]
    wmask = batch["word_mask"].to(dev)
    if a.arch == "seqxgpt":
        x = pad_to(batch["feats"], a.win).to(dev)
        lab = torch.where(pad_to(batch["word_mask"], a.win, False), 0, -1).to(dev)
        o = model(x, lab)
        return o["logits"][:, :W], o["preds"][:, :W]
    if a.arch == "seqx":
        x = pad_to(batch["feats"], a.win).to(dev)
        mask = pad_to(batch["word_mask"], a.win, False).to(dev)
        ctx = None
        if model.enc is not None:
            ctx = pad_to(model.context_states(batch["input_ids"].to(dev), batch["attention_mask"].to(dev),
                                              batch["word_index"].to(dev), W), a.win)
        logits = model(x, mask, ctx)
        return logits[:, :W], model.decode(logits, mask)[:, :W]
    if a.arch == "seqroberta":
        h = model.roberta(input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev)).last_hidden_state
        tok_logits = model.classifier(model.dropout(h))
        _, first = token_tags(batch, torch.zeros(batch["word_mask"].shape, dtype=torch.long))
        logits = tok_logits.new_zeros(h.shape[0], W, tok_logits.shape[-1])
        b, t = first.nonzero(as_tuple=True)
        w = batch["word_index"][b, t]
        ok = w < W
        logits[b[ok].to(dev), w[ok].to(dev)] = tok_logits[b[ok].to(dev), t[ok].to(dev)]
        paths, _ = model.crf.viterbi_decode(logits=logits, mask=wmask)
        return logits, paths.masked_fill(~wmask, -1)
    raise ValueError(a.arch)


# ---------------------------------------------------------------- sentence-level baseline
class SentData(torch.utils.data.Dataset):
    def __init__(self, recs, tok, max_len=128):
        self.items, self.tok, self.max_len = [], tok, max_len
        for ri, r in enumerate(recs):
            g = np.asarray(r["labels"])
            for s, e in sentence_spans(r["words"]):
                m = g[s:e] >= 0
                lab = int(np.bincount(g[s:e][m]).argmax()) if m.any() else -1
                self.items.append((ri, s, e, " ".join(r["words"][s:e]), lab))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def collate(self, b):
        enc = self.tok([x[3] for x in b], truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": torch.tensor([x[4] for x in b]), "rec_idx": [x[0] for x in b],
                "span": [(x[1], x[2]) for x in b]}


# ---------------------------------------------------------------- loops
def make_loader(a, recs, tok, fs, shuffle):
    if a.arch == "sentroberta":
        ds = SentData(recs, tok)
        return DataLoader(ds, batch_size=a.bs, shuffle=shuffle, collate_fn=ds.collate)
    ds = WindowDataset(recs, tok, fs, None, win=a.win, max_len=a.max_len)
    return DataLoader(ds, batch_size=a.bs, shuffle=shuffle, collate_fn=ds.collate)


@torch.no_grad()
def predict(model, a, recs, tok, fs, dev, n_classes):
    """uid -> {"cls": (n,) class ids, "cprob": (n, C) class probabilities}."""
    model.eval()
    out = {r["meta"]["uid"]: {"cls": np.full(len(r["words"]), -1), "cprob": np.full((len(r["words"]), n_classes), 1.0 / n_classes)}
           for r in recs}
    for batch in make_loader(a, recs, tok, fs, False):
        if a.arch == "sentroberta":
            with amp(a, dev):
                p = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev)).float().softmax(-1).cpu().numpy()
            for k, (ri, (s, e)) in enumerate(zip(batch["rec_idx"], batch["span"])):
                o = out[recs[ri]["meta"]["uid"]]
                o["cprob"][s:e] = p[k]
                o["cls"][s:e] = int(p[k].argmax())
            continue
        with amp(a, dev):
            logits, paths = word_logits(model, a, batch, dev)
        cp = class_probs(logits, n_classes).cpu().numpy()
        cls = torch.where(paths >= 0, paths // 4, -1).cpu().numpy()
        for k in range(len(cls)):
            ri, s = int(batch["rec_idx"][k]), int(batch["start"][k])
            n = int(batch["word_mask"][k].sum())
            o = out[recs[ri]["meta"]["uid"]]
            o["cls"][s:s + n] = cls[k, :n]
            o["cprob"][s:s + n] = cp[k, :n]
    n_bad = 0
    for o in out.values():   # never let NaN reach the metrics silently
        bad = ~np.isfinite(o["cprob"]).all(-1)
        if bad.any():
            n_bad += int(bad.sum())
            o["cprob"][bad] = 1.0 / n_classes
    if n_bad:
        print(f"[warn] {n_bad} words had non-finite probabilities (set to uniform)")
    for o in out.values():   # words dropped by sub-token truncation: most probable class
        bad = o["cls"] < 0
        o["cls"][bad] = o["cprob"][bad].argmax(-1)
    return out


def binary(o, hum):
    p_ai = 1.0 - o["cprob"][:, hum]
    return (o["cls"] != hum).astype(int), np.stack([1 - p_ai, p_ai], 1)


def dev_score(model, a, recs, tok, fs, dev, n_classes, hum):
    P = predict(model, a, recs, tok, fs, dev, n_classes)
    golds = [np.asarray(r["labels"]) for r in recs]
    preds = [P[r["meta"]["uid"]]["cls"] for r in recs]
    from sklearn.metrics import f1_score
    g = np.concatenate(golds)
    p = np.concatenate(preds)
    m = g >= 0
    return float(f1_score(g[m], p[m], average="macro"))


@torch.no_grad()
def dump_shapley(model, a, recs, fs, tok, dev, names, path):
    recs = cap_subset(recs, a.shap_cap)
    out = {"__players__": names + (["context"] if model.enc is not None else [])}
    model.eval()
    for batch in tqdm(make_loader(a, recs, tok, fs, False), desc="shapley", mininterval=30):
        W = batch["word_mask"].shape[1]
        x = pad_to(batch["feats"], a.win).to(dev)
        mask = pad_to(batch["word_mask"], a.win, False).to(dev)
        ctx = None
        if model.enc is not None:
            ctx = pad_to(model.context_states(batch["input_ids"].to(dev), batch["attention_mask"].to(dev),
                                              batch["word_index"].to(dev), W), a.win)
        hum = a.classes.index("human")
        with amp(a, dev):
            phi, mf, me = channel_shapley(model, x, mask, ctx, pos=1 - hum, n_classes=2)
        phi, mf, me = phi[:, :W].cpu().numpy(), mf[:, :W].cpu().numpy(), me[:, :W].cpu().numpy()
        for k in range(len(phi)):
            r = recs[int(batch["rec_idx"][k])]
            s, n = int(batch["start"][k]), int(batch["word_mask"][k].sum())
            d = out.setdefault(r["meta"]["uid"], {"phi": np.zeros((len(r["words"]), phi.shape[-1]), np.float32),
                                                  "m_full": np.zeros(len(r["words"]), np.float32),
                                                  "m_empty": np.zeros(len(r["words"]), np.float32)})
            d["phi"][s:s + n], d["m_full"][s:s + n], d["m_empty"][s:s + n] = phi[k, :n], mf[k, :n], me[k, :n]
    pickle.dump(out, open(path, "wb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", choices=["seqxgpt", "seqx", "seqroberta", "sentroberta"], required=True)
    ap.add_argument("--step", default="", help="label for the results table (S0..S4, baseline)")
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--feats", default="../data/feats")
    ap.add_argument("--out", default="../runs_ladder")
    ap.add_argument("--train", default="semeval8c")
    ap.add_argument("--eval", default="")
    ap.add_argument("--feat_kind", choices=["lp4", "contrast"], default="lp4")
    ap.add_argument("--feat_cols", default="0,1,2,3,4,5", help="contrast: columns of token_features.FEAT_NAMES")
    ap.add_argument("--context", default="", help="seqx: HF encoder for word context (S2+)")
    ap.add_argument("--encoder", default="roberta-base", help="seqroberta / sentroberta backbone")
    ap.add_argument("--evidence_dropout", type=float, default=0.0)
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast on CUDA for the head (encoder stays fp32)")
    ap.add_argument("--boundary_head", action="store_true", help="S2b Boundary Curator")
    ap.add_argument("--lam_bnd", type=float, default=0.5)
    ap.add_argument("--bnd_pos_weight", type=float, default=5.0)
    ap.add_argument("--multiclass", action="store_true", help="SeqXGPT-Bench 6-way protocol")
    ap.add_argument("--shapley", action="store_true")
    ap.add_argument("--shap_cap", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5, help="head lr (SeqXGPT default)")
    ap.add_argument("--enc_lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=int, default=25000)
    ap.add_argument("--dev_cap", type=int, default=2500)
    a = ap.parse_args()
    a.eval = a.eval or a.train
    seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    a.classes = SEQX_CLASSES if a.multiclass else ["human", "AI"]
    n_classes, hum = len(a.classes), a.classes.index("human")
    path = lambda ds: os.path.join(a.data, f"{ds}.jsonl")  # noqa: E731
    trains, evals = a.train.split(","), a.eval.split(",")
    tr = with_classes(load_records([path(d) for d in trains], {"train"}, a.limit, a.cap), a.classes, a.multiclass)
    dv = with_classes(load_records([path(d) for d in trains], {"dev"}, a.limit, a.dev_cap), a.classes, a.multiclass)
    fs, names = feature_store(a, sorted(set(trains + evals)))
    if fs is not None:
        fs.fit_norm([r["meta"]["uid"] for r in tr])
    use_tok = a.arch in ("seqroberta", "sentroberta") or (a.arch == "seqx" and a.context)
    tok = AutoTokenizer.from_pretrained(a.context if a.arch == "seqx" else a.encoder) if use_tok else None
    # always train fp32 master weights (official Seq/Sent-RoBERTa classes load with the checkpoint dtype)
    model = build(a, n_classes, len(names)).float().to(dev)
    print(f"[{a.name}] arch={a.arch} channels={names} context={a.context or '-'} train={len(tr)} dev={len(dv)} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M device={dev}")

    loader = make_loader(a, tr, tok, fs, True)
    enc_names = ("enc.", "roberta.")
    groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith(enc_names)], "lr": a.lr},
              {"params": [p for n, p in model.named_parameters() if n.startswith(enc_names)], "lr": a.enc_lr}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.98), eps=1e-8, weight_decay=a.wd)
    total = len(loader) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(a.epochs):
        model.train()
        tot, n, skipped = 0.0, 0, 0
        for batch in tqdm(loader, desc=f"epoch {ep + 1}/{a.epochs}", mininterval=30):
            with amp(a, dev):
                if a.arch == "sentroberta":
                    logits = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
                    loss = F.cross_entropy(logits.float(), batch["labels"].to(dev), ignore_index=-1)
                else:
                    loss = step_loss(model, a, batch, dev)
            opt.zero_grad()
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):              # NaN/inf gradient: never let it reach the weights
                opt.zero_grad()
                skipped += 1
                continue
            opt.step()
            sch.step()
            tot, n = tot + loss.item(), n + 1
        bad = [k for k, v in model.state_dict().items() if v.is_floating_point() and not torch.isfinite(v).all()]
        if bad:
            raise SystemExit(f"[{a.name}] non-finite weights after epoch {ep + 1}: {bad[:5]} - aborting")
        f1 = dev_score(model, a, dv, tok, fs, dev, n_classes, hum)
        print(f"epoch {ep + 1}: train_loss {tot / max(n, 1):.4f}  dev word macro-F1 {f1:.4f}  "
              f"skipped_batches {skipped}/{skipped + n}  ({(time.time() - t0) / 60:.1f} min)")
        if f1 > best:
            best, best_state = f1, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    if dev == "cuda":
        print(f"[{a.name}] peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

    rd = os.path.join(a.out, a.name)
    os.makedirs(rd, exist_ok=True)
    json.dump({**{k: v for k, v in vars(a).items()}, "best_dev_f1": best, "channels": names},
              open(os.path.join(rd, "meta.json"), "w"), indent=1)
    for ds in evals:
        recs = [r for sp in ("dev", "cal", "test") for r in load_records([path(ds)], {sp}, a.limit,
                                                                          a.dev_cap if sp == "dev" else 0)]
        recs = with_classes(recs, a.classes, a.multiclass)
        P = predict(model, a, recs, tok, fs, dev, n_classes)
        dump = {"__meta__": {"name": a.name, "step": a.step, "arch": a.arch, "train": a.train, "channels": names}}
        for r in recs:
            pr, pb = binary(P[r["meta"]["uid"]], hum)
            dump[r["meta"]["uid"]] = {"pred": pr, "prob": pb, "split": r["split"]}
        pickle.dump(dump, open(os.path.join(rd, f"pred_{ds}.pkl"), "wb"))
        te = [r for r in recs if r["split"] == "test"]
        golds_bin = [np.where(np.asarray(r["labels"]) < 0, -1, (np.asarray(r["labels"]) != hum).astype(int)) for r in te]
        m = all_metrics(golds_bin, [dump[r["meta"]["uid"]]["pred"] for r in te], [dump[r["meta"]["uid"]]["prob"] for r in te])
        m.update(sentence_metrics([r["words"] for r in te], golds_bin, [dump[r["meta"]["uid"]]["pred"] for r in te]))
        if a.multiclass:
            mc = sentence_metrics([r["words"] for r in te], [np.asarray(r["labels"]) for r in te],
                                  [P[r["meta"]["uid"]]["cls"] for r in te])
            m.update({"multiclass_" + k: v for k, v in mc.items()})
        json.dump(m, open(os.path.join(rd, f"metrics_{ds}.json"), "w"), indent=1)
        print(f"[{a.name}] {ds} test: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))
        if a.shapley and a.arch == "seqx":
            dump_shapley(model, a, te, fs, tok, dev, names, os.path.join(rd, f"shap_{ds}.pkl"))


if __name__ == "__main__":
    main()
