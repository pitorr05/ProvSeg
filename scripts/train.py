"""Train ProvSeg v2 (Evidence-Bottleneck) or an ablation/baseline, then dump word-level predictions.

  python scripts/train.py --name provseg                         # full model (all channels)
  python scripts/train.py --name abl_blackbox --blackbox         # v1 concat+linear (no bottleneck)
  python scripts/train.py --name abl_nosyntax --no_syntax
  python scripts/train.py --name abl_noedit   --no_edit          # no privileged version supervision
  python scripts/train.py --name abl_noambig  --no_ambiguity
  python scripts/train.py --name seqroberta   --encoder roberta-large --no_feats --blackbox
  python scripts/train.py --name seqxgpt_feats --encoder none --blackbox

Pred dumps (pred_<ds>.pkl): uid -> {pred, prob, contrib?, u?, edit_prob?, channels}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from provseg.data.torch_data import FeatureStore, WindowDataset, load_dep, load_records  # noqa: E402
from provseg.eval.metrics import all_metrics  # noqa: E402
from provseg.features.token_features import FEAT_NAMES  # noqa: E402
from provseg.models.efp import EFProvSeg  # noqa: E402
from provseg.models.provseg import ProvSeg  # noqa: E402

EVAL_SETS = ["seqxgpt", "semeval8c", "beemo", "pasted", "pasted_ood", "apt_eval", "editlens"]


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_predict(model, loader, device, amp, use_records=False):
    """{rec_idx: [(start, {pred, prob, ...}) ...]} per window. use_records: EFP plugs in observed edits."""
    model.eval()
    out = {}
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
            o = model.predict(b, use_records=True) if use_records else model.predict(b)
        o = {k: v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy() for k, v in o.items()}
        for k in range(len(o["pred"])):
            ri, s = int(batch["rec_idx"][k]), int(batch["start"][k])
            out.setdefault(ri, []).append((s, {kk: vv[k] for kk, vv in o.items()}))
    return out


def stitch(chunks, n):
    res = {"pred": np.full(n, -1, dtype=np.int64), "prob": np.full((n, 2), 0.5, dtype=np.float32)}
    for s, o in chunks:
        for k, v in o.items():
            if k not in res:
                res[k] = np.zeros((n,) + v.shape[1:], dtype=np.float32)
            L = min(len(v), n - s)
            res[k][s:s + L] = v[:L]
    bad = res["pred"] < 0
    res["pred"][bad] = res["prob"][bad].argmax(-1)
    if "edit_logit" in res:
        res["edit_prob"] = 1 / (1 + np.exp(-res.pop("edit_logit")))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--feats", default="../data/feats")
    ap.add_argument("--out", default="../runs")
    ap.add_argument("--train", default="seqxgpt,semeval8c,beemo,pasted,editlens")
    ap.add_argument("--eval", default=",".join(EVAL_SETS))
    ap.add_argument("--encoder", default="microsoft/deberta-v3-large")
    ap.add_argument("--no_feats", action="store_true")
    ap.add_argument("--feat_subset", default="")
    ap.add_argument("--no_crf", action="store_true")
    ap.add_argument("--no_syntax", action="store_true")
    ap.add_argument("--no_edit", action="store_true")
    ap.add_argument("--no_ambiguity", action="store_true")
    ap.add_argument("--blackbox", action="store_true")
    ap.add_argument("--model", default="efp", choices=["efp", "additive", "blackbox"],
                    help="efp = Edit-Factorized Provenance (main); additive = v2 evidence bottleneck; blackbox = v1")
    ap.add_argument("--lam_fac", type=float, default=0.5)
    ap.add_argument("--lam_edit", type=float, default=0.5)
    ap.add_argument("--lam_amb", type=float, default=0.2)
    ap.add_argument("--channel_dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--win", type=int, default=160)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="records per file (smoke test)")
    ap.add_argument("--cap_per_ds", type=int, default=25000, help="max train docs per dataset (uid-hash subset)")
    ap.add_argument("--dev_cap", type=int, default=2500, help="max dev docs per dataset (uid-hash subset)")
    a = ap.parse_args()

    if a.blackbox:
        a.model = "blackbox"
    a.blackbox = a.model == "blackbox"
    seed_all(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    run_dir = os.path.join(a.out, f"{a.name}_s{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(run_dir, "args.json"), "w"), indent=2)

    path = lambda ds: os.path.join(a.data, f"{ds}.jsonl")  # noqa: E731
    tsets = [d for d in a.train.split(",") if os.path.exists(path(d))]
    train_recs = load_records([path(d) for d in tsets], {"train"}, a.limit, a.cap_per_ds, a.seed)
    dev_recs = load_records([path(d) for d in tsets], {"dev"}, a.limit, a.dev_cap, a.seed)

    feats, feat_names = None, []
    if not a.no_feats:
        feat_names = a.feat_subset.split(",") if a.feat_subset else list(FEAT_NAMES)
        npz = [p for p in sorted(glob.glob(os.path.join(a.feats, "*.npz"))) if not p.endswith("_dep.npz")]
        feats = FeatureStore(npz, [FEAT_NAMES.index(f) for f in feat_names])
        train_recs = [r for r in train_recs if r["meta"]["uid"] in feats.arrs]
        dev_recs = [r for r in dev_recs if r["meta"]["uid"] in feats.arrs]
        feats.fit_norm([r["meta"]["uid"] for r in train_recs])
    use_syntax = not (a.no_syntax or a.blackbox)
    dep = load_dep(a.feats) if use_syntax else None
    print(f"train docs={len(train_recs)} dev docs={len(dev_recs)} feats={feat_names} syntax={use_syntax}")

    tok = None
    if a.encoder != "none":
        kw = {"add_prefix_space": True} if "roberta" in a.encoder else {}
        tok = AutoTokenizer.from_pretrained(a.encoder, use_fast=True, **kw)
    if a.model == "efp":
        model = EFProvSeg(a.encoder, feat_names, use_crf=not a.no_crf, use_syntax=use_syntax,
                          use_edit_sup=not a.no_edit, channel_dropout=a.channel_dropout).to(device)
    else:
        model = ProvSeg(a.encoder, feat_names, use_crf=not a.no_crf, use_syntax=use_syntax,
                        use_edit=not a.no_edit, use_ambiguity=not a.no_ambiguity, blackbox=a.blackbox,
                        channel_dropout=a.channel_dropout).to(device)
    channels = getattr(model, "channels", [])
    print("channels:", channels)

    tr = WindowDataset(train_recs, tok, feats, dep, a.win)
    dv = WindowDataset(dev_recs, tok, feats, dep, a.win)
    tl = DataLoader(tr, a.bs, shuffle=True, collate_fn=tr.collate, num_workers=0)
    dl = DataLoader(dv, a.bs * 2, shuffle=False, collate_fn=dv.collate)

    enc_ids = {id(p) for p in model.enc.parameters()} if model.enc is not None else set()
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) in enc_ids], "lr": a.lr},
                             {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": a.head_lr}],
                            weight_decay=0.01)
    steps = a.epochs * len(tl)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    best, log = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, parts = time.time(), 0.0, {}
        for batch in tqdm(tl, desc=f"ep{ep}", mininterval=30):
            b = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
                o = model(b, lam_edit=a.lam_edit, lam_amb=a.lam_amb, lam_fac=a.lam_fac)
            o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            tot += o["loss"].item()
            for k, v in o["logs"].items():
                parts[k] = parts.get(k, 0.0) + float(v)
        preds = run_predict(model, dl, device, amp)
        G, P, PR = [], [], []
        for ri, r in enumerate(dev_recs):
            res = stitch(preds.get(ri, []), len(r["words"]))
            G.append(np.array(r["labels"]))
            P.append(res["pred"])
            PR.append(res["prob"])
        m = all_metrics(G, P, PR)
        log.append({"epoch": ep, "train_loss": tot / max(len(tl), 1),
                    **{f"loss_{k}": v / max(len(tl), 1) for k, v in parts.items()},
                    "time": time.time() - t0, **m})
        if a.model == "additive":
            log[-1]["channel_weights"] = dict(zip(channels, model.w.detach().cpu().tolist()))
        print(json.dumps(log[-1]))
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    json.dump(log, open(os.path.join(run_dir, "train_log.json"), "w"), indent=2)

    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
    for ds in a.eval.split(","):
        if not os.path.exists(path(ds)):
            continue
        recs = load_records([path(ds)], {"dev", "cal", "test"}, a.limit)
        if feats is not None:
            recs = [r for r in recs if r["meta"]["uid"] in feats.arrs]
        if not recs:
            continue
        eds = WindowDataset(recs, tok, feats, dep, a.win)
        el = DataLoader(eds, a.bs * 2, shuffle=False, collate_fn=eds.collate)
        modes = [("", False)]
        if a.model == "efp" and any(r.get("edited") and max(r["edited"]) >= 0 for r in recs):
            modes.append(("_records", True))      # records track: observed edit indicators plugged in
        for suffix, use_rec in modes:
            preds = run_predict(model, el, device, amp, use_records=use_rec)
            dump = {"__channels__": channels, "__model__": a.model}
            for ri, r in enumerate(recs):
                dump[r["meta"]["uid"]] = stitch(preds.get(ri, []), len(r["words"]))
            d = run_dir + suffix
            os.makedirs(d, exist_ok=True)
            pickle.dump(dump, open(os.path.join(d, f"pred_{ds}.pkl"), "wb"))
            print(f"[pred{suffix}] {ds}: {len(dump) - 2} docs")


if __name__ == "__main__":
    main()
