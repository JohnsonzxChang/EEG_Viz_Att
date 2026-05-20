"""Train 3 baselines (img_only, eeg_only, eeg_img_patch) on one subject.

Output (saved under --out_dir):
    compare_summary.json   — top1/top5 per baseline + deltas
    history.json           — per-epoch loss/acc curves
    attn_examples.npz      — best test-trial attention maps (eeg_img_patch)
    state_eeg_img_patch.pt — final model weights (for re-attention later)

Usage:
    python -m process2_model_analysis.train_baselines \
        --fif data/zfn-0507/epochs_big-epo.fif \
        --session_json data/zfn-0507/session_*.json \
        --img_cache data/zfn-0507/img_embeddings_clip_vitb32.npz \
        --out_dir data/zfn-0507/process2_out
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).parent.parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from process1_data_process.data_io import load_epochs                  # noqa
from process2_model_analysis.dataset import EEGImgAttentionDataset     # noqa
from process2_model_analysis.models import (                            # noqa
    ATMEncoder, EEGImgPatchFusion, EEGOnlyClassifier, ImgOnlyClassifier,
)


def _forward(model, kind, eeg, img, patches):
    if kind == "img_only":
        return model(img)
    if kind == "eeg_only":
        return model(eeg)
    if kind == "eeg_img_patch":
        return model(eeg, img, patches)
    raise ValueError(kind)


def train_and_eval(model, train_dl, test_dl, device, kind,
                    n_epochs: int = 30, lr: float = 3e-4,
                    weight_decay: float = 1e-4):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()
    log = logging.getLogger(kind)

    history = []
    best_acc = 0.0
    best_state = None
    for ep in range(n_epochs):
        model.train()
        ep_loss = 0.0; ep_n = 0
        t0 = time.time()
        for eeg, img, patches, y, _ in train_dl:
            eeg = eeg.to(device); img = img.to(device)
            patches = patches.to(device); y = y.to(device)
            pred = _forward(model, kind, eeg, img, patches)
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * y.size(0); ep_n += y.size(0)
        sched.step()
        # eval
        model.eval()
        ys, preds, top5 = [], [], []
        with torch.no_grad():
            for eeg, img, patches, y, _ in test_dl:
                eeg = eeg.to(device); img = img.to(device)
                patches = patches.to(device)
                o = _forward(model, kind, eeg, img, patches)
                pred = o.argmax(-1).cpu().numpy()
                ys.append(y.numpy()); preds.append(pred)
                _, t5 = o.topk(min(5, o.size(-1)), dim=-1)
                top5_hit = (t5.cpu() == y.unsqueeze(-1)).any(-1).numpy()
                top5.append(top5_hit)
        y_true = np.concatenate(ys); y_pred = np.concatenate(preds)
        t5_arr = np.concatenate(top5)
        acc = float((y_true == y_pred).mean())
        top5_acc = float(t5_arr.mean())
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": ep, "train_loss": ep_loss / max(1, ep_n),
                         "test_top1": acc, "test_top5": top5_acc,
                         "lr": opt.param_groups[0]["lr"]})
        log.info("ep %2d  loss=%.4f  top1=%.4f  top5=%.4f  (%.1fs)",
                 ep, history[-1]["train_loss"], acc, top5_acc, time.time() - t0)

    return {"kind": kind, "best_test_top1": best_acc,
             "final_test_top1": float(history[-1]["test_top1"]),
             "final_test_top5": float(history[-1]["test_top5"]),
             "history": history, "best_state": best_state}


@torch.no_grad()
def extract_attention(model, test_dl, device, ds_test, top_k: int = 24):
    """Run eeg_img_patch model on test set; pick top-k correctly-classified
    high-confidence trials and save their attention map + image_id + label."""
    model.eval().to(device)
    all_attn = []
    all_logits = []
    all_idx = []
    for eeg, img, patches, y, idx in test_dl:
        eeg = eeg.to(device); img = img.to(device); patches = patches.to(device)
        logits, attn = model.forward_with_attn(eeg, img, patches)
        all_attn.append(attn.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_idx.append(idx.numpy())
    attn = np.concatenate(all_attn, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    idx = np.concatenate(all_idx, axis=0)
    y_true = ds_test.y[idx]
    y_pred = logits.argmax(-1)
    # softmax confidence
    e = np.exp(logits - logits.max(-1, keepdims=True))
    probs = e / e.sum(-1, keepdims=True)
    conf = probs[np.arange(len(probs)), y_pred]
    # Pick correct, then one highest-confidence per unique image_id
    # (avoids the gallery being dominated by repeats of the same stimulus).
    correct_mask = (y_pred == y_true)
    cand = np.nonzero(correct_mask)[0]
    cand_sorted = cand[np.argsort(-conf[cand])]
    seen = set(); order = []
    for c in cand_sorted:
        iid = str(ds_test.image_id[idx[c]])
        if iid in seen: continue
        seen.add(iid); order.append(c)
        if len(order) >= top_k: break
    order = np.asarray(order, dtype=np.int64)
    return {
        "attn": attn,         # (Ntest, P)
        "logits": logits,
        "y_true": y_true,
        "y_pred": y_pred,
        "conf": conf,
        "image_id": ds_test.image_id[idx],
        "hint": ds_test.hint[idx],
        "selected": order,
        "grid": ds_test.grid,
        "classes": ds_test.classes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fif", required=True)
    ap.add_argument("--session_json", required=True)
    ap.add_argument("--img_cache", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--decim", type=int, default=4)
    ap.add_argument("--crop_tmin", type=float, default=-0.1)
    ap.add_argument("--crop_tmax", type=float, default=0.45)
    ap.add_argument("--baseline_tmin", type=float, default=-0.1)
    ap.add_argument("--baseline_tmax", type=float, default=0.0)
    ap.add_argument("--n_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baselines",
                    default="img_only,eeg_only,eeg_img_patch")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("train")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    b = load_epochs(args.fif, args.session_json, pick_eeg=False,
                    load_data=True, decim=args.decim, dtype="float32",
                    crop_tmin=args.crop_tmin, crop_tmax=args.crop_tmax,
                    baseline=(args.baseline_tmin, args.baseline_tmax))
    img_cache = np.load(args.img_cache, allow_pickle=True)

    tr = EEGImgAttentionDataset(b, img_cache, split="train",
                                 normalize=True, want_patches=True)
    te = EEGImgAttentionDataset(b, img_cache, split="test",
                                 normalize=True, want_patches=True)
    log.info("train=%d  test=%d  classes=%d  missing_img(tr,te)=(%d,%d)",
             len(tr), len(te), len(tr.classes),
             tr.n_missing_img, te.n_missing_img)

    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)
    te_dl = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    n_channels = tr.eeg.shape[1]; n_samples = tr.eeg.shape[2]
    n_classes = len(tr.classes); img_dim = tr.img.shape[1]
    patch_dim = tr.img_patches.shape[2]
    n_patches = tr.img_patches.shape[1]
    log.info("EEG (C,T)=(%d,%d), n_classes=%d, img_dim=%d, patches=%d×%d",
             n_channels, n_samples, n_classes, img_dim, n_patches, patch_dim)

    results = {}
    selected = [s.strip() for s in args.baselines.split(",") if s.strip()]

    if "img_only" in selected:
        log.info("─── img_only ───")
        m = ImgOnlyClassifier(img_dim, n_classes)
        results["img_only"] = train_and_eval(m, tr_dl, te_dl, device,
                                              kind="img_only",
                                              n_epochs=args.n_epochs,
                                              lr=args.lr)

    if "eeg_only" in selected:
        log.info("─── eeg_only (ATM) ───")
        atm = ATMEncoder(n_channels, n_samples, n_classes,
                          d_model=128, n_heads=4, d_ff=256,
                          e_layers=1, feat_dim=256, dropout=0.3)
        m = EEGOnlyClassifier(atm)
        results["eeg_only"] = train_and_eval(m, tr_dl, te_dl, device,
                                              kind="eeg_only",
                                              n_epochs=args.n_epochs,
                                              lr=args.lr)

    if "eeg_img_patch" in selected:
        log.info("─── eeg_img_patch (ATM + Patch-CrossAttn) ───")
        atm = ATMEncoder(n_channels, n_samples, n_classes,
                          d_model=128, n_heads=4, d_ff=256,
                          e_layers=1, feat_dim=256, dropout=0.3)
        m = EEGImgPatchFusion(eeg_encoder=atm, eeg_dim=atm.feat_dim,
                               patch_dim=patch_dim, img_dim=img_dim,
                               n_classes=n_classes,
                               d_model=256, n_heads=4, dropout=0.3)
        res = train_and_eval(m, tr_dl, te_dl, device,
                              kind="eeg_img_patch",
                              n_epochs=args.n_epochs, lr=args.lr)
        # Restore best weights and extract attention
        m.load_state_dict(res["best_state"])
        attn_info = extract_attention(m, te_dl, device, te, top_k=24)
        # Save attention info
        np.savez_compressed(out_dir / "attn_examples.npz",
                             attn=attn_info["attn"],
                             logits=attn_info["logits"],
                             y_true=attn_info["y_true"],
                             y_pred=attn_info["y_pred"],
                             conf=attn_info["conf"],
                             image_id=attn_info["image_id"],
                             hint=attn_info["hint"],
                             selected=attn_info["selected"],
                             grid=np.array(attn_info["grid"]),
                             classes=np.array(attn_info["classes"]))
        torch.save(res["best_state"], out_dir / "state_eeg_img_patch.pt")
        results["eeg_img_patch"] = res

    # Strip best_state from JSON dump
    for k in results:
        results[k].pop("best_state", None)

    # Headline comparison
    summary = {
        "n_classes": n_classes, "n_train": len(tr), "n_test": len(te),
        "n_channels": n_channels, "n_samples": n_samples,
        "img_dim": img_dim, "patch_dim": patch_dim, "n_patches": n_patches,
        "baselines": {k: {"best_test_top1": v["best_test_top1"],
                           "final_test_top1": v["final_test_top1"],
                           "final_test_top5": v["final_test_top5"]}
                       for k, v in results.items()},
    }
    if {"img_only", "eeg_only"}.issubset(results):
        summary["delta_eeg_vs_img"] = (results["eeg_only"]["best_test_top1"]
                                         - results["img_only"]["best_test_top1"])
    if {"img_only", "eeg_img_patch"}.issubset(results):
        summary["delta_fusion_vs_img"] = (results["eeg_img_patch"]["best_test_top1"]
                                            - results["img_only"]["best_test_top1"])
    if {"eeg_only", "eeg_img_patch"}.issubset(results):
        summary["delta_fusion_vs_eeg"] = (results["eeg_img_patch"]["best_test_top1"]
                                            - results["eeg_only"]["best_test_top1"])

    log.info("══════ Summary ══════")
    for k, v in summary["baselines"].items():
        log.info("  %-14s best_top1=%.4f final_top5=%.4f",
                 k, v["best_test_top1"], v["final_test_top5"])
    for kk in ("delta_eeg_vs_img", "delta_fusion_vs_img", "delta_fusion_vs_eeg"):
        if kk in summary:
            log.info("  %-20s = %+.4f", kk, summary[kk])

    with open(out_dir / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_baseline": results}, f,
                   ensure_ascii=False, indent=2, default=str)
    log.info("Wrote %s", out_dir / "compare_summary.json")


if __name__ == "__main__":
    main()
