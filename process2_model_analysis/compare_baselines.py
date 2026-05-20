"""Train+eval three baselines on the same train/test split:

  img_only   : ImgOnlyClassifier
  eeg_only   : EEGOnlyClassifier(ATMEncoder)
  eeg_img    : EEGImgFusionClassifier(ATMEncoder + img MLP + cross-attn)

Outputs the gains explicitly so the user's headline claims —
  "EEG adds X% over image-only" and
  "EEG+Img adds Y% over image-only" — can be quoted directly.

Usage:
    python -m process2_model_analysis.compare_baselines \
        --fif data/zfn-0507/epochs_big-epo.fif \
        --session_json data/zfn-0507/session_rsvp_attention_lvis_pilot_20260507_201520.json \
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
    ATMEncoder, EEGImgFusionClassifier, EEGOnlyClassifier,
    ImgOnlyClassifier,
)


def train_and_eval(model: nn.Module, train_dl: DataLoader, test_dl: DataLoader,
                    device: str, kind: str, n_epochs: int = 30, lr: float = 3e-4,
                    weight_decay: float = 1e-4) -> dict:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()
    log = logging.getLogger(kind)

    history = []
    best_acc = 0.0
    for ep in range(n_epochs):
        model.train()
        ep_loss = 0.0; ep_n = 0
        t0 = time.time()
        for eeg, img, y in train_dl:
            eeg = eeg.to(device); img = img.to(device); y = y.to(device)
            if kind == "img_only":
                pred = model(img)
            elif kind == "eeg_only":
                pred = model(eeg)
            elif kind == "eeg_img":
                pred = model(eeg, img)
            else:
                raise ValueError(kind)
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * y.size(0); ep_n += y.size(0)
        sched.step()
        # eval
        model.eval()
        ys, preds, top5 = [], [], []
        with torch.no_grad():
            for eeg, img, y in test_dl:
                eeg = eeg.to(device); img = img.to(device)
                if kind == "img_only":
                    o = model(img)
                elif kind == "eeg_only":
                    o = model(eeg)
                else:
                    o = model(eeg, img)
                pred = o.argmax(-1).cpu().numpy()
                ys.append(y.numpy()); preds.append(pred)
                _, t5 = o.topk(min(5, o.size(-1)), dim=-1)
                top5_hit = (t5.cpu() == y.unsqueeze(-1)).any(-1).numpy()
                top5.append(top5_hit)
        y_true = np.concatenate(ys); y_pred = np.concatenate(preds)
        t5_arr = np.concatenate(top5)
        acc = (y_true == y_pred).mean()
        top5_acc = t5_arr.mean()
        if acc > best_acc:
            best_acc = float(acc)
        history.append({"epoch": ep, "train_loss": ep_loss / max(1, ep_n),
                         "test_top1": float(acc), "test_top5": float(top5_acc),
                         "lr": opt.param_groups[0]["lr"]})
        log.info("ep %2d  loss=%.4f  acc=%.4f  top5=%.4f  (%.1fs)",
                 ep, history[-1]["train_loss"], acc, top5_acc, time.time() - t0)

    return {"kind": kind, "best_test_top1": best_acc,
             "final_test_top1": float(history[-1]["test_top1"]),
             "final_test_top5": float(history[-1]["test_top5"]),
             "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fif", required=True)
    ap.add_argument("--session_json", required=True)
    ap.add_argument("--img_cache", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--decim", type=int, default=4)
    ap.add_argument("--n_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baselines", default="img_only,eeg_only,eeg_img")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("compare")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # Data
    b = load_epochs(args.fif, args.session_json, pick_eeg=False,
                    load_data=True, decim=args.decim, dtype="float32")
    img_cache = np.load(args.img_cache, allow_pickle=True)

    tr = EEGImgAttentionDataset(b, img_cache, split="train", normalize=True)
    te = EEGImgAttentionDataset(b, img_cache, split="test", normalize=True)
    log.info("train=%d test=%d classes=%d img_dim=%d", len(tr), len(te),
             len(tr.classes), tr.img.shape[1])

    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)
    te_dl = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    n_channels = tr.eeg.shape[1]; n_samples = tr.eeg.shape[2]
    n_classes = len(tr.classes); img_dim = tr.img.shape[1]
    log.info("EEG (C,T) = (%d,%d), n_classes=%d, img_dim=%d",
             n_channels, n_samples, n_classes, img_dim)

    results = {}
    selected = [s.strip() for s in args.baselines.split(",") if s.strip()]

    if "img_only" in selected:
        log.info("─── baseline img_only ───")
        m = ImgOnlyClassifier(img_dim, n_classes)
        results["img_only"] = train_and_eval(m, tr_dl, te_dl, device,
                                              kind="img_only",
                                              n_epochs=args.n_epochs,
                                              lr=args.lr)

    if "eeg_only" in selected:
        log.info("─── baseline eeg_only (ATM) ───")
        atm = ATMEncoder(n_channels, n_samples, n_classes,
                          d_model=128, n_heads=4, d_ff=256,
                          e_layers=1, feat_dim=256, dropout=0.3)
        m = EEGOnlyClassifier(atm)
        results["eeg_only"] = train_and_eval(m, tr_dl, te_dl, device,
                                              kind="eeg_only",
                                              n_epochs=args.n_epochs,
                                              lr=args.lr)

    if "eeg_img" in selected:
        log.info("─── fusion eeg_img (ATM + Img + cross-attn) ───")
        atm = ATMEncoder(n_channels, n_samples, n_classes,
                          d_model=128, n_heads=4, d_ff=256,
                          e_layers=1, feat_dim=256, dropout=0.3)
        m = EEGImgFusionClassifier(eeg_encoder=atm, eeg_dim=atm.feat_dim,
                                     img_dim=img_dim, n_classes=n_classes,
                                     d_model=256, n_heads=4, dropout=0.3)
        results["eeg_img"] = train_and_eval(m, tr_dl, te_dl, device,
                                              kind="eeg_img",
                                              n_epochs=args.n_epochs,
                                              lr=args.lr)

    # Headline comparison
    summary = {
        "n_classes": n_classes,
        "n_train": len(tr),
        "n_test": len(te),
        "baselines": {k: {"best_test_top1": v["best_test_top1"],
                           "final_test_top1": v["final_test_top1"],
                           "final_test_top5": v["final_test_top5"]}
                       for k, v in results.items()},
    }
    if {"img_only", "eeg_only"}.issubset(results):
        summary["delta_eeg_vs_img"] = (results["eeg_only"]["best_test_top1"]
                                         - results["img_only"]["best_test_top1"])
    if {"img_only", "eeg_img"}.issubset(results):
        summary["delta_fusion_vs_img"] = (results["eeg_img"]["best_test_top1"]
                                            - results["img_only"]["best_test_top1"])
    if {"eeg_only", "eeg_img"}.issubset(results):
        summary["delta_fusion_vs_eeg"] = (results["eeg_img"]["best_test_top1"]
                                            - results["eeg_only"]["best_test_top1"])

    log.info("══════ Summary ══════")
    for k, v in summary["baselines"].items():
        log.info("  %-10s best_top1=%.4f final_top5=%.4f",
                 k, v["best_test_top1"], v["final_test_top5"])
    if "delta_eeg_vs_img" in summary:
        log.info("  Δ (eeg − img)        = %+.4f", summary["delta_eeg_vs_img"])
    if "delta_fusion_vs_img" in summary:
        log.info("  Δ (eeg+img − img)    = %+.4f", summary["delta_fusion_vs_img"])
    if "delta_fusion_vs_eeg" in summary:
        log.info("  Δ (eeg+img − eeg)    = %+.4f", summary["delta_fusion_vs_eeg"])

    with open(out_dir / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_baseline": results}, f,
                   ensure_ascii=False, indent=2, default=str)
    log.info("Wrote %s", out_dir / "compare_summary.json")


if __name__ == "__main__":
    main()
