"""End-to-end driver for process2 visual-object decoding pipeline.

For each subject in --subjects:
    1) Build CLIP image-embedding + patch-token cache if absent.
    2) Train img_only / eeg_only / eeg_img_patch baselines.
    3) Save figures (Nat Comm style) under fig_dir.

Usage:
    python -m process2_model_analysis.run_pipeline \
        --subjects zfn-0507 zxc-0516 \
        --root . \
        --selection experiment/stimuli_select/stimuli_rsvp_attention_lvis_pilot_20260506_164009.json \
        --coco_root C:/Users/thlab/Desktop/ES_coco/data/coco \
        --fig_dir process2_model_analysis/fig \
        --n_epochs 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from process2_model_analysis import img_embeddings as IE                # noqa
from process2_model_analysis import train_baselines as TB               # noqa
from process2_model_analysis import viz_results as VR                   # noqa


def _find_fif(data_dir: Path):
    cand = list(data_dir.glob("epochs_big-epo.fif"))
    if not cand:
        cand = list(data_dir.glob("*-epo.fif"))
    if not cand:
        raise FileNotFoundError(f"No -epo.fif found in {data_dir}")
    return cand[0]


def _find_session_json(data_dir: Path):
    cand = sorted(data_dir.glob("session_*.json"))
    if not cand:
        raise FileNotFoundError(f"No session_*.json in {data_dir}")
    return cand[0]


def build_cache(selection, coco_root, out_cache, model="ViT-B-32",
                 pretrained="openai", device="cuda"):
    if out_cache.exists():
        logging.info("cache exists, skipping: %s", out_cache)
        return
    sel = json.load(open(selection, encoding="utf-8"))
    image_ids, paths = [], []
    seen = set()
    for it in sel["items"]:
        iid = str(it["image_id"])
        if iid in seen:
            continue
        seen.add(iid)
        image_ids.append(iid)
        paths.append(Path(coco_root)/it["relative_path"])
    logging.info("Embedding %d unique images with %s/%s",
                 len(paths), model, pretrained)
    G, P, grid = IE.extract_clip_with_patches(paths, device=device,
                                               model_name=model,
                                               pretrained=pretrained)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_cache,
                         image_ids=np.array(image_ids),
                         features=G, patch_tokens=P,
                         grid=np.array(grid),
                         backbone=f"{model}/{pretrained}")
    logging.info("saved %s  G=%s  P=%s  grid=%s",
                 out_cache, G.shape, P.shape, grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--n_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--decim", type=int, default=4)
    ap.add_argument("--crop_tmin", type=float, default=-0.1)
    ap.add_argument("--crop_tmax", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("pipeline")
    root = Path(args.root).resolve()
    log.info("Root: %s", root)

    subj_dirs_for_viz = []
    for subj in args.subjects:
        log.info("════════════ subject %s ════════════", subj)
        data_dir = root / "data" / subj
        fif = _find_fif(data_dir)
        sess = _find_session_json(data_dir)
        cache = data_dir / f"img_embeddings_{args.model.lower().replace('-','_')}.npz"
        out_dir = data_dir / "process2_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("fif=%s", fif.name)
        log.info("session=%s", sess.name)
        log.info("cache=%s (exists=%s)", cache.name, cache.exists())

        # 1) cache
        build_cache(Path(args.selection), args.coco_root, cache,
                     model=args.model, pretrained=args.pretrained,
                     device=args.device)

        # 2) train
        import torch
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        # Re-build datasets through TB
        from process1_data_process.data_io import load_epochs
        from process2_model_analysis.dataset import EEGImgAttentionDataset
        from process2_model_analysis.models import (
            ATMEncoder, EEGImgPatchFusion, EEGOnlyClassifier, ImgOnlyClassifier,
        )
        from torch.utils.data import DataLoader

        b = load_epochs(str(fif), str(sess), pick_eeg=False,
                         load_data=True, decim=args.decim, dtype="float32",
                         crop_tmin=args.crop_tmin, crop_tmax=args.crop_tmax,
                         baseline=(-0.1, 0.0))
        img_cache = np.load(cache, allow_pickle=True)
        tr = EEGImgAttentionDataset(b, img_cache, split="train",
                                     normalize=True, want_patches=True)
        te = EEGImgAttentionDataset(b, img_cache, split="test",
                                     normalize=True, want_patches=True)
        log.info("train=%d test=%d classes=%d", len(tr), len(te), len(tr.classes))
        tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
        te_dl = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=0)
        n_channels = tr.eeg.shape[1]; n_samples = tr.eeg.shape[2]
        n_classes = len(tr.classes); img_dim = tr.img.shape[1]
        patch_dim = tr.img_patches.shape[2]

        device = args.device if torch.cuda.is_available() else "cpu"
        results = {}
        # img_only
        log.info("─── img_only ───")
        m = ImgOnlyClassifier(img_dim, n_classes)
        results["img_only"] = TB.train_and_eval(m, tr_dl, te_dl, device,
                                                 kind="img_only",
                                                 n_epochs=args.n_epochs, lr=args.lr)
        # eeg_only
        log.info("─── eeg_only ───")
        atm = ATMEncoder(n_channels, n_samples, n_classes,
                          d_model=128, n_heads=4, d_ff=256,
                          e_layers=1, feat_dim=256, dropout=0.3)
        m = EEGOnlyClassifier(atm)
        results["eeg_only"] = TB.train_and_eval(m, tr_dl, te_dl, device,
                                                 kind="eeg_only",
                                                 n_epochs=args.n_epochs, lr=args.lr)
        # eeg_img_patch
        log.info("─── eeg_img_patch ───")
        atm = ATMEncoder(n_channels, n_samples, n_classes,
                          d_model=128, n_heads=4, d_ff=256,
                          e_layers=1, feat_dim=256, dropout=0.3)
        m = EEGImgPatchFusion(eeg_encoder=atm, eeg_dim=atm.feat_dim,
                               patch_dim=patch_dim, img_dim=img_dim,
                               n_classes=n_classes, d_model=256, n_heads=4,
                               dropout=0.3)
        res = TB.train_and_eval(m, tr_dl, te_dl, device,
                                 kind="eeg_img_patch",
                                 n_epochs=args.n_epochs, lr=args.lr)
        m.load_state_dict(res["best_state"])
        attn = TB.extract_attention(m, te_dl, device, te, top_k=30)
        np.savez_compressed(out_dir/"attn_examples.npz",
                             attn=attn["attn"], logits=attn["logits"],
                             y_true=attn["y_true"], y_pred=attn["y_pred"],
                             conf=attn["conf"], image_id=attn["image_id"],
                             hint=attn["hint"], selected=attn["selected"],
                             grid=np.array(attn["grid"]),
                             classes=np.array(attn["classes"]))
        import torch as _t
        _t.save(res["best_state"], out_dir/"state_eeg_img_patch.pt")
        results["eeg_img_patch"] = res
        for k in results: results[k].pop("best_state", None)

        # summary
        summary = {
            "n_classes": n_classes, "n_train": len(tr), "n_test": len(te),
            "n_channels": n_channels, "n_samples": n_samples,
            "img_dim": img_dim, "patch_dim": patch_dim,
            "n_patches": tr.img_patches.shape[1],
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

        log.info("subject %s — summary: %s",
                 subj, {k: round(v["best_test_top1"], 4)
                         for k, v in results.items()})
        with open(out_dir/"compare_summary.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_baseline": results}, f,
                       ensure_ascii=False, indent=2, default=str)
        subj_dirs_for_viz.append(f"{subj}:{out_dir}")

    # 3) figures
    log.info("════════════ viz ════════════")
    VR_args = ["--subject_dirs", *subj_dirs_for_viz,
               "--selection_json", args.selection,
               "--coco_root", args.coco_root,
               "--fig_dir", args.fig_dir]
    log.info("VR args: %s", VR_args)
    sys.argv = ["viz_results"] + VR_args
    VR.main()
    log.info("DONE")


if __name__ == "__main__":
    main()
