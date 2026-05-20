"""Generate Nat Comm-style figures from train_baselines.py outputs.

Figures produced (per subject):
    F1_headline_bar.{png,pdf}     — bar plot of top1 across the 3 baselines
    F2_training_curves.{png,pdf}  — train-loss + test-top1 over epochs
    F3_confusion_matrix.{png,pdf} — 40×40 confusion for eeg_img_patch
    F4_attn_overlays.{png,pdf}    — EEG→pic attention maps overlayed on COCO images

Cross-subject combined figure (if multiple subjects supplied):
    F0_cross_subject_bar.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

HERE = Path(__file__).parent.parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from process2_model_analysis.plot_style import (                       # noqa
    apply_natcomm_style, NPG, BASELINE_COLOR, BASELINE_LABEL,
)


# ─────────────────────── helpers ───────────────────────

def _bar_headline(summary, out_path, title=""):
    apply_natcomm_style()
    baselines = ["img_only", "eeg_only", "eeg_img_patch"]
    labels = [BASELINE_LABEL[b] for b in baselines]
    top1 = [summary["baselines"][b]["best_test_top1"] * 100 for b in baselines]
    top5 = [summary["baselines"][b]["final_test_top5"] * 100 for b in baselines]
    n_classes = summary["n_classes"]
    chance = 100.0 / n_classes
    colors = [BASELINE_COLOR[b] for b in baselines]

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    x = np.arange(len(baselines))
    w = 0.36
    b1 = ax.bar(x - w/2, top1, w, color=colors, label="top-1",
                 edgecolor="black", linewidth=0.5)
    b5 = ax.bar(x + w/2, top5, w, color=colors, alpha=0.45,
                 label="top-5", edgecolor="black", linewidth=0.5)
    ax.axhline(chance, ls="--", color="grey", lw=0.7, label=f"chance ({chance:.1f}%)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    for rect, val in zip(b1, top1):
        ax.text(rect.get_x()+rect.get_width()/2, val+1, f"{val:.1f}",
                ha="center", va="bottom", fontsize=7)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_ylim(0, max(100, max(top5)+8))
    plt.savefig(out_path.with_suffix(".png"))
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _training_curves(per_baseline, out_path, title=""):
    apply_natcomm_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for b, info in per_baseline.items():
        if b not in BASELINE_COLOR:
            continue
        c = BASELINE_COLOR[b]; lab = BASELINE_LABEL[b]
        hist = info["history"]
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist],
                      color=c, label=lab, lw=1.2)
        axes[1].plot(ep, [h["test_top1"]*100 for h in hist],
                      color=c, label=lab, lw=1.2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test top-1 (%)")
    axes[1].legend(loc="lower right", fontsize=7)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"))
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _confusion(attn_npz, out_path, title=""):
    apply_natcomm_style()
    y_true = attn_npz["y_true"]; y_pred = attn_npz["y_pred"]
    classes = attn_npz["classes"]
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    row_sum = cm.sum(axis=1, keepdims=True); row_sum[row_sum == 0] = 1
    cm_norm = cm / row_sum
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(cm_norm, cmap="magma", vmin=0, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=4)
    ax.set_yticklabels(classes, fontsize=4)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (HINT)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Row-normalized count", fontsize=7)
    plt.savefig(out_path.with_suffix(".png"))
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _attn_overlays(attn_npz, selection_json, coco_root, out_path,
                    n_rows=5, n_cols=6, title=""):
    apply_natcomm_style()
    sel = json.load(open(selection_json, encoding="utf-8"))
    id_to_path = {str(it["image_id"]): it["relative_path"] for it in sel["items"]}

    attn = attn_npz["attn"]
    sel_idx = attn_npz["selected"]
    image_id = attn_npz["image_id"]
    hint = attn_npz["hint"]
    classes = attn_npz["classes"]
    y_pred = attn_npz["y_pred"]
    conf = attn_npz["conf"]
    grid = tuple(attn_npz["grid"].tolist())

    n = min(n_rows * n_cols, len(sel_idx))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.8, n_rows*1.9))
    axes = np.array(axes).reshape(-1)
    for k in range(n):
        ax = axes[k]
        i = int(sel_idx[k])
        iid = str(image_id[i])
        a = attn[i].reshape(grid)
        rel = id_to_path.get(iid)
        if rel is None:
            ax.text(0.5, 0.5, f"missing\n{iid}", ha="center", va="center")
            ax.axis("off"); continue
        im = Image.open(Path(coco_root) / rel).convert("RGB")
        im = im.resize((224, 224))
        # Upsample attention map to 224×224
        from scipy.ndimage import zoom
        a_norm = (a - a.min()) / (a.max() - a.min() + 1e-9)
        au = zoom(a_norm, (224/grid[0], 224/grid[1]), order=1)
        ax.imshow(im)
        ax.imshow(au, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        h = str(hint[i])
        p = classes[int(y_pred[i])]
        ax.set_title(f"hint={h}\npred={p}  p={conf[i]:.2f}",
                      fontsize=6)
        ax.axis("off")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"))
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _cross_subject_bar(per_subject, out_path, title=""):
    """per_subject : {subj_id: summary_dict}"""
    apply_natcomm_style()
    baselines = ["img_only", "eeg_only", "eeg_img_patch"]
    subj_ids = sorted(per_subject.keys())
    nB = len(baselines); nS = len(subj_ids)
    width = 0.8 / nB
    fig, ax = plt.subplots(figsize=(1.6+nS*0.9, 2.8))
    x = np.arange(nS)
    chance = 100.0 / per_subject[subj_ids[0]]["n_classes"]
    for j, b in enumerate(baselines):
        vals = [per_subject[s]["baselines"][b]["best_test_top1"]*100
                for s in subj_ids]
        ax.bar(x + (j - (nB-1)/2)*width, vals, width,
                color=BASELINE_COLOR[b], label=BASELINE_LABEL[b],
                edgecolor="black", linewidth=0.5)
    ax.axhline(chance, ls="--", color="grey", lw=0.7,
                label=f"chance ({chance:.1f}%)")
    ax.set_xticks(x); ax.set_xticklabels(subj_ids)
    ax.set_ylabel("Best test top-1 (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7, ncols=2)
    ax.set_ylim(0, max(100, max([per_subject[s]["baselines"]["eeg_img_patch"]["best_test_top1"]
                                  for s in subj_ids])*100 + 12))
    plt.savefig(out_path.with_suffix(".png"))
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_dirs", nargs="+", required=True,
                    help="Pairs of 'subject_id:out_dir' for each subject")
    ap.add_argument("--selection_json", required=True)
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--fig_dir", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("viz")
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    per_subject = {}
    for tok in args.subject_dirs:
        sid, sdir = tok.split(":", 1)
        sdir = Path(sdir)
        rep = json.load(open(sdir/"compare_summary.json", encoding="utf-8"))
        per_subject[sid] = rep["summary"]
        # Per-subject figures
        out_sub = fig_dir / sid; out_sub.mkdir(parents=True, exist_ok=True)
        _bar_headline(rep["summary"], out_sub/"F1_headline_bar",
                       title=f"{sid} — 40-class HINT decoding")
        _training_curves(rep["per_baseline"], out_sub/"F2_training_curves",
                          title=f"{sid} — training dynamics")
        attn_path = sdir/"attn_examples.npz"
        if attn_path.exists():
            attn_npz = np.load(attn_path, allow_pickle=True)
            _confusion(attn_npz, out_sub/"F3_confusion_matrix",
                        title=f"{sid} — confusion (eeg_img_patch)")
            _attn_overlays(attn_npz, args.selection_json, args.coco_root,
                            out_sub/"F4_attn_overlays",
                            title=f"{sid} — EEG→pic attention "
                                  f"(top-confidence correct test trials)")
        log.info("subject %s done — figures in %s", sid, out_sub)

    if len(per_subject) > 1:
        _cross_subject_bar(per_subject, fig_dir/"F0_cross_subject_bar",
                            title="Cross-subject benchmark")
        log.info("cross-subject bar saved")

    # Write a markdown digest
    md = ["# process2 — visual decoding results\n",
          "*Generated by `viz_results.py`.*\n"]
    for sid, s in per_subject.items():
        md.append(f"## {sid}")
        md.append("| baseline | best top-1 | final top-5 |")
        md.append("|---|---|---|")
        for k, v in s["baselines"].items():
            md.append(f"| {BASELINE_LABEL.get(k,k)} | "
                       f"{v['best_test_top1']*100:.2f}% | "
                       f"{v['final_test_top5']*100:.2f}% |")
        md.append("")
        for kk in ("delta_eeg_vs_img", "delta_fusion_vs_img", "delta_fusion_vs_eeg"):
            if kk in s:
                md.append(f"- **{kk}** = {s[kk]*100:+.2f} pp")
        md.append("")
    (fig_dir/"results_digest.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote %s", fig_dir/"results_digest.md")


if __name__ == "__main__":
    main()
