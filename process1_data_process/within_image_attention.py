"""Within-image hint-swap analysis — the cleanest attention contrast.

This module computes the response difference for trials with IDENTICAL
visual stimulus (same image) but DIFFERENT top-down attention target
(different hint). Two complementary measures are produced:

1.  ERP-based  Δ-wave analysis
    – mean |Δ| across all (image, hint_A, hint_B) pairs (with baseline
      reference plotted), and a permutation test of the signed mean.

2.  Decoding-based attention index  (paper-grade gold standard)
    – train a per-image LDA to predict hint from EEG, using ONLY trials
      of that image. Cross-validated accuracy averaged across images is
      the canonical "feature-based attention decodability" (Stokes &
      Spaak 2016, *Trends Cogn Sci*).

The two measures are complementary: (1) is interpretable as µV; (2) is
the population-level metric that the rest of the paper (process2/) will
benchmark deep models against.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
from scipy.stats import kurtosis, skew
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from process1_data_process.data_io import EpochBundle
from process1_data_process.plot_style import (
    PALETTE, apply_publication_style, save_fig,
)

log = logging.getLogger(__name__)
apply_publication_style()


def _eeg_chs(b):
    return [i for i, n in enumerate(b.ch_names) if n not in ("gaze_x", "gaze_y")]


def _features_batch(X, sfreq):
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    nyq = sfreq / 2
    feats = []
    for lo, hi in bands:
        if hi >= nyq:
            hi = nyq - 1
        sos = sg.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        filt = sg.sosfiltfilt(sos, X, axis=2)
        bp = np.log1p(np.mean(filt ** 2, axis=2))
        feats.append(bp)
    feats.append(X.mean(axis=2))
    feats.append(X.var(axis=2))
    feats.append(skew(X, axis=2))
    feats.append(kurtosis(X, axis=2))
    return np.concatenate(feats, axis=1)


def plot_within_image_attention(b: EpochBundle, out_dir: Path,
                                 min_trials_per_cell: int = 4) -> dict:
    """ERP-based |Δ| analysis with proper baseline reference."""
    chs = _eeg_chs(b)
    t = b.times * 1000

    cells = defaultdict(list)
    for i in range(len(b)):
        if b.hint[i] == "":
            continue
        cells[(int(b.image_id[i]), b.hint[i])].append(i)

    cell_erp = {}
    for k, idxs in cells.items():
        if len(idxs) < min_trials_per_cell:
            continue
        cell_erp[k] = b.data[idxs][:, chs, :].mean(axis=(0, 1))

    img_to_hints = defaultdict(list)
    for (iid, h) in cell_erp:
        img_to_hints[iid].append(h)
    multi = {iid: hs for iid, hs in img_to_hints.items() if len(hs) >= 2}
    log.info("within-image attention: %d multi-hint images, ≥%d trials/cell",
              len(multi), min_trials_per_cell)

    diff_list = []
    for iid, hints in multi.items():
        for i in range(len(hints)):
            for j in range(i + 1, len(hints)):
                diff_list.append(cell_erp[(iid, hints[i])]
                                  - cell_erp[(iid, hints[j])])
    D = np.stack(diff_list, axis=0)
    log.info("n_pairs = %d", len(D))

    abs_D = np.abs(D)
    mean_abs = abs_D.mean(0)
    rms_D = np.sqrt((D ** 2).mean(0))

    # The expected baseline value of |Δ| under noise-only null:
    # E[|X|] = σ_pair × √(2/π) where σ_pair is the std of the per-pair Δ.
    # Compute σ_pair time-by-time, then expected_null = σ × √(2/π).
    sigma_pair = D.std(axis=0)
    expected_null = sigma_pair * np.sqrt(2.0 / np.pi)

    # Sign-flip permutation
    rng = np.random.default_rng(42)
    n_perm = 1000
    null_max_signed = np.zeros(n_perm)
    null_max_rms = np.zeros(n_perm)
    for r in range(n_perm):
        s = rng.choice([-1, 1], size=len(D)).astype(np.float32)
        flipped = D * s[:, None]
        null_max_signed[r] = np.max(np.abs(flipped.mean(0)))
        null_max_rms[r] = np.max(np.sqrt((flipped ** 2).mean(0)))
    p_thr_signed = float(np.percentile(null_max_signed, 95))
    sig_mask_signed = np.abs(D.mean(0)) > p_thr_signed

    # ── F06new ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True,
                              gridspec_kw={"height_ratios": [1.3, 1.0, 1.3]})
    fig.suptitle("Within-image attention modulation (hint-swap)\n"
                 f"N = {len(D)} pairs from {len(multi)} images   "
                 "(same stimulus, different attended target)",
                  fontweight="bold")

    ax = axes[0]
    ax.plot(t, mean_abs, color=PALETTE["target"], lw=1.8,
             label="observed mean |Δ|")
    ax.plot(t, expected_null, color="gray", lw=1.2, ls="--",
             label="noise-only null  σ·√(2/π)")
    diff_over_null = mean_abs - expected_null
    # shade where observed exceeds null by ≥ 5%
    sig = diff_over_null > 0.05 * expected_null
    ymin, ymax = ax.get_ylim()
    ax.fill_between(t, ymin, ymax, where=sig & (t > 0),
                     color=PALETTE["target"], alpha=0.08, lw=0,
                     label="|Δ| > null (post-stim)")
    ax.set_ylim(ymin, ymax)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_ylabel("|Δ ERP| (µV)")
    ax.set_title("Observed |Δ| vs noise-only null — attention adds variance "
                  "where observed > null")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.15)

    ax = axes[1]
    ax.plot(t, D.mean(0), color=PALETTE["nontarget"], lw=1.4,
             label="signed mean Δ (random pair order)")
    ax.axhline(+p_thr_signed, color="red", ls=":", lw=0.8,
                label=f"perm 95% = ±{p_thr_signed:.2f}")
    ax.axhline(-p_thr_signed, color="red", ls=":", lw=0.8)
    ymin, ymax = ax.get_ylim()
    ax.fill_between(t, ymin, ymax, where=sig_mask_signed,
                     color="red", alpha=0.10, lw=0,
                     label="perm-significant")
    ax.set_ylim(ymin, ymax)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_ylabel("Δ µV (signed)")
    ax.set_title("Signed mean ≈ 0 by construction; spikes mark moments where "
                  "hint-ordering is non-random")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.15)

    # Show |Δ| / null ratio
    ax = axes[2]
    safe_null = np.maximum(expected_null, 0.01)
    ratio = mean_abs / safe_null
    ax.plot(t, ratio, color=PALETTE["diff"], lw=1.6,
             label="|Δ| / σ·√(2/π)")
    ax.axhline(1.0, color="black", ls="--", lw=0.8,
                label="null = 1")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Attention-modulation index")
    ax.set_title("Observed/null ratio — values > 1 indicate genuine "
                  "attention-driven variance")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    save_fig(fig, out_dir / "F06new_within_image_attention.png")
    log.info("Saved F06new")

    peak_idx = int(np.argmax(diff_over_null * (t > 0)))
    return {
        "n_multi_hint_images": int(len(multi)),
        "n_pairs": int(len(D)),
        "min_trials_per_cell": min_trials_per_cell,
        "peak_t_ms": float(t[peak_idx]),
        "peak_abs_uv": float(mean_abs[peak_idx]),
        "peak_null_uv": float(expected_null[peak_idx]),
        "peak_modulation_index": float(ratio[peak_idx]),
        "perm_95_signed_uv": float(p_thr_signed),
    }


def plot_within_image_decoding(b: EpochBundle, out_dir: Path,
                                  win_ms=(0, 450),
                                  min_trials_per_cell: int = 4,
                                  n_folds: int = 3,
                                  seed: int = 42) -> dict:
    """Per-image hint-decoding — gold-standard attention readout.

    For every multi-hint image we fit an LDA to predict the hint label
    from EEG features computed on the [win_ms] window. The classifier
    sees ONLY trials of that image. We pool single-trial CV predictions
    across all images and report:
        – pooled accuracy (single-trial)
        – per-image accuracy distribution (hist)
        – relative-to-image-chance distribution (1/n_hints)
    """
    chs = _eeg_chs(b)
    t = b.times
    t0 = int((win_ms[0] / 1000.0 - t[0]) * b.sfreq)
    t1 = int((win_ms[1] / 1000.0 - t[0]) * b.sfreq)
    X_full = b.data[:, chs, t0:t1]

    rng = np.random.default_rng(seed)

    by_image = defaultdict(list)
    for i in range(len(b)):
        if b.hint[i] == "":
            continue
        by_image[int(b.image_id[i])].append(i)

    per_image = []
    pooled_correct, pooled_total = 0, 0
    for iid, idxs in by_image.items():
        labels = [b.hint[i] for i in idxs]
        uniq, cnt = np.unique(labels, return_counts=True)
        if len(uniq) < 2:
            continue
        keep = [u for u, c in zip(uniq, cnt) if c >= min_trials_per_cell]
        if len(keep) < 2:
            continue
        sel = [j for j, l in enumerate(labels) if l in keep]
        if len(sel) < 2 * min_trials_per_cell:
            continue
        X = X_full[[idxs[j] for j in sel]]
        y_raw = np.array([labels[j] for j in sel])
        y = LabelEncoder().fit(y_raw).transform(y_raw)
        # Stratified CV inside this image
        n_per_class = np.bincount(y).min()
        folds = min(n_folds, n_per_class)
        if folds < 2:
            continue
        kf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        accs = []
        correct = 0; total = 0
        for tr, va in kf.split(X, y):
            Xf_tr = _features_batch(X[tr], b.sfreq)
            Xf_va = _features_batch(X[va], b.sfreq)
            Xf_tr = np.nan_to_num(Xf_tr, nan=0, posinf=0, neginf=0)
            Xf_va = np.nan_to_num(Xf_va, nan=0, posinf=0, neginf=0)
            clf = make_pipeline(StandardScaler(),
                                  LinearDiscriminantAnalysis(solver="lsqr",
                                                              shrinkage="auto"))
            clf.fit(Xf_tr, y[tr])
            p = clf.predict(Xf_va)
            accs.append(accuracy_score(y[va], p))
            correct += (p == y[va]).sum(); total += len(va)
        per_image.append({
            "image_id": iid,
            "n_hints": int(len(keep)),
            "n_trials": int(len(sel)),
            "chance": 1.0 / len(keep),
            "acc": float(np.mean(accs)),
            "acc_minus_chance": float(np.mean(accs) - 1.0 / len(keep)),
        })
        pooled_correct += correct; pooled_total += total

    log.info("within-image decoding: %d images decoded, %d total predictions",
              len(per_image), pooled_total)

    accs = np.array([r["acc"] for r in per_image])
    chances = np.array([r["chance"] for r in per_image])
    delta = accs - chances

    # ── F21: within-image hint decoding accuracy ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    fig.suptitle(f"Within-image hint decoding  ({pooled_total} predictions "
                 f"from {len(per_image)} images, win {win_ms[0]}-{win_ms[1]} ms)",
                  fontweight="bold")

    ax = axes[0]
    ax.hist(accs, bins=20, color="steelblue", alpha=0.85, edgecolor="white",
             lw=0.4, label="per-image acc")
    ax.axvline(np.median(chances), color="black", ls="--", lw=0.8,
                label=f"median chance = {np.median(chances):.3f}")
    ax.axvline(accs.mean(), color=PALETTE["target"], lw=1.4,
                label=f"mean acc = {accs.mean():.3f}")
    ax.set_xlabel("Per-image hint accuracy")
    ax.set_ylabel("# images")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.15)
    ax.set_title("Distribution across images")

    ax = axes[1]
    ax.hist(delta, bins=20, color=PALETTE["diff"], alpha=0.85,
             edgecolor="white", lw=0.4)
    ax.axvline(0, color="black", ls="--", lw=0.8, label="chance")
    ax.axvline(delta.mean(), color=PALETTE["target"], lw=1.4,
                label=f"mean Δ = {delta.mean():+.3f}")
    # one-sample t-test vs 0
    from scipy.stats import ttest_1samp
    tstat, pval = ttest_1samp(delta, 0)
    ax.text(0.02, 0.98,
             f"t({len(delta)-1}) = {tstat:.2f}\np = {pval:.2e}",
             transform=ax.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8, lw=0))
    ax.set_xlabel("acc − chance")
    ax.set_ylabel("# images")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.15)
    ax.set_title("Above-chance distribution")

    plt.tight_layout()
    save_fig(fig, out_dir / "F21_within_image_decoding.png")
    log.info("Saved F21")

    return {
        "n_images_decoded": int(len(per_image)),
        "n_predictions_pooled": int(pooled_total),
        "pooled_accuracy": float(pooled_correct / max(pooled_total, 1)),
        "median_chance": float(np.median(chances)),
        "mean_acc": float(accs.mean()),
        "mean_acc_minus_chance": float(delta.mean()),
        "frac_above_chance": float((delta > 0).mean()),
        "ttest_t": float(tstat),
        "ttest_p": float(pval),
    }
