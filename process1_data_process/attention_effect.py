"""Attention-modulation analyses beyond the basic target vs non-target ERP.

  • time_frequency_alpha     — pre vs post alpha-band power desync
  • target_area_regression   — ERP amplitude vs target_area_frac
  • per_target_count_erp     — ERP grouped by K=# targets in image
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy import stats as ss

from process1_data_process.data_io import EpochBundle
from process1_data_process.plot_style import (
    PALETTE, apply_publication_style, save_fig,
)

log = logging.getLogger(__name__)
apply_publication_style()


def _eeg_chs(b: EpochBundle) -> list[int]:
    return [i for i, n in enumerate(b.ch_names) if n not in ("gaze_x", "gaze_y")]


# ────────── Time-frequency: alpha desync (8-13Hz) ─────────────────

def plot_alpha_desync(b: EpochBundle, out_dir: Path,
                       fmin: float = 8.0, fmax: float = 13.0,
                       chunk: int = 512) -> dict:
    """Per-trial alpha band envelope (rectified analytic signal),
    averaged across channels, salient vs non-salient.

    Chunked Hilbert keeps peak RAM bounded (~chunk * ch * t * 16 bytes).
    """
    chs = _eeg_chs(b)
    X = b.data[:, chs, :]
    sf = b.sfreq
    nyq = sf / 2
    sos = sig.butter(4, [fmin / nyq, fmax / nyq], btype="band", output="sos")
    # Process in chunks
    n, _, t_n = X.shape
    env_m = np.empty((n, t_n), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        band = sig.sosfiltfilt(sos, X[s:e].astype(np.float32), axis=2)
        env = np.abs(sig.hilbert(band, axis=2)).astype(np.float32)
        env_m[s:e] = env.mean(axis=1)
        del band, env

    t = b.times * 1000
    # split by salient vs non-salient (median of attended-object area)
    area = np.array([(b.target_areas[i].get(b.hint[i], np.nan)
                       if i < len(b.target_areas) else np.nan)
                      for i in range(len(b))], dtype=float)
    have = (b.hint != "") & np.isfinite(area)
    med = np.nanmedian(area[have])
    sel_t = have & (area >= med)
    sel_n = have & (area <  med)

    e_t = env_m[sel_t]
    e_n = env_m[sel_n]

    # normalise to pre-stim baseline (-500..0 ms)
    pre = (t < 0)
    base_t = e_t[:, pre].mean(axis=1, keepdims=True)
    base_n = e_n[:, pre].mean(axis=1, keepdims=True)
    e_t = e_t / np.maximum(base_t, 1e-6)
    e_n = e_n / np.maximum(base_n, 1e-6)

    mt, mn = e_t.mean(0), e_n.mean(0)
    st = e_t.std(0) / np.sqrt(len(e_t))
    sn = e_n.std(0) / np.sqrt(len(e_n))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(t, mt, color=PALETTE["target"], lw=1.6,
            label=f"Salient (n={sel_t.sum()})")
    ax.fill_between(t, mt - 1.96 * st, mt + 1.96 * st,
                     color=PALETTE["target"], alpha=0.18, lw=0)
    ax.plot(t, mn, color=PALETTE["nontarget"], lw=1.6,
            label=f"Non-salient (n={sel_n.sum()})")
    ax.fill_between(t, mn - 1.96 * sn, mn + 1.96 * sn,
                     color=PALETTE["nontarget"], alpha=0.18, lw=0)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.axhline(1, color="gray", alpha=0.4, lw=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("α-band envelope / baseline")
    ax.set_title(f"α-band ({fmin:.0f}-{fmax:.0f} Hz) desync — salient vs non-salient",
                 fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    save_fig(fig, out_dir / "F12_alpha_desync.png")
    log.info("Saved alpha desync")

    # quantify post-stim suppression
    post = (t >= 100) & (t <= 500)
    return {
        "alpha_post_salient":    float(mt[post].mean()),
        "alpha_post_nonsalient": float(mn[post].mean()),
        "alpha_desync_salient":    float(1 - mt[post].mean()),
        "alpha_desync_nonsalient": float(1 - mn[post].mean()),
        "median_split_area": float(med),
    }


# ────────── target_area-driven amplitude regression ───────────────

def plot_target_area_regression(b: EpochBundle, out_dir: Path,
                                  win_ms=(200, 500)) -> dict:
    """For each target trial, take the area_frac of the attended object
    and regress against post-stim mean amplitude / GFP."""
    chs = _eeg_chs(b)
    t = b.times * 1000
    win = (t >= win_ms[0]) & (t <= win_ms[1])
    X = b.data[:, chs, :][:, :, win]                   # (n, ch, t')

    sel_t = b.is_target & (b.hint != "")
    if sel_t.sum() == 0:
        log.warning("No target trials with hint — skipping target_area plot")
        return {}

    # amplitude proxy = mean absolute amplitude in window
    amp = np.abs(X).mean(axis=(1, 2))                  # (n,)
    gfp = X.std(axis=1).mean(axis=1)                   # (n,)

    area = []
    for i in range(len(b)):
        h = b.hint[i]
        d = b.target_areas[i] if i < len(b.target_areas) else {}
        area.append(d.get(h, np.nan))
    area = np.array(area, dtype=float)

    keep = sel_t & np.isfinite(area)
    A, AMP, GFP = area[keep], amp[keep], gfp[keep]
    log.info("target_area regression: n=%d", keep.sum())

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, y, lbl in [(axes[0], AMP, "Mean |amplitude| (µV)"),
                       (axes[1], GFP, "GFP (µV)")]:
        ax.scatter(A, y, s=8, alpha=0.35, color="steelblue",
                    edgecolors="none")
        # binned mean ± sem
        bins = np.linspace(0, max(0.3, A.max()), 11)
        bc = 0.5 * (bins[:-1] + bins[1:])
        ym, ys = [], []
        for k in range(len(bins) - 1):
            inb = (A >= bins[k]) & (A < bins[k + 1])
            if inb.sum() > 5:
                ym.append(y[inb].mean()); ys.append(y[inb].std() / np.sqrt(inb.sum()))
            else:
                ym.append(np.nan); ys.append(np.nan)
        ym, ys = np.array(ym), np.array(ys)
        ok = np.isfinite(ym)
        ax.errorbar(bc[ok], ym[ok], yerr=ys[ok], color=PALETTE["target"],
                     lw=1.4, capsize=2, marker="o", ms=4,
                     label="binned mean ± SEM")
        # linear fit
        m = np.isfinite(A) & np.isfinite(y)
        if m.sum() > 3:
            r, p = ss.pearsonr(A[m], y[m])
            slope, intercept = np.polyfit(A[m], y[m], 1)
            xx = np.linspace(A.min(), A.max(), 50)
            ax.plot(xx, slope * xx + intercept, color="black", ls="--", lw=0.9,
                     label=f"r={r:+.3f}  p={p:.2e}")
        ax.set_xlabel("Attended-object area fraction")
        ax.set_ylabel(lbl)
        ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.15)
    fig.suptitle(f"Target-area regression — window {win_ms[0]}-{win_ms[1]} ms",
                  fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir / "F13_target_area_regression.png")
    log.info("Saved target_area regression")

    r_amp, p_amp = ss.pearsonr(A[np.isfinite(A)], AMP[np.isfinite(A)]) \
        if np.isfinite(A).any() else (np.nan, np.nan)
    return {"n_trials": int(keep.sum()),
            "r_area_vs_amp": float(r_amp),
            "p_area_vs_amp": float(p_amp)}


# ────────── per-K (# targets) ERP comparison ──────────────────────

def plot_per_target_count_erp(b: EpochBundle, out_dir: Path) -> None:
    """ERP grouped by K — the number of target categories in an image.

    The RSVP attention paradigm shows images containing K targets; this
    isolates the within-image distractor load on the ERP."""
    chs = _eeg_chs(b)
    t = b.times * 1000
    Ks = np.array([len(tl) for tl in b.targets_in_image])
    uniq = sorted(set(Ks.tolist()))
    uniq = [k for k in uniq if (Ks == k).sum() >= 100]
    if not uniq:
        log.warning("No K-groups with ≥100 trials; skipping")
        return

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(uniq)))
    for c, k in zip(cmap, uniq):
        sel = Ks == k
        erp = b.data[sel][:, chs, :].mean(axis=(0, 1))
        ax.plot(t, erp, lw=1.3, color=c, label=f"K={k} (n={sel.sum()})")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.axhline(0, color="gray", alpha=0.3, lw=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("µV")
    ax.set_title("ERP by # targets in image (distractor load)",
                  fontweight="bold")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    save_fig(fig, out_dir / "F14_per_target_count_erp.png")
    log.info("Saved per-K ERP")
