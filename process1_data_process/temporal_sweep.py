"""Temporal sensitivity sweep — sliding-window LDA on the HINT label.

Vectorised feature extraction keeps one window over 6376 epochs at ~3s.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis, skew
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              top_k_accuracy_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from process1_data_process.data_io import EpochBundle
from process1_data_process.plot_style import (
    PALETTE, apply_publication_style, save_fig,
)

log = logging.getLogger(__name__)
apply_publication_style()


def _features_batch(X, sfreq):
    """X: (n, ch, t)  → (n, 13*ch) features (vectorised over epochs)."""
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    nyq = sfreq / 2
    n, ch, t = X.shape
    feats = []
    for lo, hi in bands:
        if hi >= nyq:
            hi = nyq - 1
        sos = sig.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        filt = sig.sosfiltfilt(sos, X, axis=2)
        bp = np.log1p(np.mean(filt ** 2, axis=2))
        feats.append(bp)
    feats.append(X.mean(axis=2))
    feats.append(X.var(axis=2))
    feats.append(skew(X, axis=2))
    feats.append(kurtosis(X, axis=2))
    q = t // 4
    feats.append(X[..., :q].mean(2))
    feats.append(X[..., q:2*q].mean(2))
    feats.append(X[..., 2*q:3*q].mean(2))
    feats.append(X[..., 3*q:].mean(2))
    return np.concatenate(feats, axis=1)


def temporal_sweep(b: EpochBundle, *, label: str = "hint",
                    t_len_ms: int = 200, t_step_ms: int = 50,
                    n_folds: int = 5, seed: int = 42,
                    lda_solver: str = "lsqr",
                    lda_shrinkage="auto") -> list[dict]:
    chs = [i for i, n in enumerate(b.ch_names) if n not in ("gaze_x", "gaze_y")]
    X_full = b.data[:, chs, :]
    sf = b.sfreq
    t = b.times
    y_raw = getattr(b, label)
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    n_classes = len(le.classes_)

    t_len = int(t_len_ms * sf / 1000)
    t_step = int(t_step_ms * sf / 1000)
    n_t = X_full.shape[2]
    starts = list(range(0, n_t - t_len + 1, t_step))
    log.info("Temporal sweep on label=%s, %d classes, %d windows",
             label, n_classes, len(starts))

    kf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    results = []
    for idx, t0 in enumerate(starts):
        t_s = time.time()
        X = _features_batch(X_full[:, :, t0:t0 + t_len], sf)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        accs, baccs, top5s = [], [], []
        for tr, va in kf.split(X, y):
            if lda_solver == "lsqr":
                clf = make_pipeline(
                    StandardScaler(),
                    LinearDiscriminantAnalysis(solver="lsqr",
                                                shrinkage=lda_shrinkage))
            else:
                clf = make_pipeline(StandardScaler(),
                                      LinearDiscriminantAnalysis())
            clf.fit(X[tr], y[tr])
            p = clf.predict(X[va])
            accs.append(accuracy_score(y[va], p))
            baccs.append(balanced_accuracy_score(y[va], p))
            try:
                pr = clf.predict_proba(X[va])
                top5s.append(top_k_accuracy_score(
                    y[va], pr, k=min(5, n_classes),
                    labels=np.arange(n_classes)))
            except Exception:
                try:
                    df = clf.decision_function(X[va])
                    top5s.append(top_k_accuracy_score(
                        y[va], df, k=min(5, n_classes),
                        labels=np.arange(n_classes)))
                except Exception:
                    top5s.append(accs[-1])
        cmid_ms = ((t0 + t_len / 2) / sf + t[0]) * 1000
        cstart = (t0 / sf + t[0]) * 1000
        cend = ((t0 + t_len) / sf + t[0]) * 1000
        results.append({
            "window_center_ms": float(cmid_ms),
            "window_start_ms": float(cstart),
            "window_end_ms": float(cend),
            "acc": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "bacc": float(np.mean(baccs)),
            "top5": float(np.mean(top5s)),
        })
        log.info("  [%2d/%d] [%+.0f..%+.0fms]  acc=%.4f bacc=%.4f top5=%.4f (%.1fs)",
                 idx + 1, len(starts), cstart, cend, results[-1]["acc"],
                 results[-1]["bacc"], results[-1]["top5"], time.time() - t_s)
    return results


def plot_sweep(results, label, n_classes, out_path):
    t = np.array([r["window_center_ms"] for r in results])
    acc = np.array([r["acc"] for r in results])
    acc_std = np.array([r["acc_std"] for r in results])
    bacc = np.array([r["bacc"] for r in results])
    top5 = np.array([r["top5"] for r in results])
    chance = 1.0 / n_classes
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.fill_between(t, acc - acc_std, acc + acc_std,
                     color="#4CAF50", alpha=0.18, lw=0)
    ax.plot(t, top5, "o-", color="#2196F3", lw=1.6, ms=4, label="Top-5")
    ax.plot(t, acc, "D-", color="#4CAF50", lw=1.6, ms=4, label="Top-1")
    ax.plot(t, bacc, "s-", color="#FF5722", lw=1.6, ms=4, label="Balanced")
    ax.axhline(chance, color="black", ls=":", lw=0.8,
                label="chance ({:.3f})".format(chance))
    ax.axvline(0, color="red", ls="--", lw=0.8, alpha=0.7, label="stim onset")
    ax.set_xlabel("Window centre (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Temporal sweep  label={}  ({} classes, LDA)".format(label, n_classes),
                  fontweight="bold")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    save_fig(fig, out_path)
    log.info("Saved %s", out_path)
