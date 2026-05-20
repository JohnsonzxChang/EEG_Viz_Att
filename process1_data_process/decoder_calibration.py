"""Decoder calibration plots: HINT confusion matrix + ERP-averaging effect.

These are the kind of plots ATM/EEG-CLIP papers use as a "linear-readout
sanity check" of the dataset, providing a chance/ceiling reference for
the downstream deep-learning experiments in process2/.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis, skew
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              confusion_matrix, top_k_accuracy_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from process1_data_process.data_io import EpochBundle
from process1_data_process.plot_style import (
    PALETTE, apply_publication_style, save_fig,
)

log = logging.getLogger(__name__)
apply_publication_style()


# ────────── reusable feature extractor ──────────────────────────────

def _features_batch(X, sfreq):
    """Vectorised features over all epochs simultaneously.

    X: (n, ch, t)  → (n, 13*ch) feature matrix.
    """
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    nyq = sfreq / 2
    n, ch, t = X.shape
    feats: list = []
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


def _eeg_chs(b):
    return [i for i, n in enumerate(b.ch_names) if n not in ("gaze_x", "gaze_y")]


# ────────── HINT confusion matrix at post-stim window ──────────────

def plot_hint_confusion(b: EpochBundle, out_dir: Path,
                          win_ms=(0, 500), n_folds: int = 5,
                          seed: int = 42) -> dict:
    """5-fold LDA on HINT label using single-trial features.

    The data here is highly imbalanced (40 classes, ~150 trials each);
    we report top-1 / top-5 / balanced accuracy and plot the row-normalised
    confusion matrix.
    """
    chs = _eeg_chs(b)
    sel = b.hint != ""
    X_full = b.data[sel][:, chs, :]
    sf = b.sfreq
    t = b.times
    t0 = int((win_ms[0] / 1000 - t[0]) * sf)
    t1 = int((win_ms[1] / 1000 - t[0]) * sf)

    y_raw = b.hint[sel]
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    n_classes = len(le.classes_)
    log.info("HINT decoding: N=%d epochs, %d classes, window=%s ms",
             len(y), n_classes, win_ms)

    log.info("Extracting features ...")
    t_s = time.time()
    Xfeat = _features_batch(X_full[:, :, t0:t1], sf)
    Xfeat = np.nan_to_num(Xfeat, nan=0, posinf=0, neginf=0)
    log.info("Features %s in %.1fs", Xfeat.shape, time.time() - t_s)

    kf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    preds = np.zeros_like(y)
    proba = np.zeros((len(y), n_classes))
    for tr, va in kf.split(Xfeat, y):
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        clf.fit(Xfeat[tr], y[tr])
        preds[va] = clf.predict(Xfeat[va])
        try:
            proba[va] = clf.predict_proba(Xfeat[va])
        except Exception:
            pass

    acc = accuracy_score(y, preds)
    bacc = balanced_accuracy_score(y, preds)
    try:
        top5 = top_k_accuracy_score(y, proba, k=min(5, n_classes),
                                      labels=np.arange(n_classes))
    except Exception:
        top5 = acc
    cm = confusion_matrix(y, preds, labels=np.arange(n_classes))
    cm_n = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    # plot row-normalised confusion matrix
    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0,
                    vmax=max(0.15, float(np.percentile(cm_n, 99))))
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(le.classes_, rotation=80, ha="right", fontsize=5)
    ax.set_yticklabels(le.classes_, fontsize=5)
    cbar = fig.colorbar(im, ax=ax, label="P(predicted | true)",
                         shrink=0.7, pad=0.02)
    cbar.outline.set_linewidth(0)
    ax.set_xlabel("Predicted HINT")
    ax.set_ylabel("True HINT")
    ax.set_title(f"HINT confusion ({n_classes} classes, LDA 5-fold)\n"
                 f"Top-1={acc:.3f}  Top-5={top5:.3f}  BAcc={bacc:.3f}  "
                 f"chance={1/n_classes:.3f}",
                  fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir / "F15_hint_confusion.png")
    log.info("Saved HINT confusion matrix")

    return {"top1": float(acc), "top5": float(top5), "balanced": float(bacc),
            "chance": float(1 / n_classes), "n_classes": n_classes,
            "n_epochs": int(len(y))}


# ────────── ERP averaging effect (k=1, 2, 4, …, all) ────────────────

def plot_erp_averaging_effect(b: EpochBundle, out_dir: Path,
                                ks=(1, 2, 4, 8, 16, "all"),
                                win_ms=(0, 500), n_folds: int = 5,
                                seed: int = 42) -> list[dict]:
    """Average k trials per (image_id, HINT) group, retrain decoder, plot
    accuracy as a function of k."""
    chs = _eeg_chs(b)
    sel = b.hint != ""
    X = b.data[sel][:, chs, :]
    y_raw = b.hint[sel]
    img = b.image_id[sel]
    sf = b.sfreq
    t = b.times
    t0 = int((win_ms[0] / 1000 - t[0]) * sf)
    t1 = int((win_ms[1] / 1000 - t[0]) * sf)

    # group by (image, hint)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(y_raw)):
        groups[(int(img[i]), y_raw[i])].append(i)
    log.info("ERP-avg: %d unique (image,hint) groups", len(groups))

    le = LabelEncoder().fit(y_raw)
    n_classes = len(le.classes_)

    rng = np.random.default_rng(seed)
    results = []
    for k in ks:
        avg_X, avg_y = [], []
        for (_, h), idxs in groups.items():
            if k == "all":
                use = idxs
            else:
                use = idxs if len(idxs) <= k else rng.choice(idxs, k, replace=False).tolist()
            if not use:
                continue
            avg_X.append(X[use].mean(axis=0))
            avg_y.append(h)
        avg_X = np.stack(avg_X)
        avg_y = le.transform(np.array(avg_y))
        if len(np.unique(avg_y)) < 2:
            continue
        Xf = _features_batch(avg_X[:, :, t0:t1], sf)
        Xf = np.nan_to_num(Xf, nan=0, posinf=0, neginf=0)
        kf = StratifiedKFold(min(n_folds, np.bincount(avg_y).min()),
                              shuffle=True, random_state=seed)
        accs, baccs, top5s = [], [], []
        for tr, va in kf.split(Xf, avg_y):
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            clf.fit(Xf[tr], avg_y[tr])
            p = clf.predict(Xf[va])
            accs.append(accuracy_score(avg_y[va], p))
            baccs.append(balanced_accuracy_score(avg_y[va], p))
            try:
                pr = clf.predict_proba(Xf[va])
                top5s.append(top_k_accuracy_score(
                    avg_y[va], pr, k=min(5, n_classes),
                    labels=np.arange(n_classes)))
            except Exception:
                top5s.append(accs[-1])
        res = {"k": k, "n_samples": int(len(avg_X)),
               "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
               "bacc": float(np.mean(baccs)),
               "top5": float(np.mean(top5s))}
        results.append(res)
        log.info("k=%-3s N=%-5d acc=%.3f bacc=%.3f top5=%.3f",
                 str(k), res["n_samples"], res["acc"], res["bacc"], res["top5"])

    # plot
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(results))
    labels = [f"k={r['k']}\n(N={r['n_samples']})" for r in results]
    ax.bar(x - 0.25, [r["top5"] for r in results], 0.25,
            color="#2196F3", alpha=0.85, label="Top-5")
    ax.bar(x, [r["acc"] for r in results], 0.25,
            yerr=[r["acc_std"] for r in results], capsize=3,
            color="#4CAF50", alpha=0.85, label="Top-1")
    ax.bar(x + 0.25, [r["bacc"] for r in results], 0.25,
            color="#FF5722", alpha=0.85, label="Balanced")
    ax.axhline(1.0 / n_classes, color="black", ls=":", lw=0.8,
                label=f"chance ({1/n_classes:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"ERP averaging effect — HINT decoding "
                 f"({n_classes} classes, LDA 5-fold)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.15, axis="y")
    # value labels
    for i, r in enumerate(results):
        ax.text(i, r["acc"] + 0.01, f"{r['acc']:.2f}", ha="center",
                 va="bottom", fontsize=6)
    plt.tight_layout()
    save_fig(fig, out_dir / "F16_erp_averaging_effect.png")
    log.info("Saved ERP averaging effect")
    return results
