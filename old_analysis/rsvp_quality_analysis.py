#!/usr/bin/env python3
"""
RSVP-COCO Data Quality Analysis & Classification Calibration

Tasks:
  1. ERP Diagnostic: Grand average, per-class ERP, GFP, butterfly plot
  2. Temporal Sensitivity Sweep: sliding window LDA classification
  3. ERP Averaging Effect: classification vs number of averaged trials
  4. 12-class classification accuracy calibration

Data: zxc-rsvp-coco-pos-0324/epochs_big-epo.fif
  - 6093 epochs, 32 EEG channels, tmin=-0.1s, tmax=+0.6s, 1000Hz
  - 360 unique events = 12 categories x 30 images x 10-20 repeats
  - Marker format: "image_id/category"
"""

import os
import gc
import json
import time
import argparse
import warnings
import numpy as np
from collections import defaultdict
from scipy import signal as sig
from scipy.stats import kurtosis, skew
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, top_k_accuracy_score)

warnings.filterwarnings('ignore')

TARGET_CATEGORIES = [
    "dog", "cat", "car", "chair", "banana", "pizza",
    "cup", "couch", "bed", "laptop", "teddy bear", "umbrella",
]
N_CLASSES = len(TARGET_CATEGORIES)


# ═══ Feature Extraction ═════════════════════════════════════════════════════

def extract_bandpower(x, sfreq=1000.0):
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
             'beta': (13, 30), 'low_gamma': (30, 45)}
    feats = []
    for bname, (flo, fhi) in bands.items():
        nyq = sfreq / 2
        if fhi >= nyq: fhi = nyq - 1
        if flo >= fhi: continue
        sos = sig.butter(4, [flo/nyq, fhi/nyq], btype='band', output='sos')
        filtered = sig.sosfiltfilt(sos, x, axis=1)
        bp = np.log1p(np.mean(filtered ** 2, axis=1))
        feats.append(bp)
    return np.concatenate(feats)


def extract_temporal_stats(x):
    m = np.mean(x, axis=1)
    v = np.var(x, axis=1)
    s = skew(x, axis=1)
    k = kurtosis(x, axis=1)
    return np.concatenate([m, v, s, k])


def extract_covariance(x):
    cov = np.cov(x)
    diag = np.log1p(np.abs(np.diag(cov)))
    iu = np.triu_indices(cov.shape[0], k=1)
    upper = cov[iu]
    return np.concatenate([diag, upper])


def extract_erp_components(x):
    n_t = x.shape[1]
    q1 = n_t // 4
    q2 = n_t // 2
    q3 = 3 * n_t // 4
    p1 = np.mean(x[:, :q1], axis=1)
    p2 = np.mean(x[:, q1:q2], axis=1)
    p3 = np.mean(x[:, q2:q3], axis=1)
    p4 = np.mean(x[:, q3:], axis=1)
    return np.concatenate([p1, p2, p3, p4])


def extract_all_features(x, sfreq=1000.0):
    feats = []
    feats.append(extract_bandpower(x, sfreq))
    feats.append(extract_temporal_stats(x))
    feats.append(extract_covariance(x))
    feats.append(extract_erp_components(x))
    return np.concatenate(feats)


# ═══ Data Loading ═══════════════════════════════════════════════════════════

def load_rsvp_data(fif_path):
    """Load RSVP-COCO epochs. Parse markers into image_id and category."""
    import mne
    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    print(f"Loaded: {len(epochs)} epochs, {len(epochs.ch_names)} channels, "
          f"tmin={epochs.tmin}s, tmax={epochs.tmax}s, sfreq={epochs.info['sfreq']}Hz")

    data = epochs.get_data().astype(np.float32) * 1e6  # uV
    events = epochs.events[:, 2]
    code_to_name = {code: name for name, code in epochs.event_id.items()}
    sfreq = epochs.info['sfreq']
    times = epochs.times

    # Parse markers
    img_ids = []
    categories = []
    for i in range(len(data)):
        code = events[i]
        name = code_to_name[code]
        parts = name.split('/')
        img_id = int(parts[0])
        cat = parts[1]
        img_ids.append(img_id)
        categories.append(cat)

    img_ids = np.array(img_ids)
    categories = np.array(categories)

    # Encode categories
    le = LabelEncoder()
    le.classes_ = np.array(TARGET_CATEGORIES)
    labels = le.transform(categories)

    # Group by image_id + category for ERP averaging
    groups = defaultdict(list)
    for i in range(len(data)):
        key = (img_ids[i], categories[i])
        groups[key].append(i)

    print(f"\nData shape: {data.shape}")
    print(f"Categories: {TARGET_CATEGORIES}")
    cat_counts = {cat: (categories == cat).sum() for cat in TARGET_CATEGORIES}
    print(f"Trials per category: {cat_counts}")
    print(f"Unique images: {len(groups)}")
    n_per_img = [len(v) for v in groups.values()]
    print(f"Trials per image: min={min(n_per_img)}, max={max(n_per_img)}, "
          f"mean={np.mean(n_per_img):.1f}")

    del epochs
    gc.collect()
    return data, labels, categories, img_ids, groups, times, sfreq


# ═══ ERP Averaging ══════════════════════════════════════════════════════════

def erp_average_data(data, labels, groups, k=None):
    """Average k trials per image. If k=None, use all trials."""
    avg_data = []
    avg_labels = []

    for (img_id, cat), indices in groups.items():
        trials = data[indices]  # (n_trials, n_ch, n_times)
        label = labels[indices[0]]

        if k is not None and k < len(indices):
            # Random subset of k trials
            sel = np.random.choice(len(indices), size=k, replace=False)
            trials = trials[sel]

        erp = trials.mean(axis=0)
        avg_data.append(erp)
        avg_labels.append(label)

    return np.stack(avg_data), np.array(avg_labels)


# ═══ Analysis 1: ERP Diagnostic ═════════════════════════════════════════════

def run_erp_diagnostic(data, labels, categories, times, sfreq, save_dir):
    """Grand average ERP, per-class ERP, GFP analysis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_ch = data.shape[1]

    # Grand average
    grand_avg = data.mean(axis=0)  # (ch, times)
    gfp = np.std(grand_avg, axis=0)

    # Per-class averages
    class_avgs = {}
    for cat in TARGET_CATEGORIES:
        mask = categories == cat
        class_avgs[cat] = data[mask].mean(axis=0)

    # ── Figure 1: Grand Average + GFP ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('RSVP-COCO ERP Diagnostic\n'
                 f'{len(data)} trials, 32 EEG channels, 12 categories',
                 fontsize=14, fontweight='bold')

    # Butterfly + GFP
    ax = axes[0]
    for ch in range(n_ch):
        ax.plot(times * 1000, grand_avg[ch], alpha=0.3, lw=0.5, color='steelblue')
    ax.plot(times * 1000, gfp, color='red', lw=2.5, label='GFP')
    ax.axvline(0, color='black', ls='--', lw=1.5, label='Stimulus onset')
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title('Grand Average ERP — Butterfly Plot + GFP (red)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    # GFP zoom with ERP components
    ax = axes[1]
    ax.plot(times * 1000, gfp, color='red', lw=2.5)
    ax.axvline(0, color='black', ls='--', lw=1.5, label='Stimulus onset')
    ax.axvline(100, color='blue', ls=':', alpha=0.6, label='P100 (~100ms)')
    ax.axvline(170, color='green', ls=':', alpha=0.6, label='N170 (~170ms)')
    ax.axvline(300, color='orange', ls=':', alpha=0.6, label='P300 (~300ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('GFP (uV)')
    ax.set_title('Global Field Power — Visual ERP Components')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    # Pre vs post GFP stats
    pre_gfp = gfp[times < 0].mean()
    post_gfp = gfp[(times >= 0.05) & (times <= 0.5)].mean()
    peak_t = times[np.argmax(gfp)] * 1000
    ax.text(0.98, 0.95,
            f'GFP pre={pre_gfp:.3f}\nGFP post={post_gfp:.3f}\n'
            f'Ratio={post_gfp/pre_gfp:.2f}\nPeak={peak_t:.0f}ms',
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Per-class ERP (mean across channels)
    ax = axes[2]
    colors = plt.cm.tab20(np.linspace(0, 1, N_CLASSES))
    for idx, cat in enumerate(TARGET_CATEGORIES):
        class_mean = class_avgs[cat].mean(axis=0)
        ax.plot(times * 1000, class_mean, lw=1.5, color=colors[idx],
                label=f'{cat} (n={(categories==cat).sum()})')
    ax.axvline(0, color='black', ls='--', lw=1.5)
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title('Per-Category ERP (mean across channels)')
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    p = os.path.join(save_dir, 'fig', 'rsvp_erp_diagnostic.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close(fig)

    # ── Figure 2: Per-class ERP heatmap ──
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Per-Category Grand Average ERP (all channels)', fontsize=14, fontweight='bold')
    for idx, cat in enumerate(TARGET_CATEGORIES):
        ax = axes[idx // 4, idx % 4]
        im = ax.imshow(class_avgs[cat], aspect='auto',
                       extent=[times[0]*1000, times[-1]*1000, n_ch-0.5, -0.5],
                       cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        ax.axvline(0, color='white', ls='--', lw=1)
        ax.set_title(f'{cat} (n={(categories==cat).sum()})', fontsize=10)
        if idx % 4 == 0:
            ax.set_ylabel('Channel')
        if idx >= 8:
            ax.set_xlabel('Time (ms)')
    plt.colorbar(im, ax=axes, label='Amplitude (uV)', shrink=0.6)
    plt.tight_layout()
    p = os.path.join(save_dir, 'fig', 'rsvp_erp_per_class_heatmap.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close(fig)

    return {
        'gfp_pre': float(pre_gfp),
        'gfp_post': float(post_gfp),
        'gfp_ratio': float(post_gfp / pre_gfp),
        'gfp_peak_ms': float(peak_t),
    }


# ═══ Analysis 2: Temporal Sensitivity Sweep ═════════════════════════════════

def run_temporal_sweep(data, labels, times, sfreq, save_dir,
                       t_len_ms=200, t_step_ms=50, n_folds=5, seed=42):
    """Slide a window across time, classify with LDA at each position."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_len = int(t_len_ms * sfreq / 1000)
    t_step = int(t_step_ms * sfreq / 1000)
    n_times = data.shape[2]

    t0_values = list(range(0, n_times - t_len + 1, t_step))
    print(f"\nTemporal sweep: {len(t0_values)} windows, "
          f"t_len={t_len_ms}ms, step={t_step_ms}ms")

    results = []
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for idx, t0 in enumerate(t0_values):
        t_start = time.time()
        # Extract features
        X = np.array([extract_all_features(data[i, :, t0:t0+t_len], sfreq)
                       for i in range(len(data))])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        fold_metrics = []
        for trn_idx, val_idx in kf.split(X, labels):
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            clf.fit(X[trn_idx], labels[trn_idx])
            y_pred = clf.predict(X[val_idx])
            acc = accuracy_score(labels[val_idx], y_pred)
            bacc = balanced_accuracy_score(labels[val_idx], y_pred)

            # Top-3 accuracy
            try:
                proba = clf.predict_proba(X[val_idx])
                top3 = top_k_accuracy_score(labels[val_idx], proba, k=3)
            except:
                top3 = acc

            fold_metrics.append({'acc': acc, 'bacc': bacc, 'top3': top3})

        avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        avg.update({k+'_std': float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]})

        t_sec_center = (t0 + t_len / 2) / sfreq + times[0]
        t_sec_start = t0 / sfreq + times[0]
        t_sec_end = (t0 + t_len) / sfreq + times[0]

        entry = {'t0_sample': t0, 'window_center_ms': t_sec_center * 1000,
                 'window_start_ms': t_sec_start * 1000,
                 'window_end_ms': t_sec_end * 1000, **avg}
        results.append(entry)

        elapsed = time.time() - t_start
        eta = elapsed * (len(t0_values) - idx - 1) / 60
        print(f"  [{idx+1:2d}/{len(t0_values)}] "
              f"[{t_sec_start*1000:+.0f}ms,{t_sec_end*1000:+.0f}ms] "
              f"Acc={avg['acc']:.4f} BAcc={avg['bacc']:.4f} Top3={avg['top3']:.4f} "
              f"| {elapsed:.1f}s (ETA: {eta:.1f}min)")

    # Plot
    t_ms = [r['window_center_ms'] for r in results]
    acc_vals = [r['acc'] for r in results]
    bacc_vals = [r['bacc'] for r in results]
    top3_vals = [r['top3'] for r in results]
    acc_std = [r['acc_std'] for r in results]
    top3_std = [r['top3_std'] for r in results]
    chance = 1.0 / N_CLASSES

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f'Temporal Sensitivity: RSVP-COCO 12-Class Classification\n'
                 f'LDA, {n_folds}-fold CV, window={t_len_ms}ms, N={len(data)} trials',
                 fontsize=13, fontweight='bold')

    for ax in axes:
        ax.axvline(0, color='red', ls='--', lw=2, alpha=0.8, label='Stimulus onset')
        ax.axvspan(min(t_ms), 0, alpha=0.06, color='gray')
        ax.axvspan(0, max(t_ms), alpha=0.03, color='#c6efce')
        ax.grid(True, alpha=0.2)

    axes[0].errorbar(t_ms, top3_vals, yerr=top3_std, fmt='o-', color='#2196F3',
                     lw=2, ms=5, capsize=3, label='Top-3 Accuracy')
    axes[0].errorbar(t_ms, acc_vals, yerr=acc_std, fmt='D-', color='#4CAF50',
                     lw=2, ms=4, capsize=3, label='Top-1 Accuracy')
    axes[0].axhline(chance, color='gray', ls=':', lw=1.5, alpha=0.6,
                    label=f'Chance ({chance:.3f})')
    axes[0].axhline(3*chance, color='gray', ls=':', lw=1, alpha=0.4,
                    label=f'3x Chance ({3*chance:.3f})')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].set_ylim(0, max(max(top3_vals) + 0.1, 0.5))

    axes[1].errorbar(t_ms, bacc_vals, yerr=[r['bacc_std'] for r in results],
                     fmt='s-', color='#FF5722', lw=2, ms=5, capsize=3,
                     label='Balanced Accuracy')
    axes[1].axhline(chance, color='gray', ls=':', lw=1.5, alpha=0.6, label='Chance')
    axes[1].set_ylabel('Balanced Accuracy')
    axes[1].set_xlabel('Window Center Time (ms, relative to stimulus onset)')
    axes[1].legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    p = os.path.join(save_dir, 'fig', 'rsvp_temporal_sweep.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close(fig)

    return results


# ═══ Analysis 3: ERP Averaging Effect ═══════════════════════════════════════

def run_erp_avg_calibration(data, labels, groups, times, sfreq, save_dir,
                            n_folds=5, seed=42):
    """Test classification accuracy with different numbers of averaged trials."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    k_values = [1, 2, 3, 5, 8, 10, 15, None]  # None = all
    k_labels_str = ['1', '2', '3', '5', '8', '10', '15', 'all']

    # Best time window from sweep: use post-stimulus [0, 500ms]
    onset_idx = int(-times[0] * sfreq)
    t0 = onset_idx  # stimulus onset
    t_len = min(int(0.5 * sfreq), data.shape[2] - t0)

    print(f"\nERP averaging calibration: window=[0, +{t_len}ms] (samples {t0}:{t0+t_len})")
    print(f"K values: {k_labels_str}")

    results = []
    for ki, k in enumerate(k_values):
        t_start = time.time()
        # Average
        np.random.seed(seed)
        avg_data, avg_labels = erp_average_data(data, labels, groups, k=k)

        # Extract features
        X = np.array([extract_all_features(avg_data[i, :, t0:t0+t_len], sfreq)
                       for i in range(len(avg_data))])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_metrics = []
        for trn_idx, val_idx in kf.split(X, avg_labels):
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            clf.fit(X[trn_idx], avg_labels[trn_idx])
            y_pred = clf.predict(X[val_idx])
            acc = accuracy_score(avg_labels[val_idx], y_pred)
            bacc = balanced_accuracy_score(avg_labels[val_idx], y_pred)
            try:
                proba = clf.predict_proba(X[val_idx])
                top3 = top_k_accuracy_score(avg_labels[val_idx], proba, k=3)
            except:
                top3 = acc
            fold_metrics.append({'acc': acc, 'bacc': bacc, 'top3': top3})

        avg_metrics = {mk: float(np.mean([m[mk] for m in fold_metrics])) for mk in fold_metrics[0]}
        avg_metrics.update({mk+'_std': float(np.std([m[mk] for m in fold_metrics])) for mk in fold_metrics[0]})
        avg_metrics['k'] = k if k else 'all'
        avg_metrics['k_label'] = k_labels_str[ki]
        avg_metrics['n_samples'] = len(avg_data)

        elapsed = time.time() - t_start
        print(f"  k={k_labels_str[ki]:>4s}: N={len(avg_data):>4d} samples | "
              f"Acc={avg_metrics['acc']:.4f} BAcc={avg_metrics['bacc']:.4f} "
              f"Top3={avg_metrics['top3']:.4f} | {elapsed:.1f}s")
        results.append(avg_metrics)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(k_values))

    top3 = [r['top3'] for r in results]
    top1 = [r['acc'] for r in results]
    bacc = [r['bacc'] for r in results]
    top3_std = [r['top3_std'] for r in results]
    top1_std = [r['acc_std'] for r in results]

    ax.bar(x_pos - 0.25, top3, 0.25, yerr=top3_std, capsize=4,
           color='#2196F3', alpha=0.8, label='Top-3 Accuracy')
    ax.bar(x_pos, top1, 0.25, yerr=top1_std, capsize=4,
           color='#4CAF50', alpha=0.8, label='Top-1 Accuracy')
    ax.bar(x_pos + 0.25, bacc, 0.25,
           color='#FF5722', alpha=0.8, label='Balanced Accuracy')

    ax.axhline(1/N_CLASSES, color='gray', ls=':', lw=1.5, label=f'Chance ({1/N_CLASSES:.3f})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'k={s}\n(N={r["n_samples"]})' for s, r in zip(k_labels_str, results)])
    ax.set_xlabel('Number of trials averaged per image (k)')
    ax.set_ylabel('Accuracy')
    ax.set_title('ERP Averaging Effect on 12-Class Classification\n'
                 f'LDA, window=[0, +500ms], {n_folds}-fold CV',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(max(top3) + 0.15, 0.5))

    # Add value labels
    for i, (t3, t1) in enumerate(zip(top3, top1)):
        ax.text(i - 0.25, t3 + 0.01, f'{t3:.3f}', ha='center', va='bottom', fontsize=7)
        ax.text(i, t1 + 0.01, f'{t1:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    p = os.path.join(save_dir, 'fig', 'rsvp_erp_averaging_effect.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close(fig)

    return results


# ═══ Analysis 4: Full Classification Report ═════════════════════════════════

def run_full_classification(data, labels, categories, groups, times, sfreq, save_dir,
                            n_folds=5, seed=42):
    """Full 12-class classification with confusion matrix, using all-trial ERP average."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ERP average (all trials)
    avg_data, avg_labels = erp_average_data(data, labels, groups, k=None)

    # Post-stimulus window
    onset_idx = int(-times[0] * sfreq)
    t0 = onset_idx
    t_len = min(int(0.5 * sfreq), avg_data.shape[2] - t0)

    X = np.array([extract_all_features(avg_data[i, :, t0:t0+t_len], sfreq)
                   for i in range(len(avg_data))])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    print(f"\nFull classification: {len(avg_data)} samples, "
          f"{X.shape[1]} features, {N_CLASSES} classes")

    # Cross-validation with confusion matrix
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_preds = np.zeros_like(avg_labels)
    all_proba = np.zeros((len(avg_labels), N_CLASSES))

    for trn_idx, val_idx in kf.split(X, avg_labels):
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        clf.fit(X[trn_idx], avg_labels[trn_idx])
        all_preds[val_idx] = clf.predict(X[val_idx])
        all_proba[val_idx] = clf.predict_proba(X[val_idx])

    acc = accuracy_score(avg_labels, all_preds)
    bacc = balanced_accuracy_score(avg_labels, all_preds)
    top3 = top_k_accuracy_score(avg_labels, all_proba, k=3)
    cm = confusion_matrix(avg_labels, all_preds)

    print(f"  Top-1 Accuracy: {acc:.4f}")
    print(f"  Balanced Accuracy: {bacc:.4f}")
    print(f"  Top-3 Accuracy: {top3:.4f}")
    print(f"  Chance: {1/N_CLASSES:.4f}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(12, 10))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=max(0.3, cm_norm.max()))

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = cm_norm[i, j]
            color = 'white' if val > 0.15 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(TARGET_CATEGORIES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(TARGET_CATEGORIES, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'RSVP-COCO 12-Class Confusion Matrix (ERP-averaged)\n'
                 f'Top-1={acc:.3f} | Top-3={top3:.3f} | BAcc={bacc:.3f} | '
                 f'Chance={1/N_CLASSES:.3f} | N={len(avg_data)} images',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Proportion', shrink=0.8)

    plt.tight_layout()
    p = os.path.join(save_dir, 'fig', 'rsvp_confusion_matrix.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close(fig)

    return {'acc': float(acc), 'bacc': float(bacc), 'top3': float(top3),
            'n_samples': len(avg_data), 'n_features': X.shape[1]}


# ═══ Main ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fif', type=str,
                        default=None,  # pass --fif <path_to_fif>
                        help='Path to .fif file')
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(os.path.join(args.save_dir, 'fig'), exist_ok=True)

    # Load
    data, labels, categories, img_ids, groups, times, sfreq = load_rsvp_data(args.fif)

    # 1. ERP diagnostic
    print("\n" + "=" * 70)
    print("ANALYSIS 1: ERP DIAGNOSTIC")
    print("=" * 70)
    erp_info = run_erp_diagnostic(data, labels, categories, times, sfreq, args.save_dir)

    # 2. Temporal sensitivity sweep
    print("\n" + "=" * 70)
    print("ANALYSIS 2: TEMPORAL SENSITIVITY SWEEP")
    print("=" * 70)
    sweep_results = run_temporal_sweep(data, labels, times, sfreq, args.save_dir,
                                       t_len_ms=200, t_step_ms=50,
                                       n_folds=args.n_folds, seed=args.seed)

    # 3. ERP averaging effect
    print("\n" + "=" * 70)
    print("ANALYSIS 3: ERP AVERAGING EFFECT")
    print("=" * 70)
    avg_results = run_erp_avg_calibration(data, labels, groups, times, sfreq,
                                          args.save_dir, n_folds=args.n_folds,
                                          seed=args.seed)

    # 4. Full classification
    print("\n" + "=" * 70)
    print("ANALYSIS 4: FULL 12-CLASS CLASSIFICATION")
    print("=" * 70)
    cls_results = run_full_classification(data, labels, categories, groups,
                                          times, sfreq, args.save_dir,
                                          n_folds=args.n_folds, seed=args.seed)

    # Save JSON summary
    summary = {
        'data': {
            'fif_path': args.fif,
            'n_epochs': len(data),
            'n_channels': data.shape[1],
            'n_times': data.shape[2],
            'tmin': float(times[0]),
            'tmax': float(times[-1]),
            'sfreq': float(sfreq),
            'n_categories': N_CLASSES,
            'categories': TARGET_CATEGORIES,
            'n_images': len(groups),
        },
        'erp_diagnostic': erp_info,
        'temporal_sweep': sweep_results,
        'erp_averaging': avg_results,
        'classification': cls_results,
    }
    jp = os.path.join(args.save_dir, 'fig', 'rsvp_analysis_results.json')
    with open(jp, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nJSON summary: {jp}")
    print("\n=== ALL ANALYSES COMPLETE ===")


if __name__ == '__main__':
    main()
