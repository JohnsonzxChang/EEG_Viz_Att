"""Signal-quality metrics: per-channel variance / kurtosis, bad-trial flags,
GFP pre/post ratio, alpha-band SNR. All operate on an EpochBundle."""
from __future__ import annotations

import logging
import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis

from process1_data_process.data_io import EpochBundle

log = logging.getLogger(__name__)


def per_epoch_quality(b: EpochBundle, eeg_channels_only: bool = True) -> dict:
    """Compute per-epoch QC metrics.

    Returns a dict of (n,)-shaped arrays:
        rms        — root-mean-square amplitude across channels & time
        kurt       — channel-mean kurtosis (eyeblink / outlier sensitive)
        peak2peak  — max peak-to-peak across channels (saturation)
        flat_count — # channels with peak2peak < 1 µV (flatline)
        gfp_post   — GFP averaged over [0, 0.5]s
        gfp_pre    — GFP averaged over [-0.5, 0]s
    """
    chs = [i for i, n in enumerate(b.ch_names)
           if not (eeg_channels_only and n in ("gaze_x", "gaze_y"))]
    X = b.data[:, chs, :]
    t = b.times

    pre_mask = t < 0
    post_mask = (t >= 0) & (t <= 0.5)

    rms = np.sqrt((X ** 2).mean(axis=(1, 2)))
    kurt = kurtosis(X, axis=2).mean(axis=1)
    p2p_per_ch = X.max(axis=2) - X.min(axis=2)
    p2p_max = p2p_per_ch.max(axis=1)
    flat_count = (p2p_per_ch < 1.0).sum(axis=1)

    gfp = X.std(axis=1)                                    # (n, t)
    gfp_pre = gfp[:, pre_mask].mean(axis=1)
    gfp_post = gfp[:, post_mask].mean(axis=1)

    return {
        "rms": rms.astype(np.float32),
        "kurt": kurt.astype(np.float32),
        "p2p_max": p2p_max.astype(np.float32),
        "flat_count": flat_count.astype(np.int32),
        "gfp_pre": gfp_pre.astype(np.float32),
        "gfp_post": gfp_post.astype(np.float32),
    }


def flag_bad_trials(qc: dict, thresholds: dict | None = None) -> np.ndarray:
    """Boolean (n,) array — True = should be excluded."""
    th = {"p2p_max_uv": 200.0, "flat_count_max": 2, "kurt_max": 8.0}
    if thresholds:
        th.update(thresholds)
    bad = ((qc["p2p_max"] > th["p2p_max_uv"]) |
           (qc["flat_count"] > th["flat_count_max"]) |
           (np.abs(qc["kurt"]) > th["kurt_max"]))
    log.info("Flagged %d/%d trials as bad (%.1f%%)", bad.sum(), len(bad),
             100.0 * bad.mean())
    return bad


def alpha_band_snr(b: EpochBundle, fmin: float = 8.0, fmax: float = 13.0) -> np.ndarray:
    """Per-epoch alpha-band SNR (post / pre stim onset).

    A higher pre-onset alpha indicates eyes-closed / drowsiness;
    healthy attended trials show alpha desync post-onset → SNR < 1 means
    desync happened, > 1 means alpha increased.
    """
    chs = [i for i, n in enumerate(b.ch_names) if n not in ("gaze_x", "gaze_y")]
    X = b.data[:, chs, :]
    sf = b.sfreq
    nyq = sf / 2
    sos = sig.butter(4, [fmin / nyq, fmax / nyq], btype="band", output="sos")
    band = sig.sosfiltfilt(sos, X, axis=2)
    p = band ** 2
    t = b.times
    pre = p[..., t < 0].mean(axis=(1, 2))
    post = p[..., (t > 0) & (t <= 0.5)].mean(axis=(1, 2))
    snr = post / np.maximum(pre, 1e-9)
    return snr.astype(np.float32)


def class_balance(b: EpochBundle) -> dict:
    """Per-HINT, per-stim-category, per-eeg-split trial counts."""
    out = {"hint": {}, "stim_category": {}, "eeg_split": {}}
    for k, arr in [("hint", b.hint), ("stim_category", b.stim_category),
                    ("eeg_split", b.eeg_split)]:
        uniq, cnt = np.unique(arr, return_counts=True)
        out[k] = dict(zip(uniq.tolist(), cnt.tolist()))
    return out


def multilabel_co_occurrence(b: EpochBundle) -> np.ndarray:
    """K×K co-occurrence matrix of `targets_in_image`.

    Each epoch's image contains K targets (one is the attended one); this
    matrix tells you how often each pair of categories co-occur across
    stimuli.
    """
    from collections import Counter
    cats: list[str] = []
    for tlist in b.targets_in_image:
        cats.extend(tlist)
    universe = sorted(set(cats))
    idx = {c: i for i, c in enumerate(universe)}
    M = np.zeros((len(universe), len(universe)), dtype=np.int32)
    for tlist in b.targets_in_image:
        ids = [idx[t] for t in tlist if t in idx]
        for i in ids:
            for j in ids:
                M[i, j] += 1
    return M, universe
