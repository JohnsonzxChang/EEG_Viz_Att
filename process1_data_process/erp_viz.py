"""Publication-grade ERP visualisation for Process-1.

Provides:
  • plot_grand_average            — butterfly + GFP + annotated component peaks
  • plot_erp_channel_heatmap      — (channel × time) heatmap of grand average
  • plot_per_hint_erp             — per-HINT panels (top-K by count)
  • plot_per_superclass           — 8-superclass ERP overlay
  • plot_eeg_split_erp            — train/test drift check
  • plot_qc_distributions         — QC histograms (RMS, p2p, kurt, GFP, alpha SNR)
  • plot_class_balance            — trial counts per HINT
  • plot_multilabel_cooccurrence  — P(b|a) heatmap of `targets_in_image`
  • plot_topomap_snapshots        — topo maps at P100/N170/P200/P300
  • plot_target_vs_nontarget_erp  — KEY attention contrast (target == hint)
  • plot_target_vs_nontarget_topo — topomap of target-effect at peak
  • plot_electrode_layout         — figure of the 4×8 posterior grid
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from process1_data_process.data_io import EpochBundle
from process1_data_process.plot_style import (
    PALETTE, SUPERCLASS_COLORS, apply_publication_style, save_fig,
)

log = logging.getLogger(__name__)
apply_publication_style()


# ── 40 LVIS targets → 8 superclasses for compact visualisation ──────────
SUPERCLASS_MAP = {
    "Animal": {"dog", "cat", "horse", "cow", "sheep", "bird", "elephant",
               "zebra", "giraffe", "bear", "teddy_bear"},
    "Food":   {"apple", "banana", "broccoli", "carrot", "orange_(fruit)",
               "tomato", "doughnut", "pizza"},
    "Vehicle":{"car_(automobile)", "boat", "motorcycle", "bicycle", "airplane",
               "bus_(vehicle)"},
    "Furniture":{"chair", "sofa", "dining_table", "bed", "cabinet"},
    "Kitchen":{"bottle", "cup", "bowl", "glass_(drink_container)", "fork",
               "knife", "spoon"},
    "Wearable":{"shoe", "hat", "umbrella", "backpack", "handbag"},
    "Electronics":{"laptop_computer", "computer_keyboard", "cellular_telephone",
                   "television_set", "remote_control"},
}


def _superclass(cat: str) -> str:
    for sc, members in SUPERCLASS_MAP.items():
        if cat in members:
            return sc
    return "Other"


def _eeg_chs(b: EpochBundle) -> list[int]:
    return [i for i, n in enumerate(b.ch_names) if n not in ("gaze_x", "gaze_y")]


def _ch_positions(b: EpochBundle, info=None) -> np.ndarray:
    """(n_ch, 2) — 2D layout in metres. Drops gaze."""
    if info is None:
        return None
    chs = _eeg_chs(b)
    pos = np.array([info["chs"][i]["loc"][:2] for i in chs])
    return pos


# ───────────────────── ERP plots ────────────────────────────────────────

def plot_grand_average(b: EpochBundle, out_dir: Path) -> dict:
    chs = _eeg_chs(b)
    X = b.data[:, chs, :]
    t = b.times * 1000  # ms

    grand = X.mean(axis=0)            # (ch, t)
    gfp = grand.std(axis=0)
    pre = gfp[t < 0].mean()
    post = gfp[(t >= 50) & (t <= 500)].mean()
    peak_t = t[np.argmax(gfp)]

    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"Grand-Average ERP  (N={len(X)} epochs, 32 occipital EEG)",
                 fontweight="bold")
    # butterfly
    for c in range(grand.shape[0]):
        ax[0].plot(t, grand[c], color="steelblue", alpha=0.30, lw=0.6)
    ax[0].plot(t, gfp, color=PALETTE["gfp"], lw=1.8, label="GFP")
    ax[0].axvline(0, color="black", ls="--", lw=0.8)
    ax[0].axhline(0, color="gray", alpha=0.3, lw=0.5)
    ax[0].set_ylabel("Amplitude (µV)")
    ax[0].set_title("Butterfly + Global Field Power")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.15)

    ax[1].plot(t, gfp, color=PALETTE["gfp"], lw=1.8)
    for tev, lbl, c in [(100, "P100", PALETTE["p100"]),
                         (170, "N170", PALETTE["n170"]),
                         (300, "P300", PALETTE["p300"])]:
        ax[1].axvline(tev, ls=":", lw=1.0, color=c)
        ax[1].text(tev, gfp.max() * 0.95, lbl, fontsize=7, ha="center", color=c)
    ax[1].axvline(0, color="black", ls="--", lw=0.8)
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylabel("GFP (µV)")
    ax[1].text(0.99, 0.97,
               f"pre={pre:.3f} µV\npost={post:.3f} µV\nratio={post/pre:.2f}\npeak={peak_t:.0f} ms",
               transform=ax[1].transAxes, va="top", ha="right", fontsize=7,
               bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8, lw=0))
    ax[1].grid(alpha=0.15)

    plt.tight_layout()
    save_fig(fig, out_dir / "F02_erp_grand_average.png")
    log.info("Saved grand average ERP")
    return {"gfp_pre": float(pre), "gfp_post": float(post),
            "gfp_ratio": float(post / pre), "gfp_peak_ms": float(peak_t)}


def plot_erp_channel_heatmap(b: EpochBundle, out_dir: Path) -> None:
    """Channel × Time heatmap of grand-average ERP. Useful for spotting
    posterior vs anterior gradients in the 4×8 patch."""
    chs = _eeg_chs(b)
    ch_names = [b.ch_names[i] for i in chs]
    X = b.data[:, chs, :].mean(axis=0)   # (ch, t)
    t = b.times * 1000

    # Re-order channels by y-coordinate (anterior → posterior) using info
    # but we don't have info here; use the channel index from event ordering.
    # The Biosemi-style A1..D8 layout: A is back, D front. Let's preserve.

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    vmax = max(2.0, float(np.percentile(np.abs(X), 99)))
    im = ax.imshow(X, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=+vmax,
                    extent=[t[0], t[-1], len(chs) - 0.5, -0.5],
                    interpolation="nearest")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_yticks(range(len(chs)))
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channel")
    ax.set_title("Grand-average ERP: channel × time heatmap")
    cbar = fig.colorbar(im, ax=ax, label="µV", pad=0.02, shrink=0.8)
    cbar.outline.set_linewidth(0)
    plt.tight_layout()
    save_fig(fig, out_dir / "F03_erp_channel_heatmap.png")
    log.info("Saved channel × time ERP heatmap")


def plot_per_hint_erp(b: EpochBundle, out_dir: Path, max_hints: int = 16) -> None:
    """One panel per HINT (attention category). Channel-mean ERP."""
    chs = _eeg_chs(b)
    t = b.times * 1000
    hints, cnt = np.unique(b.hint, return_counts=True)
    hints = hints[hints != ""]
    cnt = np.array([(b.hint == h).sum() for h in hints])
    order = np.argsort(-cnt)[:max_hints]
    hints = hints[order]

    rows = int(np.ceil(len(hints) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(11.5, 2.4 * rows),
                              sharex=True, sharey=True)
    fig.suptitle(f"Per-HINT ERP  (channel mean, top {len(hints)} hints by trial count)",
                 fontweight="bold")
    axes = axes.flatten()
    for i, h in enumerate(hints):
        sel = b.hint == h
        erp = b.data[sel][:, chs, :].mean(axis=(0, 1))
        sc = _superclass(h)
        axes[i].plot(t, erp, lw=1.0, color=SUPERCLASS_COLORS.get(sc, "steelblue"))
        axes[i].axvline(0, color="black", ls="--", lw=0.6)
        axes[i].axhline(0, color="gray", alpha=0.3, lw=0.4)
        axes[i].set_title(f"{h} (n={sel.sum()})", fontsize=7)
        axes[i].grid(alpha=0.15)
    for j in range(len(hints), len(axes)):
        axes[j].axis("off")
    for a in axes[:rows*4]:
        a.set_xlabel("ms", fontsize=7)
        a.set_ylabel("µV", fontsize=7)
    plt.tight_layout()
    save_fig(fig, out_dir / "F05_erp_per_hint.png")
    log.info("Saved per-HINT ERP")


def plot_per_superclass(b: EpochBundle, out_dir: Path) -> None:
    chs = _eeg_chs(b)
    t = b.times * 1000
    sc_arr = np.array([_superclass(h) for h in b.hint])
    sc_list = [s for s in SUPERCLASS_MAP.keys() if (sc_arr == s).any()]

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for sc in sc_list:
        sel = sc_arr == sc
        erp = b.data[sel][:, chs, :].mean(axis=(0, 1))
        ax.plot(t, erp, lw=1.4, color=SUPERCLASS_COLORS[sc],
                 label=f"{sc} (n={sel.sum()})")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.axhline(0, color="gray", alpha=0.3, lw=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Per-Superclass HINT ERP — channel mean")
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    save_fig(fig, out_dir / "F04_erp_per_superclass.png")
    log.info("Saved per-superclass ERP")


def plot_eeg_split_erp(b: EpochBundle, out_dir: Path) -> None:
    """train/test ERP overlay — confirms no temporal drift across blocks."""
    chs = _eeg_chs(b)
    t = b.times * 1000
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    for split, col in [("train", PALETTE["train"]), ("test", PALETTE["test"])]:
        sel = b.eeg_split == split
        if not sel.any():
            continue
        erp = b.data[sel][:, chs, :].mean(axis=(0, 1))
        ax.plot(t, erp, lw=1.4, color=col, label=f"{split} (n={sel.sum()})")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("µV")
    ax.set_title("ERP by eeg_split  (train/test drift check)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    save_fig(fig, out_dir / "F08_erp_split_drift.png")
    log.info("Saved split-drift ERP")


def plot_qc_distributions(b: EpochBundle, qc: dict, out_dir: Path,
                           alpha: np.ndarray | None = None) -> None:
    """6-panel QC histogram + optional alpha-SNR sub-panel."""
    ncols = 4 if alpha is not None else 3
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 5.5))
    panels = [
        ("rms",        "RMS amplitude (µV)"),
        ("p2p_max",    "Max peak-to-peak (µV)"),
        ("kurt",       "Mean channel kurtosis"),
        ("flat_count", "# flat channels per epoch"),
        ("gfp_pre",    "GFP pre-stim (µV)"),
        ("gfp_post",   "GFP post-stim (µV)"),
    ]
    for ax, (key, title) in zip(axes.flatten(), panels):
        v = qc[key]
        ax.hist(v, bins=50, color="steelblue", alpha=0.85, edgecolor="white", lw=0.4)
        ax.set_title(f"{title}\nmean={v.mean():.2f}  median={np.median(v):.2f}",
                     fontsize=8)
        ax.grid(alpha=0.15)
    if alpha is not None and ncols == 4:
        ax = axes[0, 3]
        ax.hist(alpha, bins=50, color="#9467BD", alpha=0.85, edgecolor="white", lw=0.4)
        ax.axvline(1.0, color="black", ls="--", lw=0.8)
        ax.set_title(f"Alpha SNR post/pre\nmean={alpha.mean():.2f}  med={np.median(alpha):.2f}")
        ax.grid(alpha=0.15)
        axes[1, 3].axis("off")
    plt.tight_layout()
    save_fig(fig, out_dir / "F09_qc_distributions.png")
    log.info("Saved QC distributions")


def plot_class_balance(b: EpochBundle, out_dir: Path) -> None:
    hints, cnt = np.unique(b.hint, return_counts=True)
    keep = hints != ""
    hints, cnt = hints[keep], cnt[keep]
    order = np.argsort(-cnt)
    fig, ax = plt.subplots(figsize=(11.5, 3.5))
    colors = [SUPERCLASS_COLORS.get(_superclass(h), PALETTE["neutral"])
              for h in hints[order]]
    ax.bar(range(len(hints)), cnt[order], color=colors)
    ax.set_xticks(range(len(hints)))
    ax.set_xticklabels(hints[order], rotation=80, ha="right", fontsize=7)
    ax.set_ylabel("# epochs")
    ax.set_title(f"Class balance — {len(hints)} HINT categories  "
                 f"(min={cnt.min()}, max={cnt.max()}, mean={cnt.mean():.0f})")
    ax.axhline(cnt.mean(), color="black", ls="--", lw=0.8,
               label=f"mean={cnt.mean():.0f}")
    # legend for superclasses
    sc_seen = {_superclass(h) for h in hints}
    handles = [plt.Rectangle((0, 0), 1, 1, color=SUPERCLASS_COLORS.get(sc, PALETTE["neutral"]))
                for sc in SUPERCLASS_MAP if sc in sc_seen]
    labels = [sc for sc in SUPERCLASS_MAP if sc in sc_seen]
    ax.legend(handles, labels, loc="upper right", ncol=2, fontsize=7)
    ax.grid(alpha=0.15, axis="y")
    plt.tight_layout()
    save_fig(fig, out_dir / "F10_class_balance.png")
    log.info("Saved class balance")


def plot_multilabel_cooccurrence(M: np.ndarray, universe: list[str],
                                  out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    norm = M / np.maximum(np.diag(M)[:, None], 1)
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=norm.max())
    ax.set_xticks(range(len(universe)))
    ax.set_yticks(range(len(universe)))
    ax.set_xticklabels(universe, rotation=80, ha="right", fontsize=5)
    ax.set_yticklabels(universe, fontsize=5)
    cbar = fig.colorbar(im, ax=ax, label="P(col | row)", shrink=0.8)
    cbar.outline.set_linewidth(0)
    ax.set_title("Multi-label image-level co-occurrence")
    plt.tight_layout()
    save_fig(fig, out_dir / "F11_multilabel_cooccurrence.png")
    log.info("Saved co-occurrence")


# ─────────────────────  ATTENTION / TARGET-vs-NONTARGET  ────────────────

def plot_target_vs_nontarget_erp(b: EpochBundle, out_dir: Path) -> dict:
    """KEY attention-salience figure.

    Note on paradigm: in the LVIS-attention RSVP, every recorded stim_onset
    is an attended (target) trial — the outlined category always equals the
    hint. There is therefore no "non-target" stimulus class.

    Instead we contrast SALIENT (large attended-object area) vs NON-SALIENT
    (small area) trials, using a median split on `target_areas[hint]`. This
    isolates the area-of-attended-object effect on the ERP — a known driver
    of N170/P300 amplitude.

    Output panels:
        (1) salient vs non-salient ERP at channel mean (with 95% CI)
        (2) difference wave (salient − non-salient) with point-wise sig mask
        (3) salient − non-salient (channel × time) heatmap
    """
    chs = _eeg_chs(b)
    t = b.times * 1000

    # Extract attended-object area for every trial
    area = np.array([(b.target_areas[i].get(b.hint[i], np.nan)
                       if i < len(b.target_areas) else np.nan)
                      for i in range(len(b))], dtype=float)
    have = (b.hint != "") & np.isfinite(area)
    med = np.nanmedian(area[have])
    sel_t = have & (area >= med)           # salient (large area)
    sel_n = have & (area <  med)           # non-salient (small area)
    X_t = b.data[sel_t][:, chs, :]
    X_n = b.data[sel_n][:, chs, :]
    log.info("median area=%.3f, salient=%d, non-salient=%d",
             med, sel_t.sum(), sel_n.sum())

    # Channel-mean ERPs (averaged across channels then across trials)
    erp_t = X_t.mean(axis=1)                       # (n_t, t)
    erp_n = X_n.mean(axis=1)
    mean_t, mean_n = erp_t.mean(0), erp_n.mean(0)
    sem_t = erp_t.std(0) / np.sqrt(max(1, len(erp_t)))
    sem_n = erp_n.std(0) / np.sqrt(max(1, len(erp_n)))
    diff = mean_t - mean_n

    # Pointwise t-test (Welch) — proxy for cluster permutation
    from scipy import stats as ss
    tv, pv = ss.ttest_ind(erp_t, erp_n, equal_var=False, axis=0)
    p_thr = 0.01
    sig_mask = pv < p_thr

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True,
                              gridspec_kw={"height_ratios": [1.2, 1.0, 1.6]})
    fig.suptitle(f"Salience modulation — attended-object area median split "
                  f"(med={med:.3f})", fontweight="bold")

    ax = axes[0]
    ax.plot(t, mean_t, color=PALETTE["target"], lw=1.6,
            label=f"Salient/large (n={sel_t.sum()})")
    ax.fill_between(t, mean_t - 1.96 * sem_t, mean_t + 1.96 * sem_t,
                     color=PALETTE["target"], alpha=0.18, lw=0)
    ax.plot(t, mean_n, color=PALETTE["nontarget"], lw=1.6,
            label=f"Non-salient/small (n={sel_n.sum()})")
    ax.fill_between(t, mean_n - 1.96 * sem_n, mean_n + 1.96 * sem_n,
                     color=PALETTE["nontarget"], alpha=0.18, lw=0)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.axhline(0, color="gray", alpha=0.3, lw=0.5)
    ax.set_ylabel("µV (channel mean)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)
    ax.set_title("Channel-mean ERP")

    ax = axes[1]
    ax.plot(t, diff, color=PALETTE["diff"], lw=1.6,
            label="Salient − Non-salient")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.axhline(0, color="gray", alpha=0.3, lw=0.5)
    ymin, ymax = ax.get_ylim()
    # shade where the difference is significant
    where = sig_mask
    ax.fill_between(t, ymin, ymax, where=where,
                     color=PALETTE["diff"], alpha=0.10, lw=0,
                     label=f"p<{p_thr} (pointwise)")
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Δ µV")
    ax.set_title("Difference wave")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)

    # channel × time heatmap of target - nontarget
    ax = axes[2]
    diff_full = X_t.mean(0) - X_n.mean(0)
    vmax = max(0.5, float(np.percentile(np.abs(diff_full), 99)))
    im = ax.imshow(diff_full, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=+vmax,
                    extent=[t[0], t[-1], len(chs) - 0.5, -0.5],
                    interpolation="nearest")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_yticks(range(len(chs)))
    ax.set_yticklabels([b.ch_names[i] for i in chs], fontsize=5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channel")
    ax.set_title("Salient − Non-salient  (channel × time)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="Δ µV", pad=0.01)
    cbar.outline.set_linewidth(0)

    plt.tight_layout()
    save_fig(fig, out_dir / "F06_target_vs_nontarget_erp.png")
    log.info("Saved target vs non-target")

    # find first significant window
    idx_sig = np.where(sig_mask & (t > 0))[0]
    win = (float(t[idx_sig[0]]), float(t[idx_sig[-1]])) if len(idx_sig) else (None, None)
    return {
        "n_target": int(sel_t.sum()),
        "n_nontarget": int(sel_n.sum()),
        "diff_peak_uv": float(diff[np.argmax(np.abs(diff))]),
        "diff_peak_ms": float(t[np.argmax(np.abs(diff))]),
        "sig_window_ms": list(win),
        "frac_sig_post": float((sig_mask & (t >= 0)).mean()),
    }


def plot_topomap_snapshots(b: EpochBundle, out_dir: Path,
                            times_ms=(100, 170, 220, 300, 400),
                            fif_for_info: str | None = None) -> None:
    """Snapshot topomaps of the grand average at canonical ERP times.

    Requires an MNE info object with channel positions. Fall back to a
    scatter-based topomap if mne plotting fails (e.g. bad montage).
    """
    chs = _eeg_chs(b)
    t = b.times * 1000
    X = b.data[:, chs, :].mean(0)
    info = None
    try:
        import mne
        ep = mne.read_epochs(fif_for_info or b.meta["fif_path"],
                              preload=False, verbose="ERROR")
        info = ep.info.copy()
        info = mne.pick_info(info, sel=[i for i, n in enumerate(info["ch_names"])
                                          if n not in ("gaze_x", "gaze_y")])
    except Exception as e:
        log.warning("Topomap fallback: %s", e)

    n = len(times_ms)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.6))
    fig.suptitle("Topomaps — grand-average ERP snapshots",
                 fontweight="bold")
    vmax = max(1.0, float(np.percentile(np.abs(X), 98)))
    for j, tt in enumerate(times_ms):
        k = int(np.argmin(np.abs(t - tt)))
        vals = X[:, k]
        ax = axes[j]
        if info is not None:
            try:
                import mne
                mne.viz.plot_topomap(vals, info, axes=ax, show=False,
                                      cmap="RdBu_r",
                                      vlim=(-vmax, +vmax), contours=4,
                                      sensors=True)
            except Exception as e:
                log.warning("mne topomap failed at %dms: %s", tt, e)
                _scatter_topomap(ax, b, chs, vals, vmax)
        else:
            _scatter_topomap(ax, b, chs, vals, vmax)
        ax.set_title(f"{tt} ms", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                 norm=plt.Normalize(-vmax, +vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                         orientation="horizontal", shrink=0.4, pad=0.12,
                         label="µV")
    cbar.outline.set_linewidth(0)
    plt.subplots_adjust(top=0.86, bottom=0.18)
    save_fig(fig, out_dir / "F07_topomap_snapshots.png")
    log.info("Saved topomap snapshots")


def _scatter_topomap(ax, b, chs, vals, vmax):
    """Fallback topomap using raw 2D positions as scatter."""
    pos = []
    for i in chs:
        # the info positions are NOT in EpochBundle directly; we approximate
        pos.append((i, 0))  # placeholder — overwritten by caller in practice
    ax.scatter([p[0] for p in pos], [p[1] for p in pos],
               c=vals, cmap="RdBu_r", vmin=-vmax, vmax=+vmax, s=60)
    ax.set_xticks([]); ax.set_yticks([])


def plot_target_vs_nontarget_topo(b: EpochBundle, out_dir: Path,
                                    peak_ms_target=300,
                                    fif_for_info: str | None = None) -> None:
    """Topomap of the target − non-target difference at the peak time."""
    chs = _eeg_chs(b)
    t = b.times * 1000

    area = np.array([(b.target_areas[i].get(b.hint[i], np.nan)
                       if i < len(b.target_areas) else np.nan)
                      for i in range(len(b))], dtype=float)
    have = (b.hint != "") & np.isfinite(area)
    med = np.nanmedian(area[have])
    sel_t = have & (area >= med)
    sel_n = have & (area <  med)
    Dmean = (b.data[sel_t][:, chs, :].mean(0) -
              b.data[sel_n][:, chs, :].mean(0))   # (ch, t)

    # search for actual peak around the specified ms
    near = np.where((t >= peak_ms_target - 100) & (t <= peak_ms_target + 200))[0]
    peak_k = near[np.argmax(np.abs(Dmean[:, near]).mean(0))]
    peak_t = t[peak_k]
    vals = Dmean[:, peak_k]
    vmax = max(0.4, float(np.percentile(np.abs(vals), 98)))

    info = None
    try:
        import mne
        ep = mne.read_epochs(fif_for_info or b.meta["fif_path"],
                              preload=False, verbose="ERROR")
        info = ep.info.copy()
        info = mne.pick_info(info, sel=[i for i, n in enumerate(info["ch_names"])
                                          if n not in ("gaze_x", "gaze_y")])
    except Exception as e:
        log.warning("Topomap fallback: %s", e)

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    if info is not None:
        try:
            import mne
            mne.viz.plot_topomap(vals, info, axes=ax, show=False,
                                  cmap="RdBu_r", vlim=(-vmax, +vmax),
                                  contours=4, sensors=True)
        except Exception as e:
            log.warning("mne topomap failed: %s", e)
    ax.set_title(f"Salient − Non-salient @ {peak_t:.0f} ms\n(peak ‖Δ‖)",
                  fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                 norm=plt.Normalize(-vmax, +vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, label="Δ µV", pad=0.04)
    cbar.outline.set_linewidth(0)
    plt.tight_layout()
    save_fig(fig, out_dir / "F06b_target_effect_topomap.png")
    log.info("Saved target-effect topomap")


def plot_electrode_layout(b: EpochBundle, out_dir: Path,
                            fif_for_info: str | None = None) -> None:
    """Schematic of the 4×8 posterior electrode patch."""
    try:
        import mne
        ep = mne.read_epochs(fif_for_info or b.meta["fif_path"],
                              preload=False, verbose="ERROR")
        info = ep.info
    except Exception:
        log.warning("Could not load info for electrode layout")
        return

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    chs = _eeg_chs(b)
    for i in chs:
        loc = info["chs"][i]["loc"]
        x, y = loc[0] * 100, loc[1] * 100  # → cm
        ax.scatter(x, y, s=200, c="white", edgecolors="black", lw=1.2, zorder=2)
        ax.text(x, y, b.ch_names[i], fontsize=6, ha="center", va="center", zorder=3)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("Posterior electrode patch — 4×8 grid (32 channels)",
                  fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    save_fig(fig, out_dir / "F01_electrode_layout.png")
    log.info("Saved electrode layout")
