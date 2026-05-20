"""Plan B (filter / no-baseline) + Plan C (rERP overlap deconvolution).

Plan C — finite-support Tikhonov-regularised linear deconvolution.

Model
-----
Let x(τ') be the unknown single-event response with **finite support** in
τ' ∈ [resp_tmin, resp_tmax]  (default −100…+450 ms).  With constant SOA Δ
the grand-average epoch satisfies

    ȳ(τ) = Σ_k x(τ + k·Δ)        τ ∈ [obs_tmin, obs_tmax]

We pick all k for which (τ + k·Δ) ∈ supp(x), which makes A · x = ȳ an
overdetermined linear system (≈ 2.5× more rows than columns). We solve it
with Tikhonov regularisation  min ‖A x − ȳ‖² + λ ‖x‖².

The result x(τ') is the single-event response purified of neighbour
overlap; combining with the un-deconvolved ȳ side-by-side shows how much
of the apparent "3-cycle" structure is overlap vs. genuine late activity.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from process1_data_process.data_io import load_epochs, EpochBundle
from process1_data_process.plot_style import (
    PALETTE, apply_publication_style, save_fig,
)

log = logging.getLogger(__name__)
apply_publication_style()


def _eeg_chs(b):
    return [i for i, n in enumerate(b.ch_names) if n not in ("gaze_x", "gaze_y")]


# ─────────────────────  PLAN  B  ───────────────────────────────────────

def plot_plan_b_filter_vs_baseline(fif, session_json, out_dir: Path,
                                     filter_band=(0.5, 30.0)) -> None:
    log.info("Plan B: loading two variants ...")
    b_base = load_epochs(fif, session_json, pick_eeg=True, load_data=True,
                         decim=4, dtype="float32",
                         crop_tmin=-0.5, crop_tmax=1.0,
                         baseline=(-0.1, 0.0), filter_band=None)
    b_filt = load_epochs(fif, session_json, pick_eeg=True, load_data=True,
                         decim=4, dtype="float32",
                         crop_tmin=-0.5, crop_tmax=1.0,
                         baseline=None, filter_band=filter_band)

    chs = _eeg_chs(b_base)
    t = b_base.times * 1000
    ga_base = b_base.data[:, chs, :].mean(axis=0)
    ga_filt = b_filt.data[:, chs, :].mean(axis=0)
    gfp_base = ga_base.std(0)
    gfp_filt = ga_filt.std(0)
    soa_ms = 514.0

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True,
                              gridspec_kw={"height_ratios": [1.5, 1]})
    fig.suptitle(f"Plan B — bandpass {filter_band[0]:.1f}-{filter_band[1]:.0f} Hz vs "
                 "short-baseline subtraction\n(SOA = 514 ms; dashed = neighbour-trial onsets)",
                  fontweight="bold")
    ax = axes[0]
    for c in range(ga_filt.shape[0]):
        ax.plot(t, ga_filt[c], color="steelblue", alpha=0.15, lw=0.5)
    ax.plot(t, gfp_base, color=PALETTE["target"], lw=1.6,
             label="baseline [-100,0] ms ⇒ GFP")
    ax.plot(t, gfp_filt, color=PALETTE["diff"], lw=1.6,
             label=f"bandpass {filter_band[0]:.1f}-{filter_band[1]:.0f} Hz ⇒ GFP (no baseline)")
    for k in (-1, 1, 2):
        ax.axvline(k * soa_ms, color="gray", ls=":", lw=0.6)
        ax.text(k * soa_ms, ax.get_ylim()[1]*0.95, f"n{k:+d}",
                 fontsize=6, ha="center", color="gray")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_ylabel("Amplitude (µV)  /  GFP")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)
    ax.set_title("Grand average (butterfly = bandpass variant)")

    ax = axes[1]
    ax.plot(t, gfp_filt - gfp_filt[t < 0].mean(),
             color=PALETTE["diff"], lw=1.6,
             label="bandpass GFP − pre-stim mean")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    for k in (-1, 1, 2):
        ax.axvline(k * soa_ms, color="gray", ls=":", lw=0.6)
    ax.axhline(0, color="gray", alpha=0.4, lw=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Δ GFP (µV)")
    ax.set_title("Bandpass GFP centred — neighbour-trial cycles visible at ±SOA")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)

    plt.tight_layout()
    save_fig(fig, out_dir / "F_PLANB_filter_vs_baseline.png")
    log.info("Saved plan B comparison")


# ─────────────────────  PLAN  C  ───────────────────────────────────────

def _design_finite_support(t_obs, sf, soa_s, resp_tmin, resp_tmax,
                             n_neighbours=3):
    """Build A (n_obs, n_resp) under the assumption x(τ') has support
    only in [resp_tmin, resp_tmax]."""
    soa_samp = int(round(soa_s * sf))
    # response grid
    resp_samp_min = int(round(resp_tmin * sf))
    resp_samp_max = int(round(resp_tmax * sf))
    n_resp = resp_samp_max - resp_samp_min + 1
    obs_samp = np.round(t_obs / 1000.0 * sf).astype(int)
    n_obs = len(obs_samp)
    A = np.zeros((n_obs, n_resp), dtype=np.float32)
    for i, sj in enumerate(obs_samp):
        for k in range(-n_neighbours, n_neighbours + 1):
            # x(τ + k·Δ) is non-zero iff τ_i + k·Δ ∈ supp(x)
            # i.e. obs_sample[i] + k·soa_samp ∈ [resp_samp_min, resp_samp_max]
            tau_resp = sj + k * soa_samp
            if resp_samp_min <= tau_resp <= resp_samp_max:
                j = tau_resp - resp_samp_min
                A[i, j] = 1.0
    t_resp = np.arange(resp_samp_min, resp_samp_max + 1) / sf * 1000  # ms
    return A, t_resp


def rerp_deconvolve(b, *, soa_ms=514.0, ridge=0.5,
                     resp_tmin_ms=-100.0, resp_tmax_ms=450.0,
                     n_neighbours=3) -> dict:
    """Tikhonov-regularised deconvolution under finite-support constraint.

    Returns dict with x_obs, x_dec, t_obs (ms), t_dec (ms), cond, A_shape.
    """
    chs = _eeg_chs(b)
    X = b.data[:, chs, :].mean(axis=0)
    sf = b.sfreq
    t_obs = b.times * 1000
    A, t_dec = _design_finite_support(t_obs, sf, soa_ms / 1000.0,
                                        resp_tmin_ms / 1000.0,
                                        resp_tmax_ms / 1000.0, n_neighbours)
    log.info("rERP design matrix: A=%s, ridge=%.2g, neighbours=±%d",
              A.shape, ridge, n_neighbours)
    AtA = A.T @ A + ridge * np.eye(A.shape[1], dtype=np.float32)
    AtY = A.T @ X.T
    x_dec = np.linalg.solve(AtA, AtY).T
    cond = float(np.linalg.cond(AtA))
    log.info("Normal-matrix cond=%.1e", cond)
    return dict(x_obs=X, x_dec=x_dec, t_obs=t_obs, t_dec=t_dec,
                 cond=cond, A_shape=A.shape)


def plot_plan_c_rerp(b, out_dir: Path, soa_ms=514.0, ridge=0.5,
                       fif_for_info=None,
                       resp_tmin_ms=-100.0, resp_tmax_ms=450.0) -> dict:
    res = rerp_deconvolve(b, soa_ms=soa_ms, ridge=ridge,
                            resp_tmin_ms=resp_tmin_ms,
                            resp_tmax_ms=resp_tmax_ms)
    t_obs, t_dec = res["t_obs"], res["t_dec"]
    X_obs, X_dec = res["x_obs"], res["x_dec"]
    gfp_obs = X_obs.std(0)
    gfp_dec = X_dec.std(0)

    # ── F18: observed vs deconvolved GFP ────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5),
                              gridspec_kw={"height_ratios":[1,1]})
    fig.suptitle(f"Plan C — overlap deconvolution  (SOA={soa_ms:.0f} ms, "
                 f"ridge={ridge}, response support [{resp_tmin_ms:.0f}, {resp_tmax_ms:.0f}] ms)",
                  fontweight="bold")
    ax = axes[0]
    ax.plot(t_obs, gfp_obs, color=PALETTE["nontarget"], lw=1.6,
             label="Observed grand-average GFP (overlapped)")
    for k in (-2, -1, 1, 2):
        ax.axvline(k * soa_ms, color="gray", ls=":", lw=0.6)
        ax.text(k * soa_ms, ax.get_ylim()[1]*0.92, f"n{k:+d}",
                 fontsize=6, ha="center", color="gray")
    ax.axvline(0, color="black", ls="--", lw=0.8, label="current onset")
    ax.set_xlim(t_obs[0], t_obs[-1])
    ax.set_ylabel("GFP (µV)")
    ax.set_title("Observed — periodicity at ±SOA visible")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)

    ax = axes[1]
    ax.plot(t_dec, gfp_dec, color=PALETTE["target"], lw=1.8,
             label="Deconvolved x(τ) GFP")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.axhline(0, color="gray", alpha=0.3, lw=0.4)
    ax.set_xlim(t_dec[0], t_dec[-1])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("GFP (µV)")
    ax.set_title(f"Deconvolved single-event response  (cond={res['cond']:.1f})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)

    plt.tight_layout()
    save_fig(fig, out_dir / "F18_rerp_gfp_before_after.png")

    # ── F19: deconvolved butterfly + observed butterfly overlay ─
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    for c in range(X_obs.shape[0]):
        ax.plot(t_obs, X_obs[c], color="steelblue", alpha=0.3, lw=0.6)
    ax.plot(t_obs, gfp_obs, color=PALETTE["nontarget"], lw=1.6, label="GFP")
    for k in (-1, 1):
        ax.axvline(k * soa_ms, color="gray", ls=":", lw=0.6)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Observed grand-average butterfly")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)

    ax = axes[1]
    for c in range(X_dec.shape[0]):
        ax.plot(t_dec, X_dec[c], color="firebrick", alpha=0.3, lw=0.6)
    ax.plot(t_dec, gfp_dec, color=PALETTE["target"], lw=1.6, label="GFP (deconv.)")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.axhline(0, color="gray", alpha=0.3, lw=0.4)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Deconvolved butterfly")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)

    fig.suptitle("Plan C — observed vs deconvolved butterfly  (same y-axis units)",
                  fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir / "F19_rerp_butterfly.png")

    # ── F20: salient vs non-salient deconvolved ──────────────────
    area = np.array([(b.target_areas[i].get(b.hint[i], np.nan)
                       if i < len(b.target_areas) else np.nan)
                     for i in range(len(b))], dtype=float)
    have = (b.hint != "") & np.isfinite(area)
    med = np.nanmedian(area[have])
    sel_S = have & (area >= med)
    sel_N = have & (area <  med)

    res_S = rerp_deconvolve(_subset_bundle(b, sel_S), soa_ms=soa_ms, ridge=ridge,
                              resp_tmin_ms=resp_tmin_ms, resp_tmax_ms=resp_tmax_ms)
    res_N = rerp_deconvolve(_subset_bundle(b, sel_N), soa_ms=soa_ms, ridge=ridge,
                              resp_tmin_ms=resp_tmin_ms, resp_tmax_ms=resp_tmax_ms)

    gfp_S = res_S["x_dec"].std(0)
    gfp_N = res_N["x_dec"].std(0)
    mean_S = res_S["x_dec"].mean(0)
    mean_N = res_N["x_dec"].mean(0)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.5), sharex=True)
    fig.suptitle("Plan C — deconvolved salient vs non-salient (area median split)",
                  fontweight="bold")
    ax = axes[0]
    ax.plot(res_S["t_dec"], gfp_S, color=PALETTE["target"], lw=1.6,
             label=f"Salient (n={int(sel_S.sum())})")
    ax.plot(res_N["t_dec"], gfp_N, color=PALETTE["nontarget"], lw=1.6,
             label=f"Non-salient (n={int(sel_N.sum())})")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_ylabel("Deconvolved GFP (µV)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)

    ax = axes[1]
    ax.plot(res_S["t_dec"], mean_S, color=PALETTE["target"], lw=1.4, label="Salient")
    ax.plot(res_N["t_dec"], mean_N, color=PALETTE["nontarget"], lw=1.4, label="Non-salient")
    ax.plot(res_S["t_dec"], mean_S - mean_N, color=PALETTE["diff"], lw=1.4,
             label="Δ (S − NS)")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.axhline(0, color="gray", alpha=0.3, lw=0.4)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channel-mean (µV)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)

    plt.tight_layout()
    save_fig(fig, out_dir / "F20_rerp_salient_vs_nonsalient.png")
    log.info("Saved plan C figures (F18-F20)")

    return dict(soa_ms=soa_ms, ridge=ridge, cond=res["cond"],
                 A_shape=list(res["A_shape"]),
                 gfp_obs_max_uv=float(gfp_obs.max()),
                 gfp_dec_max_uv=float(gfp_dec.max()),
                 gfp_pre_after=float(gfp_dec[t_dec < 0].mean()),
                 gfp_post_after=float(gfp_dec[(t_dec > 50) & (t_dec < 350)].mean()),
                 ratio_after=float(gfp_dec[(t_dec > 50) & (t_dec < 350)].mean()
                                    / max(gfp_dec[t_dec < 0].mean(), 1e-9)),
                 resp_window_ms=[resp_tmin_ms, resp_tmax_ms])


def _subset_bundle(b, sel):
    return EpochBundle(
        data=b.data[sel],
        times=b.times, sfreq=b.sfreq, ch_names=b.ch_names,
        stim_category=b.stim_category[sel],
        image_id=b.image_id[sel],
        hint=b.hint[sel],
        eeg_split=b.eeg_split[sel],
        is_target=b.is_target[sel],
        repeat_index=b.repeat_index[sel],
        targets_in_image=[b.targets_in_image[i] for i in np.where(sel)[0]],
        target_areas=[b.target_areas[i] for i in np.where(sel)[0]],
        meta=dict(b.meta),
    )
