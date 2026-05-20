"""Process-1 driver — single --config YAML, no other CLI args.

    python -B -m process1_data_process.run_qc \
        --config process1_data_process/configs/process1_clean.yaml

The resolved config is dumped next to the figures so the run is
fully reproducible.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent.parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from process1_data_process.config import load_config, dump_resolved_config
from process1_data_process.data_io import load_epochs
from process1_data_process.qc_signal import (
    alpha_band_snr, class_balance, flag_bad_trials,
    multilabel_co_occurrence, per_epoch_quality,
)
from process1_data_process.erp_viz import (
    plot_class_balance, plot_eeg_split_erp, plot_grand_average,
    plot_multilabel_cooccurrence, plot_per_hint_erp, plot_per_superclass,
    plot_qc_distributions, plot_erp_channel_heatmap,
    plot_target_vs_nontarget_erp, plot_topomap_snapshots,
    plot_target_vs_nontarget_topo, plot_electrode_layout,
)
from process1_data_process.attention_effect import (
    plot_alpha_desync, plot_target_area_regression,
    plot_per_target_count_erp,
)
from process1_data_process.decoder_calibration import (
    plot_hint_confusion, plot_erp_averaging_effect,
)
from process1_data_process.temporal_sweep import plot_sweep, temporal_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                     help="YAML config (see process1_data_process/configs/).")
    args = ap.parse_args()
    cfg = load_config(args.config)

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")
    log = logging.getLogger("process1")
    t_master = time.time()

    fig_dir = Path(cfg["output"]["fig_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    dump_resolved_config(cfg, fig_dir / "_resolved_config.yaml")

    # ── preprocessing args from YAML ──────────────────────────────
    p = cfg["preproc"]
    baseline = None
    if p.get("baseline"):
        baseline = (p["baseline"]["tmin_s"], p["baseline"]["tmax_s"])
    filter_band = None
    if p.get("filter_band") and p["filter_band"]["lo_hz"] is not None:
        filter_band = (p["filter_band"]["lo_hz"], p["filter_band"]["hi_hz"])

    b = load_epochs(cfg["dataset"]["fif"], cfg["dataset"]["session_json"],
                    pick_eeg=p["pick_eeg"], load_data=True,
                    decim=p["decim"], dtype=p["dtype"],
                    crop_tmin=p["crop_tmin_s"], crop_tmax=p["crop_tmax_s"],
                    baseline=baseline, filter_band=filter_band)
    log.info("Bundle: %d epochs, ch=%d, sfreq=%.0fHz, t=[%.3f, %.3f]s",
             len(b), len(b.ch_names), b.sfreq, b.times[0], b.times[-1])

    qc_th = dict(p2p_max_uv=cfg["qc"]["p2p_max_uv"],
                  flat_count_max=cfg["qc"]["flat_count_max"],
                  kurt_max=cfg["qc"]["kurt_max"])
    qc = per_epoch_quality(b)
    bad = flag_bad_trials(qc, thresholds=qc_th)
    alpha_band = cfg["viz"]["alpha_band_hz"]
    alpha = alpha_band_snr(b, fmin=alpha_band[0], fmax=alpha_band[1])
    balance = class_balance(b)
    M, universe = multilabel_co_occurrence(b)

    log.info("=== FIG: electrode layout ===")
    plot_electrode_layout(b, fig_dir, fif_for_info=cfg["dataset"]["fif"])
    log.info("=== FIG: QC ===")
    plot_qc_distributions(b, qc, fig_dir, alpha=alpha)
    plot_class_balance(b, fig_dir)
    plot_multilabel_cooccurrence(M, universe, fig_dir)

    log.info("=== FIG: ERP ===")
    ga_stats = plot_grand_average(b, fig_dir)
    plot_erp_channel_heatmap(b, fig_dir)
    plot_per_hint_erp(b, fig_dir, max_hints=cfg["viz"]["max_hints_panel"])
    plot_per_superclass(b, fig_dir)
    plot_eeg_split_erp(b, fig_dir)
    plot_topomap_snapshots(b, fig_dir, fif_for_info=cfg["dataset"]["fif"],
                             times_ms=cfg["viz"]["topomap_times_ms"])

    log.info("=== FIG: salience modulation ===")
    att_stats = plot_target_vs_nontarget_erp(b, fig_dir)
    plot_target_vs_nontarget_topo(b, fig_dir,
                                    fif_for_info=cfg["dataset"]["fif"],
                                    peak_ms_target=cfg["viz"]["target_effect_peak_ms"])
    alpha_stats = plot_alpha_desync(b, fig_dir,
                                      fmin=alpha_band[0], fmax=alpha_band[1])
    area_win = cfg["viz"]["target_area_window_ms"]
    area_stats = plot_target_area_regression(b, fig_dir,
                                                win_ms=tuple(area_win))
    plot_per_target_count_erp(b, fig_dir)

    sweep_results = None
    if cfg["sweep"]["enable"]:
        log.info("=== FIG: temporal sweep ===")
        s = cfg["sweep"]
        sweep_results = temporal_sweep(b, label="hint",
                                        t_len_ms=s["t_len_ms"],
                                        t_step_ms=s["t_step_ms"],
                                        n_folds=s["n_folds"],
                                        seed=cfg["decoder"]["seed"],
                                        lda_solver=s["lda_solver"],
                                        lda_shrinkage=s["lda_shrinkage"])
        n_classes = len(set([h for h in b.hint if h]))
        plot_sweep(sweep_results, "hint", n_classes,
                    fig_dir / "F17_temporal_sweep_hint.png")

    log.info("=== FIG: HINT confusion ===")
    win = tuple(cfg["decoder"]["win_ms"])
    conf_stats = plot_hint_confusion(b, fig_dir, win_ms=win,
                                      n_folds=cfg["decoder"]["n_folds"],
                                      seed=cfg["decoder"]["seed"])

    log.info("=== FIG: ERP averaging effect ===")
    avg_results = plot_erp_averaging_effect(b, fig_dir,
                                              ks=tuple(cfg["decoder"]["erp_avg_ks"]),
                                              win_ms=win,
                                              n_folds=cfg["decoder"]["n_folds"],
                                              seed=cfg["decoder"]["seed"])

    summary = {
        "config_file": str(Path(args.config).resolve()),
        "crop_window_s": [p["crop_tmin_s"], p["crop_tmax_s"]],
        "baseline_s": [baseline[0], baseline[1]] if baseline else None,
        "filter_band_hz": list(filter_band) if filter_band else None,
        "n_epochs": int(len(b)),
        "n_channels": int(len(b.ch_names)),
        "ch_names": list(b.ch_names),
        "sfreq": float(b.sfreq),
        "n_dropped_during_join": int(b.meta.get("n_dropped_during_join", 0)),
        "n_bad_trials_flagged": int(bad.sum()),
        "alpha_snr_post_pre_mean": float(alpha.mean()),
        "grand_average_stats": ga_stats,
        "attention_stats": att_stats,
        "alpha_desync_stats": alpha_stats,
        "target_area_regression": area_stats,
        "class_balance": balance,
        "n_hint_classes": int(len(set([h for h in b.hint if h]))),
        "n_unique_images": int(len(set(b.image_id))),
        "qc_summary": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                            "max": float(np.max(v))} for k, v in qc.items()},
        "temporal_sweep_hint": sweep_results,
        "hint_confusion": conf_stats,
        "erp_averaging_effect": avg_results,
        "elapsed_s": time.time() - t_master,
    }
    sp = fig_dir / "process1_summary.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log.info("Wrote %s", sp)
    log.info("ALL DONE in %.1fs", time.time() - t_master)


if __name__ == "__main__":
    main()
