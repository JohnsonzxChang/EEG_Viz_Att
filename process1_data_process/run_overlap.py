"""Plan B + Plan C driver — single --config YAML.

    python -B -m process1_data_process.run_overlap \
        --config process1_data_process/configs/process1_overlap.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from process1_data_process.config import load_config, dump_resolved_config
from process1_data_process.data_io import load_epochs
from process1_data_process.overlap_correction import (
    plot_plan_b_filter_vs_baseline, plot_plan_c_rerp,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")
    log = logging.getLogger("overlap")
    t_master = time.time()

    fig_dir = Path(cfg["output"]["fig_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    dump_resolved_config(cfg, fig_dir / "_resolved_config.yaml")

    summary = {"config_file": str(Path(args.config).resolve()),
                "soa_ms": cfg["rerp"]["soa_ms"],
                "ridge": cfg["rerp"]["ridge"]}

    fif = cfg["dataset"]["fif"]
    sjs = cfg["dataset"]["session_json"]

    if cfg["output"].get("enable_plan_b", True):
        log.info("=== PLAN B: bandpass vs short-baseline ===")
        b_band = cfg["plan_b"]["filter_band"]
        plot_plan_b_filter_vs_baseline(
            fif, sjs, fig_dir,
            filter_band=(b_band["lo_hz"], b_band["hi_hz"]))
        summary["plan_b"] = {"filter_band": [b_band["lo_hz"], b_band["hi_hz"]]}

    if cfg["output"].get("enable_plan_c", True):
        log.info("=== PLAN C: rERP overlap deconvolution ===")
        # Plan C uses bandpass-filtered, no-baseline data
        b_filt = cfg["preproc"]["filter_band"]
        b = load_epochs(fif, sjs, pick_eeg=cfg["preproc"]["pick_eeg"],
                        load_data=True, decim=cfg["preproc"]["decim"],
                        dtype=cfg["preproc"]["dtype"],
                        crop_tmin=cfg["preproc"]["crop_tmin_s"],
                        crop_tmax=cfg["preproc"]["crop_tmax_s"],
                        baseline=None,
                        filter_band=(b_filt["lo_hz"], b_filt["hi_hz"]))
        log.info("Loaded for rERP: %d epochs, %d ch, %d samples",
                  len(b), len(b.ch_names), len(b.times))
        r = cfg["rerp"]
        stats = plot_plan_c_rerp(b, fig_dir,
                                   soa_ms=r["soa_ms"],
                                   ridge=r["ridge"],
                                   fif_for_info=fif,
                                   resp_tmin_ms=r["resp_tmin_ms"],
                                   resp_tmax_ms=r["resp_tmax_ms"])
        summary["plan_c"] = stats

    sp = fig_dir / "overlap_summary.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log.info("Wrote %s", sp)
    log.info("ALL DONE in %.1fs", time.time() - t_master)


if __name__ == "__main__":
    main()
