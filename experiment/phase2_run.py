"""Phase 2 — stimulus presentation + timing/eye-tracking recording.

Loads:
  - YAML config (display, paradigm, marker, eye-tracking parameters)
  - Phase-1 selection JSON (image_id + relative_path list)

Then opens the GLFW/OpenGL display, optionally talks to the EEG TTL
serial line (controlled by `markers.serial.enabled`), records eye-tracker
samples (Tobii ET5 via the C bridge), and runs the RSVP-attention
paradigm.

Usage:
    C:\\Users\\thlab\\.conda\\envs\\VIZ\\python.exe phase2_run.py \
        --config configs/default.yaml \
        --stimuli stimuli_select/stimuli_*.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from core.display import DisplayEngine                                # noqa: E402
from core.logger import EventLogger                                   # noqa: E402
from datasets import DATASET_REGISTRY                                 # noqa: E402
from eyetracking import NullEyeTracker                                # noqa: E402
from markers.base import MarkerManager                                # noqa: E402
from markers.photodiode import PhotodiodeMarker                       # noqa: E402
from markers.serial_marker import SerialMarker                        # noqa: E402
from paradigms import PARADIGM_REGISTRY                               # noqa: E402


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_eyetracker(cfg: dict):
    et_cfg = cfg.get("eyetracking", {}) or {}
    if not et_cfg.get("enabled", False):
        return NullEyeTracker()
    device = et_cfg.get("device", "tobii_et5")
    if device == "tobii_et5":
        from eyetracking.tobii_et5 import TobiiET5
        # Backwards-compat: previously `eyetracking.monitor` was used as the
        # screen-index hint for TGI (an int). The new schema reserves
        # `monitor` as a sub-mapping of live-monitoring options. The screen
        # index now lives under `display_monitor`. Fall back gracefully.
        screen_monitor = et_cfg.get("display_monitor")
        if screen_monitor is None:
            legacy = et_cfg.get("monitor")
            if isinstance(legacy, int):
                screen_monitor = legacy
        if screen_monitor is None:
            screen_monitor = cfg.get("display", {}).get("monitor_index")
        return TobiiET5(
            dll_path=et_cfg.get("dll_path"),
            monitor=screen_monitor,
            recorder_format=str(et_cfg.get("format", "csv")),
            hdf5_chunk_size=int(et_cfg.get("hdf5_chunk_size", 2048)),
            hdf5_compression=et_cfg.get("hdf5_compression", "gzip"),
            hdf5_compression_level=int(
                et_cfg.get("hdf5_compression_level", 4)),
            hdf5_flush_every_s=float(
                et_cfg.get("hdf5_flush_every_s", 2.0)),
            stale_threshold_s=float(
                et_cfg.get("stale_threshold_s", 0.08)),
        )
    raise ValueError(f"Unknown eye-tracker device: {device}")


def find_latest_selection(default_dir: Path, exp_name: str) -> Path | None:
    """If --stimuli is not given, pick the most recent matching JSON."""
    if not default_dir.exists():
        return None
    candidates = sorted(
        p for p in default_dir.glob(f"stimuli_{exp_name}_*.json")
        if not p.stem.endswith("_stats")
    )
    return candidates[-1] if candidates else None


def main() -> None:
    setup_logging()
    log = logging.getLogger("phase2")

    ap = argparse.ArgumentParser(description="Phase 2 — RSVP attention experiment")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--stimuli", type=str, default=None,
                    help="Path to Phase-1 selection JSON. If omitted, the most "
                         "recent matching file in stimuli_select/ is used.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load config + selection, do NOT open display.")
    ap.add_argument("--coco-root", type=str, default=None,
                    help="Override dataset root recorded in the selection JSON.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    exp_name = cfg.get("experiment_name", "unnamed")
    log.info("Experiment: %s", exp_name)

    # ── Phase-1 selection ─────
    if args.stimuli:
        sel_path = Path(args.stimuli)
    else:
        sel_path = find_latest_selection(HERE / "stimuli_select", exp_name)
        if sel_path is None:
            log.error("No --stimuli given and no matching JSON in stimuli_select/. "
                      "Run phase1_select.py first.")
            sys.exit(1)
        log.info("Auto-selected most recent: %s", sel_path)

    if not sel_path.exists():
        log.error("Selection JSON not found: %s", sel_path); sys.exit(1)
    with open(sel_path, encoding="utf-8") as f:
        selection = json.load(f)

    # ── dataset bundle ─────
    ds_cfg = cfg["dataset"]
    dataset = DATASET_REGISTRY[ds_cfg["type"]]()
    # We do NOT call dataset.load() here — Phase 2 doesn't need LVIS metadata.
    coco_root_override = args.coco_root or ds_cfg.get("coco_root")
    bundle = dataset.bundle_from_selection(selection, dataset_root=coco_root_override)
    if len(bundle) == 0:
        log.error("Bundle empty — image files likely missing under %s",
                  coco_root_override); sys.exit(1)
    log.info("Bundle: %d unique images, K=%d, %d targets",
             len(bundle), bundle.K, len(bundle.targets))

    if args.dry_run:
        par_cfg = cfg.get("paradigm", {}) or {}
        if par_cfg.get("type") == "rsvp_attention":
            from paradigms.rsvp_attention import RSVPAttentionParadigm
            groups, pool_info = RSVPAttentionParadigm.plan_groups(bundle, par_cfg)
            timing_info = RSVPAttentionParadigm.estimate_timing(groups, par_cfg)
            total_presentations = sum(len(g.stimuli) for g in groups)
            log.info("Plan: %d RSVP groups, %d stimulus presentations, "
                     "%d pooled images",
                     len(groups), total_presentations,
                     pool_info["n_pool_images"])
            log.info("Plan repeats: eeg_train=%s eeg_test=%s",
                     pool_info["eeg_train_repeats_per_image_label"],
                     pool_info["eeg_test_repeats_per_image_label"])
            log.info("Plan first groups: %s",
                     pool_info.get("group_order", [])[:20])
            RSVPAttentionParadigm._log_timing_summary(timing_info)
        log.info("Dry-run complete; exiting.")
        return

    # ── Display ─────
    disp_cfg = cfg.get("display", {}) or {}
    engine = DisplayEngine(
        width=disp_cfg.get("width", 1920),
        height=disp_cfg.get("height", 1080),
        fullscreen=disp_cfg.get("fullscreen", False),
        vsync=disp_cfg.get("vsync", 1),
        warmup_frames=disp_cfg.get("warmup_frames", 10),
        window_name=f"EEG_Viz_Att - {exp_name}",
        monitor_index=disp_cfg.get("monitor_index", 0),
    )
    log.info("Display: %dx%d @ %.1f Hz",
             engine.width, engine.height, engine.refresh_hz)

    # ── Markers ─────
    mk_cfg = cfg.get("markers", {}) or {}
    marker_mgr = MarkerManager()
    pd_cfg = mk_cfg.get("photodiode", {}) or {}
    if pd_cfg.get("enabled", False):
        engine.set_photodiode(PhotodiodeMarker(size=pd_cfg.get("size", 60),
                                                corner=pd_cfg.get("corner",
                                                                    "bottom_right")))
        log.info("Photodiode: size=%d corner=%s",
                 pd_cfg.get("size", 60), pd_cfg.get("corner", "bottom_right"))

    ser_cfg = mk_cfg.get("serial", {}) or {}
    ttl_enabled = bool(ser_cfg.get("enabled", False))
    engine.set_ttl_enabled(ttl_enabled)
    if ttl_enabled:
        sm = SerialMarker(port=ser_cfg.get("port", "COM3"),
                           baudrate=ser_cfg.get("baudrate", 115200))
        marker_mgr.add_sender(sm)
        log.info("Serial TTL enabled on %s @ %d", ser_cfg.get("port"),
                 ser_cfg.get("baudrate"))
    else:
        log.info("Serial TTL DISABLED — onsets will be timed via cv2.getTickCount "
                 "(field `cv2_tick_qpc` on every stim_onset event)")

    engine.set_marker_manager(marker_mgr)

    # ── Logger ─────
    event_logger = EventLogger(engine.clock)
    engine.set_logger(event_logger)
    event_logger.log("experiment_start",
                      experiment_name=exp_name,
                      config_path=str(Path(args.config).resolve()),
                      stimuli_json=str(sel_path),
                      refresh_hz=engine.refresh_hz,
                      ttl_enabled=ttl_enabled,
                      n_targets=len(bundle.targets),
                      K=bundle.K)

    # ── Eye-tracker ─────
    eyetracker = build_eyetracker(cfg)
    log_dir = Path(cfg.get("logging", {}).get("output_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    et_fmt = str((cfg.get("eyetracking") or {}).get("format", "csv")).lower()
    suffix = ".h5" if et_fmt in {"hdf5", "h5"} else ".csv"
    eye_path = log_dir / f"eye_{exp_name}_{timestamp}{suffix}"
    eyetracker.start(eye_path)
    # Some recorders (HDF5) may have rewritten the suffix; capture the truth.
    actual_eye_path = getattr(eyetracker, "output_path", None) or eye_path
    event_logger.log(
        "eyetracker_start",
        path=str(actual_eye_path),
        format=et_fmt,
    )

    # ── Verify Tobii is actually producing data ─────
    # Distinguish two failure modes:
    #   (a) hardware not connected / DLL not loaded / no service link
    #       → fail FAST here, before we burn through subject patience
    #         on an empty recording
    #   (b) eyes briefly out of view during the session (blinks, look
    #       away) → handled later by the in-paradigm pause-on-loss
    #         watchdog (auto-resume).
    et_cfg_full = cfg.get("eyetracking", {}) or {}
    if et_cfg_full.get("enabled", False):
        verify_timeout_s = float(et_cfg_full.get("startup_verify_timeout_s", 3.0))
        log.info("Verifying Tobii data flow (up to %.1f s)...", verify_timeout_s)
        if not eyetracker.wait_for_data(timeout_s=verify_timeout_s):
            log.error(
                "❌ Tobii ET5 produced no gaze samples within %.1f s. "
                "Check: (1) device USB connected, (2) Tobii Experience "
                "running, (3) subject's eyes in tracking volume. "
                "Aborting before stimulus presentation.",
                verify_timeout_s,
            )
            try:
                diag = eyetracker.diagnostics()  # type: ignore[attr-defined]
                log.error("Tobii diagnostics: %s", diag)
            except Exception:
                pass
            event_logger.log(
                "eyetracker_no_data_at_startup",
                timeout_s=verify_timeout_s,
            )
            try:
                eyetracker.stop()
            except Exception:
                pass
            engine.close()
            marker_mgr.close()
            sys.exit(2)
        log.info("✓ Tobii data flow confirmed.")

    # ── Paradigm ─────
    par_cfg = cfg["paradigm"]
    par_type = par_cfg["type"]
    if par_type not in PARADIGM_REGISTRY:
        log.error("Unknown paradigm: %s", par_type); engine.close(); sys.exit(1)

    paradigm = PARADIGM_REGISTRY[par_type](engine, event_logger, marker_mgr)
    if hasattr(paradigm, "eyetracker"):
        paradigm.eyetracker = eyetracker

    # Forward the eyetracking.monitor block into paradigm config so the
    # paradigm can self-configure its live cursor / threshold without
    # reaching into the eyetracker object. This keeps the paradigm
    # decoupled from the specific tracker implementation.
    et_cfg = cfg.get("eyetracking", {}) or {}
    monitor_block = et_cfg.get("monitor")
    if monitor_block is not None and "monitor" not in par_cfg:
        par_cfg = dict(par_cfg)
        par_cfg["monitor"] = dict(monitor_block)

    try:
        paradigm.run(bundle, par_cfg)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        event_logger.log("experiment_end", aborted=engine.should_close())
        eyetracker.stop()
        event_logger.log("eyetracker_stop")

        log_path = log_dir / f"session_{exp_name}_{timestamp}.json"
        session_info = {
            "experiment_name": exp_name,
            "configuration": cfg,
            "stimuli_json": str(sel_path),
            "refresh_hz": engine.refresh_hz,
            "n_unique_images": len(bundle),
            "K": bundle.K,
            "targets": bundle.targets,
            "ttl_enabled": ttl_enabled,
            # Keep `eye_csv` for backward-compat with downstream tooling
            # that expects this key; also expose the format-aware path.
            "eye_csv": str(actual_eye_path),
            "eye_path": str(actual_eye_path),
            "eye_format": et_fmt,
        }
        saved = event_logger.save(log_path, session_info)
        log.info("Session log saved: %s", saved)

        engine.close()
        marker_mgr.close()


if __name__ == "__main__":
    main()
