r"""Standalone Tobii ET5 debug flow.

Checks the complete eye-tracking path used by Phase 2:
  1. bridge DLL/config loading;
  2. device/API initialization;
  3. recording start and marker insertion;
  4. gaze/head-pose CSV writing;
  5. recorded sample counts, rates, gaps, and marker rows.

Usage:
    C:\Users\thlab\.conda\envs\VIZ\python.exe debug_eyetracking.py \
        --config configs/default.yaml --duration 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from core.clock import PerfClock  # noqa: E402
from core.logger import EventLogger  # noqa: E402
from eyetracking.tobii_et5 import TobiiET5  # noqa: E402


EXPECTED_COLUMNS = [
    "tick_qpc_s",
    "kind",
    "gaze_x",
    "gaze_y",
    "gaze_origin_x",
    "gaze_origin_y",
    "gaze_origin_z",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "head_x",
    "head_y",
    "head_z",
    "validity",
    "tag",
]


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_output_dir_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".eye_debug_write_test_{os.getpid()}.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise PermissionError(f"Output directory is not writable: {path}") from exc
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def span_seconds(ticks: list[float]) -> float:
    if len(ticks) < 2:
        return 0.0
    return max(ticks) - min(ticks)


def max_gap_seconds(ticks: list[float]) -> float:
    if len(ticks) < 2:
        return 0.0
    ordered = sorted(ticks)
    return max(b - a for a, b in zip(ordered, ordered[1:]))


def range_or_none(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [min(values), max(values)]


def analyze_eye_csv(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "csv_path": str(path),
        "csv_exists": path.exists(),
        "csv_bytes": path.stat().st_size if path.exists() else 0,
        "header_ok": False,
        "missing_columns": EXPECTED_COLUMNS,
        "total_rows": 0,
        "kind_counts": {},
        "gaze_rows": 0,
        "head_pose_rows": 0,
        "mark_rows": 0,
        "unknown_rows": 0,
        "bad_timestamp_rows": 0,
        "gaze_rate_hz": 0.0,
        "head_pose_rate_hz": 0.0,
        "recorded_span_s": 0.0,
        "gaze_span_s": 0.0,
        "max_gaze_gap_s": 0.0,
        "gaze_x_range": None,
        "gaze_y_range": None,
        "gaze_in_bounds_fraction": None,
        "tags": [],
    }
    if not path.exists() or path.stat().st_size == 0:
        return summary

    all_ticks: list[float] = []
    gaze_ticks: list[float] = []
    head_ticks: list[float] = []
    gaze_x: list[float] = []
    gaze_y: list[float] = []
    in_bounds = 0
    kind_counts: Counter[str] = Counter()
    tags: list[str] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [col for col in EXPECTED_COLUMNS if col not in fieldnames]
        summary["header_ok"] = not missing
        summary["missing_columns"] = missing

        for row in reader:
            if not any(row.values()):
                continue
            summary["total_rows"] += 1
            kind = (row.get("kind") or "").strip()
            kind_counts[kind or "<empty>"] += 1

            tick = parse_float(row.get("tick_qpc_s"))
            if tick is None:
                summary["bad_timestamp_rows"] += 1
            else:
                all_ticks.append(tick)

            if kind == "mark":
                summary["mark_rows"] += 1
                tag = (row.get("tag") or "").strip()
                if tag:
                    tags.append(tag)
                continue

            if kind != "gaze":
                summary["unknown_rows"] += 1
                continue

            validity = parse_int(row.get("validity"))
            if validity == 1:
                summary["gaze_rows"] += 1
                if tick is not None:
                    gaze_ticks.append(tick)
                gx = parse_float(row.get("gaze_x"))
                gy = parse_float(row.get("gaze_y"))
                if gx is not None and gy is not None:
                    gaze_x.append(gx)
                    gaze_y.append(gy)
                    if -1.05 <= gx <= 1.05 and -1.05 <= gy <= 1.05:
                        in_bounds += 1
            elif validity == 2:
                summary["head_pose_rows"] += 1
                if tick is not None:
                    head_ticks.append(tick)
            else:
                summary["unknown_rows"] += 1

    recorded_span = span_seconds(all_ticks)
    gaze_span = span_seconds(gaze_ticks)
    head_span = span_seconds(head_ticks)
    summary["kind_counts"] = dict(kind_counts)
    summary["recorded_span_s"] = recorded_span
    summary["gaze_span_s"] = gaze_span
    summary["max_gaze_gap_s"] = max_gap_seconds(gaze_ticks)
    summary["gaze_rate_hz"] = (
        len(gaze_ticks) / gaze_span if gaze_span > 0 else 0.0
    )
    summary["head_pose_rate_hz"] = (
        len(head_ticks) / head_span if head_span > 0 else 0.0
    )
    summary["gaze_x_range"] = range_or_none(gaze_x)
    summary["gaze_y_range"] = range_or_none(gaze_y)
    summary["gaze_in_bounds_fraction"] = (
        in_bounds / len(gaze_x) if gaze_x else None
    )
    summary["tags"] = tags
    return summary


def log_check(log: logging.Logger, ok: bool, name: str, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    level = logging.INFO if ok else logging.ERROR
    log.log(level, "[%s] %s - %s", status, name, detail)


def run_recording(
    *,
    tracker: TobiiET5,
    eye_csv: Path,
    duration_s: float,
    mark_interval_s: float,
    event_logger: EventLogger,
    log: logging.Logger,
) -> int:
    tracker.start(eye_csv)
    if not tracker.is_started:
        return 0

    written_marks = 0

    def mark(tag: str) -> None:
        nonlocal written_marks
        tracker.mark(tag)
        written_marks += 1
        event_logger.log(
            "eyetracker_debug_mark",
            tag=tag,
            bridge_qpc=tracker.now_qpc_seconds(),
        )

    start_tick = time.perf_counter()
    event_logger.log(
        "eyetracker_debug_recording_start",
        csv=str(eye_csv),
        duration_s=duration_s,
        mark_interval_s=mark_interval_s,
        bridge_qpc=tracker.now_qpc_seconds(),
    )
    mark("debug_start")

    next_mark = start_tick + max(0.1, mark_interval_s)
    next_status = start_tick + 1.0
    mark_index = 0
    deadline = start_tick + duration_s

    log.info(
        "Recording for %.1fs. Move gaze across the screen and blink naturally.",
        duration_s,
    )
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        if now >= next_mark:
            mark_index += 1
            mark(f"debug_tick_{mark_index:03d}")
            next_mark += max(0.1, mark_interval_s)
        if now >= next_status:
            elapsed = now - start_tick
            log.info("Recording %.1f/%.1fs, debug marks=%d",
                     elapsed, duration_s, written_marks)
            next_status += 1.0
        time.sleep(0.02)

    mark("debug_end")
    event_logger.log(
        "eyetracker_debug_recording_end",
        elapsed_s=time.perf_counter() - start_tick,
        debug_marks=written_marks,
        bridge_qpc=tracker.now_qpc_seconds(),
    )
    return written_marks


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Debug Tobii ET5 online status, data reception, and CSV recording."
    )
    ap.add_argument("--config", type=str, default=str(HERE / "configs" / "default.yaml"))
    ap.add_argument("--duration", type=float, default=10.0,
                    help="Recording duration in seconds.")
    ap.add_argument("--mark-interval", type=float, default=1.0,
                    help="Insert a debug marker every N seconds.")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Override output directory for debug CSV/JSON files.")
    ap.add_argument("--dll-path", type=str, default=None,
                    help="Override Tobii bridge DLL path.")
    ap.add_argument("--monitor", type=int, default=None,
                    help="Monitor index for TrackRectangle (0=primary, 1=secondary). "
                         "Defaults to display.monitor_index from config.")
    ap.add_argument("--min-gaze-samples", type=int, default=30,
                    help="Minimum gaze samples required for PASS.")
    ap.add_argument("--min-gaze-rate-hz", type=float, default=20.0,
                    help="Minimum gaze sample rate required for PASS.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("eye_debug")

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    exp_name = cfg.get("experiment_name", "unnamed")
    et_cfg = cfg.get("eyetracking", {}) or {}
    if et_cfg.get("enabled") is False:
        log.warning("eyetracking.enabled is false in config; debug still probes the device.")
    device = et_cfg.get("device", "tobii_et5")
    if device != "tobii_et5":
        log.error("Only tobii_et5 is supported by this debug script, got %r", device)
        sys.exit(2)

    output_dir = Path(
        args.output_dir
        or cfg.get("logging", {}).get("output_dir", "logs")
    )
    ensure_output_dir_writable(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eye_csv = output_dir / f"eye_debug_{exp_name}_{timestamp}.csv"
    events_json = output_dir / f"eye_debug_events_{exp_name}_{timestamp}.json"
    summary_json = output_dir / f"eye_debug_summary_{exp_name}_{timestamp}.json"

    event_logger = EventLogger(PerfClock())
    dll_path = args.dll_path if args.dll_path is not None else et_cfg.get("dll_path")
    monitor = args.monitor
    if monitor is None:
        monitor = et_cfg.get("monitor")
    if monitor is None:
        monitor = cfg.get("display", {}).get("monitor_index")
    event_logger.log(
        "eyetracker_debug_start",
        config_path=str(cfg_path),
        dll_path=dll_path,
        monitor=monitor,
        output_csv=str(eye_csv),
    )

    log.info("Config: %s", cfg_path)
    log.info("Output CSV: %s", eye_csv)
    tracker = TobiiET5(dll_path=dll_path, monitor=monitor)
    diagnostics = tracker.diagnostics()
    event_logger.log("eyetracker_debug_diagnostics", **diagnostics)

    log_check(
        log,
        bool(diagnostics["dll_exists"]),
        "bridge DLL exists",
        str(diagnostics["dll_path"]),
    )
    log_check(
        log,
        bool(diagnostics["dll_loaded"]),
        "bridge DLL loaded",
        "loaded" if diagnostics["dll_loaded"]
        else str(diagnostics.get("load_error") or "not loaded"),
    )
    log_check(
        log,
        tracker.is_available and tracker.init_rc == 0,
        "device/API online",
        f"tb_init rc={tracker.init_rc}",
    )
    if not tracker.is_available:
        summary = {
            "diagnostics": diagnostics,
            "analysis": analyze_eye_csv(eye_csv),
            "passed": False,
            "failure": "Tobii bridge/API is not available.",
        }
        summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        event_logger.log("eyetracker_debug_failed", reason=summary["failure"])
        event_logger.save(events_json, {"experiment_name": exp_name})
        log.error("Debug failed before recording. Summary: %s", summary_json)
        sys.exit(2)

    py_qpc = time.perf_counter()
    bridge_qpc = tracker.now_qpc_seconds()
    qpc_delta = abs(py_qpc - bridge_qpc)
    event_logger.log(
        "eyetracker_debug_clock_check",
        python_perf_counter=py_qpc,
        bridge_qpc=bridge_qpc,
        abs_delta_s=qpc_delta,
    )
    log.info("Clock check: |python_qpc - bridge_qpc| = %.6fs", qpc_delta)

    written_marks = 0
    try:
        written_marks = run_recording(
            tracker=tracker,
            eye_csv=eye_csv,
            duration_s=max(0.1, args.duration),
            mark_interval_s=max(0.1, args.mark_interval),
            event_logger=event_logger,
            log=log,
        )
        log_check(
            log,
            tracker.last_start_rc == 0 and written_marks > 0,
            "recording started",
            f"tb_start rc={tracker.last_start_rc}, debug marks={written_marks}",
        )
    finally:
        tracker.stop()
        event_logger.log("eyetracker_debug_stop", diagnostics=tracker.diagnostics())

    analysis = analyze_eye_csv(eye_csv)
    expected_debug_marks = written_marks
    checks = {
        "csv_written": (
            analysis["csv_exists"]
            and analysis["csv_bytes"] > 0
            and analysis["header_ok"]
        ),
        "gaze_received": analysis["gaze_rows"] >= args.min_gaze_samples,
        "gaze_rate_ok": analysis["gaze_rate_hz"] >= args.min_gaze_rate_hz,
        "markers_recorded": analysis["mark_rows"] >= expected_debug_marks,
    }
    passed = all(checks.values())

    log_check(
        log,
        checks["csv_written"],
        "CSV recording",
        f"{analysis['csv_bytes']} bytes, header_ok={analysis['header_ok']}",
    )
    log_check(
        log,
        checks["gaze_received"],
        "gaze data received",
        f"{analysis['gaze_rows']} samples "
        f"(required >= {args.min_gaze_samples})",
    )
    log_check(
        log,
        checks["gaze_rate_ok"],
        "gaze sample rate",
        f"{analysis['gaze_rate_hz']:.1f} Hz "
        f"(required >= {args.min_gaze_rate_hz:.1f} Hz)",
    )
    log_check(
        log,
        checks["markers_recorded"],
        "marker rows recorded",
        f"{analysis['mark_rows']} rows, expected >= {expected_debug_marks}",
    )
    log.info(
        "CSV summary: gaze=%d head_pose=%d marks=%d span=%.2fs max_gap=%.3fs",
        analysis["gaze_rows"],
        analysis["head_pose_rows"],
        analysis["mark_rows"],
        analysis["recorded_span_s"],
        analysis["max_gaze_gap_s"],
    )
    log.info("Gaze x range=%s y range=%s in_bounds=%s",
             analysis["gaze_x_range"],
             analysis["gaze_y_range"],
             analysis["gaze_in_bounds_fraction"])

    summary = {
        "passed": passed,
        "checks": checks,
        "diagnostics": diagnostics,
        "analysis": analysis,
        "config_path": str(cfg_path),
        "events_json": str(events_json),
        "summary_json": str(summary_json),
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    event_logger.log("eyetracker_debug_summary", **summary)
    event_logger.save(events_json, {"experiment_name": exp_name})

    if passed:
        log.info("Eye-tracking debug PASSED. Summary: %s", summary_json)
        sys.exit(0)

    log.error("Eye-tracking debug FAILED. Summary: %s", summary_json)
    sys.exit(1)


if __name__ == "__main__":
    main()
