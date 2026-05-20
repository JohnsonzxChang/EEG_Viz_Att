"""Phase 1 — stimulus selection.

Reads the YAML config's `dataset` block, runs LVIS filtering once, and
emits a JSON file containing only image_id + relative_path (relative to
the dataset root) plus the metadata needed by Phase 2 (targets per image,
bboxes, areas).

The JSON file is the single source of truth handed to Phase 2 — that way
the expensive filtering pass runs once and is reproducible.

Usage:
    C:\\Users\\thlab\\.conda\\envs\\VIZ\\python.exe phase1_select.py \
        --config configs/default.yaml \
        --out stimuli_select/stimuli_<name>_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from datasets import DATASET_REGISTRY  # noqa: E402


def _short_label(label: str, max_len: int = 30) -> str:
    return label if len(label) <= max_len else label[:max_len - 1] + "~"


def compute_multilabel_stats(selection: dict[str, Any]) -> dict[str, Any]:
    targets = list(selection.get("targets", []))
    target_index = {t: i for i, t in enumerate(targets)}
    target_counts: Counter[str] = Counter({t: 0 for t in targets})
    eeg_split_counts: Counter[str] = Counter()
    target_counts_by_eeg_split: dict[str, Counter[str]] = {}
    label_count_distribution: Counter[str] = Counter()
    matrix = [[0 for _ in targets] for _ in targets]

    for rec in selection.get("items", []):
        eeg_split = str(rec.get("eeg_split", "unknown"))
        eeg_split_counts[eeg_split] += 1
        if eeg_split not in target_counts_by_eeg_split:
            target_counts_by_eeg_split[eeg_split] = Counter({t: 0 for t in targets})
        labels = sorted(
            {t for t in rec.get("targets", []) if t in target_index},
            key=lambda t: target_index[t],
        )
        label_count_distribution[str(len(labels))] += 1
        for t in labels:
            target_counts[t] += 1
            target_counts_by_eeg_split[eeg_split][t] += 1
        for a, b in combinations(labels, 2):
            ia = target_index[a]
            ib = target_index[b]
            matrix[ia][ib] += 1
            matrix[ib][ia] += 1

    for t, i in target_index.items():
        matrix[i][i] = target_counts[t]

    count_values = [target_counts[t] for t in targets]
    if count_values:
        mean_count = sum(count_values) / len(count_values)
        std_count = math.sqrt(
            sum((x - mean_count) ** 2 for x in count_values) / len(count_values)
        )
        nonzero = [x for x in count_values if x > 0]
        min_nonzero = min(nonzero) if nonzero else 0
        max_count = max(count_values)
    else:
        mean_count = 0.0
        std_count = 0.0
        min_nonzero = 0
        max_count = 0

    top_pairs = []
    for i, a in enumerate(targets):
        for j in range(i + 1, len(targets)):
            b = targets[j]
            c = matrix[i][j]
            if c > 0:
                top_pairs.append({"pair": [a, b], "count": c})
    top_pairs.sort(key=lambda x: (-int(x["count"]), x["pair"][0], x["pair"][1]))

    return {
        "n_unique_images": int(selection.get("n_unique_images", 0)),
        "K": int(selection.get("K", 0)),
        "targets": targets,
        "target_counts": {t: int(target_counts[t]) for t in targets},
        "eeg_split_counts": {
            k: int(eeg_split_counts[k]) for k in sorted(eeg_split_counts)
        },
        "target_counts_by_eeg_split": {
            split: {t: int(counter[t]) for t in targets}
            for split, counter in sorted(target_counts_by_eeg_split.items())
        },
        "target_count_summary": {
            "min": int(min(count_values)) if count_values else 0,
            "min_nonzero": int(min_nonzero),
            "max": int(max_count),
            "mean": round(mean_count, 3),
            "std": round(std_count, 3),
            "max_to_min_nonzero": round(max_count / min_nonzero, 3)
            if min_nonzero else None,
        },
        "label_count_distribution": {
            k: int(label_count_distribution[k])
            for k in sorted(label_count_distribution, key=lambda x: int(x))
        },
        "cooccurrence_matrix": matrix,
        "top_cooccurrences": top_pairs[:40],
    }


def draw_multilabel_stats_png(stats: dict[str, Any], out_path: Path) -> None:
    import cv2
    import numpy as np

    targets = list(stats["targets"])
    counts = stats["target_counts"]
    matrix = stats["cooccurrence_matrix"]
    if not targets:
        return

    canvas = np.full((1800, 2400, 3), 255, dtype=np.uint8)
    ink = (35, 35, 35)
    muted = (115, 115, 115)
    grid = (225, 225, 225)
    blue = (210, 125, 45)
    warn = (65, 105, 210)

    def put(text: str, x: int, y: int, scale: float = 0.48,
            color: tuple[int, int, int] = ink, thick: int = 1) -> None:
        cv2.putText(
            canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thick, cv2.LINE_AA,
        )

    summary = stats["target_count_summary"]
    put("Phase 1 multi-label selection statistics", 55, 58, 0.9, ink, 2)
    put(
        f"images={stats['n_unique_images']}  K={stats['K']}  "
        f"targets={len(targets)}  count range={summary['min']}..{summary['max']}  "
        f"max/min_nonzero={summary['max_to_min_nonzero']}",
        55, 92, 0.48, muted, 1,
    )

    ordered = sorted(targets, key=lambda t: (-int(counts[t]), t))
    max_count = max(int(counts[t]) for t in targets) or 1
    mean_count = float(summary["mean"])
    std_count = float(summary["std"])

    bar_x = 430
    bar_y = 150
    bar_w = 650
    row_h = 24
    put("Per-target image counts", 55, 130, 0.62, ink, 2)
    for k in range(0, max_count + 1, max(1, max_count // 4)):
        x = bar_x + int(k / max_count * bar_w)
        cv2.line(canvas, (x, bar_y - 16), (x, bar_y + row_h * len(ordered)),
                 grid, 1)
        put(str(k), x - 12, bar_y - 24, 0.34, muted, 1)

    for idx, t in enumerate(ordered):
        y = bar_y + idx * row_h
        c = int(counts[t])
        color = warn if c > mean_count + 2 * std_count else blue
        put(f"{idx + 1:02d} {_short_label(t, 32)}", 55, y + 6, 0.38, ink, 1)
        cv2.rectangle(canvas, (bar_x, y - 10),
                      (bar_x + int(c / max_count * bar_w), y + 7),
                      color, -1)
        put(str(c), bar_x + int(c / max_count * bar_w) + 8, y + 6,
            0.36, ink, 1)

    dist = stats["label_count_distribution"]
    dist_x = 55
    dist_y = 1185
    put("Labels per selected image", dist_x, dist_y, 0.62, ink, 2)
    if dist:
        dist_max = max(int(v) for v in dist.values()) or 1
        x0 = dist_x
        y0 = dist_y + 260
        slot_w = 95
        for idx, key in enumerate(sorted(dist, key=lambda x: int(x))):
            val = int(dist[key])
            bar_h = int(val / dist_max * 210)
            x = x0 + idx * slot_w
            cv2.rectangle(canvas, (x, y0 - bar_h), (x + 52, y0),
                          (75, 155, 95), -1)
            put(str(val), x - 2, y0 - bar_h - 10, 0.36, ink, 1)
            put(f"{key} lbl", x - 2, y0 + 24, 0.36, muted, 1)

    heat_x = 1210
    heat_y = 150
    n = len(targets)
    cell = max(12, min(22, 900 // n))
    heat_size = cell * n
    target_to_index = {t: i for i, t in enumerate(targets)}
    max_pair = 1
    for a in targets:
        ia = target_to_index[a]
        for b in targets:
            ib = target_to_index[b]
            if ia != ib:
                max_pair = max(max_pair, int(matrix[ia][ib]))

    put("Target co-occurrence matrix", heat_x, 130, 0.62, ink, 2)
    put("axis numbers match the ranked target list on the left",
        heat_x, heat_y + heat_size + 34, 0.4, muted, 1)
    for r, a in enumerate(ordered):
        ia = target_to_index[a]
        y = heat_y + r * cell
        put(str(r + 1), heat_x - 34, y + cell - 4, 0.32, muted, 1)
        put(str(r + 1), heat_x + r * cell + 2, heat_y - 10, 0.28, muted, 1)
        for c, b in enumerate(ordered):
            ib = target_to_index[b]
            x = heat_x + c * cell
            if ia == ib:
                color = (235, 235, 235)
            else:
                norm = math.sqrt(int(matrix[ia][ib]) / max_pair)
                shade = int(255 - 205 * norm)
                color = (255, shade, shade)
            cv2.rectangle(canvas, (x, y), (x + cell - 1, y + cell - 1),
                          color, -1)
    cv2.rectangle(canvas, (heat_x, heat_y),
                  (heat_x + heat_size, heat_y + heat_size), ink, 1)

    pairs_x = heat_x
    pairs_y = heat_y + heat_size + 86
    put("Top co-occurring target pairs", pairs_x, pairs_y, 0.62, ink, 2)
    for idx, rec in enumerate(stats["top_cooccurrences"][:18]):
        a, b = rec["pair"]
        put(f"{idx + 1:02d}. {_short_label(a, 21)} + {_short_label(b, 21)}",
            pairs_x, pairs_y + 34 + idx * 24, 0.38, ink, 1)
        put(str(rec["count"]), pairs_x + 610, pairs_y + 34 + idx * 24,
            0.38, ink, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def write_multilabel_report(selection: dict[str, Any],
                            out_path: Path) -> tuple[Path, Path]:
    stats = compute_multilabel_stats(selection)
    stats_path = out_path.with_name(f"{out_path.stem}_stats.json")
    plot_path = out_path.with_name(f"{out_path.stem}_stats.png")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    draw_multilabel_stats_png(stats, plot_path)
    return stats_path, plot_path


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    setup_logging()
    log = logging.getLogger("phase1")

    ap = argparse.ArgumentParser(description="Phase 1 — stimulus selection")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Default: "
                         "stimuli_select/stimuli_<exp>_<ts>.json")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", "unnamed")
    ds_cfg = cfg["dataset"]
    ds_type = ds_cfg["type"]
    if ds_type not in DATASET_REGISTRY:
        log.error("Unknown dataset type: %s", ds_type); sys.exit(1)

    dataset = DATASET_REGISTRY[ds_type]()
    dataset.load(ds_cfg)
    selection = dataset.select_stimuli(ds_cfg)

    if not selection["items"]:
        log.error("Selection empty — relax the filter thresholds.")
        sys.exit(1)

    # default output path
    if args.out:
        out_path = Path(args.out)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = HERE / "stimuli_select" / f"stimuli_{exp_name}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selection["meta"] = {
        "experiment_name": exp_name,
        "config_path": str(Path(args.config).resolve()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

    stats_path, plot_path = write_multilabel_report(selection, out_path)

    log.info("Phase 1 complete:")
    log.info("  unique images = %d", selection["n_unique_images"])
    log.info("  K              = %d", selection["K"])
    log.info("  targets        = %d", len(selection["targets"]))
    log.info("  output JSON    = %s", out_path)
    log.info("  stats JSON     = %s", stats_path)
    log.info("  stats PNG      = %s", plot_path)


if __name__ == "__main__":
    main()
