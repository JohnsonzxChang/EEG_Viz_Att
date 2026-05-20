"""LVIS-on-COCO dataset adapter for the multi-target attention RSVP paradigm.

LVIS schema (v1):
    categories[]: {id, name, frequency('r'/'c'/'f'), synonyms, image_count, instance_count}
    images[]:     {id, height, width, coco_url, ...}
    annotations[]:{id, image_id, category_id, area, bbox=[x,y,w,h], segmentation}

Two-phase API:
    select_stimuli(cfg)        -> dict (Phase 1: filter + JSON-serialise)
    bundle_from_selection(...) -> StimulusBundle (Phase 2: rebuild in memory)

Selection rule:
  For each (image, target_category) pair, the category is *qualifying* iff
  it has exactly `instances_per_target` instances (default 1, the
  uniqueness rule that prevents subject confusion) AND every instance has
  bbox area fraction in [min_area_frac, max_area_frac]. Image is kept iff
  qualifying-count >= K (default 2 — see notes/design.md for why K>=4 is
  effectively unattainable on natural LVIS data).
"""

from __future__ import annotations

import json
import logging
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets.base import ImageDataset, ImageItem, StimulusBundle

log = logging.getLogger(__name__)


# 40 perceptually-distinct LVIS categories. All are LVIS 'f' (frequent),
# validated against lvis_v1_val.json. They span man-made vs natural and
# animate vs inanimate. No human faces.
DEFAULT_TARGETS_40: list[str] = [
    # Animals (animate, natural) — 10
    "dog", "cat", "horse", "cow", "sheep",
    "bird", "elephant", "zebra", "giraffe", "bear",
    # Plants / food (natural, inanimate) — 8
    "apple", "banana", "broccoli", "carrot",
    "orange_(fruit)", "tomato", "doughnut", "pizza",
    # Vehicles (man-made, inanimate, large) — 5
    "car_(automobile)", "boat", "motorcycle", "bicycle", "airplane",
    # Furniture (man-made, inanimate, large) — 5
    "chair", "sofa", "dining_table", "bed", "cabinet",
    # Kitchen / containers (man-made, inanimate, small) — 4
    "bottle", "cup", "bowl", "glass_(drink_container)",
    # Wearables / accessories (man-made, inanimate, small) — 4
    "shoe", "hat", "umbrella", "backpack",
    # Electronics (man-made, inanimate, mid) — 4
    "laptop_computer", "cellular_telephone", "television_set", "remote_control",
]


class LVISCOCODataset(ImageDataset):

    def __init__(self) -> None:
        self._splits_loaded: dict[str, dict[str, Any]] = {}
        self._coco_root: Path | None = None
        self._cat_id_to_name: dict[int, str] = {}
        self._cat_name_to_id: dict[str, int] = {}

    # ── load ──────────────────────────────────────────────────────────

    def load(self, config: dict[str, Any]) -> None:
        coco_root = Path(config["coco_root"])
        if not coco_root.exists():
            raise FileNotFoundError(f"COCO root not found: {coco_root}")
        self._coco_root = coco_root

        ann_paths = config.get("lvis_annotations", {})
        for split in ("train", "val"):
            p = ann_paths.get(split)
            if not p:
                continue
            data = self._load_lvis_json(Path(p))
            self._splits_loaded[split] = data
            log.info("Loaded LVIS %s: %d images, %d annotations, %d categories",
                     split, len(data["images"]), len(data["annotations"]),
                     len(data["categories"]))
            for c in data["categories"]:
                self._cat_id_to_name[c["id"]] = c["name"]
                self._cat_name_to_id[c["name"]] = c["id"]
            
        if not self._splits_loaded:
            raise ValueError("No LVIS annotation file was loaded — check `lvis_annotations` config.")

    @staticmethod
    def _load_lvis_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"LVIS file not found: {path}")
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                inner = [n for n in z.namelist() if n.endswith(".json")][0]
                with z.open(inner) as f:
                    return json.load(f)
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ── inspection ────────────────────────────────────────────────────

    def get_categories(self) -> list[str]:
        return sorted(self._cat_name_to_id.keys())

    # ── Phase 1: filter + serialise ───────────────────────────────────

    def select_stimuli(self, config: dict[str, Any]) -> dict[str, Any]:
        assert self._coco_root is not None
        kept_items, used_targets, K, balance_info = self._select_internal(config)
        eeg_split_info = self._assign_eeg_splits(
            kept_items,
            used_targets,
            test_fraction=float(config.get("eeg_test_fraction", 0.2)),
            split_seed=int(config.get("eeg_split_seed",
                                      config.get("shuffle_seed", 42))),
        )

        coco_root = str(Path(self._coco_root).resolve()).replace("\\", "/")
        items_payload = []
        for it in kept_items:
            items_payload.append({
                "image_id": it.image_id,
                "relative_path": it.relative_path,
                "split": it.split,
                "eeg_split": it.metadata.get("eeg_split", "train"),
                "targets": list(it.targets),
                "target_areas": {k: round(v, 6) for k, v in it.target_areas.items()},
                "bboxes": {k: list(v) for k, v in it.bboxes.items()},
                "segmentations": it.metadata.get("segmentations", {}),
                "all_target_areas": it.metadata.get("all_target_areas", {}),
                "all_bboxes": it.metadata.get("all_bboxes", {}),
                "all_segmentations": it.metadata.get("all_segmentations", {}),
                "target_instance_counts": it.metadata.get("target_instance_counts", {}),
                "n_qualifying": int(it.metadata.get("n_qualifying", len(it.targets))),
                "img_h": int(it.metadata.get("img_h", 0)),
                "img_w": int(it.metadata.get("img_w", 0)),
            })

        return {
            "selection_config": {
                "target_categories": list(used_targets),
                "K": K,
                "instances_per_target": config.get("instances_per_target"),
                "min_area_frac": float(config.get("min_area_frac", 0.005)),
                "max_area_frac": float(config.get("max_area_frac", 0.60)),
                "max_images_per_target": config.get("max_images_per_target"),
                "balance_selection": bool(config.get("balance_selection", True)),
                "preferred_targets_per_image": int(
                    config.get("preferred_targets_per_image", 4)),
                "target_images_per_target": config.get("target_images_per_target"),
                "eeg_test_fraction": float(config.get("eeg_test_fraction", 0.2)),
                "eeg_split_seed": int(config.get(
                    "eeg_split_seed", config.get("shuffle_seed", 42))),
                "splits": list(config.get("splits") or ["val", "train"]),
                "shuffle_seed": config.get("shuffle_seed", 42),
            },
            "balance": balance_info,
            "eeg_split": eeg_split_info,
            "dataset_root": coco_root,
            "K": K,
            "targets": list(used_targets),
            "n_unique_images": len(items_payload),
            "items": items_payload,
        }

    # ── Phase 2: rebuild bundle from JSON ─────────────────────────────

    def bundle_from_selection(self, selection: dict[str, Any],
                               dataset_root: str | None = None) -> StimulusBundle:
        targets: list[str] = list(selection["targets"])
        K: int = int(selection["K"])
        root = Path(dataset_root or selection.get("dataset_root", "."))

        items: list[ImageItem] = []
        n_missing = 0
        for rec in selection["items"]:
            rel = rec["relative_path"]
            abs_path = root / rel
            if not abs_path.exists():
                n_missing += 1
                continue
            items.append(ImageItem(
                image_id=str(rec["image_id"]),
                file_path=abs_path,
                relative_path=rel,
                targets=list(rec["targets"]),
                target_areas={k: float(v) for k, v in rec["target_areas"].items()},
                bboxes={k: [float(x) for x in v] for k, v in rec["bboxes"].items()},
                split=str(rec.get("split", "")),
                metadata={"img_h": rec.get("img_h", 0),
                          "img_w": rec.get("img_w", 0),
                          "eeg_split": rec.get("eeg_split", "train"),
                          "segmentations": rec.get("segmentations", {}),
                          "all_target_areas": rec.get("all_target_areas", {}),
                          "all_bboxes": rec.get("all_bboxes", {}),
                          "all_segmentations": rec.get("all_segmentations", {}),
                          "target_instance_counts": rec.get("target_instance_counts", {}),
                          "n_qualifying": rec.get("n_qualifying", len(rec["targets"]))},
            ))
        if n_missing:
            log.warning("Phase2: %d/%d images missing on disk under %s",
                        n_missing, len(selection["items"]), root)

        images_by_target: dict[str, list[ImageItem]] = {t: [] for t in targets}
        for it in items:
            for t in it.targets:
                if t in images_by_target:
                    images_by_target[t].append(it)

        return StimulusBundle(targets=targets, K=K,
                               images_by_target=images_by_target,
                               all_images=items)

    # ── shared filter pass ────────────────────────────────────────────

    def _select_internal(
        self, config: dict[str, Any]
    ) -> tuple[list[ImageItem], list[str], int, dict[str, Any]]:
        cfg = config
        targets: list[str] = list(cfg.get("target_categories") or DEFAULT_TARGETS_40)
        K: int = int(cfg.get("targets_per_image", 2))
        instances_raw = cfg.get("instances_per_target")
        if instances_raw is None or int(instances_raw) <= 0:
            instances_per_target: int | None = None
        else:
            instances_per_target = int(instances_raw)
        min_area_frac: float = float(cfg.get("min_area_frac", 0.005))
        max_area_frac: float = float(cfg.get("max_area_frac", 0.60))
        max_per_target: int | None = cfg.get("max_images_per_target")
        target_per_target: int | None = cfg.get("target_images_per_target")
        if target_per_target is None:
            target_per_target = max_per_target
        preferred_targets_per_image: int = int(
            cfg.get("preferred_targets_per_image", 4))
        balance_selection: bool = bool(cfg.get("balance_selection", True))
        min_images_warn: int = int(cfg.get("min_images_per_target_warning", 15))
        shuffle_seed: int | None = cfg.get("shuffle_seed", 42)
        splits_to_use: list[str] = list(cfg.get("splits") or ["val", "train"])

        target_ids: set[int] = set()
        valid_targets: list[str] = []
        for name in targets:
            cid = self._cat_name_to_id.get(name)
            if cid is None:
                log.warning("Target category %r not in LVIS; skipping", name)
            else:
                target_ids.add(cid)
                valid_targets.append(name)
        if not target_ids:
            raise ValueError("No valid target categories.")

        target_id_to_name = {cid: n for cid, n in self._cat_id_to_name.items()
                              if cid in target_ids}

        kept_items: list[ImageItem] = []
        for split in splits_to_use:
            if split not in self._splits_loaded:
                log.warning("Split %r not loaded — skipping", split)
                continue
            kept_items.extend(self._filter_split(
                split=split, target_ids=target_ids,
                target_id_to_name=target_id_to_name, K=K,
                instances_per_target=instances_per_target,
                min_area_frac=min_area_frac, max_area_frac=max_area_frac,
            ))

        instance_rule = (
            "any" if instances_per_target is None else str(instances_per_target)
        )
        log.info("Pre-balance: %d images, K=%d instances=%s area=[%.3f, %.3f]",
                 len(kept_items), K, instance_rule,
                 min_area_frac, max_area_frac)

        balance_info: dict[str, Any] = {
            "enabled": False,
            "candidate_images": len(kept_items),
        }
        if balance_selection:
            kept_items, balance_info = self._balance_multilabel_items(
                kept_items,
                valid_targets,
                preferred_targets_per_image=preferred_targets_per_image,
                target_images_per_target=target_per_target,
                shuffle_seed=shuffle_seed,
            )
            log.info(
                "Post-balance: %d images, preferred labels/image=%d, "
                "target=%s/target",
                len(kept_items), preferred_targets_per_image,
                target_per_target if target_per_target else "auto",
            )
        elif max_per_target is not None and max_per_target > 0:
            kept_items = self._cap_per_target(kept_items, max_per_target,
                                              shuffle_seed=shuffle_seed)
            log.info("Post-balance: %d images, cap=%d/target",
                     len(kept_items), max_per_target)

        for t in valid_targets:
            cnt = sum(1 for it in kept_items if t in it.targets)
            log.info("  %-25s %5d images", t, cnt)
            if 0 < cnt < min_images_warn:
                log.warning(
                    "Target %r has only %d selected images (<%d); Phase 2 "
                    "may not form trials for it unless images_per_trial is lower.",
                    t, cnt, min_images_warn,
                )

        return kept_items, valid_targets, K, balance_info

    # ── per-split filter ─────────────────────────────────────────────

    def _filter_split(self, *, split: str, target_ids: set[int],
                       target_id_to_name: dict[int, str], K: int,
                       instances_per_target: int | None, min_area_frac: float,
                       max_area_frac: float) -> list[ImageItem]:
        assert self._coco_root is not None
        data = self._splits_loaded[split]
        img_dir_name = "train2017" if split == "train" else "val2017"
        img_dir = self._coco_root / img_dir_name

        id_to_img = {im["id"]: im for im in data["images"]}
        ann_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in data["annotations"]:
            if ann["category_id"] in target_ids:
                ann_by_img[ann["image_id"]].append(ann)

        kept: list[ImageItem] = []
        n_total = 0
        for img_id, anns in ann_by_img.items():
            n_total += 1
            img_meta = id_to_img.get(img_id)
            if not img_meta:
                continue
            img_h = img_meta["height"]
            img_w = img_meta["width"]
            img_area = float(img_h * img_w)

            by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for ann in anns:
                by_cat[ann["category_id"]].append(ann)

            target_areas: dict[str, float] = {}
            bboxes: dict[str, list[float]] = {}
            segmentations: dict[str, Any] = {}
            all_target_areas: dict[str, list[float]] = {}
            all_bboxes: dict[str, list[list[float]]] = {}
            all_segmentations: dict[str, list[Any]] = {}
            target_instance_counts: dict[str, int] = {}
            for cid, anns_i in by_cat.items():
                if (
                    instances_per_target is not None
                    and len(anns_i) != instances_per_target
                ):
                    continue

                qualified: list[tuple[dict[str, Any], float]] = []
                for ann in anns_i:
                    frac_i = float(ann["area"]) / img_area
                    if min_area_frac <= frac_i <= max_area_frac:
                        qualified.append((ann, frac_i))
                if instances_per_target is not None and len(qualified) != len(anns_i):
                    continue
                if not qualified:
                    continue

                qualified.sort(key=lambda pair: pair[1], reverse=True)
                a, frac = qualified[0]
                name = target_id_to_name[cid]
                target_areas[name] = frac
                bboxes[name] = list(a["bbox"])
                segmentations[name] = a.get("segmentation")
                all_target_areas[name] = [
                    round(float(q_frac), 6) for _, q_frac in qualified
                ]
                all_bboxes[name] = [list(q_ann["bbox"]) for q_ann, _ in qualified]
                all_segmentations[name] = [
                    q_ann.get("segmentation") for q_ann, _ in qualified
                ]
                target_instance_counts[name] = len(qualified)

            if len(target_areas) < K:
                continue

            file_name = img_meta.get("file_name") or f"{img_id:012d}.jpg"
            file_path = img_dir / file_name
            relative_path = f"{img_dir_name}/{file_name}"
            if not file_path.exists():
                continue

            kept.append(ImageItem(
                image_id=str(img_id),
                file_path=file_path,
                relative_path=relative_path,
                targets=sorted(target_areas.keys()),
                target_areas=target_areas,
                bboxes=bboxes,
                split=split,
                metadata={"img_h": img_h, "img_w": img_w,
                          "segmentations": segmentations,
                          "all_target_areas": all_target_areas,
                          "all_bboxes": all_bboxes,
                          "all_segmentations": all_segmentations,
                          "target_instance_counts": target_instance_counts,
                          "n_qualifying": len(target_areas)},
            ))
        log.info("split=%s: scanned=%d kept=%d (>=%d qualifying targets each)",
                 split, n_total, len(kept), K)
        return kept

    @staticmethod
    def _cap_per_target(items: list[ImageItem], cap: int, *,
                         shuffle_seed: int | None) -> list[ImageItem]:
        rng = random.Random(shuffle_seed)
        per_target_count: dict[str, int] = defaultdict(int)
        order = list(items)
        rng.shuffle(order)
        kept: list[ImageItem] = []
        for it in order:
            if all(per_target_count[t] < cap for t in it.targets):
                kept.append(it)
                for t in it.targets:
                    per_target_count[t] += 1
        return kept

    @staticmethod
    def _assign_eeg_splits(
        items: list[ImageItem],
        targets: list[str],
        *,
        test_fraction: float,
        split_seed: int,
    ) -> dict[str, Any]:
        """Assign deterministic EEG train/test labels inside the selected subset.

        This is independent from the original LVIS/COCO split. The number of
        test images is fixed by `test_fraction`; selection is greedy
        multi-label stratified so target-level test counts stay close to the
        requested fraction.
        """
        test_fraction = max(0.0, min(1.0, float(test_fraction)))
        n_test = int(round(len(items) * test_fraction))
        rng = random.Random(split_seed)

        for item in items:
            item.metadata["eeg_split"] = "train"

        if not items or n_test <= 0:
            return {
                "test_fraction": test_fraction,
                "split_seed": split_seed,
                "n_train": len(items),
                "n_test": 0,
                "target_train_counts": {t: sum(1 for it in items if t in it.targets)
                                        for t in targets},
                "target_test_counts": {t: 0 for t in targets},
            }

        target_counts = {
            t: sum(1 for it in items if t in it.targets) for t in targets
        }
        target_goals = {
            t: int(round(target_counts[t] * test_fraction)) for t in targets
        }
        test_counts = {t: 0 for t in targets}
        target_set = set(targets)
        remaining: set[int] = set(range(len(items)))
        selected: list[int] = []

        while remaining and len(selected) < n_test:
            best_idx: int | None = None
            best_key: tuple[float, ...] | None = None
            for idx in remaining:
                labels = [t for t in items[idx].targets if t in target_set]
                under = [t for t in labels if test_counts[t] < target_goals[t]]
                gain = sum(
                    (target_goals[t] - test_counts[t]) / max(target_goals[t], 1)
                    for t in under
                )
                over_penalty = sum(
                    max(0, test_counts[t] + 1 - target_goals[t]) /
                    max(target_goals[t], 1)
                    for t in labels
                )
                key = (
                    float(len(under)),
                    gain,
                    -over_penalty,
                    float(len(labels)),
                    rng.random(),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx
            if best_idx is None:
                break
            remaining.remove(best_idx)
            selected.append(best_idx)
            for t in items[best_idx].targets:
                if t in test_counts:
                    test_counts[t] += 1

        if len(selected) < n_test:
            fill = list(remaining)
            rng.shuffle(fill)
            selected.extend(fill[:n_test - len(selected)])

        selected_set = set(selected[:n_test])
        for idx, item in enumerate(items):
            item.metadata["eeg_split"] = "test" if idx in selected_set else "train"

        target_test_counts = {
            t: sum(1 for it in items
                   if it.metadata.get("eeg_split") == "test" and t in it.targets)
            for t in targets
        }
        target_train_counts = {
            t: sum(1 for it in items
                   if it.metadata.get("eeg_split") == "train" and t in it.targets)
            for t in targets
        }
        return {
            "test_fraction": test_fraction,
            "split_seed": split_seed,
            "n_train": len(items) - len(selected_set),
            "n_test": len(selected_set),
            "target_test_goals": target_goals,
            "target_train_counts": target_train_counts,
            "target_test_counts": target_test_counts,
        }

    @staticmethod
    def _balance_multilabel_items(
        items: list[ImageItem],
        targets: list[str],
        *,
        preferred_targets_per_image: int,
        target_images_per_target: int | None,
        shuffle_seed: int | None,
    ) -> tuple[list[ImageItem], dict[str, Any]]:
        """Greedily select a balanced multi-label subset.

        The filter stage builds all valid candidates. This stage chooses a
        subset that fills every configured target up to a shared goal where
        possible, while preferring images whose number of target labels is
        close to `preferred_targets_per_image`.
        """
        if not items:
            return [], {
                "enabled": True,
                "candidate_images": 0,
                "selected_images": 0,
                "target_images_per_target": target_images_per_target,
                "preferred_targets_per_image": preferred_targets_per_image,
            }

        target_set = set(targets)
        candidate_counts: dict[str, int] = {t: 0 for t in targets}
        for it in items:
            for t in it.targets:
                if t in candidate_counts:
                    candidate_counts[t] += 1

        available_nonzero = [c for c in candidate_counts.values() if c > 0]
        if target_images_per_target is None or target_images_per_target <= 0:
            sorted_counts = sorted(available_nonzero)
            if sorted_counts:
                target_images_per_target = sorted_counts[len(sorted_counts) // 2]
            else:
                target_images_per_target = 0

        goals = {
            t: min(candidate_counts[t], int(target_images_per_target))
            for t in targets
        }

        rng = random.Random(shuffle_seed)
        order = list(items)
        rng.shuffle(order)
        remaining: set[int] = set(range(len(order)))
        selected: list[ImageItem] = []
        selected_counts: dict[str, int] = {t: 0 for t in targets}

        while remaining:
            underfilled = {t for t in targets if selected_counts[t] < goals[t]}
            if not underfilled:
                break

            best_idx: int | None = None
            best_key: tuple[float, ...] | None = None

            for idx in remaining:
                it = order[idx]
                labels = [t for t in it.targets if t in target_set]
                under_labels = [t for t in labels if t in underfilled]
                if not under_labels:
                    continue

                n_labels = len(labels)
                under_gain = sum(
                    (goals[t] - selected_counts[t]) / max(goals[t], 1)
                    for t in under_labels
                )
                scarcity_gain = sum(
                    1.0 / max(candidate_counts[t], 1) for t in under_labels
                )
                over_penalty = sum(
                    max(0, selected_counts[t] + 1 - goals[t]) / max(goals[t], 1)
                    for t in labels if t not in underfilled
                )
                closeness = abs(n_labels - preferred_targets_per_image)
                exact_bonus = 1.0 if n_labels == preferred_targets_per_image else 0.0
                multi_label_bonus = min(n_labels, preferred_targets_per_image) / max(
                    preferred_targets_per_image, 1)

                key = (
                    float(len(under_labels)),
                    under_gain,
                    exact_bonus,
                    -float(closeness),
                    multi_label_bonus,
                    scarcity_gain,
                    -over_penalty,
                    rng.random() * 1e-6,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx

            if best_idx is None:
                break

            item = order[best_idx]
            selected.append(item)
            remaining.remove(best_idx)
            for t in item.targets:
                if t in selected_counts:
                    selected_counts[t] += 1

        return selected, {
            "enabled": True,
            "candidate_images": len(items),
            "selected_images": len(selected),
            "preferred_targets_per_image": preferred_targets_per_image,
            "target_images_per_target": int(target_images_per_target),
            "candidate_counts": candidate_counts,
            "target_goals": goals,
            "selected_counts": selected_counts,
        }
