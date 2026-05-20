"""Attention-driven RSVP groups.

The current protocol is organized by attention label:

* build one shared stimulus pool by sampling N images per configured label;
* each RSVP group has exactly one attention label / HINT;
* every image in that group contains the attention label;
* each image-label pair is repeated several times for averaging;
* after M groups, show a rest screen and wait for Enter before continuing.

All image textures for a group are loaded before the group starts. When
configured, the attention object's annotation outline is drawn into the
preloaded texture, so presentation does not perform disk I/O or annotation
rendering.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from core.display import DisplayEngine, OverlayState, TextureFrame
from core.graphics import (
    draw_annotation_outline,
    fit_image_to_canvas_with_transform,
    load_image,
    make_blank,
    make_fixation,
    make_text_canvas,
    make_text_canvas_pil,
)
from core.logger import EventLogger
from datasets.base import ImageItem, StimulusBundle
from markers.base import MarkerManager
from paradigms.base import Paradigm

log = logging.getLogger(__name__)


# Reserved system markers.
M_HINT_PREFIX = 250
M_TRIAL_START = 251
M_TRIAL_END = 252
M_SESSION_START = 253
M_SESSION_END = 254
M_EXPERIMENT_END = 255


@dataclass(frozen=True)
class RSVPStim:
    item: ImageItem
    repeat_index: int
    repeat_total: int


@dataclass(frozen=True)
class RSVPGroup:
    group_index: int
    hint: str
    stimuli: list[RSVPStim]


class RSVPAttentionParadigm(Paradigm):
    def __init__(self, display: DisplayEngine, logger: EventLogger,
                 marker_mgr: MarkerManager,
                 eyetracker: Any = None) -> None:
        super().__init__(display, logger, marker_mgr)
        self.eyetracker = eyetracker
        # Persistent gaze watchdog state. Tracks the most recent local
        # wall-clock time at which the eye-tracker delivered a fresh
        # sample. The previous design counted None frames *within a
        # single ON or ISI*, but at typical RSVP timing (e.g.,
        # stim_on_ms=350, ~21 frames @ 60 Hz) one trial is shorter than
        # `gaze_loss_grace_ms=400`, so a per-trial counter could never
        # accumulate enough loss frames to trigger. The watchdog now
        # spans trials: any continuous None duration (inside or across
        # ON/ISI/HINT/REST) of >= `gaze_loss_grace_ms` triggers the
        # pause.
        self._gaze_last_seen_qpc: float | None = None

    def run(self, bundle: StimulusBundle, config: dict[str, Any]) -> None:
        on_ms = int(config.get("stim_on_ms", 300))
        off_ms = int(config.get("stim_off_ms", 200))
        off_jitter_ms = max(0, int(config.get("stim_off_ms_jitter", 0)))
        hint_ms = int(config.get("hint_ms", 2000))
        fixation_ms = int(config.get("fixation_ms", 600))
        initial_rest_ms = int(config.get("initial_rest_ms", 3000))
        final_rest_ms = int(config.get("final_rest_ms", 2000))
        group_rest_ms = int(config.get("group_rest_ms", 1000))
        groups_per_rest = int(config.get("groups_per_rest", 40))
        rest_after_groups_ms = int(config.get("rest_after_groups_ms", 300000))
        wait_for_enter = bool(config.get("wait_for_enter_after_rest", True))
        draw_outline = bool(config.get("draw_target_outline", True))
        outline_color = tuple(config.get("outline_color_rgb", [255, 255, 0]))
        outline_thickness = int(config.get("outline_thickness", 5))
        seed = config.get("shuffle_seed", 42)

        # ── HINT rendering — Chinese / English / both ────────────────
        hint_lang = str(config.get("hint_language", "en")).lower()
        hint_font_path = config.get(
            "hint_font_path", "C:/Windows/Fonts/msyh.ttc")
        hint_font_size = int(config.get("hint_font_size", 240))
        hint_subtitle_size = int(config.get("hint_subtitle_size", 56))
        hint_zh_map = dict(config.get("category_translation_zh", {}) or {})
        self._hint_render_cfg = {
            "language": hint_lang,
            "font_path": hint_font_path,
            "font_size": hint_font_size,
            "subtitle_size": hint_subtitle_size,
            "zh_map": hint_zh_map,
        }

        # ── Live gaze monitoring during ON / live cursor during ISI ──
        # The `eyetracking.monitor.*` block is forwarded into the paradigm
        # config by phase2_run.py — that keeps the paradigm self-contained
        # and avoids reaching into the eyetracker object for config.
        monitor_cfg = config.get("monitor", {}) or {}
        self._monitor_cfg = {
            "enabled": bool(monitor_cfg.get("enabled", False))
                        and self.eyetracker is not None,
            "cursor_radius_px": int(monitor_cfg.get("cursor_radius_px", 16)),
            "cursor_color_rgb": tuple(
                monitor_cfg.get("cursor_color_rgb", [0, 255, 0])),
            "cursor_y_invert": bool(monitor_cfg.get("cursor_y_invert", False)),
            # Dual-monitor / off-screen cursor handling.
            # False (default): out-of-bounds gaze → cursor naturally
            # disappears (clipped by GL viewport). True: cursor pinned to
            # the screen edge in `cursor_offscreen_color_rgb` so it
            # remains a visible direction indicator.
            "cursor_clip_to_edge": bool(
                monitor_cfg.get("cursor_clip_to_edge", False)),
            "cursor_offscreen_color_rgb": tuple(
                monitor_cfg.get("cursor_offscreen_color_rgb", [255, 200, 80])),
            "fixation_marker_radius_px": int(
                monitor_cfg.get("fixation_marker_radius_px", 0)),
            "fixation_marker_color_rgb": tuple(
                monitor_cfg.get("fixation_marker_color_rgb", [255, 255, 255])),
            "threshold_normalized": float(
                monitor_cfg.get("threshold_normalized", 0.15)),
            "min_consecutive_frames": int(
                monitor_cfg.get("min_consecutive_frames", 1)),
            "warning_ring_radius_px": int(
                monitor_cfg.get("warning_ring_radius_px", 64)),
            "warning_ring_color_rgb": tuple(
                monitor_cfg.get("warning_ring_color_rgb", [255, 50, 50])),
            "warning_ring_thickness_px": int(
                monitor_cfg.get("warning_ring_thickness_px", 4)),
            # Gaze-loss watchdog
            "pause_on_gaze_loss": bool(
                monitor_cfg.get("pause_on_gaze_loss", True)),
            # Abort the ON stimulus mid-presentation when gaze loss
            # exceeds grace. Default True per request — once eyes are
            # confirmed lost there is no value in continuing the
            # remaining ON budget; the EEG epoch is already invalid.
            "abort_on_loss_during_on": bool(
                monitor_cfg.get("abort_on_loss_during_on", True)),
            "gaze_loss_grace_ms": int(
                monitor_cfg.get("gaze_loss_grace_ms", 400)),
            "gaze_resume_window_ms": int(
                monitor_cfg.get("gaze_resume_window_ms", 300)),
            "pause_message_zh": str(
                monitor_cfg.get("pause_message_zh", "未检测到眼动信号")),
            "pause_message_zh_sub": str(
                monitor_cfg.get(
                    "pause_message_zh_sub", "请保持注视屏幕中心")),
            "pause_font_size": int(
                monitor_cfg.get("pause_font_size", 96)),
            "pause_font_subtitle_size": int(
                monitor_cfg.get("pause_font_subtitle_size", 48)),
            "pause_check_interval_ms": int(
                monitor_cfg.get("pause_check_interval_ms", 50)),
        }
        log.info(
            "Live gaze monitor: enabled=%s threshold=%.3f cursor_r=%dpx "
            "grace=%dms abort_on_during_on=%s",
            self._monitor_cfg["enabled"],
            self._monitor_cfg["threshold_normalized"],
            self._monitor_cfg["cursor_radius_px"],
            self._monitor_cfg["gaze_loss_grace_ms"],
            self._monitor_cfg["abort_on_loss_during_on"],
        )
        # Initialise the persistent watchdog. wait_for_data() in
        # phase2_run.py has already verified the tracker is delivering
        # samples, so we can claim "just seen" right now.
        self._gaze_last_seen_qpc = time.perf_counter()
        # Also remember whether we have already logged the entry into
        # a loss state for the *current* loss episode, to avoid
        # duplicate `gaze_lost` log entries when both ON and ISI
        # callbacks observe the same persistent loss.
        self._gaze_loss_episode_logged = False

        rng = random.Random(seed)
        d = self.display
        w, h = d.width, d.height

        cat_to_code = {t: i + 1 for i, t in enumerate(bundle.targets)}
        if cat_to_code and max(cat_to_code.values()) >= M_HINT_PREFIX:
            log.warning("Category marker codes collide with reserved markers")
        log.info("HINT/category code mapping: %s", cat_to_code)

        groups, pool_info = self.plan_groups(bundle, config, rng)
        total_presentations = sum(len(g.stimuli) for g in groups)
        timing_info = self.estimate_timing(groups, config)
        log.info(
            "Planned %d RSVP groups, %d stimulus presentations, pool=%d images",
            len(groups), total_presentations, pool_info["n_pool_images"],
        )
        self._log_timing_summary(timing_info)

        blank_frame = d.to_texture_frame(d.stamp_photodiode(
            make_blank(w, h), active=False))
        fixation_frame = d.to_texture_frame(d.stamp_photodiode(
            make_fixation(w, h), active=False))

        self.logger.log(
            "experiment_start_paradigm",
            n_targets=len(bundle.targets),
            K=bundle.K,
            n_groups=len(groups),
            total_presentations=total_presentations,
            stimulus_pool=pool_info,
            timing_estimate=timing_info,
        )
        self._safe_eyetracker_mark("experiment_start")

        if not d.present_for(blank_frame, initial_rest_ms):
            return

        self.marker_mgr.on_flip(M_SESSION_START)
        self.logger.log("session_start", session_index=0, n_groups=len(groups))
        self._safe_eyetracker_mark("session_0_start")

        if not d.present_for(fixation_frame, fixation_ms):
            return
        self.logger.log("session_fixation_offset", session_index=0)

        run_started = time.perf_counter()
        completed_presentations = 0
        for group in groups:
            if (
                group.group_index > 0
                and groups_per_rest > 0
                and group.group_index % groups_per_rest == 0
            ):
                if not self._show_group_break(
                    completed_groups=group.group_index,
                    total_groups=len(groups),
                    rest_ms=rest_after_groups_ms,
                    wait_for_enter=wait_for_enter,
                ):
                    break

            if not self._run_group(
                group,
                cat_to_code,
                blank_frame,
                on_ms=on_ms,
                off_ms=off_ms,
                off_jitter_ms=off_jitter_ms,
                hint_ms=hint_ms,
                group_rest_ms=group_rest_ms,
                draw_outline=draw_outline,
                outline_color=outline_color,  # type: ignore[arg-type]
                outline_thickness=outline_thickness,
                timing_info=timing_info,
                run_started=run_started,
                completed_presentations=completed_presentations,
                rng=rng,
            ):
                break
            completed_presentations += len(group.stimuli)

        self.marker_mgr.on_flip(M_SESSION_END)
        self.logger.log("session_end", session_index=0)
        self._safe_eyetracker_mark("session_0_end")

        d.present_for(blank_frame, final_rest_ms)
        self.marker_mgr.on_flip(M_EXPERIMENT_END)
        self.logger.log("experiment_end_paradigm")
        self._safe_eyetracker_mark("experiment_end")

    @staticmethod
    def plan_groups(
        bundle: StimulusBundle,
        config: dict[str, Any],
        rng: random.Random | None = None,
    ) -> tuple[list[RSVPGroup], dict[str, Any]]:
        rng = rng or random.Random(config.get("shuffle_seed", 42))
        images_per_target = int(config.get("pool_images_per_target", 10))
        group_images_per_label = int(config.get("group_images_per_label", 40))
        rsvp_group_size = int(config.get("rsvp_group_size", 15))
        preferred_labels = int(config.get("preferred_targets_per_image", 4))
        train_repeats = int(config.get("eeg_train_repeats_per_image_label", 8))
        test_repeats = int(config.get("eeg_test_repeats_per_image_label", 40))
        stimulus_splits = config.get("stimulus_splits")
        allowed_splits = set(stimulus_splits) if stimulus_splits else None

        pool, pool_by_seed_target = RSVPAttentionParadigm._sample_image_pool(
            bundle=bundle,
            images_per_target=images_per_target,
            preferred_labels=preferred_labels,
            allowed_splits=allowed_splits,
            rng=rng,
        )
        pool_ids = {it.image_id for it in pool}

        groups_by_hint: dict[str, list[RSVPGroup]] = {}
        group_image_counts: dict[str, int] = {}
        group_presentation_counts: dict[str, int] = {}
        group_counts_by_hint: dict[str, int] = {}
        eeg_split_repeat_counts: dict[str, int] = defaultdict(int)

        for hint in bundle.targets:
            label_items = [
                it for it in pool
                if it.image_id in pool_ids and hint in it.targets
            ]
            label_items = RSVPAttentionParadigm._fit_group_image_count(
                label_items=label_items,
                candidates=[
                    it for it in bundle.images_by_target.get(hint, [])
                    if allowed_splits is None or it.split in allowed_splits
                ],
                target_count=group_images_per_label,
                preferred_labels=preferred_labels,
                rng=rng,
            )
            if not label_items:
                log.warning("No pooled images contain attention label %r", hint)
                continue
            if group_images_per_label > 0 and len(label_items) < group_images_per_label:
                log.warning(
                    "Attention label %r has only %d/%d group images",
                    hint, len(label_items), group_images_per_label,
                )
            rng.shuffle(label_items)

            raw_stimuli: list[RSVPStim] = []
            for item in label_items:
                eeg_split = str(item.metadata.get("eeg_split", "train"))
                repeat_total = (
                    test_repeats if eeg_split == "test" else train_repeats
                )
                eeg_split_repeat_counts[eeg_split] += repeat_total
                for rep_idx in range(repeat_total):
                    raw_stimuli.append(RSVPStim(
                        item=item,
                        repeat_index=rep_idx + 1,
                        repeat_total=repeat_total,
                    ))

            ordered_stimuli = RSVPAttentionParadigm._shuffle_no_adjacent(
                raw_stimuli, rng)
            group_image_counts[hint] = len(label_items)
            group_presentation_counts[hint] = len(ordered_stimuli)

            hint_groups: list[RSVPGroup] = []
            for start in range(0, len(ordered_stimuli), max(1, rsvp_group_size)):
                chunk = ordered_stimuli[start:start + max(1, rsvp_group_size)]
                hint_groups.append(RSVPGroup(
                    group_index=-1,
                    hint=hint,
                    stimuli=chunk,
                ))
            groups_by_hint[hint] = hint_groups
            group_counts_by_hint[hint] = len(hint_groups)

        groups = RSVPAttentionParadigm._shuffle_groups_no_adjacent_hint(
            groups_by_hint, rng)
        groups = [
            RSVPGroup(group_index=i, hint=g.hint, stimuli=g.stimuli)
            for i, g in enumerate(groups)
        ]

        pool_info = {
            "n_pool_images": len(pool),
            "pool_images_per_target": images_per_target,
            "group_images_per_label": group_images_per_label,
            "rsvp_group_size": rsvp_group_size,
            "preferred_targets_per_image": preferred_labels,
            "eeg_train_repeats_per_image_label": train_repeats,
            "eeg_test_repeats_per_image_label": test_repeats,
            "stimulus_splits": list(allowed_splits) if allowed_splits else None,
            "pool_by_seed_target": {
                t: [it.image_id for it in items]
                for t, items in pool_by_seed_target.items()
            },
            "group_image_counts": group_image_counts,
            "group_presentation_counts": group_presentation_counts,
            "group_counts_by_hint": group_counts_by_hint,
            "group_order": [g.hint for g in groups],
            "eeg_split_repeat_counts": dict(eeg_split_repeat_counts),
        }
        return groups, pool_info

    @staticmethod
    def _sample_image_pool(
        *,
        bundle: StimulusBundle,
        images_per_target: int,
        preferred_labels: int,
        allowed_splits: set[str] | None,
        rng: random.Random,
    ) -> tuple[list[ImageItem], dict[str, list[ImageItem]]]:
        selected: dict[str, ImageItem] = {}
        by_seed_target: dict[str, list[ImageItem]] = {}

        targets = list(bundle.targets)
        rng.shuffle(targets)
        for target in targets:
            candidates = [
                it for it in bundle.images_by_target.get(target, [])
                if allowed_splits is None or it.split in allowed_splits
            ]
            rng.shuffle(candidates)
            candidates.sort(
                key=lambda it: (
                    abs(len(it.targets) - preferred_labels),
                    -len(it.targets),
                    rng.random(),
                )
            )

            already = [
                it for it in selected.values()
                if target in it.targets
            ]
            chosen = list(already[:images_per_target])
            for item in candidates:
                if len(chosen) >= images_per_target:
                    break
                if item.image_id in {it.image_id for it in chosen}:
                    continue
                chosen.append(item)
                selected[item.image_id] = item

            if len(chosen) < images_per_target:
                log.warning(
                    "Target %r has only %d/%d pooled images under current filters",
                    target, len(chosen), images_per_target,
                )
            by_seed_target[target] = chosen

        return list(selected.values()), by_seed_target

    @staticmethod
    def _fit_group_image_count(
        *,
        label_items: list[ImageItem],
        candidates: list[ImageItem],
        target_count: int,
        preferred_labels: int,
        rng: random.Random,
    ) -> list[ImageItem]:
        if target_count <= 0:
            return label_items

        chosen: dict[str, ImageItem] = {it.image_id: it for it in label_items}
        remaining = [it for it in candidates if it.image_id not in chosen]
        rng.shuffle(remaining)
        remaining.sort(
            key=lambda it: (
                abs(len(it.targets) - preferred_labels),
                -len(it.targets),
                rng.random(),
            )
        )
        for item in remaining:
            if len(chosen) >= target_count:
                break
            chosen[item.image_id] = item

        out = list(chosen.values())
        rng.shuffle(out)
        out.sort(
            key=lambda it: (
                abs(len(it.targets) - preferred_labels),
                -len(it.targets),
                rng.random(),
            )
        )
        return out[:target_count]

    @staticmethod
    def _shuffle_no_adjacent(
        stimuli: list[RSVPStim],
        rng: random.Random,
    ) -> list[RSVPStim]:
        by_image: dict[str, list[RSVPStim]] = defaultdict(list)
        for stim in stimuli:
            by_image[stim.item.image_id].append(stim)
        for bucket in by_image.values():
            rng.shuffle(bucket)

        out: list[RSVPStim] = []
        last_image: str | None = None
        while by_image:
            candidates = [
                (image_id, bucket)
                for image_id, bucket in by_image.items()
                if image_id != last_image
            ]
            if not candidates:
                candidates = list(by_image.items())

            max_len = max(len(bucket) for _, bucket in candidates)
            top = [
                (image_id, bucket)
                for image_id, bucket in candidates
                if len(bucket) == max_len
            ]
            image_id, bucket = rng.choice(top)
            out.append(bucket.pop())
            last_image = image_id
            if not bucket:
                del by_image[image_id]
        return out

    @staticmethod
    def _shuffle_groups_no_adjacent_hint(
        groups_by_hint: dict[str, list[RSVPGroup]],
        rng: random.Random,
    ) -> list[RSVPGroup]:
        remaining = {
            hint: list(groups)
            for hint, groups in groups_by_hint.items()
            if groups
        }
        for groups in remaining.values():
            rng.shuffle(groups)

        out: list[RSVPGroup] = []
        last_hint: str | None = None
        while remaining:
            total_remaining = sum(len(groups) for groups in remaining.values())
            candidates = [
                (hint, groups)
                for hint, groups in remaining.items()
                if hint != last_hint
            ]
            if not candidates:
                candidates = list(remaining.items())

            forced = [
                (hint, groups)
                for hint, groups in candidates
                if len(groups) > total_remaining - len(groups) + 1
            ]
            if forced:
                hint, groups = max(forced, key=lambda pair: len(pair[1]))
            else:
                weights = [float(len(groups)) for _, groups in candidates]
                hint, groups = rng.choices(candidates, weights=weights, k=1)[0]
            out.append(groups.pop())
            last_hint = hint
            if not groups:
                del remaining[hint]
        return out

    def _run_group(
        self,
        group: RSVPGroup,
        cat_to_code: dict[str, int],
        blank_frame: TextureFrame,
        *,
        on_ms: int,
        off_ms: int,
        off_jitter_ms: int,
        hint_ms: int,
        group_rest_ms: int,
        draw_outline: bool,
        outline_color: tuple[int, int, int],
        outline_thickness: int,
        timing_info: dict[str, Any],
        run_started: float,
        completed_presentations: int,
        rng: random.Random,
    ) -> bool:
        d = self.display
        code = cat_to_code[group.hint]
        group_est_s = self._estimate_group_seconds(
            len(group.stimuli), hint_ms, on_ms, off_ms, group_rest_ms)
        completed_groups = group.group_index
        total_groups = int(timing_info["n_groups"])
        total_presentations = int(timing_info["n_presentations"])
        elapsed_wall = max(0.0, time.perf_counter() - run_started)
        estimated_remaining_s = max(
            0.0,
            float(timing_info["active_seconds"]) -
            self._estimate_presentations_seconds(
                completed_presentations, on_ms, off_ms),
        )

        # ── HINT canvas (Chinese / English / both) ────────────────
        group_number = group.group_index + 1
        hint_canvas = self._make_hint_canvas(
            hint=group.hint,
            group_number=group_number,
            total_groups=total_groups,
            width=d.width,
            height=d.height,
        )
        hint_frame = d.to_texture_frame(d.stamp_photodiode(
            hint_canvas, active=False))

        log.info(
            "[group %04d/%04d] label=%s code=%d presentations=%d "
            "est_group=%s progress=%d/%d remaining~%s elapsed=%s",
            group.group_index + 1,
            total_groups,
            group.hint,
            code,
            len(group.stimuli),
            self._format_duration(group_est_s),
            completed_presentations,
            total_presentations,
            self._format_duration(estimated_remaining_s),
            self._format_duration(elapsed_wall),
        )
        tex_cache = self._precache_group_textures(
            group,
            draw_outline=draw_outline,
            outline_color=outline_color,
            outline_thickness=outline_thickness,
        )

        self.marker_mgr.on_flip(M_TRIAL_START)
        self.logger.log(
            "rsvp_group_start",
            group_index=group.group_index,
            hint=group.hint,
            hint_code=code,
            n_presentations=len(group.stimuli),
            n_unique_images=len({stim.item.image_id for stim in group.stimuli}),
        )
        self._safe_eyetracker_mark(
            f"group_{group.group_index}_start_hint={group.hint}")

        if not d.present_for_logged(
            hint_frame,
            hint_ms,
            marker_code=M_HINT_PREFIX,
            event_name="hint",
            group_index=group.group_index,
            group_number=group_number,
            total_groups=total_groups,
            hint=group.hint,
            hint_code=code,
        ):
            return False
        self.marker_mgr.on_flip(code)

        prev_image: str | None = None
        consecutive_repeats = 0
        # Live gaze monitoring state for THIS group.
        mon = self._monitor_cfg
        threshold = float(mon["threshold_normalized"])
        min_consec = int(mon["min_consecutive_frames"])
        bad_fixation_groups = 0  # number of trials with breach in this group
        next_isi_warn = False    # warning ring set for the upcoming ISI

        for stim_idx, stim in enumerate(group.stimuli):
            item = stim.item
            frame = tex_cache.get(item.image_id)
            if frame is None:
                continue

            if prev_image == item.image_id:
                consecutive_repeats += 1
            prev_image = item.image_id

            # Per-trial random ISI in [base - jitter, base + jitter]
            if off_jitter_ms > 0:
                isi_ms = max(
                    1,
                    rng.randint(
                        max(1, off_ms - off_jitter_ms),
                        off_ms + off_jitter_ms,
                    ),
                )
            else:
                isi_ms = off_ms

            # ── ON: clear overlay (no cursor on stimulus) and monitor gaze
            d.clear_overlay()
            on_state = {
                "consec": 0,
                "max_consec": 0,
                "max_dev": 0.0,
                "bad": False,
                "loss_triggered": False,
                "any_sample": False,
                "frames_no_gaze": 0,
            }

            def _check_gaze(_elapsed_s: float, _state=on_state) -> None:
                if not mon["enabled"] or self.eyetracker is None:
                    return
                g = self.eyetracker.get_latest_gaze()
                # Persistent watchdog: returns True if cumulative loss
                # (across trials) has reached gaze_loss_grace_ms.
                if self._gaze_watchdog_step(g):
                    _state["loss_triggered"] = True
                if g is None:
                    _state["frames_no_gaze"] += 1
                    return
                _state["any_sample"] = True
                gx, gy = g
                dev = max(abs(gx), abs(gy))
                if dev > _state["max_dev"]:
                    _state["max_dev"] = dev
                if dev > threshold:
                    _state["consec"] += 1
                    if _state["consec"] > _state["max_consec"]:
                        _state["max_consec"] = _state["consec"]
                    if _state["consec"] >= min_consec:
                        _state["bad"] = True
                else:
                    _state["consec"] = 0

            # Abort callback: if we're configured to interrupt the
            # stimulus mid-ON when the eyes go missing, return True from
            # `should_abort` once the loss watchdog has fired. The
            # display loop exits early, the EEG epoch is logged with
            # `aborted=True`, and we go straight to the pause loop —
            # no point holding a flashing image on a screen the subject
            # is no longer looking at.
            def _should_abort_on(
                _elapsed_s: float,
                _state=on_state,
            ) -> bool:
                return bool(_state["loss_triggered"])

            on_abort_cb = (
                _should_abort_on
                if (mon["enabled"] and mon["abort_on_loss_during_on"])
                else None
            )

            if not d.present_for_logged(
                frame,
                on_ms,
                marker_code=code,
                event_name="stim",
                on_frame=_check_gaze if mon["enabled"] else None,
                should_abort=on_abort_cb,
                group_index=group.group_index,
                stim_index=stim_idx,
                image_id=item.image_id,
                split=item.split,
                eeg_split=item.metadata.get("eeg_split", "train"),
                hint=group.hint,
                targets_in_image=item.targets,
                repeat_index=stim.repeat_index,
                repeat_total=stim.repeat_total,
                outlined_label=group.hint if draw_outline else None,
                target_areas=item.target_areas,
            ):
                return False

            # If ON aborted mid-stim due to gaze loss, pause IMMEDIATELY
            # and skip the upcoming ISI for this trial — that ISI would
            # only display a flashing cursor on top of nothing useful.
            if (mon["enabled"] and mon["pause_on_gaze_loss"]
                    and on_state["loss_triggered"]
                    and mon["abort_on_loss_during_on"]):
                self.logger.log(
                    "gaze_lost",
                    group_index=group.group_index,
                    stim_index=stim_idx,
                    image_id=item.image_id,
                    on_frames_no_gaze=on_state["frames_no_gaze"],
                    grace_loss_ms=mon["gaze_loss_grace_ms"],
                    detected_in="on",
                    aborted_on=True,
                )
                self._safe_eyetracker_mark(
                    f"gaze_lost_on_g{group.group_index}_s{stim_idx}")
                d.clear_overlay()
                if not self._wait_for_gaze_return(
                    group_index=group.group_index,
                    stim_index=stim_idx,
                ):
                    return False
                continue   # skip ISI for this trial; advance to next stim

            if mon["enabled"] and on_state["bad"]:
                bad_fixation_groups += 1
                next_isi_warn = True
                self.logger.log(
                    "bad_fixation",
                    group_index=group.group_index,
                    stim_index=stim_idx,
                    image_id=item.image_id,
                    threshold_normalized=threshold,
                    max_deviation_normalized=on_state["max_dev"],
                    max_consecutive_frames=on_state["max_consec"],
                )
                self._safe_eyetracker_mark(
                    f"bad_fix_g{group.group_index}_s{stim_idx}")

            # ── ISI: install cursor + (warning if last trial was bad) ──
            self._enable_isi_overlay(warning=next_isi_warn)
            isi_warn_shown = next_isi_warn
            next_isi_warn = False  # consumed
            isi_state = {
                "loss_triggered": False,
                "frames_no_gaze": 0,
            }

            def _update_cursor(
                _elapsed_s: float,
                _invert_y=mon["cursor_y_invert"],
                _state=isi_state,
            ) -> None:
                if self.eyetracker is None:
                    return
                g = self.eyetracker.get_latest_gaze()
                # Persistent watchdog (shared with ON callback).
                if self._gaze_watchdog_step(g):
                    _state["loss_triggered"] = True
                if g is None:
                    d.set_cursor(None)
                    _state["frames_no_gaze"] += 1
                    return
                gx, gy = g
                if _invert_y:
                    gy = -gy
                d.set_cursor((float(gx), float(gy)))

            # ISI also honours the watchdog: short-circuit if loss has
            # accumulated past grace inside the ISI window. present_for
            # supports an on_frame; we leverage present_for's loop by
            # signalling abort via the shared isi_state flag — checked
            # by the tiny inline predicate below using a closure.
            # `present_for` itself does not expose should_abort; we
            # achieve the same by raising in on_frame, but that pollutes
            # the loop. Simpler: just let present_for finish ISI
            # normally — the ISI is at most ~150 ms, the cost of waiting
            # it out is negligible compared to clean code.
            if not d.present_for(
                blank_frame,
                isi_ms,
                on_frame=_update_cursor if mon["enabled"] else None,
            ):
                return False

            d.clear_overlay()
            self.logger.log(
                "isi_offset",
                group_index=group.group_index,
                stim_index=stim_idx,
                isi_ms=isi_ms,
                isi_warning_shown=isi_warn_shown,
                on_frames_no_gaze=on_state["frames_no_gaze"],
                isi_frames_no_gaze=isi_state["frames_no_gaze"],
            )

            # ── Gaze-loss watchdog (post-trial) ────────────────────────
            # If the persistent watchdog fired during either ON or ISI
            # of THIS trial (and we did not already abort + pause
            # mid-ON above), pause now until the tracker returns
            # continuous samples for resume_window_ms. Tracker
            # disconnected at startup is a separate, fatal error
            # already handled in phase2_run.py.
            if mon["enabled"] and mon["pause_on_gaze_loss"] and (
                on_state["loss_triggered"] or isi_state["loss_triggered"]
            ):
                detected_in = (
                    "on" if on_state["loss_triggered"] else "isi"
                )
                self.logger.log(
                    "gaze_lost",
                    group_index=group.group_index,
                    stim_index=stim_idx,
                    image_id=item.image_id,
                    on_frames_no_gaze=on_state["frames_no_gaze"],
                    isi_frames_no_gaze=isi_state["frames_no_gaze"],
                    grace_loss_ms=mon["gaze_loss_grace_ms"],
                    detected_in=detected_in,
                    aborted_on=False,
                )
                self._safe_eyetracker_mark(
                    f"gaze_lost_g{group.group_index}_s{stim_idx}")
                if not self._wait_for_gaze_return(
                    group_index=group.group_index,
                    stim_index=stim_idx,
                ):
                    return False  # user closed window during pause

        self.marker_mgr.on_flip(M_TRIAL_END)
        self.logger.log(
            "rsvp_group_end",
            group_index=group.group_index,
            hint=group.hint,
            consecutive_same_image_count=consecutive_repeats,
            bad_fixation_trials=bad_fixation_groups,
        )
        self._safe_eyetracker_mark(f"group_{group.group_index}_end")

        if group_rest_ms > 0:
            if not d.present_for(blank_frame, group_rest_ms):
                return False
            self.logger.log(
                "rsvp_group_rest_offset",
                group_index=group.group_index,
                rest_ms=group_rest_ms,
            )
        return True

    def _precache_group_textures(
        self,
        group: RSVPGroup,
        *,
        draw_outline: bool,
        outline_color: tuple[int, int, int],
        outline_thickness: int,
    ) -> dict[str, TextureFrame]:
        d = self.display
        cache: dict[str, TextureFrame] = {}
        for stim in group.stimuli:
            item = stim.item
            if item.image_id in cache:
                continue
            raw = load_image(str(item.file_path))
            if raw is None:
                log.warning("Cannot read image: %s", item.file_path)
                continue
            canvas, scale, x0, y0 = fit_image_to_canvas_with_transform(
                raw, d.width, d.height)
            if draw_outline:
                all_bboxes = item.metadata.get("all_bboxes", {}).get(group.hint)
                all_segmentations = item.metadata.get(
                    "all_segmentations", {}).get(group.hint)
                if all_bboxes:
                    for idx, bbox in enumerate(all_bboxes):
                        segmentation = None
                        if isinstance(all_segmentations, list) and idx < len(all_segmentations):
                            segmentation = all_segmentations[idx]
                        draw_annotation_outline(
                            canvas,
                            bbox=bbox,
                            segmentation=segmentation,
                            scale=scale,
                            offset_x=x0,
                            offset_y=y0,
                            color=outline_color,
                            thickness=outline_thickness,
                        )
                else:
                    segmentations = item.metadata.get("segmentations", {})
                    draw_annotation_outline(
                        canvas,
                        bbox=item.bboxes.get(group.hint),
                        segmentation=segmentations.get(group.hint),
                        scale=scale,
                        offset_x=x0,
                        offset_y=y0,
                        color=outline_color,
                        thickness=outline_thickness,
                    )
            frame = d.to_texture_frame(d.stamp_photodiode(canvas, active=True))
            cache[item.image_id] = frame
        log.info(
            "[group %02d] cached %d unique images",
            group.group_index, len(cache),
        )
        return cache

    @staticmethod
    def estimate_timing(
        groups: list[RSVPGroup],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        on_ms = int(config.get("stim_on_ms", 300))
        off_ms = int(config.get("stim_off_ms", 200))
        hint_ms = int(config.get("hint_ms", 2000))
        group_rest_ms = int(config.get("group_rest_ms", 1000))
        initial_rest_ms = int(config.get("initial_rest_ms", 3000))
        fixation_ms = int(config.get("fixation_ms", 600))
        final_rest_ms = int(config.get("final_rest_ms", 2000))
        groups_per_rest = int(config.get("groups_per_rest", 40))
        rest_after_groups_ms = int(config.get("rest_after_groups_ms", 300000))
        wait_for_enter = bool(config.get("wait_for_enter_after_rest", True))

        n_groups = len(groups)
        n_presentations = sum(len(g.stimuli) for g in groups)
        group_seconds = [
            RSVPAttentionParadigm._estimate_group_seconds(
                len(g.stimuli), hint_ms, on_ms, off_ms, group_rest_ms)
            for g in groups
        ]
        long_rest_count = (
            (n_groups - 1) // groups_per_rest
            if groups_per_rest > 0 and n_groups > 0 else 0
        )
        active_seconds = (
            (initial_rest_ms + fixation_ms + final_rest_ms) / 1000.0
            + sum(group_seconds)
        )
        scheduled_rest_seconds = long_rest_count * (rest_after_groups_ms / 1000.0)
        total_without_enter_seconds = active_seconds + scheduled_rest_seconds
        return {
            "n_groups": n_groups,
            "n_presentations": n_presentations,
            "stim_on_ms": on_ms,
            "stim_off_ms": off_ms,
            "hint_ms": hint_ms,
            "group_rest_ms": group_rest_ms,
            "initial_rest_ms": initial_rest_ms,
            "fixation_ms": fixation_ms,
            "final_rest_ms": final_rest_ms,
            "groups_per_rest": groups_per_rest,
            "long_rest_count": long_rest_count,
            "rest_after_groups_ms": rest_after_groups_ms,
            "wait_for_enter_after_rest": wait_for_enter,
            "group_seconds_min": min(group_seconds) if group_seconds else 0.0,
            "group_seconds_max": max(group_seconds) if group_seconds else 0.0,
            "group_seconds_mean": (
                sum(group_seconds) / len(group_seconds) if group_seconds else 0.0
            ),
            "active_seconds": active_seconds,
            "scheduled_rest_seconds": scheduled_rest_seconds,
            "total_without_enter_seconds": total_without_enter_seconds,
        }

    @staticmethod
    def _estimate_group_seconds(
        n_presentations: int,
        hint_ms: int,
        on_ms: int,
        off_ms: int,
        group_rest_ms: int,
    ) -> float:
        return (
            hint_ms
            + n_presentations * (on_ms + off_ms)
            + group_rest_ms
        ) / 1000.0

    @staticmethod
    def _estimate_presentations_seconds(
        n_presentations: int,
        on_ms: int,
        off_ms: int,
    ) -> float:
        return n_presentations * (on_ms + off_ms) / 1000.0

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        total = int(round(seconds))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}h{m:02d}m{s:02d}s"
        return f"{m:d}m{s:02d}s"

    @classmethod
    def _log_timing_summary(cls, timing: dict[str, Any]) -> None:
        log.info(
            "Timing estimate: active=%s scheduled_rest=%s total(no Enter wait)=%s",
            cls._format_duration(float(timing["active_seconds"])),
            cls._format_duration(float(timing["scheduled_rest_seconds"])),
            cls._format_duration(float(timing["total_without_enter_seconds"])),
        )
        log.info(
            "Timing detail: groups=%d presentations=%d group[min/mean/max]=%s/%s/%s "
            "long_rests=%d every=%d groups",
            timing["n_groups"],
            timing["n_presentations"],
            cls._format_duration(float(timing["group_seconds_min"])),
            cls._format_duration(float(timing["group_seconds_mean"])),
            cls._format_duration(float(timing["group_seconds_max"])),
            timing["long_rest_count"],
            timing["groups_per_rest"],
        )

    def _show_group_break(
        self,
        *,
        completed_groups: int,
        total_groups: int,
        rest_ms: int,
        wait_for_enter: bool,
    ) -> bool:
        d = self.display
        rest_canvas = make_text_canvas(
            "REST",
            d.width,
            d.height,
            font_scale=2.4,
            thickness=5,
            subtitle=f"{completed_groups}/{total_groups} groups complete",
        )
        rest_frame = d.to_texture_frame(d.stamp_photodiode(
            rest_canvas, active=False))

        self.logger.log(
            "group_block_rest_start",
            completed_groups=completed_groups,
            total_groups=total_groups,
            rest_ms=rest_ms,
            wait_for_enter=wait_for_enter,
        )
        self._safe_eyetracker_mark(
            f"group_block_rest_start_{completed_groups}")

        if rest_ms > 0 and not d.present_for(rest_frame, rest_ms):
            return False

        if wait_for_enter:
            enter_canvas = make_text_canvas(
                "PRESS ENTER",
                d.width,
                d.height,
                font_scale=2.0,
                thickness=4,
                subtitle="Continue the experiment",
            )
            enter_frame = d.to_texture_frame(d.stamp_photodiode(
                enter_canvas, active=False))
            if not d.wait_for_enter(enter_frame):
                return False

        self.logger.log(
            "group_block_rest_end",
            completed_groups=completed_groups,
            total_groups=total_groups,
        )
        self._safe_eyetracker_mark(f"group_block_rest_end_{completed_groups}")
        return True

    def _make_hint_canvas(
        self,
        *,
        hint: str,
        group_number: int,
        total_groups: int,
        width: int,
        height: int,
    ):
        """Render the HINT cue. Chinese needs a TTF font (Pillow); English
        keeps the legacy cv2.putText path so a system without Pillow still
        runs."""
        cfg = getattr(self, "_hint_render_cfg", None) or {}
        lang = str(cfg.get("language", "en")).lower()
        zh_map: dict[str, str] = cfg.get("zh_map", {}) or {}
        zh_label = zh_map.get(hint) or zh_map.get(hint.lower())
        en_label = hint.replace("_", " ").upper()

        # English-only fallback path (legacy behaviour).
        if lang == "en" or not zh_label:
            return make_text_canvas(
                en_label,
                width,
                height,
                subtitle=(
                    f"Attend to: {hint} | RSVP Group "
                    f"{group_number}/{total_groups}"
                ),
            )

        font_path = cfg.get("font_path", "C:/Windows/Fonts/msyh.ttc")
        font_size = int(cfg.get("font_size", 240))
        sub_size = int(cfg.get("subtitle_size", 56))

        if lang == "both":
            big = zh_label
            sub = (
                f"{en_label}  ·  Group {group_number}/{total_groups}"
            )
        else:  # 'zh'
            big = zh_label
            sub = f"第 {group_number}/{total_groups} 组"

        return make_text_canvas_pil(
            big,
            width,
            height,
            font_path=font_path,
            font_size=font_size,
            subtitle=sub,
            subtitle_size=sub_size,
        )

    def _wait_for_gaze_return(
        self,
        *,
        group_index: int,
        stim_index: int,
    ) -> bool:
        """Pause loop for gaze loss. Shows a Chinese pause screen with the
        live gaze cursor (so the subject can self-correct), and resumes
        only after the tracker delivers continuous valid samples for
        ``gaze_resume_window_ms``. Returns False if the operator aborted
        the session via the close button.

        Importantly, this does NOT poll the eye-tracker for connection
        state — that is done at startup. By the time we reach this
        function, the tracker hardware is known good; we are merely
        waiting for the *subject's eyes* to come back into view.
        """
        d = self.display
        mon = self._monitor_cfg
        cfg = getattr(self, "_hint_render_cfg", None) or {}

        # Render the pause screen once. PIL TTF caching means re-creation
        # is cheap, but we only need it once per pause event anyway.
        canvas = make_text_canvas_pil(
            mon["pause_message_zh"],
            d.width,
            d.height,
            font_path=cfg.get("font_path", "C:/Windows/Fonts/msyh.ttc"),
            font_size=mon["pause_font_size"],
            subtitle=mon["pause_message_zh_sub"],
            subtitle_size=mon["pause_font_subtitle_size"],
            fg_color=(255, 200, 80),       # warm amber — distinct from HINT
            subtitle_color=(200, 200, 200),
        )
        # Photodiode INACTIVE — pause screens must not generate spurious
        # onset markers in offline analysis.
        pause_frame = d.to_texture_frame(d.stamp_photodiode(canvas, active=False))

        # Install overlay: live cursor + center fixation; no warning ring.
        self._enable_isi_overlay(warning=False)

        frame_ms = max(1.0, d.frame_interval_s * 1000.0)
        resume_target_frames = max(
            1,
            int(round(mon["gaze_resume_window_ms"] / frame_ms)),
        )

        state = {"consec_ok_frames": 0, "resumed": False}
        pause_start = time.perf_counter()
        log.warning(
            "Gaze lost mid-experiment (group=%d stim=%d) — pausing until "
            "tracker delivers %d consecutive valid samples (~%d ms).",
            group_index, stim_index,
            resume_target_frames, mon["gaze_resume_window_ms"],
        )
        self.logger.log(
            "experiment_paused",
            reason="gaze_lost",
            group_index=group_index,
            stim_index=stim_index,
            resume_target_frames=resume_target_frames,
        )

        invert_y = mon["cursor_y_invert"]

        def _on_frame(_elapsed: float, _state=state) -> None:
            if self.eyetracker is None:
                return
            g = self.eyetracker.get_latest_gaze()
            if g is None:
                d.set_cursor(None)
                _state["consec_ok_frames"] = 0
                return
            gx, gy = g
            if invert_y:
                gy = -gy
            d.set_cursor((float(gx), float(gy)))
            _state["consec_ok_frames"] += 1
            if _state["consec_ok_frames"] >= resume_target_frames:
                _state["resumed"] = True

        def _stop(_elapsed: float, _state=state) -> bool:
            return _state["resumed"]

        ok = d.present_until(
            pause_frame,
            stop_predicate=_stop,
            on_frame=_on_frame,
            max_duration_s=600.0,    # 10 min hard cap — operator can ESC
        )
        d.clear_overlay()

        elapsed_s = time.perf_counter() - pause_start
        if ok:
            log.info(
                "Gaze re-acquired after %.2f s pause — resuming experiment.",
                elapsed_s,
            )
            # Reset the persistent watchdog so the next trial does not
            # immediately re-trigger on the residual loss timer.
            self._gaze_last_seen_qpc = time.perf_counter()
            self._gaze_loss_episode_logged = False
            self.logger.log(
                "experiment_resumed",
                reason="gaze_recovered",
                group_index=group_index,
                stim_index=stim_index,
                pause_duration_s=elapsed_s,
            )
            self._safe_eyetracker_mark(
                f"gaze_recovered_g{group_index}_s{stim_index}")
        else:
            log.warning(
                "Gaze pause aborted by operator after %.2f s.",
                elapsed_s,
            )
            self.logger.log(
                "experiment_resumed",
                reason="aborted",
                group_index=group_index,
                stim_index=stim_index,
                pause_duration_s=elapsed_s,
            )
        return ok

    def _gaze_watchdog_step(
        self,
        gaze: tuple[float, float] | None,
    ) -> bool:
        """Advance the persistent gaze-loss watchdog with the latest
        sample (None = stale). Returns True if continuous loss has
        reached or exceeded ``gaze_loss_grace_ms``.

        Centralises the watchdog logic so ON's gaze check and ISI's
        cursor update share the same notion of "lost vs present", and
        the loss timer accumulates ACROSS trial boundaries — fixing
        the prior bug where a single ON window (e.g. 350 ms) was too
        short to ever exceed grace (400 ms) on its own.
        """
        mon = self._monitor_cfg
        if not mon["enabled"]:
            return False
        now = time.perf_counter()
        if gaze is not None:
            self._gaze_last_seen_qpc = now
            self._gaze_loss_episode_logged = False
            return False
        if self._gaze_last_seen_qpc is None:
            # First call ever and still no sample — treat current time
            # as the start of the loss window.
            self._gaze_last_seen_qpc = now
            return False
        loss_ms = (now - self._gaze_last_seen_qpc) * 1000.0
        return loss_ms >= mon["gaze_loss_grace_ms"]

    def _enable_isi_overlay(self, *, warning: bool) -> None:
        """Push monitor settings into DisplayEngine.OverlayState before
        the ISI loop. Called once per ISI; the per-frame callback only
        flips the cursor coordinate, which is the cheapest possible
        update path."""
        mon = self._monitor_cfg
        ov = OverlayState(
            cursor_xy_norm=None,                         # filled per frame
            cursor_radius_px=mon["cursor_radius_px"],
            cursor_color_rgb=mon["cursor_color_rgb"],
            cursor_clip_to_edge=mon["cursor_clip_to_edge"],
            cursor_offscreen_color_rgb=mon["cursor_offscreen_color_rgb"],
            fixation_marker_radius_px=mon["fixation_marker_radius_px"],
            fixation_marker_color_rgb=mon["fixation_marker_color_rgb"],
            warning_active=bool(warning),
            warning_ring_radius_px=mon["warning_ring_radius_px"],
            warning_ring_color_rgb=mon["warning_ring_color_rgb"],
            warning_ring_thickness_px=mon["warning_ring_thickness_px"],
        )
        self.display.set_overlay(ov)

    def _safe_eyetracker_mark(self, tag: str) -> None:
        if self.eyetracker is None:
            return
        try:
            self.eyetracker.mark(tag)
        except Exception:
            log.warning("eyetracker.mark(%r) failed", tag, exc_info=True)
