"""Load epochs and join with the session-JSON per-stim metadata.

The epochs (.fif) carry only `image_id/category` in the event_id mapping.
The HINT (attention category), `targets_in_image`, `eeg_split`, repeat
information lives in the session JSON. We chronologically align the i-th
*surviving* stim_onset event from the session JSON to the i-th epoch.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class EpochBundle:
    data: np.ndarray
    times: np.ndarray
    sfreq: float
    ch_names: list[str]
    stim_category: np.ndarray
    image_id: np.ndarray
    hint: np.ndarray
    eeg_split: np.ndarray
    is_target: np.ndarray
    repeat_index: np.ndarray
    targets_in_image: list[list[str]]
    target_areas: list[dict]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.data)


def _parse_event_name(name: str) -> tuple[int, str]:
    img_str, cat = name.split("/", 1)
    return int(img_str), cat


def load_epochs(fif_path, session_json_path,
                pick_eeg: bool = False, load_data: bool = True,
                decim: int = 1, dtype: str = "float32",
                crop_tmin: float | None = None,
                crop_tmax: float | None = None,
                baseline: tuple[float, float] | None = None,
                filter_band: tuple[float, float] | None = None) -> EpochBundle:
    """Load .fif epochs, join session JSON metadata, return a flat bundle.

    Args
    ----
    crop_tmin/crop_tmax : restrict epoch window (s). For SOA=514 ms RSVP,
        use (-0.1, 0.45) to avoid neighbour-trial spillover.
    baseline : (tmin, tmax) in s for mean subtraction. Pass None to skip
        (recommended for fast RSVP — see plan B).
    filter_band : (lo, hi) Hz. If given, apply a 4-th order zero-phase
        Butterworth bandpass to each epoch (plan B).
    """
    import mne

    ep = mne.read_epochs(str(fif_path), preload=False, verbose="ERROR")
    log.info("Loaded %d epochs, %d channels @ %.0fHz, tmin/tmax=%.2f/%.2f s",
             len(ep), len(ep.ch_names), ep.info["sfreq"], ep.tmin, ep.tmax)

    code_to_name = {v: k for k, v in ep.event_id.items()}
    stim_cat, img_ids = [], []
    for code in ep.events[:, 2]:
        iid, cat = _parse_event_name(code_to_name[int(code)])
        stim_cat.append(cat); img_ids.append(iid)
    stim_cat_arr = np.array(stim_cat)
    img_id_arr = np.array(img_ids, dtype=np.int64)

    with open(session_json_path, encoding="utf-8") as f:
        sess = json.load(f)
    stim_events = [e for e in sess["events"] if e["name"] == "stim_onset"]
    log.info("Session JSON has %d stim_onset events, epochs have %d → %.1f%% survival",
             len(stim_events), len(ep),
             100.0 * len(ep) / max(1, len(stim_events)))

    sess_pointer = 0
    hint_out, split_out, rep_idx_out, targets_out, areas_out = [], [], [], [], []
    n_dropped = n_matched = 0
    for ep_idx in range(len(ep)):
        want_img = int(img_id_arr[ep_idx])
        want_cat = stim_cat_arr[ep_idx]
        while sess_pointer < len(stim_events):
            p = stim_events[sess_pointer]["payload"]
            if int(p["image_id"]) == want_img and p["outlined_label"] == want_cat:
                hint_out.append(p["hint"])
                split_out.append(p.get("eeg_split", ""))
                rep_idx_out.append(int(p.get("repeat_index", 0)))
                targets_out.append(list(p.get("targets_in_image", [])))
                areas_out.append(dict(p.get("target_areas", {})))
                sess_pointer += 1; n_matched += 1
                break
            sess_pointer += 1; n_dropped += 1
        else:
            hint_out.append(""); split_out.append("")
            rep_idx_out.append(-1); targets_out.append([]); areas_out.append({})
    log.info("Joined: %d/%d epochs got HINT metadata (%d session events skipped)",
             n_matched, len(ep), n_dropped)
    hint_arr = np.array(hint_out)
    split_arr = np.array(split_out)
    rep_idx_arr = np.array(rep_idx_out, dtype=np.int64)

    if load_data:
        if pick_eeg:
            keep_idx = [i for i, n in enumerate(ep.ch_names)
                         if n not in ("gaze_x", "gaze_y")]
        else:
            keep_idx = list(range(len(ep.ch_names)))
        ch_names = [ep.ch_names[i] for i in keep_idx]

        full_times = ep.times
        t_mask = np.ones(full_times.shape, dtype=bool)
        if crop_tmin is not None:
            t_mask &= (full_times >= crop_tmin)
        if crop_tmax is not None:
            t_mask &= (full_times <= crop_tmax)
        keep_t = np.where(t_mask)[0]
        if len(keep_t) < len(full_times):
            log.info("Cropping epoch window to [%.3f, %.3f] s (%d→%d samples)",
                     full_times[keep_t[0]], full_times[keep_t[-1]],
                     len(full_times), len(keep_t))

        chunk = 256
        parts = []
        for s in range(0, len(ep), chunk):
            e = min(len(ep), s + chunk)
            d = ep.get_data(item=range(s, e), picks=keep_idx,
                             verbose="ERROR").astype(dtype)
            d = d[..., keep_t]
            if decim > 1:
                d = d[..., ::decim]
            parts.append(d)
        data = np.concatenate(parts, axis=0) * 1e6
        del parts
        times = full_times[keep_t]
        if decim > 1:
            times = times[::decim]
        sfreq = float(ep.info["sfreq"]) / float(decim if decim > 1 else 1)

        if filter_band is not None:
            from scipy import signal as sg
            lo, hi = filter_band
            nyq = sfreq / 2
            if hi >= nyq: hi = nyq - 1
            sos = sg.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
            data = sg.sosfiltfilt(sos, data, axis=2).astype(dtype)
            log.info("Applied bandpass %.1f-%.1f Hz", lo, hi)

        if baseline is not None:
            b0, b1 = baseline
            bmask = (times >= b0) & (times <= b1)
            if bmask.any():
                base = data[..., bmask].mean(axis=2, keepdims=True)
                data = data - base
                log.info("Subtracted baseline mean over [%.3f, %.3f] s", b0, b1)
            else:
                log.warning("Baseline window [%.3f, %.3f] outside epoch", b0, b1)
    else:
        data = np.empty((0, 0, 0), dtype=np.float32)
        ch_names = list(ep.ch_names)
        if pick_eeg:
            ch_names = [n for n in ch_names if n not in ("gaze_x", "gaze_y")]
        times = ep.times
        sfreq = float(ep.info["sfreq"])

    is_target = (stim_cat_arr == hint_arr)
    return EpochBundle(
        data=data,
        times=np.asarray(times, dtype=np.float32),
        sfreq=sfreq,
        ch_names=ch_names,
        stim_category=stim_cat_arr,
        image_id=img_id_arr,
        hint=hint_arr,
        eeg_split=split_arr,
        is_target=is_target,
        repeat_index=rep_idx_arr,
        targets_in_image=targets_out,
        target_areas=areas_out,
        meta={
            "fif_path": str(fif_path),
            "session_json": str(session_json_path),
            "n_session_stim_events": len(stim_events),
            "n_dropped_during_join": n_dropped,
            "crop_tmin": crop_tmin, "crop_tmax": crop_tmax,
            "baseline": baseline, "filter_band": filter_band,
        },
    )
