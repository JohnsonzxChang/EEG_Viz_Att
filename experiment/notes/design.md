# Experiment Design Notes

## 0. Two-phase architecture
The pipeline is split into:

- **Phase 1 — selection** (`phase1_select.py`):
  reads `configs/*.yaml`, runs LVIS filtering once, and writes a JSON
  that contains only `image_id` + `relative_path` (relative to
  `dataset.coco_root`) per item. The JSON is the single source of truth
  handed to Phase 2 — that way the (slow) filtering pass runs once and
  is reproducible across subjects/sessions.
- **Phase 2 — presentation** (`phase2_run.py`):
  reads YAML + the Phase-1 JSON, opens the GLFW/OpenGL window, talks to
  the EEG TTL (or skips it — see §4), records eye-tracking samples, and
  runs the RSVP-attention paradigm.

`run.py` is a one-shot dispatcher that invokes both back-to-back.

## 1. Paradigm overview
RSVP with **attention modulation**: every trial is preceded by a HINT (text)
that names the *to-be-attended* category. The same stimulus image is
re-used across multiple trials with different HINTs (K=4 targets present
per image), so within-image neural variance is dominated by attention,
not by visual content.

```
[HINT 2.0s]  →  [15 × (ON 0.3s + OFF 0.2s) = 7.5s]  →  [REST rand(2.5..5.5)s]
└ "DOG"          ├ same HINT for all 15 imgs                    └ blank
                 ─ images sampled so HINT category is always present
```

Trial total: 12.0–15.0 s (deterministic = 9.5 s, plus random rest).

## 2. Experiment timeline
- Session: ≤ 10 min of continuous trials (auto-split).
- Inter-session rest: 5 min (text screen + blank).
- Total experiment cap: 2 h.
- ≈ 8 sessions × ~44 trials ≈ 350 trials × 15 imgs = **~5300 stimulus
  presentations** (with image repetition across attention groups).

## 3. Stimulus selection (LVIS-on-COCO)
Constraints (all configurable in YAML):
- `targets_per_image: 4` — exactly K=4 LVIS categories from the 40-class
  universe present in the image.
- `instances_per_target: 1` — each target appears exactly once
  (no ambiguity for the subject).
- `min_area_frac: 0.02`, `max_area_frac: 0.40` — bbox area fraction
  bounded to keep stimuli neither tiny (unrecognizable) nor overwhelming
  (single-object close-up).
- `max_images_per_target: 800` — per-category cap to prevent dominant
  classes (e.g. dog, car) from skewing the marginals.

The 40 default categories span the man-made vs natural × animate vs
inanimate space. No `human face` / `person` to avoid face-specific ERPs
from leaking in.

## 4. TTL toggle
`markers.serial.enabled` controls whether 1-byte TTL bytes are written
to the EEG amplifier:

| Mode | Behavior |
|---|---|
| **enabled = true**  | Serial bytes are sent on each stim ON, HINT, and trial/session boundary. The amplifier's hardware trigger drives offline epoching. |
| **enabled = false** | No bytes are sent. Every `stim_onset` event in the session JSON carries an additional `cv2_tick_qpc` field — read from `cv2.getTickCount()` at the exact same screen-flip the QPC tick is read on. Use this when the EEG TTL line isn't connected (e.g. dry-run rehearsals or mobile rigs). |

Because `cv2.getTickCount` wraps `QueryPerformanceCounter` on Windows,
its values agree with both the Python `PerfClock` and the Tobii bridge's
QPC reading down to QPC resolution. Offline alignment uses the same
merge-asof recipe regardless of which mode was used.

## 5. Marker scheme (1-byte UART)
| Code | Meaning |
|---|---|
| 1..40 | HINT category (also stamped on every stimulus ON during the trial) |
| 250 | HINT_PREFIX — sent on the screen flip that brings up the HINT text |
| 251 | trial_start |
| 252 | trial_end (start of REST) |
| 253 | session_start |
| 254 | session_end |
| 255 | experiment_end |

The combination "(250 → C) within 50 ms" unambiguously marks a HINT for
category C, so offline parsing can disambiguate HINT-onset codes from
stimulus-ON codes carrying the same C.

## 5. Time alignment
- **Python clock** = `time.perf_counter()` (Windows QPC).
- **Tobii bridge clock** = `QueryPerformanceCounter` (same QPC).
- **EEG marker latency** = 1-byte UART write @ 115200 baud + FTDI buffer
  ≈ 0.5–1 ms after `glfwSwapBuffers`.
- **Photodiode** is the ground truth for offline alignment if a
  photo-diode-to-trigger box is wired in parallel to the serial line.

## 6. File outputs per session
```
logs/
  session_<exp>_<ts>.json     # all paradigm events with QPC ticks
  eye_<exp>_<ts>.csv          # gaze + head pose stream from Tobii bridge
```
The two streams share the QPC tick axis and can be merged with
`pandas.merge_asof`.

## 7. Things to verify on the experiment rig
- [ ] Display refresh rate matches `display.width × display.height`
      (check `engine.refresh_hz` in the run log).
- [ ] Photodiode square is visible to the photo-sensor in the chosen
      corner (`markers.photodiode.corner`).
- [ ] EEG amplifier triggers fire on the chosen COM port at 115200 baud.
      Test with `python -c "from markers.serial_marker import
      SerialMarker; s = SerialMarker('COM3'); s.send(1)"`.
- [ ] Tobii Game Hub recognizes ET5 (`tobii_bridge.dll` returns 0 from
      `tb_init`).
- [ ] Run with `--dry-run` first to confirm dataset filtering yields
      enough images per target (~ ≥ 60 to fill 4 trials per category).
