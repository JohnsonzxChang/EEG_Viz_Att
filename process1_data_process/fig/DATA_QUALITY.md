# Data-quality assessment — `zfn-0507` pilot

Compiled from `process1_summary.json` (fig/, fig_clean/, fig_overlap/) and
all rendered figures. Use this to prioritise pipeline fixes vs. paradigm
changes before scaling to N≥8 subjects.

## TL;DR — overall grade

| Aspect | Grade | Comment |
|---|---|---|
| **Trial yield / rejection** | A | 0.3% bad-trial rate (16/6376) — excellent |
| **Class balance** | A | 40 HINT classes, min=105, max=205, mean=159 |
| **Channel integrity** | A | 0 flat channels per epoch, no saturated trials |
| **Single-trial SNR** | B | RMS 14 µV w/ bimodal distribution (see issue #1) |
| **ERP component shape** | B− | GFP peak at 52 ms (too early, see #2) |
| **Pre-stim baseline cleanliness** | C | GFP_pre/post ≈ 1.0 (RSVP overlap, see #3) |
| **Alpha desync (attention marker)** | C− | post/pre = 1.44 (alpha *increases*, see #4) |
| **HINT decoding accuracy** | B | top-1 = 0.076 (3.0× chance), top-5 = 0.276 (11× chance) |
| **Salience effect amplitude** | B− | Δ = −1.75 µV @ 172 ms; rERP-corrected ≈ −0.5 µV |
| **Target-area regression** | D | r = −0.014, p = 0.28 — no relation (see #5) |
| **Eye-tracking integration** | F | Not used yet (see #6) |

Overall: **good enough for a methods/dataset paper, but four issues
(2-5) need fixing or explicit caveats before submission.**

---

## Issue 1 · Bimodal RMS distribution (F09_qc_distributions)

The RMS-amplitude histogram is clearly **two-peaked** (≈ 12 µV and ≈ 20 µV).
Possible sources:

1. **Block-to-block impedance drift** — the flexible posterior patch likely
   loses contact slightly across the 30+ minutes of recording.
2. **Two distinct artefact regimes** — e.g., eyes-open vs. brief micro-
   sleep, or pre/post a saline re-application.
3. **Different eeg_split conditions** — train vs. test blocks at different
   times of the session.

**Fix:**
- Split `qc.rms` by `eeg_split`, by block_index, and by time-of-session,
  re-plot. If RMS correlates with block_index, do per-block z-score
  normalisation OR add per-block re-referencing.
- For multi-subject pipeline: report `coefficient of bimodality` per
  subject; exclude subjects with > 0.6.

## Issue 2 · GFP peak at 52 ms (F02_erp_grand_average)

A visual P1 should peak at ≈ 100 ms (visual cortex L1-L4). 52 ms is
*before* feedforward V1 activation in healthy adults (Foxe & Simpson 2002).
Most likely causes:

1. **Display/stimulation latency not subtracted.** If your TTL marker
   fires *before* the actual photons hit the retina (typical LCD delay
   ~15-60 ms), the trial onset is shifted forward and the apparent
   peak moves earlier.
2. **Filtering edge artefact** if a high-pass was applied during raw→fif
   without sufficient padding.

**Fix:**
- Re-measure the **photodiode-to-marker** delay (check `adc_*.bin` /
  `adc_*.meta` files — they may already contain the photodiode trace).
- Correct event onsets by this constant. After correction the P1 should
  fall in the standard 80-130 ms band.

## Issue 3 · GFP_pre / GFP_post ≈ 1 (overlap contamination)

Confirmed: 514 ms SOA means the previous trial's N1/P2 falls inside
[-500, 0] ms baseline. **Already addressed** by:
- Plan A: 95 ms tight baseline → `fig_clean/`
- Plan C: rERP deconvolution → `fig_overlap/F18-F20`

**Recommendation for future subjects:**
- Add ≥ 200 ms ISI jitter (e.g., uniform 100-300 ms) so that the SOA is
  not a sharp delta-function and the rERP system is *fully* identifiable
  from continuous EEG (instead of relying on finite-support assumption).
- Optionally insert random "blank" trials every 5-10 stims to refresh
  baseline.

## Issue 4 · Alpha SNR post/pre = 1.44 (NO desync)

Attended-target presentation should cause **alpha desynchronisation**
(post/pre < 1, typical 0.6-0.9 over occipital cortex). This dataset shows
the *opposite* — alpha is **stronger** post-stim.

Possible causes:

1. **Patch placement.** The 4×8 grid covers occipital but may sit too
   high (parietal/medial occipital), missing the classic α-generating
   region around O1/O2/Oz.
2. **Hint cue period bleed-in.** The HINT cue lands BEFORE the stim
   onset. If hint→stim < α-cycle period, the "post" window captures the
   hint-induced **alpha rebound** rather than the stim-induced desync.
3. **Subject drowsiness** — passive RSVP at 5 Hz is hypnotic. Check eye-
   tracking pupil dilation for arousal.

**Fix:**
- Run a **simple visual-evoked attention block** (8-13 Hz flicker or
  attention-cued grating) before the RSVP and verify classic desync —
  use it as electrode-placement validation.
- For the current data: compute α power in [-500, -300] ms (deep pre,
  no hint) vs [+200, +500] ms (full stim window) instead of
  ±[100] ms — that should yield the classic desync.

## Issue 5 · target_area regression: r = −0.014, p = 0.28

The attended-object area-fraction explains **zero variance** of
post-stim mean amplitude. Interpretation options:
- Genuinely small effect — single subject, single session
- Wrong dependent variable — try **GFP @ 170 ms peak** instead of
  mean-|amplitude| over a 300 ms window
- Confound with category — large objects are over-represented in some
  HINT classes (e.g., teddy_bear), so the cat regression eats the area
  effect

**Fix:**
- Use per-HINT residuals: regress amplitude on category, then regress
  the residual on log(area)
- Try **logistic mixed-effects model** in R/lme4 with random intercept
  per image_id

## Issue 6 · Eye-tracking unused

`eye_*.csv` (24 MB) and `gaze_x` / `gaze_y` channels in the fif are
recorded but currently dropped (`pick_eeg=true`). The HINT-attention
paradigm makes gaze a critical confound — saccades to the outlined
target produce a CORNEO-RETINAL artefact at exactly the same latency
as the visual ERP.

**Fix (high priority):**
- Add `attention_effect.gaze_diagnostics()` module:
  - Per-trial gaze deviation from fixation centre
  - Per-trial fixation-stability score
  - Drop trials with saccade > 2° within [0, +300] ms
- Use gaze as an auxiliary feature in `process2/` deep models

## Issue 7 · Single-subject statistics

All numbers reported here are single-session, single-subject. No
group-level CI is possible. **The full N≥8 collection is the next
critical milestone** before any of the above issues can be cleanly
disentangled from individual variability.

---

## Concrete to-do list (priority-ordered)

1. **[paradigm]** Add ISI jitter 100-300 ms for next pilot (fixes #3 fully)
2. **[acquisition]** Photodiode latency calibration (fixes #2)
3. **[acquisition]** Re-position patch with O1/Oz/O2 covered (fixes #4)
4. **[pipeline]** Gaze-based artefact rejection (fixes #6)
5. **[pipeline]** Per-block RMS normalisation (fixes #1)
6. **[stats]** Mixed-effects model for area regression (improves #5)
7. **[paper]** Run 8 subjects with all the above, expect:
   - HINT top-1 LDA ≈ 0.10-0.15
   - HINT top-5 LDA ≈ 0.35-0.45
   - Plan-C-corrected salience effect = -1 µV @ 170 ms (currently -0.5
     after overlap correction, will grow with proper electrode placement)

## What's already great about this dataset

- **0.3% rejection rate** — flexible cap is performing well
- **40 categories × 159 trials / class** — comparable to ATM 12×500 in
  effective data volume per class
- **Multi-label image-level structure** — every image has K=2-3 targets,
  enabling within-image attention contrasts that simple oddball designs
  can't (this is the unique selling-point for process2/)
- **eeg_split column** — train/test pre-split at acquisition time avoids
  data leakage in downstream deep models
- **Full provenance** in session JSON (image_id, marker_code, target_areas,
  repeat_index) — sufficient for replication

