# Process-1 Figure Plan — Attention-RSVP EEG dataset paper

**Pilot subject:** `zfn-0507` (6376 epochs, 32 occipital EEG, 1000→250 Hz).
**Recorded SOA:** **514 ± 8 ms** (352 ms stim + 156 ms ISI), confirmed from
session JSON inter-onset deltas. The fif epoch window [-500, +1000] ms
spans ≈ 3 SOAs ⇒ 3 visible ERP cycles in the grand average.

## Directory layout

```
process1_data_process/
├── fig/                       # main pipeline (baseline-subtracted, full window)
│   ├── F01-F17_*.png/.svg
│   ├── process1_summary.json
│   ├── temporal_sweep_results.json
│   ├── decode_results.json
│   └── FIGURE_PLAN.md         (this file)
├── fig_clean/                 # tight-window analyses (-100…+450 ms)
└── fig_overlap/               # plans B + C (overlap-aware)
    ├── F_PLANB_filter_vs_baseline.png/.svg
    ├── F18_rerp_gfp_before_after.png/.svg
    ├── F19_rerp_butterfly.png/.svg
    ├── F20_rerp_salient_vs_nonsalient.png/.svg
    └── overlap_summary.json
```

## Article-level figure composition

### MF1 · Paradigm & dataset
| Panel | Source |
|---|---|
| A | (illustrator) RSVP block schematic, hint cue → image stream |
| B | `F01_electrode_layout` 4×8 posterior patch |
| C | `F10_class_balance` 40 HINT categories, super-class colored |
| D | `F11_multilabel_cooccurrence` P(b\|a) of K targets per image |

### MF2 · Signal quality & ERP baseline
| Panel | Source |
|---|---|
| A | `F09_qc_distributions` (< 1% bad trials) |
| B | `F02_erp_grand_average` butterfly + GFP — visible 3-cycle overlap structure |
| C | `F07_topomap_snapshots` posterior negativity at 170 ms |
| D | `F03_erp_channel_heatmap` channel × time |

### MF3 · Overlap is real (NEW) + clean response after correction
**This is the SOA = 514 ms story panel.**

| Panel | Source |
|---|---|
| A | `F_PLANB_filter_vs_baseline` — bandpass vs short-baseline GFP, dashed lines at ±SOA show the periodicity |
| B | `F18_rerp_gfp_before_after` (top) — observed grand-average GFP with 3 visible cycles |
| C | `F18_rerp_gfp_before_after` (bottom) — Tikhonov-deconvolved single-event response (clean N1 / P2 / late) |
| D | `F19_rerp_butterfly` observed vs deconvolved butterfly, same y-axis |

**Caption**: *Constant 514 ms SOA produces overlapping responses in the
grand-average ERP. A linear Tikhonov-regularised deconvolution
(`x ∈ ℝ^{n_resp}`, support [-100, +450] ms, A ∈ ℝ^{376×138}, ridge λ=1.0)
recovers the single-event response (cond=7.0; 2.7× over-determined).
Cleaned-up N1 peaks at ≈ 80 ms, P2 at ≈ 150 ms, with a late component
between 200-300 ms.*

### MF4 · Salience modulation on overlap-corrected data
| Panel | Source |
|---|---|
| A | `F06_target_vs_nontarget_erp` (salient vs non-salient median split, observed ERP, before correction) |
| B | `F20_rerp_salient_vs_nonsalient` deconvolved salient vs non-salient (NEW) |
| C | `F06b_target_effect_topomap` |
| D | `F13_target_area_regression` |
| E | `F12_alpha_desync` |
| F | `F14_per_target_count_erp` |

**Caption**: *After overlap correction (B) the salience effect is
isolated to the 100-300 ms window with peak Δ ≈ 0.5 µV; the pre-correction
"cycles" outside this window in (A) are now confirmed to be neighbour-
trial responses, not genuine late effects.*

### MF5 · Linear-readout calibration (baseline for process2)
| Panel | Source |
|---|---|
| A | `F17_temporal_sweep_hint` sliding-window LDA (top-1/5/balanced) |
| B | `F15_hint_confusion` 40-class confusion at [0, 500] ms |
| C | `F16_erp_averaging_effect` accuracy vs k |
| D | `F04_erp_per_superclass` per-superclass ERP |

### SF · Per-category atlas / drift / reproducibility
SF1 = `F05_erp_per_hint`,  SF2 = `F08_erp_split_drift`,
SF3 = `process1_summary.json` + `overlap_summary.json`.

---

## How to handle overlap — 3-tier methodology summary

| Plan | What it does | Time sweep | Data segmentation | Code |
|---|---|---|---|---|
| **A** | Crop epoch to [-100, +450] ms, baseline [-100, 0] | sweep in [-50, +450] ms only | unchanged | `run_qc.py --crop_tmin -0.1 --crop_tmax 0.45` → `fig_clean/` |
| **B** | Bandpass 0.5-30 Hz, **no baseline** | full window with bandpass | unchanged | `run_overlap.py` (Plan B step) → `F_PLANB_*` |
| **C** | rERP linear deconvolution (Tikhonov), support-constrained | sweep on the deconvolved x(τ) (≈ 138 samples) | reads epochs.fif as-is; output is x ∈ ℝ^{n_chan × n_resp_grid} | `overlap_correction.rerp_deconvolve` → `F18-F20` |

**Recommendation**: report Plan A as the **primary** analysis (clearest
single-trial signal-to-noise), Plan C as the **methods-figure validation**
(overlap-corrected ERP confirms 100-300 ms responses are not artifactual).

## Headline numbers

| metric | observed | overlap-corrected (Plan C) |
|---|---|---|
| GFP peak (µV) | 2.56 | 1.78 (post-stim) |
| GFP pre / post ratio | 1.01 (problematic) | post=0.96 vs pre=0.43 ⇒ 2.2× |
| salience peak Δ (µV) | -1.75 @ 172 ms | ≈ -0.5 @ 200 ms (clean) |
| HINT top-1 LDA, [0,500] ms | 0.076 | TBD (rERP-projected features) |
| HINT top-5 LDA | 0.276 | TBD |
| Sweep peak top-5 | 0.249 @ +200 ms | TBD |

## Caveats fixed in Plan C
✔ GFP pre/post ratio was ≈ 1.0 because [-500, 0] ms baseline contained
   the prior trial's N1/P2 → fixed by either Plan A short baseline or
   Plan C deconvolution.
✔ Top-5 ≈ 0.20 in pre-stim window of the original sweep was due to the
   prior-trial response landing inside the "pre-stim" sweep slot → Plan A
   sweep is now restricted to [-50, +450] ms.

## How to regenerate

```bash
# Main pipeline — generates F01-F17 in fig/
python -B -m process1_data_process.run_qc \
    --fif        data/zfn-0507/epochs_big-epo.fif \
    --session_json data/zfn-0507/session_rsvp_attention_lvis_pilot_20260507_201520.json \
    --fig_dir    process1_data_process/fig \
    --crop_tmin -0.5 --crop_tmax 1.0                         # original full window

# Tight (Plan A) variant → fig_clean/
python -B -m process1_data_process.run_qc \
    --fif … --session_json … --fig_dir process1_data_process/fig_clean \
    --crop_tmin -0.1 --crop_tmax 0.45 \
    --baseline_tmin -0.1 --baseline_tmax 0.0

# Plans B + C overlap analysis → fig_overlap/
python -B -m process1_data_process.run_overlap \
    --fif … --session_json … --fig_dir process1_data_process/fig_overlap \
    --soa_ms 514 --ridge 1.0 --filter_lo 0.5 --filter_hi 30
```
