"""Nat Comm-style colour palette + matplotlib defaults used by process2."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# NPG (ggsci) palette — frequently used in Nature Communications figures
NPG = [
    "#E64B35",  # red
    "#4DBBD5",  # cyan
    "#00A087",  # teal
    "#3C5488",  # navy
    "#F39B7F",  # salmon
    "#8491B4",  # slate
    "#91D1C2",  # mint
    "#DC0000",  # crimson
    "#7E6148",  # brown
    "#B09C85",  # tan
]

# Baseline-specific colours (consistent across all figures)
BASELINE_COLOR = {
    "img_only": "#8491B4",
    "eeg_only": "#4DBBD5",
    "eeg_img":  "#E64B35",
    "eeg_img_patch": "#E64B35",
}
BASELINE_LABEL = {
    "img_only": "Pic embedding only",
    "eeg_only": "EEG only",
    "eeg_img":  "EEG + Pic",
    "eeg_img_patch": "EEG + Pic (patch attn)",
}

NATCOMM_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_natcomm_style():
    mpl.rcParams.update(NATCOMM_RC)
