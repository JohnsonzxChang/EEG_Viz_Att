"""Publication-quality matplotlib style for Process-1 figures.

Targets Nature-Communications-level rendering: vector-friendly SVG/PDF,
300 dpi PNG fallback, sans-serif Arial-like font, consistent palette.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# Nature Comms palette — colorblind-safe, distinguishable in grayscale
PALETTE = {
    "target":     "#D62728",   # red — attended / target
    "nontarget":  "#1F77B4",   # blue — non-attended / standard
    "diff":       "#2CA02C",   # green — difference wave
    "gfp":        "#E45756",
    "p100":       "#4C78A8",
    "n170":       "#F58518",
    "p300":       "#54A24B",
    "train":      "#4C78A8",
    "test":       "#E45756",
    "neutral":    "#7F7F7F",
}

SUPERCLASS_COLORS = {
    "Animal":     "#1F77B4",
    "Food":       "#FF7F0E",
    "Vehicle":    "#2CA02C",
    "Furniture":  "#D62728",
    "Kitchen":    "#9467BD",
    "Wearable":   "#8C564B",
    "Electronics":"#E377C2",
    "Other":      "#7F7F7F",
}


def apply_publication_style() -> None:
    """Call once before plotting. Idempotent."""
    mpl.rcParams.update({
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":           9,
        "axes.titlesize":      10,
        "axes.titleweight":    "bold",
        "axes.labelsize":      9,
        "axes.linewidth":      0.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "xtick.major.width":   0.8,
        "ytick.major.width":   0.8,
        "xtick.major.size":    3,
        "ytick.major.size":    3,
        "legend.fontsize":     8,
        "legend.frameon":      False,
        "figure.dpi":          150,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
        "savefig.transparent": False,
        "lines.linewidth":     1.2,
        "lines.markersize":    4,
        "pdf.fonttype":        42,  # embed as TrueType (editable in Illustrator)
        "ps.fonttype":         42,
        "svg.fonttype":        "none",
    })


def save_fig(fig, path, *, also_svg: bool = True, also_pdf: bool = False) -> None:
    """Save PNG (raster, 300dpi) and optionally SVG/PDF (vector).

    For Nature-level submissions, prefer SVG/PDF; PNG is for quick previews.
    """
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    if also_svg:
        fig.savefig(str(path).replace(".png", ".svg"), bbox_inches="tight")
    if also_pdf:
        fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
