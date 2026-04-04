# src/helpers/style_guide.py
"""
SCTT Project — Centralized Style Guide
=======================================
Import this module to get consistent colours, fonts, sizes, and
a one-call `apply_theme()` that configures seaborn + matplotlib.

Usage:
    from src_phase1.helpers.style_guide import S, apply_theme
    apply_theme()
    ...
    ax.scatter(..., color=S.MODEL_COLORS['7b'], alpha=S.SCATTER_ALPHA)
"""

import matplotlib.pyplot as plt
import seaborn as sns


class S:
    """Queryable namespace of every style constant used in the project."""

    # ── Seaborn / Matplotlib theme ────────────────────────────────────────
    SNS_STYLE      = "whitegrid"
    SNS_FONT_SCALE = 1.1

    # ── Model palette ─────────────────────────────────────────────────────
    MODEL_COLORS = {
        "7b":  "#2196F3",   # Material Blue
        "13b": "#FF5722",   # Material Deep Orange
    }
    MODEL_LABELS = {
        "7b":  "Llama-2-7B",
        "13b": "Llama-2-13B",
    }
    MODELS = [("7b", "Llama-2-7B"), ("13b", "Llama-2-13B")]

    # ── Accent / utility colours ──────────────────────────────────────────
    NEGATIVE_BIAS  = "#EF5350"      # red bars for negative bias
    ANNOTATION_TXT = "#333333"      # dark-gray item labels
    IDEAL_LINE     = "k--"          # black dashed for ideal / zero lines
    IDEAL_LINE_LW  = 1
    ERROR_BAR      = "black"
    BAR_EDGE       = "white"

    # ── Scatter defaults ──────────────────────────────────────────────────
    SCATTER_ALPHA        = 0.4
    SCATTER_SIZE         = 20
    SCATTER_RASTERIZED   = True
    # Lighter variants used in specific plots
    SCATTER_ALPHA_LIGHT  = 0.25   # residual scatters
    SCATTER_SIZE_SMALL   = 10
    SCATTER_ALPHA_HELDOUT = 0.35
    SCATTER_SIZE_HELDOUT  = 15

    # ── Bar chart defaults ────────────────────────────────────────────────
    BAR_ALPHA     = 0.85
    BAR_WIDTH     = 0.35

    # ── KDE / line defaults ───────────────────────────────────────────────
    KDE_LINEWIDTH = 2

    # ── Box-plot defaults ─────────────────────────────────────────────────
    BOX_ALPHA         = 0.75
    BOX_MEDIAN_COLOR  = "black"
    BOX_MEDIAN_LW     = 1.5

    # ── OLS fit line ──────────────────────────────────────────────────────
    OLS_STYLE = "k--"
    OLS_LW    = 1

    # ── Font sizes (pt) ──────────────────────────────────────────────────
    SUPTITLE_SIZE   = 14
    TITLE_SIZE      = 13
    AXIS_LABEL_SIZE = 12
    TICK_LABEL_SIZE = 11
    LEGEND_SIZE     = 10       # 9–10 range; use 9 for tighter plots
    LEGEND_SIZE_SM  = 9
    ANNOT_SIZE      = 11       # r-value text box
    ITEM_LABEL_SIZE = 8.5      # per-item annotations
    BAR_VALUE_SIZE  = 8        # value labels on bars
    COUNT_LABEL_SIZE = 6.5     # n= labels inside bars
    CAPTION_SIZE       = 10
    CAPTION_STYLE      = "italic"
    CAPTION_FONT       = "Arial"
    CAPTION_LINESPACING = 1.0

    # ── Font weights ──────────────────────────────────────────────────────
    TITLE_WEIGHT = "bold"

    # ── Annotation text-box ───────────────────────────────────────────────
    ANNOT_BBOX = dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="lightgray",
        alpha=0.8,
    )

    # ── Error-bar styling ─────────────────────────────────────────────────
    ERRBAR_CAPSIZE = 3
    ERRBAR_LW      = 1.2

    # ── Calibration scatter (phase-2 bias plot) ───────────────────────────
    CALIB_MARKER    = "D"       # diamond for MAE twin-axis
    CALIB_MARKER_MS = 5
    CALIB_LW        = 1.5

    # ── Common figure sizes (width, height) ───────────────────────────────
    FIG_2x2       = (13, 10)
    FIG_2x2_TALL  = (13, 11)
    FIG_2x2_WIDE  = (16, 14)
    FIG_1x4       = (16, 5)
    FIG_1x2       = (14, 5)
    FIG_1xN_UNIT  = 7           # per-panel width for variable-column figs

    # ── Save defaults ─────────────────────────────────────────────────────
    SAVE_DPI_HI   = 300         # publication-quality scatters
    SAVE_DPI_LO   = 150         # supporting / summary plots
    SAVE_BBOX     = "tight"

    # ── Binning ───────────────────────────────────────────────────────────
    BINS_START = 0.0
    BINS_END   = 1.1
    BINS_STEP  = 0.1


def apply_theme():
    """Call once at the top of a notebook / script to set the global theme."""
    sns.set_theme(style=S.SNS_STYLE, font_scale=S.SNS_FONT_SCALE)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']


def model_color(key: str) -> str:
    """Return the hex colour for a model key ('7b', '13b', …)."""
    return S.MODEL_COLORS[key]


def model_label(key: str) -> str:
    """Return the display label for a model key."""
    return S.MODEL_LABELS[key]


def add_caption(fig, text, y=-0.02):
    """Add an APA-style figure caption below the figure."""
    fig.text(
        0.5, y, text,
        ha='center', va='top',
        fontsize=S.CAPTION_SIZE,
        fontstyle=S.CAPTION_STYLE,
        fontfamily=S.CAPTION_FONT,
        linespacing=S.CAPTION_LINESPACING,
    )


def savefig(fig, path, hi_res=True):
    """Convenience wrapper with project-standard DPI and bbox."""
    dpi = S.SAVE_DPI_HI if hi_res else S.SAVE_DPI_LO
    fig.savefig(path, dpi=dpi, bbox_inches=S.SAVE_BBOX)