"""Shared chart colours so every figure and dashboard view encodes things the same way.

Values come from the validated reference palette (dataviz skill). Congestion severity uses
the fixed status palette (good -> critical) and is ALWAYS shown with its text label, so colour
never carries the meaning alone.
"""

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots in fixed order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Congestion severity (status palette), in Free-flow -> Severe order.
CONGESTION_COLORS = {
    "Free-flow": "#0ca30c",
    "Moderate": "#fab219",
    "Heavy": "#ec835a",
    "Severe": "#d03b3b",
}

# Sequential blue ramp (light -> dark) for magnitude heatmaps.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Diverging poles for signed values (e.g. correlation), neutral grey midpoint.
DIVERGING_NEGATIVE = "#e34948"
DIVERGING_POSITIVE = "#2a78d6"
DIVERGING_MID = "#f0efec"


def apply_matplotlib_style():
    """Set recessive axes/grid and the system sans for static report figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "lines.linewidth": 2.0,
        "font.family": "sans-serif",
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })
    return plt
