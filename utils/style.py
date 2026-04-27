"""
Visual constants aligned with TPE2 plotters/plot_experiment_comparisons.py.
"""

FIG_DPI = 144
FIG_SIZE = (11.0, 6.2)
SAVE_PAD_INCHES = 0.2

STYLE = {
    "figure_bg": "#fff5ec",
    "axes_bg": "#fff5ec",
    "text_title": "#343434",
    "text_axis": "#343434",
    "grid": "#e8dcd0",
    "grid_minor": "#d4c8bc",
    "stats_text": "#555555",
}

PLOT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "#343434",
    "axes.labelcolor": "#343434",
    "axes.linewidth": 0.8,
    "xtick.color": "#343434",
    "ytick.color": "#343434",
}

BOXPLOT_STYLE = {
    "box_face": "#4a90d9",
    "box_edge": "#2c6cb0",
    "median_color": "#c0392b",
    "whisker_color": "#343434",
    "cap_color": "#343434",
    "flier_color": "#95a5a6",
    "point_color": "#e67e22",
    "point_edge": "#c45f1a",
    "mean_color": "#27ae60",
}

PAIRWISE_COLORS = [
    {"box_face": "#4a90d9", "box_edge": "#2c6cb0"},
    {"box_face": "#e67e22", "box_edge": "#c45f1a"},
]
