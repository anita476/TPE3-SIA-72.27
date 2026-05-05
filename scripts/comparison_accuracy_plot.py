"""
comparison_accuracy_plot.py
============================
Plots accuracy learning curves and final-accuracy bar charts from the CSVs
produced by comparison_underfitting.py (the patched version that emits
*_linear_accuracy.csv and *_nonlinear_accuracy.csv).

Usage:
    python comparison_accuracy_plot.py \
        --linear    results/experiment_linear_accuracy.csv \
        --nonlinear results/experiment_nonlinear_accuracy.csv \
        --out       results/plots

Optional flags (mirror the MSE plotting script):
    --no-std        hide ±1 std confidence bands / error bars
    --max-epochs N  only plot up to epoch N
    --lr 0.01 0.1   filter to specific learning rates

Output PNGs (written to --out/):
    accuracy_learning_curves.png   — per-epoch accuracy, all configs
    accuracy_final_bars.png        — final-epoch accuracy, bar chart

Note on imbalanced datasets:
    Accuracy can be deceptively high on skewed label distributions.
    A model that always predicts the majority class will score high accuracy
    while recall stays at 0.  Always read this plot alongside the recall curves.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

BG_COLOR = "#fff5ec"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e8e8e8",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "medium",
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#dddddd",
})

# ── palette (identical to MSE script) ────────────────────────────────────────

BASE_COLOR = {
    "linear":             "#378ADD",
    "nonlinear_tanh":     "#7F77DD",
    "nonlinear_logistic": "#1D9E75",
    "nonlinear_relu":     "#D85A30",
}

LINESTYLE = {
    "linear":             (0, ()),           # solid
    "nonlinear_tanh":     (0, (5, 2)),       # dashed
    "nonlinear_logistic": (0, (3, 1, 1, 1)), # dash-dot
    "nonlinear_relu":     (0, (1, 1)),       # dotted
}

_COLOR_MAP: dict = {}


def _hex_shade(hex_color: str, factor: float) -> str:
    """Darken (factor < 1) or lighten (factor > 1) a hex color."""
    import colorsys
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r2 * 255), int(g2 * 255), int(b2 * 255))


def build_color_map(all_configs: list) -> None:
    from collections import defaultdict
    _COLOR_MAP.clear()
    family_lrs: dict = defaultdict(set)
    for cfg in all_configs:
        row0 = cfg.iloc[0]
        family_lrs[palette_key(row0)].add(row0["lr"])

    for family, lrs in family_lrs.items():
        sorted_lrs = sorted(lrs)
        n = len(sorted_lrs)
        base = BASE_COLOR.get(family, "#888888")
        factors = [0.6 + 0.7 * i / max(n - 1, 1) for i in range(n)]
        for lr, factor in zip(sorted_lrs, factors):
            _COLOR_MAP[(family, lr)] = _hex_shade(base, factor)


def palette_key(row: pd.Series) -> str:
    if row["model"] == "linear":
        return "linear"
    act = str(row.get("activation", "")).lower()
    return f"nonlinear_{act}" if f"nonlinear_{act}" in BASE_COLOR else "nonlinear_tanh"


def config_color(row: pd.Series) -> str:
    family = palette_key(row)
    return _COLOR_MAP.get((family, row["lr"]), BASE_COLOR.get(family, "#888888"))


def group_label(row: pd.Series) -> str:
    if row["model"] == "linear":
        return f"linear  lr={row['lr']}"
    return f"nonlinear [{row['activation']}]  lr={row['lr']}"


# ── data loading & aggregation ────────────────────────────────────────────────

def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "test_per" in df.columns:
        df = df[df["test_per"].isin(["full", "None"]) | df["test_per"].isna()]
    return df


def average_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average train_accuracy over seeds for each (model, activation, lr, epoch).
    Also computes train_accuracy_std (population std, ddof=0).
    """
    group_cols = ["model", "activation", "lr", "epoch"]
    col = "train_accuracy"

    grp     = df.groupby(group_cols, sort=False)[col]
    mean_df = grp.mean().reset_index()
    std_df  = grp.std(ddof=0).reset_index()
    mean_df["train_accuracy_std"] = std_df[col].values
    return mean_df


def configs(df: pd.DataFrame) -> list[pd.DataFrame]:
    group_cols = ["model", "activation", "lr"]
    return [grp.sort_values("epoch") for _, grp in df.groupby(group_cols, sort=False)]


# ── plotting helpers ──────────────────────────────────────────────────────────

def plot_accuracy_curves(
    ax: plt.Axes,
    all_configs: list[pd.DataFrame],
    show_std: bool = True,
):
    col     = "train_accuracy"
    std_col = "train_accuracy_std"

    for cfg in all_configs:
        if col not in cfg.columns or cfg[col].isna().all():
            continue
        row0  = cfg.iloc[0]
        key   = palette_key(row0)
        label = group_label(row0)
        color = config_color(row0)
        ls    = LINESTYLE.get(key, (0, ()))

        ax.plot(
            cfg["epoch"], cfg[col],
            color=color, linestyle=ls, linewidth=1.6,
            label=label, alpha=0.9,
        )

        if show_std and std_col in cfg.columns:
            std = cfg[std_col].fillna(0)
            if std.abs().max() > 0:
                ax.fill_between(
                    cfg["epoch"],
                    cfg[col] - std,
                    cfg[col] + std,
                    color=color, alpha=0.18, linewidth=0,
                )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        borderaxespad=0.0,
    )


def final_accuracy_bars(
    ax: plt.Axes,
    all_configs: list[pd.DataFrame],
    show_std: bool = True,
):
    col     = "train_accuracy"
    std_col = "train_accuracy_std"

    labels, values, errors, colors = [], [], [], []
    for cfg in all_configs:
        if col not in cfg.columns or cfg[col].isna().all():
            continue
        row0      = cfg.iloc[0]
        final_row = cfg.iloc[-1]
        labels.append(group_label(row0))
        values.append(final_row[col])
        errors.append(float(final_row[std_col]) if (show_std and std_col in cfg.columns) else 0.0)
        colors.append(config_color(row0))

    if not labels:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Sort descending by accuracy
    order  = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    errors = [errors[i] for i in order]
    colors = [colors[i] for i in order]

    x    = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.55, zorder=3)

    has_error = show_std and any(e > 0 for e in errors)
    if has_error:
        ax.errorbar(
            x, values, yerr=errors,
            fmt="none", ecolor="#333333", elinewidth=1.2,
            capsize=4, capthick=1.2, zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Accuracy final por configuración", pad=10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))

    # Annotate bars
    for bar, val, err in zip(bars, values, errors):
        label_text = (
            f"{val:.3f}" if not (has_error and err > 0)
            else f"{val:.3f}\n±{err:.3f}"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.005,
            label_text,
            ha="center", va="bottom", fontsize=7,
        )

    # Majority-class baseline annotation
    ax.axhline(
        y=max(values),  # just a visual reference line at the top value
        color="#aaaaaa", linestyle="--", linewidth=0.8, zorder=1,
    )


def annotate_ceiling(ax: plt.Axes, all_configs: list[pd.DataFrame]):
    """Draw a reference band over the final-value range (mirrors annotate_underfitting)."""
    finals = []
    for cfg in all_configs:
        col = "train_accuracy"
        if col in cfg.columns and not cfg[col].isna().all():
            finals.append(cfg.iloc[-1][col])
    if not finals:
        return
    lo, hi = min(finals), max(finals)
    ax.axhspan(lo * 0.999, hi * 1.001, color="#f0f0f0", zorder=0, label="final value range")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy learning curves from comparison_underfitting.py output."
    )
    parser.add_argument("--linear",     default=None, help="Path to linear accuracy CSV")
    parser.add_argument("--nonlinear",  default=None, help="Path to non-linear accuracy CSV")
    parser.add_argument("--no-std",     action="store_true", help="Hide ±1 std bands and error bars")
    parser.add_argument("--out",        default="results/plots", help="Output directory")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Only plot up to this epoch (inclusive).")
    parser.add_argument("--lr",         type=float, default=None, nargs="+",
                        help="Filter to specific learning-rate values (e.g. --lr 0.01 0.001).")
    args = parser.parse_args()

    if not args.linear and not args.nonlinear:
        parser.error("Provide at least one of --linear or --nonlinear.")

    show_std = not args.no_std
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if args.linear:
        df_lin = load(args.linear)
        df_lin["activation"] = "—"
        print(f"Linear   : {len(df_lin):,} rows | "
              f"LRs={sorted(df_lin['lr'].unique())} | "
              f"seeds={sorted(df_lin['seed'].unique())}")
        frames.append(df_lin)

    if args.nonlinear:
        df_nln = load(args.nonlinear)
        print(f"Non-linear: {len(df_nln):,} rows | "
              f"LRs={sorted(df_nln['lr'].unique())} | "
              f"acts={sorted(df_nln['activation'].unique())} | "
              f"seeds={sorted(df_nln['seed'].unique())}")
        frames.append(df_nln)

    combined = pd.concat(frames, ignore_index=True)

    if args.lr is not None:
        requested = set(args.lr)
        available = set(combined["lr"].unique())
        matched   = requested & available
        missing   = requested - available
        if missing:
            print(f"  [--lr] warning: LR values not found in data: {sorted(missing)}")
        if not matched:
            parser.error("None of the requested --lr values exist in the data.")
        combined = combined[combined["lr"].isin(matched)]
        print(f"  [--lr] keeping: {sorted(matched)}")

    if args.max_epochs is not None:
        max_in_data = combined["epoch"].max()
        if args.max_epochs > max_in_data:
            print(f"  [--max-epochs] {args.max_epochs} exceeds data range "
                  f"(max epoch = {max_in_data}); using all epochs.")
        else:
            combined = combined[combined["epoch"] <= args.max_epochs]
            print(f"  [--max-epochs] plotting up to epoch {args.max_epochs}.")

    averaged  = average_over_seeds(combined)
    all_cfgs  = configs(averaged)

    print(f"\nPlotting {len(all_cfgs)} configurations (averaged over seeds).")
    build_color_map(all_cfgs)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Accuracy en entrenamiento — perceptrón lineal vs no-lineal",
        fontsize=14, y=1.01,
    )
    plot_accuracy_curves(ax, all_cfgs, show_std=show_std)
    ax.set_title("Curvas de aprendizaje (Accuracy)")
    annotate_ceiling(ax, all_cfgs)
    fig.tight_layout()
    p = out_dir / "accuracy_learning_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"  saved → {p}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        "Accuracy en entrenamiento — perceptrón lineal vs no-lineal",
        fontsize=14, y=1.01,
    )
    final_accuracy_bars(ax, all_cfgs, show_std=show_std)
    fig.tight_layout()
    p = out_dir / "accuracy_final_bars.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"  saved → {p}")
    plt.close(fig)

    print("\nDone. All plots saved to:", out_dir)


if __name__ == "__main__":
    main()