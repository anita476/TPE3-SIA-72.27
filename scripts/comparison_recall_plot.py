"""
comparison_recall_plot.py
=========================
Plots recall learning curves from the recall CSVs produced by
comparison_underfitting.py.

Usage:
    python comparison_recall_plot.py \\
        --linear   results/experiment_linear_recall.csv \\
        --nonlinear results/experiment_nonlinear_recall.csv \\
        --out      results/plots

Outputs (saved to --out directory):
    recall_curves.png          — learning curves (recall vs epoch) + final bar chart
    recall_saturation.png      — flat-zone analysis: highlights configs where
                                 recall stopped improving (capacity saturation signal)
"""

import argparse
from pathlib import Path
from collections import defaultdict

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

BASE_COLOR = {
    "linear":             "#378ADD",
    "nonlinear_tanh":     "#7F77DD",
    "nonlinear_logistic": "#1D9E75",
    "nonlinear_relu":     "#D85A30",
}

LINESTYLE = {
    "linear":             (0, ()),
    "nonlinear_tanh":     (0, (5, 2)),
    "nonlinear_logistic": (0, (3, 1, 1, 1)),
    "nonlinear_relu":     (0, (1, 1)),
}

_COLOR_MAP: dict = {}


def _hex_shade(hex_color: str, factor: float) -> str:
    import colorsys
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r2 * 255), int(g2 * 255), int(b2 * 255))


def build_color_map(all_configs: list) -> None:
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


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "test_per" in df.columns:
        df = df[df["test_per"].isin(["full", "None"]) | df["test_per"].isna()]
    return df


def average_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Average recall over seeds; compute std for confidence bands."""
    group_cols = ["model", "activation", "lr", "epoch"]
    existing   = [c for c in ["train_recall"] if c in df.columns]

    grp     = df.groupby(group_cols, sort=False)[existing]
    mean_df = grp.mean().reset_index()
    std_df  = grp.std(ddof=0).reset_index()

    for col in existing:
        mean_df[f"{col}_std"] = std_df[col].values

    return mean_df


def configs(df: pd.DataFrame) -> list[pd.DataFrame]:
    group_cols = ["model", "activation", "lr"]
    return [grp.sort_values("epoch") for _, grp in df.groupby(group_cols, sort=False)]

def plot_recall_curves(
    ax: plt.Axes,
    all_configs: list[pd.DataFrame],
    show_std: bool = True,
):
    """Learning curves: recall vs epoch for every config."""
    col     = "train_recall"
    std_col = "train_recall_std"

    for cfg in all_configs:
        if col not in cfg.columns or cfg[col].isna().all():
            continue
        row0  = cfg.iloc[0]
        key   = palette_key(row0)
        color = config_color(row0)
        ls    = LINESTYLE.get(key, (0, ()))

        ax.plot(
            cfg["epoch"], cfg[col],
            color=color, linestyle=ls, linewidth=1.6,
            label=group_label(row0), alpha=0.9,
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
    ax.set_ylabel("Recall (clase fraude)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="lower right", ncol=1)
    ax.set_title("Recall por época")


def final_recall_bars(
    ax: plt.Axes,
    all_configs: list[pd.DataFrame],
    show_std: bool = True,
):
    col     = "train_recall"
    std_col = "train_recall_std"
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
    ax.set_ylabel("Final recall (fraude)")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Recall final por configuración", pad=10)

    for bar, val, err in zip(bars, values, errors):
        label_text = (
            f"{val:.1%}" if not (has_error and err > 0)
            else f"{val:.1%}\n±{err:.3f}"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            label_text,
            ha="center", va="bottom", fontsize=7,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot recall learning curves from comparison_underfitting.py output."
    )
    parser.add_argument("--linear",    default=None, help="Path to linear recall CSV")
    parser.add_argument("--nonlinear", default=None, help="Path to non-linear recall CSV")
    parser.add_argument("--no-std",    action="store_true", help="Hide ±1 std bands")
    parser.add_argument(
        "--window", type=int, default=50,
        help="Epoch window for plateau detection (default: 50)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.005,
        help="Recall improvement threshold below which a config is considered plateaued (default: 0.005)"
    )
    parser.add_argument("--out", default="results/plots", help="Output directory")
    parser.add_argument(
        "--lr", type=float, default=None, nargs="+",
        help="One or more learning-rate values to keep (e.g. --lr 0.01 0.001). "
             "Values absent from the data are silently skipped.",
    )
    args = parser.parse_args()

    if not args.linear and not args.nonlinear:
        parser.error("Provide at least one of --linear or --nonlinear.")

    show_std = not args.no_std
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if args.linear:
        df_lin = load(args.linear)
        if "activation" not in df_lin.columns:
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

    # ── optional LR filter ──────────────────────────────────────────────────
    if args.lr is not None:
        requested = set(args.lr)
        available = set(combined["lr"].unique())
        matched   = requested & available
        missing   = requested - available
        if missing:
            print(f"  [--lr] warning: LR values not found in data and skipped: {sorted(missing)}")
        if not matched:
            parser.error("None of the requested --lr values exist in the data.")
        combined = combined[combined["lr"].isin(matched)]
        print(f"  [--lr] keeping: {sorted(matched)}")
    averaged = average_over_seeds(combined)
    all_cfgs = configs(averaged)

    print(f"\nPlotting recall for {len(all_cfgs)} configurations.")
    build_color_map(all_cfgs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    fig.suptitle(
        "Recall clase fraude — lineal vs no lineal",
        fontsize=14, y=1.01,
    )

    plot_recall_curves(axes[0], all_cfgs, show_std=show_std)
    final_recall_bars(axes[1], all_cfgs, show_std=show_std)

    fig.tight_layout()
    p = out_dir / "recall_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"  saved → {p}")
    plt.close(fig)


    print("\nDone. Plot saved to:", out_dir)


if __name__ == "__main__":
    main()