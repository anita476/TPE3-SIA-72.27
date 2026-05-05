import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

BG_COLOR = "#f7f4ef"

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

# Base hues per model/activation family.
# Linestyle encodes the family; color shade encodes the learning rate.
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

# Populated at runtime by build_color_map(); keyed by (family, lr).
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
    """
    For each (family, lr) pair assign a unique shade of the family's base color.
    Smallest lr → darkest shade, largest lr → lightest shade.
    """
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
        # spread luminance from 60 % (dark) to 130 % (light) of the base
        factors = [0.6 + 0.7 * i / max(n - 1, 1) for i in range(n)]
        for lr, factor in zip(sorted_lrs, factors):
            _COLOR_MAP[(family, lr)] = _hex_shade(base, factor)


# ── helpers ─────────────────────────────────────────────────────────────────

def palette_key(row: pd.Series) -> str:
    """Return the family key (used for linestyle and as base-color selector)."""
    if row["model"] == "linear":
        return "linear"
    act = str(row.get("activation", "")).lower()
    return f"nonlinear_{act}" if f"nonlinear_{act}" in BASE_COLOR else "nonlinear_tanh"


def config_color(row: pd.Series) -> str:
    """Return the lr-specific shade for this config row."""
    family = palette_key(row)
    return _COLOR_MAP.get((family, row["lr"]), BASE_COLOR.get(family, "#888888"))


def group_label(row: pd.Series) -> str:
    """Human-readable label for a config (without seed — seeds are averaged)."""
    if row["model"] == "linear":
        return f"linear  lr={row['lr']}"
    return f"nonlinear [{row['activation']}]  lr={row['lr']}"


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep only full-dataset runs (no val split)
    if "test_per" in df.columns:
        df = df[df["test_per"].isin(["full", "None"]) | df["test_per"].isna()]
    return df


def average_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average MSE and BCE over seeds for each (model, activation, lr, epoch).
    Also computes std columns (train_mse_std, train_bce_std) for confidence bands.
    ddof=0 (population std) avoids NaN when only one seed is present.
    """
    group_cols = ["model", "activation", "lr", "epoch"]
    numeric = ["train_mse", "train_bce"]
    existing = [c for c in numeric if c in df.columns]

    grp     = df.groupby(group_cols, sort=False)[existing]
    mean_df = grp.mean().reset_index()
    std_df  = grp.std(ddof=0).reset_index()

    for col in existing:
        mean_df[f"{col}_std"] = std_df[col].values

    return mean_df


def configs(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split averaged df into one sub-df per (model, activation, lr) config."""
    group_cols = ["model", "activation", "lr"]
    return [grp.sort_values("epoch") for _, grp in df.groupby(group_cols, sort=False)]


# ── plotting ────────────────────────────────────────────────────────────────

def plot_metric(
    ax: plt.Axes,
    all_configs: list[pd.DataFrame],
    col: str,
    ylabel: str,
    log_scale: bool = False,
    show_std: bool = True,
):
    std_col = f"{col}_std"
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

        # ±1 std shaded band — only rendered when std > 0 (i.e. multiple seeds)
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
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        borderaxespad=0.0,
    )


def final_value_bars(ax: plt.Axes, all_configs: list[pd.DataFrame], col: str, title: str,
                     show_std: bool = True):
    """Bar chart of the final-epoch value for each config, with optional std error bars."""
    std_col = f"{col}_std"
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
    ax.set_ylabel(f"Final {col.upper()}")
    ax.set_title(title, pad=10)

    for bar, val, err in zip(bars, values, errors):
        label_text = f"{val:.4f}" if not (has_error and err > 0) else f"{val:.4f}\n±{err:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            label_text,
            ha="center", va="bottom", fontsize=7,
        )


def annotate_underfitting(ax: plt.Axes, all_configs: list[pd.DataFrame], col: str):
    """
    Draw a horizontal reference band showing the range of final values.
    A flat curve that barely descends from its start is a sign of underfitting.
    """
    finals = []
    for cfg in all_configs:
        if col in cfg.columns and not cfg[col].isna().all():
            finals.append(cfg.iloc[-1][col])
    if not finals:
        return
    lo, hi = min(finals), max(finals)
    ax.axhspan(lo * 0.95, hi * 1.05, color="#f0f0f0", zorder=0, label="final value range")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot MSE and BCE learning curves from run_comparison.py output."
    )
    parser.add_argument("--linear",     default=None,  help="Path to linear curves CSV")
    parser.add_argument("--nonlinear",  default=None,  help="Path to non-linear curves CSV")
    parser.add_argument("--log",        action="store_true", help="Use log scale on y-axis")
    parser.add_argument("--no-std",     action="store_true", help="Hide ±1 std bands and error bars")
    parser.add_argument("--out",        default="results/plots", help="Output directory")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Only plot up to this epoch (inclusive). "
                             "Silently ignored if the value exceeds the data range.")
    parser.add_argument("--lr",         type=float, default=None, nargs="+",
                        help="One or more learning-rate values to keep (e.g. --lr 0.01 0.001). "
                             "Values absent from the data are silently skipped.")
    args = parser.parse_args()

    if not args.linear and not args.nonlinear:
        parser.error("Provide at least one of --linear or --nonlinear.")

    show_std = not args.no_std

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if args.linear:
        df_lin = load(args.linear)
        df_lin["activation"] = "—"          # sentinel so groupby key is uniform
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
            print(f"  [--lr] warning: the following LR values were not found in the data "
                  f"and will be skipped: {sorted(missing)}")
        if not matched:
            parser.error("None of the requested --lr values exist in the data.")
        combined = combined[combined["lr"].isin(matched)]
        print(f"  [--lr] keeping: {sorted(matched)}")

    if args.max_epochs is not None:
        max_epoch_in_data = combined["epoch"].max()
        if args.max_epochs > max_epoch_in_data:
            print(f"  [--max-epochs] {args.max_epochs} exceeds data range "
                  f"(max epoch = {max_epoch_in_data}); using all epochs.")
        else:
            combined = combined[combined["epoch"] <= args.max_epochs]
            print(f"  [--max-epochs] plotting up to epoch {args.max_epochs}.")

    # ── aggregate & split ───────────────────────────────────────────────────
    averaged = average_over_seeds(combined)
    all_cfgs = configs(averaged)

    print(f"\nPlotting {len(all_cfgs)} configurations (averaged over seeds).")
    build_color_map(all_cfgs)

    # ── Figure 1a: MSE learning curves ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Training MSE — linear vs non-linear perceptron", fontsize=14, y=1.01)
    plot_metric(ax, all_cfgs, "train_mse", "MSE", log_scale=args.log, show_std=show_std)
    ax.set_title("Learning curves (MSE)")
    annotate_underfitting(ax, all_cfgs, "train_mse")
    fig.tight_layout()
    p = out_dir / "learning_curves_mse.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"  saved → {p}")
    plt.close(fig)

    # ── Figure 1b: MSE final-value bars ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Training MSE — linear vs non-linear perceptron", fontsize=14, y=1.01)
    final_value_bars(ax, all_cfgs, "train_mse", "Final MSE per config", show_std=show_std)
    fig.tight_layout()
    p = out_dir / "final_bars_mse.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"  saved → {p}")
    plt.close(fig)

    # ── Figure 2a: BCE learning curves ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Training BCE — linear vs non-linear perceptron", fontsize=14, y=1.01)
    plot_metric(ax, all_cfgs, "train_bce", "BCE", log_scale=args.log, show_std=show_std)
    ax.set_title("Learning curves (BCE)")
    annotate_underfitting(ax, all_cfgs, "train_bce")
    fig.tight_layout()
    p = out_dir / "learning_curves_bce.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"  saved → {p}")
    plt.close(fig)

    # ── Figure 2b: BCE final-value bars ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Training BCE — linear vs non-linear perceptron", fontsize=14, y=1.01)
    final_value_bars(ax, all_cfgs, "train_bce", "Final BCE per config", show_std=show_std)
    fig.tight_layout()
    p = out_dir / "final_bars_bce.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"  saved → {p}")
    plt.close(fig)

    # ── Figure 3: MSE vs BCE scatter (final values) ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Final MSE vs final BCE per config", pad=10)

    for cfg in all_cfgs:
        if "train_mse" not in cfg.columns or "train_bce" not in cfg.columns:
            continue
        row0  = cfg.iloc[-1]
        key   = palette_key(row0)
        label = group_label(row0)
        ax.scatter(
            row0["train_mse"], row0["train_bce"],
            color=config_color(row0),
            s=70, zorder=3, label=label,
            marker="o" if "linear" in key else "s",
        )

    ax.set_xlabel("Final train MSE")
    ax.set_ylabel("Final train BCE")
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=8, ncol=2, borderaxespad=0.0)
    fig.tight_layout()
    p = out_dir / "final_mse_vs_bce.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"  saved → {p}")
    plt.close(fig)

    print("\nDone. All plots saved to:", out_dir)


if __name__ == "__main__":
    main()