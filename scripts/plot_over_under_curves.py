"""
plot_fitting_curves.py
======================
Generates two figure types for each unique (model_type, activation) group:

  1. TRAINING CURVES  (over/undertraining)
     Train MSE and Test MSE vs epoch for every learning rate.
     → Reveals whether a given LR trains for too few epochs (undertraining,
       test MSE still falling) or too many (overtraining, test MSE rising
       while train MSE keeps falling).

  2. FITTING CURVES  (over/underfitting)
     Final train MSE and test MSE vs learning rate (log-scale x-axis).
     The gap between the two lines diagnoses fitting behaviour:
       - Large gap (test >> train) → overfitting
       - Both high               → underfitting
       - Both low and close      → good generalisation
     The best learning rate (lowest test MSE, averaged across seeds) is
     annotated with a vertical marker.

Usage
-----
    python plot_fitting_curves.py                          # uses default paths
    python plot_fitting_curves.py --curves results/linear_vs_nonlinear_curves.csv \
                                   --fitting results/linear_vs_nonlinear_fitting_curves.csv \
                                   --out plots/

Outputs
-------
    plots/training_curves__{model_type}__{activation}.png   (one per group)
    plots/fitting_curves__{model_type}__{activation}.png    (one per group)
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                        # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CURVES_CSV  = os.path.join("../results", "linear_vs_nonlinear_curves.csv")
DEFAULT_FITTING_CSV = os.path.join("../results", "linear_vs_nonlinear_fitting_curves.csv")
DEFAULT_OUT_DIR     = os.path.join("plots")

# Colour palette — one colour per LR (cycles if more LRs than colours)
_PALETTE = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#E91E63",  # pink
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lr_colors(lrs: list) -> dict:
    """Map each learning rate to a colour."""
    return {lr: _PALETTE[i % len(_PALETTE)] for i, lr in enumerate(sorted(lrs))}


def _lr_label(lr: float) -> str:
    return f"lr={lr:.0e}" if lr < 0.01 else f"lr={lr}"


def _safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _group_key(row) -> tuple[str, str]:
    """(model_type, activation) — the grouping used for both plot types."""
    return (str(row["model_type"]), str(row["activation"]))


# ---------------------------------------------------------------------------
# Plot 1: Training curves (over/undertraining)
# ---------------------------------------------------------------------------

def plot_training_curves(
    df_curves: pd.DataFrame,
    out_dir: str,
) -> None:
    """
    For each (model_type, activation) group, draw one figure with:
      - A solid line per LR for train_mse vs epoch
      - A dashed line per LR for test_mse vs epoch  (if available)
    Multiple seeds for the same LR are averaged; a shaded band shows ±1 std.
    """
    if df_curves.empty:
        print("  [training curves] curves CSV is empty — skipping.")
        return

    required = {"model_type", "activation", "lr", "seed", "epoch", "train_mse"}
    missing = required - set(df_curves.columns)
    if missing:
        print(f"  [training curves] missing columns {missing} — skipping.")
        return

    has_test_mse = "test_mse" in df_curves.columns

    groups = df_curves.groupby(["model_type", "activation"], sort=True)

    for (model_type, activation), grp in groups:
        lrs = sorted(grp["lr"].unique())
        colors = _lr_colors(lrs)

        fig, ax = plt.subplots(figsize=(9, 5))
        legend_handles = []

        for lr in lrs:
            lr_grp = grp[grp["lr"] == lr]
            color  = colors[lr]

            # ---- train MSE: mean ± std across seeds ----
            pivot_train = (
                lr_grp.groupby(["seed", "epoch"])["train_mse"]
                .mean()
                .reset_index()
                .pivot(index="epoch", columns="seed", values="train_mse")
            )
            epochs      = pivot_train.index.values
            train_mean  = pivot_train.mean(axis=1).values
            train_std   = pivot_train.std(axis=1).fillna(0).values

            h_train, = ax.plot(
                epochs, train_mean,
                color=color, linewidth=1.8,
                label=f"{_lr_label(lr)} train",
            )
            ax.fill_between(
                epochs,
                train_mean - train_std,
                train_mean + train_std,
                color=color, alpha=0.12,
            )
            legend_handles.append(h_train)

            # ---- test MSE (dashed) ----
            if has_test_mse:
                test_col = lr_grp[["seed", "epoch", "test_mse"]].dropna(subset=["test_mse"])
                if not test_col.empty:
                    pivot_test = (
                        test_col.groupby(["seed", "epoch"])["test_mse"]
                        .mean()
                        .reset_index()
                        .pivot(index="epoch", columns="seed", values="test_mse")
                    )
                    t_epochs    = pivot_test.index.values
                    test_mean   = pivot_test.mean(axis=1).values
                    test_std    = pivot_test.std(axis=1).fillna(0).values

                    h_test, = ax.plot(
                        t_epochs, test_mean,
                        color=color, linewidth=1.8, linestyle="--",
                        label=f"{_lr_label(lr)} test",
                    )
                    ax.fill_between(
                        t_epochs,
                        test_mean - test_std,
                        test_mean + test_std,
                        color=color, alpha=0.08,
                    )
                    legend_handles.append(h_test)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MSE", fontsize=12)
        title_act = activation if model_type == "non-linear" else "identity"
        ax.set_title(
            f"Training curves — {model_type}  [{title_act}]\n"
            f"Solid = train MSE · Dashed = test MSE · Shading = ±1 std (seeds)",
            fontsize=11,
        )
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

        # Compact legend: two columns
        ax.legend(
            handles=legend_handles,
            fontsize=8,
            ncol=2,
            loc="upper right",
            framealpha=0.85,
        )

        fname = f"training_curves__{model_type.replace('-', '_')}__{activation}.png"
        fpath = os.path.join(out_dir, fname)
        fig.tight_layout()
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fpath}")


# ---------------------------------------------------------------------------
# Plot 2: Fitting curves (over/underfitting vs LR)
# ---------------------------------------------------------------------------

def plot_fitting_curves(
    df_fitting: pd.DataFrame,
    out_dir: str,
) -> None:
    """
    For each (model_type, activation) group, draw one figure with:
      - Train MSE vs LR  (solid blue)
      - Test  MSE vs LR  (solid red, dashed)
      - Vertical line + annotation marking the best LR (lowest test MSE)
    Multiple seeds are averaged; error bars show ±1 std.

    A secondary axes shows the gap = test_mse - train_mse so readers can
    see overfitting (gap > 0 and growing) vs underfitting (both high) at a
    glance.
    """
    if df_fitting.empty:
        print("  [fitting curves] fitting CSV is empty — skipping.")
        return

    required = {"model_type", "activation", "lr", "seed", "final_train_mse", "final_test_mse"}
    missing = required - set(df_fitting.columns)
    if missing:
        print(f"  [fitting curves] missing columns {missing} — skipping.")
        return

    groups = df_fitting.groupby(["model_type", "activation"], sort=True)

    for (model_type, activation), grp in groups:
        # Aggregate over seeds
        agg = (
            grp.groupby("lr")
            .agg(
                train_mse_mean=("final_train_mse", "mean"),
                train_mse_std =("final_train_mse", "std"),
                test_mse_mean =("final_test_mse",  "mean"),
                test_mse_std  =("final_test_mse",  "std"),
            )
            .reset_index()
            .sort_values("lr")
        )
        agg["train_mse_std"] = agg["train_mse_std"].fillna(0)
        agg["test_mse_std"]  = agg["test_mse_std"].fillna(0)
        agg["gap_mean"]      = agg["test_mse_mean"] - agg["train_mse_mean"]

        lrs = agg["lr"].values

        # Best LR = LR with lowest mean test MSE
        best_idx = int(np.argmin(agg["test_mse_mean"].values))
        best_lr  = float(lrs[best_idx])
        best_test_mse = float(agg["test_mse_mean"].iloc[best_idx])

        # ---- figure layout: main axes + smaller gap axes ----
        fig, (ax_main, ax_gap) = plt.subplots(
            2, 1,
            figsize=(9, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.08)

        # ---- main: train + test MSE ----
        ax_main.errorbar(
            lrs,
            agg["train_mse_mean"],
            yerr=agg["train_mse_std"],
            color="#2196F3", linewidth=2, marker="o", markersize=6,
            capsize=4, label="Train MSE",
        )
        ax_main.errorbar(
            lrs,
            agg["test_mse_mean"],
            yerr=agg["test_mse_std"],
            color="#F44336", linewidth=2, marker="s", markersize=6,
            linestyle="--", capsize=4, label="Test MSE",
        )

        # Mark best LR
        ax_main.axvline(best_lr, color="#4CAF50", linewidth=1.5, linestyle=":", alpha=0.9)
        ax_main.annotate(
            f"Best LR\n{_lr_label(best_lr)}\n(test MSE={best_test_mse:.4f})",
            xy=(best_lr, best_test_mse),
            xytext=(best_lr * 1.25 if best_lr != lrs[-1] else best_lr * 0.6,
                    best_test_mse * 1.35),
            fontsize=8,
            color="#2E7D32",
            arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#4CAF50", alpha=0.85),
        )

        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax_main.set_ylabel("MSE (log scale)", fontsize=11)
        title_act = activation if model_type == "non-linear" else "identity"
        ax_main.set_title(
            f"Fitting curves — {model_type}  [{title_act}]\n"
            f"Over/underfitting diagnosis across learning rates",
            fontsize=11,
        )
        ax_main.legend(fontsize=10, loc="upper left", framealpha=0.85)
        ax_main.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

        # Shade overfitting region (where test > train substantially)
        ax_main.fill_between(
            lrs,
            agg["train_mse_mean"],
            agg["test_mse_mean"],
            where=(agg["gap_mean"].values > 0),
            color="#F44336", alpha=0.07, label="_nolegend_",
        )

        # ---- gap axes ----
        gap_color = np.where(agg["gap_mean"].values > 0, "#F44336", "#2196F3")
        ax_gap.bar(
            lrs,
            agg["gap_mean"].values,
            color=gap_color,
            width=np.array(lrs) * 0.4,   # proportional bar width on log scale
            alpha=0.65,
        )
        ax_gap.axhline(0, color="black", linewidth=0.8)
        ax_gap.axvline(best_lr, color="#4CAF50", linewidth=1.5, linestyle=":", alpha=0.9)
        ax_gap.set_ylabel("Gap\n(test−train)", fontsize=9)
        ax_gap.set_xlabel("Learning rate (log scale)", fontsize=11)
        ax_gap.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
        ax_gap.set_xscale("log")
        ax_gap.xaxis.set_major_formatter(ticker.LogFormatterMathtext())

        # Annotation: region labels
        y_range = ax_gap.get_ylim()
        if y_range[1] - y_range[0] > 1e-9:
            mid_y_pos = (ax_gap.get_ylim()[1]) * 0.6
            mid_y_neg = (ax_gap.get_ylim()[0]) * 0.6
            ax_gap.text(
                lrs[0], mid_y_pos, "  overfit →",
                fontsize=7, color="#C62828", va="center",
            )
            if mid_y_neg < 0:
                ax_gap.text(
                    lrs[0], mid_y_neg, "  underfit →",
                    fontsize=7, color="#1565C0", va="center",
                )

        fname = f"fitting_curves__{model_type.replace('-', '_')}__{activation}.png"
        fpath = os.path.join(out_dir, fname)
        fig.tight_layout()
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot over/undertraining and over/underfitting curves."
    )
    p.add_argument(
        "--curves",
        default=DEFAULT_CURVES_CSV,
        help=f"Path to curves CSV (default: {DEFAULT_CURVES_CSV})",
    )
    p.add_argument(
        "--fitting",
        default=DEFAULT_FITTING_CSV,
        help=f"Path to fitting curves CSV (default: {DEFAULT_FITTING_CSV})",
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for PNG files (default: {DEFAULT_OUT_DIR})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _safe_mkdir(args.out)

    # ---- load CSVs ----
    df_curves: pd.DataFrame = pd.DataFrame()
    if os.path.isfile(args.curves):
        df_curves = pd.read_csv(args.curves)
        print(f"Loaded curves CSV:  {len(df_curves)} rows from {args.curves}")
    else:
        print(f"Curves CSV not found ({args.curves}) — training-curve plots will be skipped.")

    df_fitting: pd.DataFrame = pd.DataFrame()
    if os.path.isfile(args.fitting):
        df_fitting = pd.read_csv(args.fitting)
        print(f"Loaded fitting CSV: {len(df_fitting)} rows from {args.fitting}")
    else:
        print(f"Fitting CSV not found ({args.fitting}) — fitting-curve plots will be skipped.")

    # ---- generate plots ----
    print("\n--- Training curves (over/undertraining) ---")
    plot_training_curves(df_curves, args.out)

    print("\n--- Fitting curves (over/underfitting) ---")
    plot_fitting_curves(df_fitting, args.out)

    print(f"\nDone. All plots written to: {args.out}/")


if __name__ == "__main__":
    main()