"""
plot_activation_tanh_vs_logistic.py

Compares Tanh vs Logistic activations across learning rates and produces:
  1. tanh_vs_logistica_lr_sweep.png  — 3 key metrics vs LR (1x3 semilog)
  2. tanh_vs_logistica_comparacion.png — head-to-head bar chart at best LRs

Usage:
    python scripts/plot_activation_tanh_vs_logistic.py \
        [--config configs/lr_exploration_tanh_logistic.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.style import FIG_DPI, FIG_SIZE, PLOT_RC, SAVE_PAD_INCHES, STYLE

PLOTS = ROOT / "plots" / "ej1"
RESULTS = ROOT / "results"
DEFAULT_CONFIG = ROOT / "configs" / "lr_exploration_tanh_logistic.json"
DEFAULT_SUMMARY = RESULTS / "linear_vs_nonlinear_summary.csv"
DEFAULT_CURVES = RESULTS / "linear_vs_nonlinear_curves.csv"
DEFAULT_ROC = RESULTS / "linear_vs_nonlinear_roc.csv"

COLORS_ACT = {"tanh": "#27ae60", "logistic": "#8e44ad"}
LABEL_ACT  = {"tanh": "Tanh", "logistic": "Logística"}
MARKERS    = {"tanh": "o", "logistic": "s"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_act(s: str) -> str:
    a = str(s).strip().lower()
    return "logistic" if a in ("logistics", "sigmoid") else a


def _apply_style(fig: plt.Figure, *axes: plt.Axes) -> None:
    fig.patch.set_facecolor(STYLE["figure_bg"])
    for ax in axes:
        ax.set_facecolor(STYLE["axes_bg"])
        ax.grid(axis="y", which="major", linestyle="-", linewidth=0.6,
                alpha=0.55, color=STYLE["grid"], zorder=0)
        ax.minorticks_on()
        ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.4,
                alpha=0.45, color=STYLE["grid_minor"], zorder=0)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.tick_params(axis="both", colors=STYLE["text_axis"])
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(STYLE["text_axis"])
        ax.title.set_color(STYLE["text_title"])
        ax.xaxis.label.set_color(STYLE["text_axis"])
        ax.yaxis.label.set_color(STYLE["text_axis"])


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / name
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight",
                pad_inches=SAVE_PAD_INCHES, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Guardado: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _is_no_split(v: object) -> bool:
    return str(v).strip().lower() in ("true", "1")


def _filter_curves_scope(
    curves_df: pd.DataFrame,
    test_per: float | None,
    activations: list[str] | None = None,
) -> pd.DataFrame:
    """Apply the same split/test_per scope filtering used across all curve-based views."""
    work = curves_df.copy()
    if "activation" in work.columns:
        work["activation"] = work["activation"].map(_norm_act)
    if activations is not None and "activation" in work.columns:
        work = work[work["activation"].isin(activations)]
    if "no_split" in work.columns:
        ns = work["no_split"].map(_is_no_split)
        work = work[ns] if test_per is None else work[~ns]
    if test_per is not None and "test_per" in work.columns:
        work = work[np.isclose(work["test_per"].astype(float), test_per, rtol=0, atol=1e-9)]
    return work


def _std0(s: pd.Series) -> float:
    return float(s.std(ddof=0))


def _pr_auc_from_curve(recall: np.ndarray, precision: np.ndarray) -> float:
    r = np.asarray(recall, dtype=float)
    p = np.asarray(precision, dtype=float)
    mask = np.isfinite(r) & np.isfinite(p)
    r = r[mask]
    p = p[mask]
    if r.size < 2:
        return float("nan")
    order = np.argsort(r, kind="stable")
    r = r[order]
    p = p[order]
    _, idx = np.unique(r, return_index=True)
    r = r[idx]
    p = p[idx]
    if r.size < 2:
        return float("nan")
    return float(np.trapezoid(p, r))


def _pr_auc_seed_table(
    roc_df: pd.DataFrame,
    test_per: float | None,
) -> pd.DataFrame:
    """PR-AUC per (activation, lr, seed) from precision/recall curves."""
    if roc_df.empty:
        return pd.DataFrame(columns=["activation", "lr", "seed", "pr_auc"])

    work = roc_df.copy()
    if "activation" in work.columns:
        work["activation"] = work["activation"].map(_norm_act)
    work = work[
        (work["model_type"].astype(str) == "non-linear")
        & work["activation"].isin(["tanh", "logistic"])
    ]
    if "no_split" in work.columns:
        ns = work["no_split"].map(_is_no_split)
        work = work[ns] if test_per is None else work[~ns]
    if test_per is not None and "test_per" in work.columns:
        work = work[np.isclose(work["test_per"].astype(float), test_per, rtol=0, atol=1e-9)]
    if work.empty:
        return pd.DataFrame(columns=["activation", "lr", "seed", "pr_auc"])

    rows: list[dict] = []
    for (act, lr, seed), grp in work.groupby(["activation", "lr", "seed"], sort=False):
        auc_pr = _pr_auc_from_curve(
            grp["recall"].to_numpy(float),
            grp["precision"].to_numpy(float),
        )
        if np.isfinite(auc_pr):
            rows.append(
                {"activation": act, "lr": float(lr), "seed": int(seed), "pr_auc": auc_pr}
            )
    return pd.DataFrame(rows)


def _curves_mean_std_by_epoch(work: pd.DataFrame) -> pd.DataFrame:
    """Mean/std over seeds by (activation, lr, epoch), matching underfitting plotter std convention.

    Uses population std (ddof=0), same as comparison_underfitting_plot.py.
    """
    grp = (
        work.groupby(["activation", "lr", "epoch"], sort=False)["train_mse"]
        .agg(mean="mean", std=_std0)
        .reset_index()
    )
    grp["std"] = grp["std"].fillna(0.0)
    return grp


def _load(summary_path: Path, test_per: float | None) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    df["activation"] = df["activation"].map(_norm_act)
    base_mask = (
        (df["model_type"].astype(str) == "non-linear") &
        df["activation"].isin(["tanh", "logistic"])
    )
    ns = df["no_split"].map(_is_no_split)
    if test_per is None:
        scope_mask = ns
    else:
        scope_mask = (~ns) & np.isclose(df["test_per"].astype(float), test_per, rtol=0, atol=1e-9)
    df = df[base_mask & scope_mask].copy()
    return df


def _aggregate(df: pd.DataFrame, pr_auc_seed: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (activation, lr): mean+std over seeds for all key metrics."""
    # Compute gen gap per row first
    df = df.copy()
    df["gen_gap"] = df["train_f1_at_threshold"] - df["f1_at_threshold"]

    cols = [
        "best_f1", "best_recall", "best_precision",
        "final_train_mse", "gen_gap",
        "recall_at_threshold", "precision_at_threshold",
        "f1_at_threshold", "fpr_at_threshold",
    ]
    cols = [c for c in cols if c in df.columns]

    agg_kw: dict = {}
    for c in cols:
        agg_kw[f"{c}_mean"] = (c, "mean")
        agg_kw[f"{c}_std"]  = (c, _std0)

    out = df.groupby(["activation", "lr"], sort=False).agg(**agg_kw).reset_index()

    if pr_auc_seed is not None and not pr_auc_seed.empty:
        pr = (
            pr_auc_seed.groupby(["activation", "lr"], sort=False)["pr_auc"]
            .agg(pr_auc_mean="mean", pr_auc_std=_std0)
            .reset_index()
        )
        out = out.merge(pr, on=["activation", "lr"], how="left")

    for c in out.columns:
        if c.endswith("_std"):
            out[c] = out[c].fillna(0.0)
    return out


def _epochs_to_fraction(
    curve: np.ndarray,
    fraction: float = 0.90,
) -> int | None:
    """Return first 1-based epoch where the curve reaches `fraction` of total improvement.

    Improvement is measured as drop from first to best (minimum) value.
    If there is no improvement or curve is invalid, returns None.
    """
    if curve.size < 2 or not np.all(np.isfinite(curve)):
        return None
    start = float(curve[0])
    best = float(np.min(curve))
    total_gain = start - best
    if total_gain <= 0:
        return None
    target = start - fraction * total_gain
    hit = np.where(curve <= target)[0]
    if hit.size == 0:
        return None
    return int(hit[0] + 1)


def _best_lr(agg: pd.DataFrame, act: str, converge_tol: float = 0.02) -> float | None:
    """Return the best LR for `act`.

    Two-stage selection:
      1. Convergence gate — keep only LRs whose mean final train MSE is within
         `converge_tol` (2 %) of the minimum.  Filters out LRs that are too low
         (model not trained enough) or too high (oscillating / diverging).
      2. Among converged LRs, pick the one with the highest mean test F1
         (best_f1_mean) — the primary deployment metric.

    Falls back to argmax best_f1 if train MSE data is unavailable.
    """
    sub = agg[agg["activation"] == act]

    if "final_train_mse_mean" in sub.columns and "best_f1_mean" in sub.columns:
        mse_sub = sub.dropna(subset=["final_train_mse_mean"])
        if not mse_sub.empty:
            min_mse = float(mse_sub["final_train_mse_mean"].min())
            converged = mse_sub[mse_sub["final_train_mse_mean"] <= min_mse * (1 + converge_tol)]
            if converged.empty:
                converged = mse_sub
            f1_sub = converged.dropna(subset=["best_f1_mean"])
            if not f1_sub.empty:
                return float(f1_sub.loc[f1_sub["best_f1_mean"].idxmax(), "lr"])
            return float(converged.loc[converged["final_train_mse_mean"].idxmin(), "lr"])

    # fallback: argmax best F1
    f1_sub = sub.dropna(subset=["best_f1_mean"])
    if not f1_sub.empty:
        return float(f1_sub.loc[f1_sub["best_f1_mean"].idxmax(), "lr"])
    return None


def _best_lr_train_from_agg(agg: pd.DataFrame, act: str) -> float | None:
    """Best LR by minimizing final train MSE (summary-aggregate fallback)."""
    if "final_train_mse_mean" not in agg.columns:
        return None
    sub = agg[agg["activation"] == act].dropna(subset=["final_train_mse_mean"])
    if sub.empty:
        return None
    return float(sub.loc[sub["final_train_mse_mean"].idxmin(), "lr"])


def _best_lr_train_from_curves(
    curves_df: pd.DataFrame,
    act: str,
    test_per: float | None,
) -> float | None:
    """Best LR by minimizing last train MSE, matching the curves-based bar logic."""
    if curves_df.empty or "train_mse" not in curves_df.columns:
        return None

    work = _filter_curves_scope(curves_df, test_per=test_per, activations=[act])
    work = work[work["train_mse"].notna()]
    if work.empty:
        return None

    grp = _curves_mean_std_by_epoch(work)
    candidates: list[tuple[float, float]] = []  # (lr, last_train_mse)
    for lr in sorted(grp["lr"].astype(float).unique().tolist()):
        cfg = grp[np.isclose(grp["lr"].astype(float), lr, atol=1e-12)].sort_values("epoch")
        if cfg.empty:
            continue
        candidates.append((float(lr), float(cfg.iloc[-1]["mean"])))
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])[0]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — LR sweep (3 panels)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_sweep_panel(
    ax: plt.Axes,
    agg: pd.DataFrame,
    mean_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    best_lrs: dict[str, float | None],
    *,
    log_y: bool = False,
    minimize: bool = False,
) -> None:
    present = [a for a in ["tanh", "logistic"] if a in agg["activation"].values]

    for act in present:
        sub = agg[agg["activation"] == act].sort_values("lr")
        if sub.empty or mean_col not in sub.columns:
            continue
        xs = sub["lr"].to_numpy(float)
        m  = sub[mean_col].to_numpy(float)
        s  = sub[std_col].to_numpy(float) if std_col in sub.columns else np.zeros_like(m)
        color = COLORS_ACT[act]

        ax.semilogx(xs, m, color=color, marker=MARKERS[act],
                    markersize=5, linewidth=2, label=LABEL_ACT[act])
        lo = np.maximum(m - s, 1e-9) if log_y else np.maximum(m - s, 0.0)
        ax.fill_between(xs, lo, m + s, color=color, alpha=0.18, linewidth=0)

        # Star at metric optimum
        best_val = m.min() if minimize else m.max()
        best_idx = int(np.argmin(m) if minimize else np.argmax(m))
        ax.scatter([xs[best_idx]], [best_val], s=180, marker="*",
                   color=color, edgecolors="#2c3e50", linewidths=0.8, zorder=8)

    # Vertical reference lines at best-AUC LR
    for act in present:
        lr_best = best_lrs.get(act)
        if lr_best is not None:
            ax.axvline(lr_best, color=COLORS_ACT[act], linestyle="--",
                       linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Learning rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if log_y:
        ax.set_yscale("log")


def plot_lr_sweep(agg: pd.DataFrame, best_lrs: dict[str, float | None], n_seeds: int) -> None:
    panels = [
        ("best_recall_mean", "best_recall_std", "Recall optimo (test)",      "Recall optimo en test",          False, False),
        ("best_f1_mean",     "best_f1_std",     "F1 optimo (test)",          "F1 optimo en test",              False, False),
        ("gen_gap_mean",     "gen_gap_std",     "Brecha de generalizacion",  "Brecha (F1 train - F1 test)",    False, False),
    ]

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 3, figsize=(FIG_SIZE[0] * 1.2, FIG_SIZE[1] * 0.82))

        for ax, (mc, sc, ylabel, title, log_y, minimize) in zip(axes, panels):
            _plot_sweep_panel(ax, agg, mc, sc, ylabel, title, best_lrs,
                              log_y=log_y, minimize=minimize)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8, loc="best")

        # Shared best-LR legend entries
        present = [a for a in ["tanh", "logistic"] if a in agg["activation"].values]
        extra_handles = [
            plt.Line2D([0], [0], color=COLORS_ACT[a], linestyle="--", linewidth=1.2,
                       label=f"Mejor LR {LABEL_ACT[a]} = {best_lrs[a]:g}")
            for a in present if best_lrs.get(a) is not None
        ] + [plt.Line2D([0], [0], marker="*", color="gray", linestyle="None",
                        markersize=9, label="Optimo del panel")]
        fig.legend(handles=extra_handles, loc="lower center", ncol=len(extra_handles),
                   fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.04))

        fig.suptitle(
            f"Tanh vs Logistica -- Exploracion de learning rate  |  banda = +-1 std ({n_seeds} semillas)",
            fontsize=10,
        )
        _apply_style(fig, *axes)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        _save(fig, "tanh_vs_logistica_lr_sweep.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Head-to-head comparison at best LRs
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparacion(
    agg: pd.DataFrame,
    best_lrs: dict[str, float | None],
) -> None:
    present = [a for a in ["tanh", "logistic"] if a in agg["activation"].values]

    # (label, mean_col, std_col, higher_is_better)
    metric_defs = [
        ("Recall óptimo",   "best_recall_mean",         "best_recall_std",         True),
        ("F1 óptimo",       "best_f1_mean",             "best_f1_std",             True),
        ("Especificidad",   "fpr_at_threshold_mean",    "fpr_at_threshold_std",    False),  # shown as 1-FPR
    ]
    # keep only metrics present in agg
    metric_defs = [(lbl, mc, sc, hib) for lbl, mc, sc, hib in metric_defs if mc in agg.columns]

    n_metrics = len(metric_defs)
    bar_h = 0.32
    gap   = 0.1
    n_acts = len(present)

    # Extract values per activation at its best LR
    vals: dict[str, dict[str, tuple[float, float]]] = {}  # act -> label -> (mean, std)
    for act in present:
        lr = best_lrs.get(act)
        if lr is None:
            continue
        row = agg[(agg["activation"] == act) & np.isclose(agg["lr"].astype(float), lr, atol=1e-11)]
        if row.empty:
            continue
        vals[act] = {}
        for lbl, mc, sc, hib in metric_defs:
            m_val = float(row[mc].iloc[0]) if mc in row.columns else 0.0
            s_val = float(row[sc].iloc[0]) if sc in row.columns else 0.0
            # Invert FPR -> Especificidad
            if "fpr" in mc:
                m_val = 1.0 - m_val
            vals[act][lbl] = (m_val, s_val)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.92, FIG_SIZE[1] * 1.05))

        y_centers = np.arange(n_metrics, dtype=float)

        for ai, act in enumerate(present):
            if act not in vals:
                continue
            color = COLORS_ACT[act]
            offset = (ai - (n_acts - 1) / 2) * (bar_h + gap / 2)
            for mi, (lbl, mc, sc, hib) in enumerate(metric_defs):
                m_val, s_val = vals[act].get(lbl, (0.0, 0.0))
                y = y_centers[mi] + offset
                ax.barh(y, m_val, height=bar_h, color=color, alpha=0.75,
                        edgecolor=color, linewidth=0.8, xerr=s_val,
                        error_kw=dict(ecolor="#2c3e50", capsize=3, linewidth=1.2))
                # value label
                ax.text(min(m_val + s_val + 0.012, 0.97), y, f"{m_val:.3f}",
                        va="center", ha="left", fontsize=7.5,
                        color=STYLE["text_title"], fontweight="bold")

        # Mark winner per metric
        for mi, (lbl, mc, sc, hib) in enumerate(metric_defs):
            act_vals = {a: vals[a].get(lbl, (0.0, 0.0))[0] for a in present if a in vals}
            if len(act_vals) < 2:
                continue
            winner = max(act_vals, key=act_vals.get)
            loser_val = min(act_vals.values())
            # only mark if gap is meaningful
            if act_vals[winner] - loser_val > 1e-4:
                ai = present.index(winner)
                offset = (ai - (n_acts - 1) / 2) * (bar_h + gap / 2)
                ax.scatter([-0.015], [y_centers[mi] + offset],
                           s=60, marker="D", color=COLORS_ACT[winner],
                           zorder=9, clip_on=False)

        ax.set_yticks(y_centers)
        ax.set_yticklabels([m[0] for m in metric_defs], fontsize=9)
        ax.set_xlabel("Valor (0-1)")
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()

        # Winner annotation box
        if len(present) == 2 and all(a in vals for a in present):
            a0, a1 = present
            r0 = vals[a0].get("Recall óptimo", (0, 0))[0]
            r1 = vals[a1].get("Recall óptimo", (0, 0))[0]
            winner_act = a0 if r0 >= r1 else a1
            winner_metric = max(r0, r1)
            metric_line = f"Recall óptimo = {winner_metric:.4f}"
            winner_lr = best_lrs[winner_act]
            f1_val = vals[winner_act].get("F1 óptimo", (0, 0))[0]
            ax.text(0.98, 0.02,
                    f"Mejor modelo:\n{LABEL_ACT[winner_act]} (lr={winner_lr:g})"
                    f"\n{metric_line}"
                    f"\nF1 óptimo = {f1_val:.4f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                    fontweight="bold", color=COLORS_ACT[winner_act],
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=COLORS_ACT[winner_act], alpha=0.92))

        lr_strs = [f"{LABEL_ACT[a]} lr={best_lrs[a]:g}" for a in present if best_lrs.get(a)]
        title_str = "Comparacion directa -- " + " vs ".join(lr_strs)
        ax.set_title(title_str, fontsize=10)

        legend_handles = [
            mpatches.Patch(facecolor=COLORS_ACT[a], alpha=0.8, label=LABEL_ACT[a])
            for a in present
        ] + [plt.Line2D([0], [0], marker="D", color="gray", linestyle="None",
                        markersize=6, label="Ganador (diferencia > 0.0001)")]
        ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right")

        _apply_style(fig, ax)
        # suppress minor y-grid (it's a horizontal bar chart)
        ax.grid(axis="y", which="both", visible=False)
        ax.grid(axis="x", which="major", linestyle="-", linewidth=0.6,
                alpha=0.4, color=STYLE["grid"], zorder=0)
        fig.tight_layout()
        _save(fig, "tanh_vs_logistica_comparacion.png")


def plot_convergencia(
    curves_df: pd.DataFrame,
    best_lrs: dict[str, float | None],
    test_per: float | None = None,
) -> None:
    """Convergence-speed chart: train MSE vs epoch at best LR per activation.

    Train MSE is used as the main convergence criterion.
    Test MSE is shown as a secondary dashed reference for generalization context.
    """
    if curves_df.empty or "train_mse" not in curves_df.columns:
        print("  [Fig 3] Sin curvas de train_mse para graficar convergencia.")
        return

    work = _filter_curves_scope(curves_df, test_per=test_per, activations=["tanh", "logistic"])

    present = [a for a in ["tanh", "logistic"] if best_lrs.get(a) is not None]
    if not present:
        print("  [Fig 3] No hay activaciones con mejor LR para graficar convergencia.")
        return

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.95, FIG_SIZE[1] * 0.92))
        legend_rows: list[str] = []

        for act in present:
            lr = float(best_lrs[act])
            _cols = ["seed", "epoch", "train_mse"] + (["test_mse"] if "test_mse" in work.columns else [])
            sub = work[
                (work["activation"] == act) &
                np.isclose(work["lr"].astype(float), lr, atol=1e-12)
            ][_cols]

            if sub.empty:
                continue

            pivot_train = (
                sub.dropna(subset=["train_mse"])
                .pivot_table(index="epoch", columns="seed", values="train_mse", aggfunc="mean")
                .sort_index()
            )
            if pivot_train.empty:
                continue

            y_mean = pivot_train.mean(axis=1).to_numpy(float)
            y_std = pivot_train.std(axis=1, ddof=0).fillna(0.0).to_numpy(float)
            x = pivot_train.index.to_numpy(int)

            color = COLORS_ACT[act]
            ax.plot(x, y_mean, color=color, linewidth=2.2, label=f"{LABEL_ACT[act]} (train)")
            ax.fill_between(x, np.maximum(y_mean - y_std, 1e-12), y_mean + y_std,
                            color=color, alpha=0.18, linewidth=0)

            pivot_test = (
                sub.dropna(subset=["test_mse"])
                .pivot_table(index="epoch", columns="seed", values="test_mse", aggfunc="mean")
                .sort_index()
            ) if "test_mse" in sub.columns else pd.DataFrame()
            if not pivot_test.empty:
                yt_mean = pivot_test.mean(axis=1).to_numpy(float)
                xt = pivot_test.index.to_numpy(int)
                ax.plot(
                    xt,
                    yt_mean,
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.8,
                    label=f"{LABEL_ACT[act]} (test)",
                )

            best_idx = int(np.argmin(y_mean))
            ax.scatter([x[best_idx]], [y_mean[best_idx]], s=120, marker="*",
                       color=color, edgecolors="#2c3e50", linewidths=0.7, zorder=8)

            ep90 = _epochs_to_fraction(y_mean, fraction=0.90)
            if ep90 is not None:
                ax.axvline(ep90, color=color, linestyle="--", linewidth=1.2, alpha=0.75)
                legend_rows.append(f"{LABEL_ACT[act]}: ep90={ep90} (lr={lr:g})")
            else:
                legend_rows.append(f"{LABEL_ACT[act]}: ep90=n/a (lr={lr:g})")

        ax.set_yscale("log")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("MSE (escala log)")
        ax.set_title("Velocidad de convergencia (train) en el mejor LR por activacion", fontsize=10.5)
        ax.legend(fontsize=8.5, loc="upper right")

        if legend_rows:
            ax.text(
                0.02,
                0.02,
                "\n".join(legend_rows),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.3,
                color=STYLE["text_title"],
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
            )

        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "tanh_vs_logistica_convergencia.png")


def plot_convergencia_todos_lrs(curves_df: pd.DataFrame, test_per: float | None = None) -> None:
    """Convergence chart using train MSE for all LRs per activation."""
    if curves_df.empty or "train_mse" not in curves_df.columns:
        print("  [Fig 4] Sin curvas de train_mse para graficar todos los LR.")
        return

    work = _filter_curves_scope(curves_df, test_per=test_per, activations=["tanh", "logistic"])
    if work.empty:
        print("  [Fig 4] Sin activaciones tanh/logistic para graficar todos los LR.")
        return

    present = [a for a in ["tanh", "logistic"] if a in set(work["activation"])]
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, len(present), figsize=(FIG_SIZE[0] * 1.3, FIG_SIZE[1] * 0.9), squeeze=False)
        ax_list = axes.ravel().tolist()

        for i, act in enumerate(present):
            ax = ax_list[i]
            sub = work[work["activation"] == act].dropna(subset=["train_mse"])
            if sub.empty:
                ax.set_title(f"{LABEL_ACT[act]} (sin datos)")
                continue

            lrs = sorted(sub["lr"].astype(float).unique().tolist())
            cmap = plt.cm.viridis(np.linspace(0.12, 0.92, len(lrs)))

            for color, lr in zip(cmap, lrs):
                lr_sub = sub[np.isclose(sub["lr"].astype(float), lr, atol=1e-12)][["seed", "epoch", "train_mse"]]
                pivot = (
                    lr_sub.pivot_table(index="epoch", columns="seed", values="train_mse", aggfunc="mean")
                    .sort_index()
                )
                if pivot.empty:
                    continue

                x = pivot.index.to_numpy(int)
                y_mean = pivot.mean(axis=1).to_numpy(float)
                y_std = pivot.std(axis=1, ddof=0).fillna(0.0).to_numpy(float)

                ax.plot(x, y_mean, color=color, linewidth=1.7, label=f"lr={lr:g}")
                ax.fill_between(
                    x,
                    np.maximum(y_mean - y_std, 1e-12),
                    y_mean + y_std,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )

            ax.set_yscale("log")
            ax.set_xlabel("Epoca")
            ax.set_ylabel("Train MSE (escala log)")
            ax.set_title(f"{LABEL_ACT[act]} -- todos los LR", fontsize=10)
            ax.legend(fontsize=7.2, ncol=2, loc="upper right", framealpha=0.9)

        _apply_style(fig, *ax_list)
        fig.suptitle("Convergencia por activacion con todos los learning rates", fontsize=10.5)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, "tanh_vs_logistica_convergencia_todos_lrs.png")


def plot_last_mse_vs_lr(
    agg: pd.DataFrame,
    curves_df: pd.DataFrame | None = None,
    test_per: float | None = None,
) -> None:
    """Final train MSE as bar chart per (activation, lr) configuration.

    Priority source (to match comparison_underfitting_plot):
      1) curves_df: mean/std over seeds at each epoch, then take the last epoch.
      2) agg fallback: final_train_mse_mean/std from summary aggregation.
    """
    use_curves = curves_df is not None and (not curves_df.empty) and ("train_mse" in curves_df.columns)
    if not use_curves and "final_train_mse_mean" not in agg.columns:
        print("  [Fig 5] No hay fuente de datos para last MSE.")
        return

    present = [a for a in ["tanh", "logistic"] if a in agg["activation"].values]
    if not present:
        print("  [Fig 5] Sin activaciones tanh/logistic para graficar last MSE.")
        return

    rows: list[tuple[str, float, float, float]] = []

    if use_curves:
        work = _filter_curves_scope(curves_df, test_per=test_per, activations=present)
        work = work[work["train_mse"].notna()]

        # Same logic as comparison_underfitting_plot:
        # average over seeds by (activation, lr, epoch), then take final epoch per config.
        grp = _curves_mean_std_by_epoch(work)

        for act in present:
            act_df = grp[grp["activation"] == act]
            if act_df.empty:
                continue
            for lr in sorted(act_df["lr"].astype(float).unique().tolist()):
                cfg = act_df[np.isclose(act_df["lr"].astype(float), lr, atol=1e-12)].sort_values("epoch")
                if cfg.empty:
                    continue
                final = cfg.iloc[-1]
                rows.append((act, float(lr), float(final["mean"]), float(final["std"])))
    else:
        for act in present:
            sub = agg[agg["activation"] == act].sort_values("lr")
            for _, r in sub.iterrows():
                rows.append(
                    (
                        act,
                        float(r["lr"]),
                        float(r["final_train_mse_mean"]),
                        float(r["final_train_mse_std"]) if "final_train_mse_std" in sub.columns else 0.0,
                    )
                )

    if not rows:
        print("  [Fig 5] No hay filas para construir el grafico de barras.")
        return

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 1.15, FIG_SIZE[1] * 0.95))

        x = np.arange(len(rows))
        means = np.array([r[2] for r in rows], dtype=float)
        stds = np.array([r[3] for r in rows], dtype=float)
        colors = [COLORS_ACT[r[0]] for r in rows]
        labels = [f"{LABEL_ACT[r[0]]}\nlr={r[1]:g}" for r in rows]

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            alpha=0.82,
            edgecolor=colors,
            linewidth=0.8,
            error_kw=dict(ecolor="#2c3e50", capsize=3, linewidth=1.2),
        )

        # Highlight best configuration (minimum last mse) for each activation
        for act in present:
            idxs = [i for i, r in enumerate(rows) if r[0] == act]
            if not idxs:
                continue
            best_idx = min(idxs, key=lambda i: rows[i][2])
            bars[best_idx].set_linewidth(1.8)
            bars[best_idx].set_edgecolor("#1f2d3d")

        for i, (m, s) in enumerate(zip(means, stds)):
            y_txt = m + s + max(0.00002, 0.03 * np.nanmax(means))
            ax.text(
                i,
                y_txt,
                f"{m:.4f}\n±{s:.4f}",
                ha="center",
                va="bottom",
                fontsize=7.1,
                color=STYLE["text_title"],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Last MSE (train)")
        ax.set_title("Last MSE por configuracion (alineado con curvas)", fontsize=10.5)

        legend_handles = [
            mpatches.Patch(facecolor=COLORS_ACT[a], edgecolor=COLORS_ACT[a], alpha=0.82, label=LABEL_ACT[a])
            for a in present
        ]
        ax.legend(handles=legend_handles, fontsize=8.5, loc="upper right")

        _apply_style(fig, ax)
        ax.grid(axis="x", which="both", visible=False)
        fig.tight_layout()
        _save(fig, "tanh_vs_logistica_last_mse.png")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(agg: pd.DataFrame, best_lrs: dict[str, float | None], criterion_label: str) -> None:
    present = [a for a in ["tanh", "logistic"] if a in agg["activation"].values]

    print(f"\n--- Mejor LR por activacion (criterio: {criterion_label}) ---")
    for act in present:
        lr = best_lrs.get(act)
        if lr is None:
            continue
        row = agg[(agg["activation"] == act) & np.isclose(agg["lr"].astype(float), lr, atol=1e-11)]
        if row.empty:
            continue
        mse_m = float(row["final_train_mse_mean"].iloc[0]) if "final_train_mse_mean" in row.columns else float("nan")
        rec_m = float(row["best_recall_mean"].iloc[0]) if "best_recall_mean" in row.columns else float("nan")
        f1_m  = float(row["best_f1_mean"].iloc[0])    if "best_f1_mean"  in row.columns else float("nan")
        print(f"  {LABEL_ACT[act]:12s}: lr = {lr:g}   (train_mse={mse_m:.6f}  recall={rec_m:.4f}  F1={f1_m:.4f})")

    print("\n--- Comparacion directa al mejor LR ---")
    metric_defs = [
        ("Recall optimo",  "best_recall_mean",         "best_recall_std",         True),
        ("F1 optimo",      "best_f1_mean",             "best_f1_std",             True),
        ("FPR@umbral",     "fpr_at_threshold_mean",    "fpr_at_threshold_std",    False),
    ]
    metric_defs = [(l, mc, sc, h) for l, mc, sc, h in metric_defs if mc in agg.columns]

    header = f"  {'Metrica':<20}" + "".join(f"  {LABEL_ACT[a]:<22}" for a in present) + "  Ganador"
    print(header)
    print("  " + "-" * (len(header) - 2))

    winner_counts: dict[str, int] = {a: 0 for a in present}
    for lbl, mc, sc, higher_is_better in metric_defs:
        row_vals: dict[str, tuple[float, float]] = {}
        for act in present:
            lr = best_lrs.get(act)
            if lr is None:
                continue
            row = agg[(agg["activation"] == act) & np.isclose(agg["lr"].astype(float), lr, atol=1e-11)]
            if row.empty or mc not in row.columns:
                continue
            row_vals[act] = (float(row[mc].iloc[0]), float(row[sc].iloc[0]))

        if len(row_vals) < 1:
            continue

        line = f"  {lbl:<20}"
        for act in present:
            if act in row_vals:
                m, s = row_vals[act]
                line += f"  {m:.4f} +- {s:.4f}    "
            else:
                line += "  -                    "

        if len(row_vals) == 2:
            a0, a1 = present
            m0, m1 = row_vals[a0][0], row_vals[a1][0]
            if higher_is_better:
                winner = a0 if m0 > m1 else (a1 if m1 > m0 else "Empate")
            else:
                winner = a0 if m0 < m1 else (a1 if m1 < m0 else "Empate")
            line += LABEL_ACT.get(winner, winner)
            if winner in winner_counts:
                winner_counts[winner] += 1
        print(line)

    if len(present) == 2:
        a0, a1 = present
        r = {}
        for a in present:
            lr = best_lrs.get(a)
            if lr is None:
                continue
            row = agg[(agg["activation"] == a) & np.isclose(agg["lr"].astype(float), lr, atol=1e-11)]
            if not row.empty and "best_recall_mean" in row.columns:
                r[a] = float(row["best_recall_mean"].iloc[0])
        deploy_winner = None
        if len(r) == 2:
            deploy_winner = max(r, key=r.get) if abs(r[a0] - r[a1]) > 1e-4 else None

        if deploy_winner:
            lr_best = best_lrs.get(deploy_winner)
            print(f"\n  Veredicto: {LABEL_ACT[deploy_winner]} (lr={lr_best:g})"
                  f" — mayor Recall optimo")
        else:
            overall = max(winner_counts, key=winner_counts.get)
            lr_best = best_lrs.get(overall)
            print(f"\n  Veredicto: {LABEL_ACT[overall]} (lr={lr_best:g})"
                  f" gana en {winner_counts[overall]}/{len(metric_defs)} metricas"
                  f" (modelos practicamente equivalentes)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tanh vs Logistica -- LR sweep + comparacion final")
    p.add_argument("--config",  type=Path, default=DEFAULT_CONFIG,
                   help=f"Config JSON (default: {DEFAULT_CONFIG.name})")
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                   help=f"Summary CSV (default: {DEFAULT_SUMMARY.name})")
    p.add_argument("--curves", type=Path, default=DEFAULT_CURVES,
                   help=f"Curves CSV (default: {DEFAULT_CURVES.name})")
    p.add_argument("--roc", type=Path, default=DEFAULT_ROC,
                   help=f"ROC/PR points CSV (default: {DEFAULT_ROC.name})")
    return p.parse_args()


def _test_per_from_config(cfg: dict) -> float | None:
    vals = cfg.get("grid", {}).get("test_per", [])
    if any(v is None for v in vals):
        return None
    nums = [v for v in vals if v is not None]
    return float(nums[0]) if nums else 0.20


def _threshold_from_config(cfg: dict) -> float:
    return float(cfg.get("base", {}).get("threshold", 0.5))


def main() -> None:
    args = parse_args()
    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config no encontrado: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    test_per = _test_per_from_config(cfg)
    thr      = _threshold_from_config(cfg)
    summary_path = args.summary.resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Summary no encontrado: {summary_path}")
    curves_path = args.curves.resolve()
    roc_path = args.roc.resolve()

    print(f"Config: {cfg_path.name}  |  test_per={test_per}  |  umbral={thr}")

    df = _load(summary_path, test_per)
    if df.empty:
        raise SystemExit("Sin datos tras filtrar. Corre el experiment_runner primero.")

    present = sorted(df["activation"].unique())
    n_seeds = df["seed"].nunique()
    print(f"Activaciones: {present}  |  LRs por activacion: {df.groupby('activation')['lr'].nunique().to_dict()}  |  Semillas: {n_seeds}")

    pr_seed = pd.DataFrame(columns=["activation", "lr", "seed", "pr_auc"])
    if roc_path.is_file():
        roc_df = pd.read_csv(roc_path)
        pr_seed = _pr_auc_seed_table(roc_df, test_per=test_per)
    agg = _aggregate(df, pr_auc_seed=pr_seed)
    criterion_label = "mejor F1 optimo en test entre LRs convergidos (MSE ≤ min+2%)"
    best_lrs: dict[str, float | None] = {"tanh": None, "logistic": None}
    curves_df: pd.DataFrame | None = None
    if curves_path.is_file():
        curves_df = pd.read_csv(curves_path)

    if test_per is None:
        for act in ("tanh", "logistic"):
            lr = None
            if curves_df is not None:
                lr = _best_lr_train_from_curves(curves_df, act, test_per=None)
            best_lrs[act] = lr if lr is not None else _best_lr_train_from_agg(agg, act)
    else:
        best_lrs = {
            "tanh": _best_lr(agg, "tanh"),
            "logistic": _best_lr(agg, "logistic"),
        }

    print("\n[Fig 1] LR sweep...")
    plot_lr_sweep(agg, best_lrs, n_seeds)

    print("[Fig 2] Comparacion final...")
    plot_comparacion(agg, best_lrs)

    if curves_df is not None:
        print("[Fig 5] Last MSE por LR...")
        plot_last_mse_vs_lr(agg, curves_df=curves_df, test_per=test_per)
        print("[Fig 3] Velocidad de convergencia...")
        plot_convergencia(curves_df, best_lrs, test_per=test_per)
        print("[Fig 4] Convergencia con todos los LR...")
        plot_convergencia_todos_lrs(curves_df, test_per=test_per)
    else:
        print("[Fig 5] Last MSE por LR...")
        plot_last_mse_vs_lr(agg)
        print(f"[Fig 3] Omitida: curves CSV no encontrado ({curves_path})")

    _print_summary(agg, best_lrs, criterion_label=criterion_label)
    print("\nListo. Figuras generadas en plots/ej1/")


if __name__ == "__main__":
    main()
