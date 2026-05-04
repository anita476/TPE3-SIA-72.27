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

COLORS_ACT = {"tanh": "#27ae60", "logistic": "#8e44ad"}
LABEL_ACT  = {"tanh": "Tanh", "logistic": "Logistica"}
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

def _load(summary_path: Path, test_per: float) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    df["activation"] = df["activation"].map(_norm_act)
    ns = df["no_split"].map(lambda x: str(x).strip().lower() in ("false", "0"))
    df = df[
        (df["model_type"].astype(str) == "non-linear") &
        ns &
        df["activation"].isin(["tanh", "logistic"]) &
        np.isclose(df["test_per"].astype(float), test_per, rtol=0, atol=1e-9)
    ].copy()
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (activation, lr): mean+std over seeds for all key metrics."""
    # Compute gen gap per row first
    df = df.copy()
    df["gen_gap"] = df["train_roc_auc"] - df["roc_auc"]

    cols = [
        "roc_auc", "best_f1", "best_recall", "best_precision",
        "train_roc_auc", "final_train_mse", "gen_gap",
        "recall_at_threshold", "precision_at_threshold",
        "f1_at_threshold", "fpr_at_threshold",
    ]
    cols = [c for c in cols if c in df.columns]

    agg_kw: dict = {}
    for c in cols:
        agg_kw[f"{c}_mean"] = (c, "mean")
        agg_kw[f"{c}_std"]  = (c, "std")

    out = df.groupby(["activation", "lr"], sort=False).agg(**agg_kw).reset_index()
    for c in out.columns:
        if c.endswith("_std"):
            out[c] = out[c].fillna(0.0)
    return out


def _best_lr(agg: pd.DataFrame, act: str) -> float | None:
    sub = agg[agg["activation"] == act].dropna(subset=["roc_auc_mean"])
    if sub.empty:
        return None
    mx = sub["roc_auc_mean"].max()
    tied = sub[np.isclose(sub["roc_auc_mean"], mx, atol=1e-9)]
    return float(tied.sort_values("lr").iloc[0]["lr"])


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
        ("roc_auc_mean",  "roc_auc_std",  "ROC-AUC (test)",          "ROC-AUC en test",             False, False),
        ("best_f1_mean",  "best_f1_std",  "F1 optimo (test)",         "F1 optimo en test",            False, False),
        ("gen_gap_mean",  "gen_gap_std",  "Brecha de generalizacion", "Brecha (AUC train - AUC test)", False, True),
    ]

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 3, figsize=(FIG_SIZE[0] * 1.2, FIG_SIZE[1] * 0.82))

        for ax, (mc, sc, ylabel, title, log_y, minimize) in zip(axes, panels):
            _plot_sweep_panel(ax, agg, mc, sc, ylabel, title, best_lrs,
                              log_y=log_y, minimize=minimize)
            ax.legend(fontsize=8, loc="best")

        # Add y=0 reference line to gen gap panel
        axes[2].axhline(0, color="#7f8c8d", linestyle=":", linewidth=1.2,
                        label="Sin sobreajuste")
        axes[2].legend(fontsize=8, loc="best")

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

    # Metrics to compare (label, mean_col, std_col, higher_is_better)
    metric_defs = [
        ("ROC-AUC (test)",       "roc_auc_mean",              "roc_auc_std",              True),
        ("F1 optimo",            "best_f1_mean",              "best_f1_std",              True),
        ("Recall optimo",        "best_recall_mean",          "best_recall_std",          True),
        ("Precision optima",     "best_precision_mean",       "best_precision_std",       True),
        ("Recall @ umbral",      "recall_at_threshold_mean",  "recall_at_threshold_std",  True),
        ("Precision @ umbral",   "precision_at_threshold_mean","precision_at_threshold_std",True),
        ("F1 @ umbral",          "f1_at_threshold_mean",      "f1_at_threshold_std",      True),
        ("Especificidad (1-FPR)","fpr_at_threshold_mean",     "fpr_at_threshold_std",     True),
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
            auc0 = vals[a0].get("ROC-AUC (test)", (0, 0))[0]
            auc1 = vals[a1].get("ROC-AUC (test)", (0, 0))[0]
            winner_act = a0 if auc0 >= auc1 else a1
            winner_lr  = best_lrs[winner_act]
            winner_auc = max(auc0, auc1)
            ax.text(0.98, 0.02,
                    f"Mejor modelo:\n{LABEL_ACT[winner_act]} (lr={winner_lr:g})\nAUC = {winner_auc:.4f}",
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


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(agg: pd.DataFrame, best_lrs: dict[str, float | None]) -> None:
    present = [a for a in ["tanh", "logistic"] if a in agg["activation"].values]

    print("\n--- Mejor LR por activacion (criterio: ROC-AUC test) ---")
    for act in present:
        lr = best_lrs.get(act)
        if lr is None:
            continue
        row = agg[(agg["activation"] == act) & np.isclose(agg["lr"].astype(float), lr, atol=1e-11)]
        if row.empty:
            continue
        m = float(row["roc_auc_mean"].iloc[0])
        s = float(row["roc_auc_std"].iloc[0])
        print(f"  {LABEL_ACT[act]:12s}: lr = {lr:g}   (AUC = {m:.4f} +- {s:.4f})")

    print("\n--- Comparacion directa al mejor LR ---")
    metric_defs = [
        ("ROC-AUC (test)",   "roc_auc_mean",              "roc_auc_std",              True),
        ("F1 optimo",        "best_f1_mean",              "best_f1_std",              True),
        ("Recall optimo",    "best_recall_mean",          "best_recall_std",          True),
        ("Precision optima", "best_precision_mean",       "best_precision_std",       True),
        ("Recall@umbral",    "recall_at_threshold_mean",  "recall_at_threshold_std",  True),
        ("FPR@umbral",       "fpr_at_threshold_mean",     "fpr_at_threshold_std",     False),
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
        overall = max(winner_counts, key=winner_counts.get)
        lr_best = best_lrs.get(overall)
        print(f"\n  Veredicto: {LABEL_ACT[overall]} (lr={lr_best:g}) gana en {winner_counts[overall]}/{len(metric_defs)} metricas")


# ─────────────────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tanh vs Logistica -- LR sweep + comparacion final")
    p.add_argument("--config",  type=Path, default=DEFAULT_CONFIG,
                   help=f"Config JSON (default: {DEFAULT_CONFIG.name})")
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                   help=f"Summary CSV (default: {DEFAULT_SUMMARY.name})")
    return p.parse_args()


def _test_per_from_config(cfg: dict) -> float:
    vals = [v for v in cfg.get("grid", {}).get("test_per", []) if v is not None]
    return float(vals[0]) if vals else 0.20


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

    print(f"Config: {cfg_path.name}  |  test_per={test_per}  |  umbral={thr}")

    df = _load(summary_path, test_per)
    if df.empty:
        raise SystemExit("Sin datos tras filtrar. Corre el experiment_runner primero.")

    present = sorted(df["activation"].unique())
    n_seeds = df["seed"].nunique()
    print(f"Activaciones: {present}  |  LRs por activacion: {df.groupby('activation')['lr'].nunique().to_dict()}  |  Semillas: {n_seeds}")

    agg = _aggregate(df)
    best_lrs: dict[str, float | None] = {
        "tanh":    _best_lr(agg, "tanh"),
        "logistic": _best_lr(agg, "logistic"),
    }

    print("\n[Fig 1] LR sweep...")
    plot_lr_sweep(agg, best_lrs, n_seeds)

    print("[Fig 2] Comparacion final...")
    plot_comparacion(agg, best_lrs)

    _print_summary(agg, best_lrs)
    print("\nListo. 2 figuras en plots/ej1/")


if __name__ == "__main__":
    main()
