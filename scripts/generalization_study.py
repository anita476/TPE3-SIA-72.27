"""
generalization_study.py — Estudio de generalización (Q2a / Q2b / Q2c)

Genera 8 figuras respondiendo las tres preguntas del trabajo:
  Q2a (3 figs): distribución de clases, métricas vs baseline, curva PR
  Q2b (2 figs): PR-AUC vs porcentaje de test, boxplots de PR-AUC por porcentaje de test
  Q2c (3 figs): curva ROC (diagnóstica), P/R/F2 vs umbral, matriz de confusión + tabla de métricas

Uso:
    python scripts/generalization_study.py [--config configs/experiments_generalization.json]

El config puede incluir ``activation_lr_pairs`` con una sola activación (p. ej. solo tanh):
los gráficos y la selección del mejor modelo usan solo las filas presentes en los CSV.
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

RESULTS = ROOT / "results"
PLOTS   = ROOT / "plots" / "ej1"

COLORS_ACT = {"tanh": "#27ae60", "logistic": "#8e44ad"}
LABEL_ACT  = {"tanh": "Tanh", "logistic": "Logística"}
BETA       = 2.0   # Fβ: β=2 penaliza más los fraudes no detectados que las falsas alarmas

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _lr_match(series: pd.Series, lr: float) -> pd.Series:
    return np.isclose(series.astype(float), float(lr), rtol=0, atol=1e-11)


def _norm_act(s: str) -> str:
    a = str(s).strip().lower()
    return "logistic" if a in ("logistics", "sigmoid") else a


def _present_binary_activations(df: pd.DataFrame) -> list[str]:
    """Tanh y/o logistic presentes en `df`, en orden fijo."""
    if df.empty or "activation" not in df.columns:
        return []
    vals = set(df["activation"].astype(str).map(_norm_act))
    return [a for a in ("tanh", "logistic") if a in vals]


def _fbeta(p: np.ndarray, r: np.ndarray, beta: float = BETA) -> np.ndarray:
    """Fβ score elemento a elemento. β=2 pondera recall el doble que precision."""
    b2 = beta ** 2
    denom = b2 * p + r
    return np.where(denom == 0, 0.0, (1 + b2) * p * r / denom)


def _pr_auc_from_curve(recall: np.ndarray, precision: np.ndarray) -> float:
    """AUC-PR robusta: ordena por recall y deduplica recalls repetidos."""
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


def _pr_auc_by_seed(roc_rows: pd.DataFrame) -> pd.DataFrame:
    """Calcula PR-AUC por (activation, test_per, seed) desde precision/recall."""
    if roc_rows.empty:
        return pd.DataFrame(columns=["activation", "test_per", "seed", "pr_auc"])
    out_rows: list[dict] = []
    for (act, tp, seed), grp in roc_rows.groupby(["activation", "test_per", "seed"], sort=False):
        auc_pr = _pr_auc_from_curve(
            grp["recall"].to_numpy(float),
            grp["precision"].to_numpy(float),
        )
        if np.isfinite(auc_pr):
            out_rows.append(
                {"activation": act, "test_per": float(tp), "seed": int(seed), "pr_auc": auc_pr}
            )
    return pd.DataFrame(out_rows)


def _best_f2_by_seed(roc_rows: pd.DataFrame, beta: float = BETA) -> pd.DataFrame:
    """Mejor Fβ (y su precisión/recall) por (activation, test_per, seed)."""
    if roc_rows.empty:
        return pd.DataFrame(columns=[
            "activation", "test_per", "seed",
            "best_f2", "best_recall_f2", "best_precision_f2",
        ])
    b2 = beta ** 2
    out: list[dict] = []
    for (act, tp, seed), grp in roc_rows.groupby(
        ["activation", "test_per", "seed"], sort=False
    ):
        p = grp["precision"].to_numpy(float)
        r = grp["recall"].to_numpy(float)
        fb = _fbeta(p, r, beta)
        best_idx = int(np.argmax(fb))
        out.append({
            "activation":        act,
            "test_per":          float(tp),
            "seed":              int(seed),
            "best_f2":           float(fb[best_idx]),
            "best_recall_f2":    float(r[best_idx]),
            "best_precision_f2": float(p[best_idx]),
        })
    return pd.DataFrame(out)


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


def _load(
    tanh_lr: float | None,
    logistic_lr: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga y filtra summary + roc.
    Retorna (summary_all, split_df, nosplit_df, roc_df).
    split_df   → no_split=False (train/test split runs)
    nosplit_df → no_split=True  (full-dataset runs)
    """
    summary_path = RESULTS / "linear_vs_nonlinear_summary.csv"
    roc_path = RESULTS / "linear_vs_nonlinear_roc.csv"
    if not summary_path.is_file():
        raise SystemExit(f"Summary no encontrado: {summary_path}")
    if not roc_path.is_file():
        raise SystemExit(f"ROC no encontrado: {roc_path}")
    summary = pd.read_csv(summary_path)
    roc_df  = pd.read_csv(roc_path)

    summary["activation"] = summary["activation"].map(_norm_act)
    roc_df["activation"]  = roc_df["activation"].map(_norm_act)

    def _act_mask(df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        if tanh_lr is not None:
            mask |= (df["activation"] == "tanh") & _lr_match(df["lr"], tanh_lr)
        if logistic_lr is not None:
            mask |= (df["activation"] == "logistic") & _lr_match(df["lr"], logistic_lr)
        if tanh_lr is None and logistic_lr is None:
            mask = pd.Series(True, index=df.index)
        return mask

    base_mask = (summary["model_type"].astype(str) == "non-linear") & _act_mask(summary)
    ns_false  = summary["no_split"].map(lambda x: str(x).strip().lower() in ("false", "0"))
    ns_true   = summary["no_split"].map(lambda x: str(x).strip().lower() in ("true", "1"))

    split   = summary[base_mask & ns_false].copy()
    nosplit = summary[base_mask & ns_true].copy()

    roc_base = (roc_df["model_type"].astype(str) == "non-linear") & _act_mask(roc_df)
    roc_ns   = roc_df["no_split"].map(lambda x: str(x).strip().lower() in ("false", "0"))
    roc = roc_df[roc_base & roc_ns].copy()

    return summary, split, nosplit, roc


# ──────────────────────────────────────────────────────────────────────────────
# Q2a — Métricas
# ──────────────────────────────────────────────────────────────────────────────

def plot_q2a_distribucion(split: pd.DataFrame, test_per: float) -> None:
    """Gráfico de barras mostrando el desbalance de clases."""
    sub = split[np.isclose(split["test_per"].astype(float), test_per, rtol=0, atol=1e-9)]
    if sub.empty:
        print(f"  [Q2a-dist] Sin datos para test_per={test_per}")
        return

    fraud_rate = float(sub["fraud_rate_test"].mean())
    n_total    = float(sub["n_test"].mean() + sub["n_train"].mean())
    n_fraud    = n_total * fraud_rate
    n_no       = n_total * (1 - fraud_rate)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.55, FIG_SIZE[1] * 0.85))

        bars = ax.bar(
            ["No fraude", "Fraude"],
            [n_no, n_fraud],
            color=["#2980b9", "#e74c3c"],
            edgecolor="#2c3e50",
            linewidth=0.8,
            width=0.5,
        )
        for bar, pct in zip(bars, [1 - fraud_rate, fraud_rate]):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + n_total * 0.012,
                f"{pct * 100:.1f}%\n({int(h):,})",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=STYLE["text_title"],
            )

        ax.set_ylabel("Número de transacciones")
        ax.set_ylim(0, n_total * 1.18)
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "gen_study_q2a_distribucion.png")


def plot_q2a_metricas(split: pd.DataFrame, test_per: float) -> None:
    """Boxplots de métricas clave vs baseline de clasificador trivial."""
    sub = split[np.isclose(split["test_per"].astype(float), test_per, rtol=0, atol=1e-9)]
    if sub.empty:
        print(f"  [Q2a-metricas] Sin datos para test_per={test_per}")
        return

    fraud_rate = float(sub["fraud_rate_test"].mean())
    baseline_acc = 1.0 - fraud_rate

    metrics = [
        ("test_acc",           "Accuracy"),
        ("best_precision_f2",  "Precisión (F2-ópt.)"),
        ("best_recall_f2",     "Recall (F2-ópt.)"),
        ("best_f2",            "F2 (opt.)"),
    ]
    activations = _present_binary_activations(sub)
    if not activations:
        print(f"  [Q2a-metricas] Sin activaciones tanh/logistic en test_per={test_per}")
        return
    n_metrics   = len(metrics)
    n_acts      = len(activations)
    width       = 0.5 if n_acts <= 1 else 0.32
    gap         = 0.08

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.95, FIG_SIZE[1] * 0.85))

        for i, (col, label) in enumerate(metrics):
            for j, act in enumerate(activations):
                rows = sub[sub["activation"] == act]
                if rows.empty or col not in rows.columns:
                    continue
                vals = rows[col].dropna().values
                pos = (
                    float(i)
                    if n_acts <= 1
                    else i + (j - 0.5) * (width + gap / 2)
                )
                ax.boxplot(
                    vals,
                    positions=[pos],
                    widths=width,
                    patch_artist=True,
                    medianprops=dict(color="#c0392b", linewidth=2),
                    whiskerprops=dict(color=COLORS_ACT[act], linewidth=1.2),
                    capprops=dict(color=COLORS_ACT[act], linewidth=1.2),
                    flierprops=dict(marker="o", color=COLORS_ACT[act], alpha=0.5, markersize=4),
                    boxprops=dict(facecolor=COLORS_ACT[act], alpha=0.55, edgecolor=COLORS_ACT[act]),
                )

        ax.axhline(baseline_acc, color="#e67e22", linestyle="--", linewidth=1.6,
                   label=f"Clasificador trivial ({baseline_acc * 100:.1f}%)")

        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels([m[1] for m in metrics])
        ax.set_ylabel("Valor (0–1)")
        ax.set_ylim(0, 1.08)

        legend_handles = [
            mpatches.Patch(facecolor=COLORS_ACT[act], alpha=0.7, label=LABEL_ACT[act])
            for act in activations
        ] + [plt.Line2D([0], [0], color="#e67e22", linestyle="--", linewidth=1.6,
                        label=f"Clasificador trivial ({baseline_acc * 100:.1f}%)")]
        ax.legend(handles=legend_handles, fontsize=9, loc="lower right")


        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "gen_study_q2a_metricas.png")


def plot_q2a_pr_curve(roc: pd.DataFrame, test_per: float) -> None:
    """Curva Precisión-Recall interpolada y promediada sobre semillas."""
    sub = roc[np.isclose(roc["test_per"].astype(float), test_per, rtol=0, atol=1e-9)]
    if sub.empty:
        print(f"  [Q2a-PR] Sin datos para test_per={test_per}")
        return

    recall_grid = np.linspace(0, 1, 200)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.82, FIG_SIZE[1] * 0.85))

        present_acts = _present_binary_activations(sub)
        for act in present_acts:
            rows = sub[sub["activation"] == act]
            curves: list[np.ndarray] = []
            for seed, grp in rows.groupby("seed"):
                g = grp.sort_values("recall")
                r = g["recall"].values
                p = g["precision"].values
                _, idx = np.unique(r, return_index=True)
                r, p = r[idx], p[idx]
                if len(r) < 2:
                    continue
                curves.append(np.interp(recall_grid, r, p, left=p[0], right=p[-1]))
            if not curves:
                continue
            mat  = np.stack(curves)
            mean = mat.mean(0)
            std  = mat.std(0)
            auc_pr = float(np.trapezoid(mean, recall_grid))
            ax.plot(recall_grid, mean, color=COLORS_ACT[act], linewidth=2,
                    label=f"{LABEL_ACT[act]} (área = {auc_pr:.3f})")
            ax.fill_between(recall_grid, np.maximum(mean - std, 0), mean + std,
                            color=COLORS_ACT[act], alpha=0.18, linewidth=0)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precisión")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="upper right")

        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "gen_study_q2a_pr_curve.png")


# ──────────────────────────────────────────────────────────────────────────────
# Q2b — Generalización
# ──────────────────────────────────────────────────────────────────────────────

_MAX_REC_TEST_PER = 0.30   # never recommend using less than 70 % for training


def _recommend_test_per(
    split: pd.DataFrame,
    gain_threshold: float = 0.010,
    std_target: float = 0.016,
) -> float:
    """Devuelve el test_per recomendado (máximo _MAX_REC_TEST_PER).

    Usa best_f2 (umbral F2-óptimo por semilla, β=2) para medir rendimiento.
    Si la curva es plana (rango < gain_threshold), recomienda el menor test_per
    cuyo std cae por debajo de std_target — estabilidad de estimación.
    Si la curva no es plana, recomienda el punto anterior a la primera caída
    significativa (> gain_threshold).
    Siempre acotado por _MAX_REC_TEST_PER.
    """
    tp_vals = sorted(split["test_per"].dropna().unique())  # low → high

    if len(tp_vals) <= 1:
        return float(tp_vals[0]) if tp_vals else 0.20

    auc_per_tp: dict[float, float] = {}
    std_per_tp: dict[float, float] = {}
    for tp in tp_vals:
        rows = split[np.isclose(split["test_per"].astype(float), tp, rtol=0, atol=1e-9)]
        auc_per_tp[float(tp)] = float(rows["best_f2"].mean())
        std_per_tp[float(tp)] = float(rows["best_f2"].std(ddof=0))

    auc_vals = list(auc_per_tp.values())
    curve_is_flat = (max(auc_vals) - min(auc_vals)) < gain_threshold

    if curve_is_flat:
        for tp in tp_vals:
            if float(tp) > _MAX_REC_TEST_PER:
                break
            if std_per_tp[float(tp)] < std_target:
                return float(tp)
        candidates = [tp for tp in tp_vals if float(tp) <= _MAX_REC_TEST_PER]
        return float(max(candidates)) if candidates else float(tp_vals[0])
    else:
        rec_tp = float(tp_vals[0])
        for i in range(1, len(tp_vals)):
            tp_prev = float(tp_vals[i - 1])
            tp_curr = float(tp_vals[i])
            drop = auc_per_tp[tp_prev] - auc_per_tp[tp_curr]
            if drop > gain_threshold:
                rec_tp = tp_prev
                break
        else:
            candidates = [tp for tp in tp_vals if float(tp) <= _MAX_REC_TEST_PER]
            rec_tp = float(max(candidates)) if candidates else float(tp_vals[0])
        return min(rec_tp, _MAX_REC_TEST_PER)


def _agg_by_tp(split: pd.DataFrame, activation: str) -> dict[float, tuple[float, float]]:
    """Mean ± std best_f2 per test_per for one activation."""
    out = {}
    for tp, grp in split[split["activation"] == activation].groupby("test_per"):
        out[float(tp)] = (float(grp["best_f2"].mean()), float(grp["best_f2"].std(ddof=0)))
    return out


def plot_q2b_auc_line(split: pd.DataFrame, rec_tp: float) -> None:
    """F2 óptimo vs porcentaje de test — líneas con banda de error."""
    present_acts = _present_binary_activations(split)
    markers = {"tanh": "o", "logistic": "s"}

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.92, FIG_SIZE[1] * 0.85))

        all_vals: list[float] = []
        for act in present_acts:
            agg = _agg_by_tp(split, act)
            if not agg:
                continue
            tps = sorted(agg)
            xs  = np.array([tp * 100 for tp in tps])
            ms  = np.array([agg[tp][0] for tp in tps])
            ss  = np.array([agg[tp][1] for tp in tps])

            ax.plot(xs, ms, color=COLORS_ACT[act], marker=markers[act], markersize=6,
                    linewidth=2, label=LABEL_ACT[act])
            ax.fill_between(xs, np.maximum(ms - ss, 0), ms + ss,
                            color=COLORS_ACT[act], alpha=0.18, linewidth=0)
            all_vals.extend(ms.tolist())

        rec_pct = rec_tp * 100
        ax.axvline(rec_pct, color="#e67e22", linestyle="--", linewidth=1.5)
        if all_vals:
            yrange = max(all_vals) - min(all_vals)
            ypos   = min(all_vals) + yrange * 0.15
        else:
            ypos = 0.90
        ax.annotate(
            f"Recomendado:\n{rec_pct:.0f}% test\n(varianza estable)",
            xy=(rec_pct, ypos),
            xytext=(14, 0),
            textcoords="offset points",
            fontsize=8.5, color="#c0392b", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        )

        ax.set_xlabel("Porcentaje de test (%)")
        ax.set_ylabel("F2 óptimo (por semilla)")
        ax.legend(fontsize=9)
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "gen_study_q2b_auc_vs_fraccion.png")


def plot_q2b_auc_boxplots(split: pd.DataFrame, rec_tp: float) -> None:
    """Boxplots de F2 óptimo por porcentaje de test."""
    tp_vals      = sorted(split["test_per"].dropna().unique())
    present_acts = _present_binary_activations(split)
    x_labels     = [f"{tp*100:.0f}%" for tp in tp_vals]

    n_acts = len(present_acts)
    width  = 0.32 if n_acts > 1 else 0.5
    gap    = 0.08

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 1.05, FIG_SIZE[1] * 0.85))

        for i, tp in enumerate(tp_vals):
            rows = split[np.isclose(split["test_per"].astype(float), tp, rtol=0, atol=1e-9)]
            for j, act in enumerate(present_acts):
                vals = rows[rows["activation"] == act]["best_f2"].dropna().values
                if len(vals) == 0:
                    continue
                pos = i + (j - 0.5) * (width + gap / 2) if n_acts > 1 else i
                ax.boxplot(
                    vals,
                    positions=[pos],
                    widths=width,
                    patch_artist=True,
                    medianprops=dict(color="#c0392b", linewidth=2),
                    whiskerprops=dict(color=COLORS_ACT[act], linewidth=1.2),
                    capprops=dict(color=COLORS_ACT[act], linewidth=1.2),
                    flierprops=dict(marker="o", color=COLORS_ACT[act], alpha=0.5, markersize=3.5),
                    boxprops=dict(facecolor=COLORS_ACT[act], alpha=0.55, edgecolor=COLORS_ACT[act]),
                )

        rec_i = next((i for i, tp in enumerate(tp_vals) if np.isclose(float(tp), rec_tp, atol=1e-9)), None)
        if rec_i is not None:
            ax.axvline(rec_i, color="#e67e22", linestyle="--", linewidth=1.5,
                       label=f"Recomendado: {rec_tp*100:.0f}% test")

        ax.set_xticks(range(len(tp_vals)))
        ax.set_xticklabels(x_labels, fontsize=8.5)
        ax.set_xlabel("Porcentaje de test (%)")
        ax.set_ylabel("F2 óptimo (por semilla)")

        legend_handles = [
            mpatches.Patch(facecolor=COLORS_ACT[act], alpha=0.7, label=LABEL_ACT[act])
            for act in present_acts
        ]
        if rec_i is not None:
            legend_handles.append(
                plt.Line2D([0], [0], color="#e67e22", linestyle="--",
                           label=f"Recomendado: {rec_tp*100:.0f}% test")
            )
        ax.legend(handles=legend_handles, fontsize=9)
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "gen_study_q2b_auc_boxplots.png")


def _print_q2b(split: pd.DataFrame, rec_tp: float) -> None:
    n_seeds = split["seed"].nunique()
    print("\n-- Q2b ----------------------------------------------------")
    print(f"Estrategia: particion estratificada por clase, repetida sobre {n_seeds} semillas.")
    print(f"Porcentaje de test recomendado: {rec_tp*100:.0f}% (varianza estable, max training)")
    for act in _present_binary_activations(split):
        rows = split[
            (split["activation"] == act) &
            np.isclose(split["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
        ]["best_f2"]
        if rows.empty:
            continue
        print(f"  {LABEL_ACT[act]:10s} F2 opt = {rows.mean():.4f} +- {rows.std(ddof=0):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Q2c — Mejor modelo
# ──────────────────────────────────────────────────────────────────────────────

def _best_activation(
    split: pd.DataFrame, rec_tp: float,
    tanh_lr: float | None, logistic_lr: float | None,
) -> tuple[str, float]:
    sub_tp = split[np.isclose(split["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)]
    acts_here = [a for a in ("tanh", "logistic") if a in sub_tp["activation"].values]
    best_act, best_val = None, -1.0
    for act in acts_here:
        rows = sub_tp[sub_tp["activation"] == act]
        if rows.empty:
            continue
        m = float(rows["best_f2"].mean())
        if m > best_val:
            best_val = m
            best_act = act
    if best_act is None:
        raise SystemExit("No se encontro ninguna activacion valida en los datos.")
    lr_map = {"tanh": tanh_lr, "logistic": logistic_lr}
    best_lr = lr_map.get(best_act) or float(
        split[split["activation"] == best_act]["lr"].iloc[0]
    )
    return best_act, best_lr


def _recommend_threshold(
    roc: pd.DataFrame, best_act: str, rec_tp: float, beta: float = BETA
) -> float:
    """Umbral que maximiza Fβ promediado sobre semillas (β=2 por defecto)."""
    rows = roc[
        (roc["activation"] == best_act) &
        np.isclose(roc["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
    ]
    thr_grid = np.linspace(0, 1, 200)
    fb_curves: list[np.ndarray] = []
    b2 = beta ** 2
    for _, grp in rows.groupby("seed"):
        g = grp.sort_values("threshold")
        t = g["threshold"].values
        p = g["precision"].values
        r = g["recall"].values
        fb = np.where((b2 * p + r) == 0, 0.0, (1 + b2) * p * r / (b2 * p + r))
        _, idx = np.unique(t, return_index=True)
        t, fb = t[idx], fb[idx]
        if len(t) < 2:
            continue
        fb_curves.append(np.interp(thr_grid, t, fb, left=fb[0], right=fb[-1]))
    if not fb_curves:
        return 0.5
    mean_fb = np.stack(fb_curves).mean(0)
    return float(thr_grid[np.argmax(mean_fb)])


def plot_q2c_metrics_bar(
    split: pd.DataFrame, best_act: str, rec_tp: float,
) -> None:
    """Bar chart: precision, recall, F2, accuracy at F2-optimal threshold vs trivial baseline."""
    rows = split[
        (split["activation"] == best_act) &
        np.isclose(split["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
    ]
    if rows.empty:
        print("  [Q2c-metricas] Sin datos.")
        return

    fraud_rate  = float(rows["fraud_rate_test"].mean())
    trivial_acc = 1.0 - fraud_rate

    metric_defs = [
        ("F2 (óptimo)",        "best_f2",             True),
        ("Recall (óptimo)",    "best_recall_f2",       True),
        ("Precisión (óptima)", "best_precision_f2",    True),
        ("Accuracy",           "test_acc",             False),
    ]

    vals  = [float(rows[col].mean()) for _, col, _ in metric_defs]
    errs  = [float(rows[col].std(ddof=0)) for _, col, _ in metric_defs]
    labels = [lbl for lbl, _, _ in metric_defs]
    ys = np.arange(len(metric_defs))

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.72, FIG_SIZE[1] * 0.72))

        ax.barh(ys, vals, xerr=errs, color=COLORS_ACT[best_act], alpha=0.75,
                height=0.5, capsize=4,
                error_kw=dict(linewidth=1.2, ecolor="#444444"))

        ax.axvline(trivial_acc, color="#e67e22", linestyle="--", linewidth=1.4,
                   label=f"Clasificador trivial — accuracy = {trivial_acc*100:.1f}%")

        for i, (val, err) in enumerate(zip(vals, errs)):
            ax.text(val + err + 0.008, i, f"{val:.3f}",
                    va="center", fontsize=9, color=STYLE["text_title"])

        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("Valor de métrica")
        ax.legend(fontsize=9, loc="lower right")
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "gen_study_q2c_metricas.png")


def plot_q2c_umbral(roc: pd.DataFrame, best_act: str, rec_tp: float, best_thr: float) -> None:
    rows = roc[
        (roc["activation"] == best_act) &
        np.isclose(roc["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
    ]
    thr_grid = np.linspace(0, 1, 200)

    b2 = BETA ** 2
    prec_list, rec_list, f2_list = [], [], []
    for _, grp in rows.groupby("seed"):
        g = grp.sort_values("threshold")
        t  = g["threshold"].values
        pr = g["precision"].values
        rc = g["recall"].values
        fb = np.where((b2 * pr + rc) == 0, 0.0, (1 + b2) * pr * rc / (b2 * pr + rc))
        _, idx = np.unique(t, return_index=True)
        t, pr, rc, fb = t[idx], pr[idx], rc[idx], fb[idx]
        if len(t) < 2:
            continue
        prec_list.append(np.interp(thr_grid, t, pr, left=pr[0], right=pr[-1]))
        rec_list.append(np.interp(thr_grid, t, rc, left=rc[0], right=rc[-1]))
        f2_list.append(np.interp(thr_grid, t, fb, left=fb[0], right=fb[-1]))

    if not prec_list:
        print("  [Q2c-umbral] Sin curvas disponibles.")
        return

    mean_p  = np.stack(prec_list).mean(0)
    mean_r  = np.stack(rec_list).mean(0)
    mean_f2 = np.stack(f2_list).mean(0)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.82, FIG_SIZE[1] * 0.88))

        ax.plot(thr_grid, mean_p,  color="#2980b9", linewidth=2,   label="Precisión")
        ax.plot(thr_grid, mean_r,  color="#27ae60", linewidth=2,   label="Recall")
        ax.plot(thr_grid, mean_f2, color="#c0392b", linewidth=2.2, label=f"F2")

        ax.axvline(best_thr, color="#e67e22", linestyle="--", linewidth=1.8,
                   label=f"Umbral recomendado = {best_thr:.3f}")

        ax.set_xlabel("Umbral de decisión")
        ax.set_ylabel("Valor de métrica")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower right")
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "gen_study_q2c_umbral.png")


def plot_q2c_confusion_tabla(
    split: pd.DataFrame,
    roc: pd.DataFrame,
    best_act: str,
    best_lr: float,
    rec_tp: float,
    best_thr: float,
) -> None:
    rows = split[
        (split["activation"] == best_act) &
        np.isclose(split["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
    ]
    if rows.empty:
        print("  [Q2c-confusion] Sin datos.")
        return

    n_test     = float(rows["n_test"].mean())
    fraud_rate = float(rows["fraud_rate_test"].mean())
    # Use the per-seed F2-optimal metrics so that the confusion matrix numbers
    # are consistent with the F2-optimal threshold (β=2, recall-weighted).
    recall_thr = float(rows["best_recall_f2"].mean())
    prec_thr   = float(rows["best_precision_f2"].mean())
    f2_m       = float(rows["best_f2"].mean())
    f2_s       = float(rows["best_f2"].std(ddof=0))
    acc_m      = float(rows["test_acc"].mean())

    P  = round(n_test * fraud_rate)
    TP = round(recall_thr * P)
    FP = round(TP / prec_thr - TP) if prec_thr > 0 else 0
    FN = P - TP
    TN = round(n_test) - P - FP
    TN = max(TN, 0)

    cm = np.array([[TP, FN], [FP, TN]], dtype=float)

    with plt.rc_context(PLOT_RC):
        fig, (ax_cm, ax_tbl) = plt.subplots(
            1, 2, figsize=(FIG_SIZE[0] * 1.1, FIG_SIZE[1] * 0.82),
            gridspec_kw={"width_ratios": [1, 1.1]},
        )

        # — Confusion matrix —
        im = ax_cm.imshow(cm, cmap="Blues", aspect="auto", vmin=0)
        labels_cm = [["VP", "FN"], ["FP", "VN"]]
        for r in range(2):
            for c in range(2):
                val = int(cm[r, c])
                color = "white" if cm[r, c] > cm.max() * 0.6 else STYLE["text_title"]
                ax_cm.text(c, r, f"{labels_cm[r][c]}\n{val:,}",
                           ha="center", va="center", fontsize=11,
                           fontweight="bold", color=color)
        ax_cm.set_xticks([0, 1])
        ax_cm.set_xticklabels(["Fraude", "No fraude"])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_yticklabels(["Fraude", "No fraude"])
        ax_cm.set_xlabel("Predicción del modelo")
        ax_cm.set_ylabel("Clase real")
        ax_cm.set_facecolor(STYLE["axes_bg"])

        # — Metrics table —
        ax_tbl.axis("off")
        table_data = [
            ("Modelo",              LABEL_ACT[best_act]),
            ("Learning rate",       f"{best_lr:g}"),
            ("F2 óptimo (media)",   f"{f2_m:.4f} ± {f2_s:.4f}"),
            ("Precisión (F2-ópt.)", f"{prec_thr:.4f}"),
            ("Recall (F2-ópt.)",    f"{recall_thr:.4f}"),
            ("Accuracy",            f"{acc_m:.4f}"),
            ("Umbral recomendado",  f"{best_thr:.3f}"),
            ("Tasa de fraude",      f"{fraud_rate * 100:.1f}%"),
            ("Muestras test",       f"{int(n_test):,}"),
        ]
        n_rows = len(table_data)
        row_h  = 1.0 / (n_rows + 1)
        for i, (lbl, val) in enumerate(table_data):
            y = 1.0 - (i + 1) * row_h
            ax_tbl.text(0.04, y + row_h * 0.3, lbl, transform=ax_tbl.transAxes,
                        fontsize=9.5, color=STYLE["text_axis"], va="center")
            ax_tbl.text(0.96, y + row_h * 0.3, val, transform=ax_tbl.transAxes,
                        fontsize=9.5, color=STYLE["text_title"], va="center",
                        ha="right", fontweight="bold")
            ax_tbl.plot([0.02, 0.98], [y, y],
                        color=STYLE["grid"], linewidth=0.7,
                        transform=ax_tbl.transAxes, clip_on=False)

        ax_tbl.set_facecolor(STYLE["axes_bg"])

        fig.patch.set_facecolor(STYLE["figure_bg"])
        for spine in ax_cm.spines.values():
            spine.set_color(STYLE["text_axis"])
        ax_cm.tick_params(colors=STYLE["text_axis"])
        ax_cm.xaxis.label.set_color(STYLE["text_axis"])
        ax_cm.yaxis.label.set_color(STYLE["text_axis"])
        fig.tight_layout()
        _save(fig, "gen_study_q2c_confusion_tabla.png")


def _print_q2c(split: pd.DataFrame, roc: pd.DataFrame,
               best_act: str, best_lr: float, rec_tp: float, best_thr: float) -> None:
    rows = split[
        (split["activation"] == best_act) &
        np.isclose(split["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
    ]
    print("\n-- Q2c ----------------------------------------------------")
    print(f"Mejor modelo: {LABEL_ACT[best_act]} (lr={best_lr:g})")
    if not rows.empty:
        print(f"  F2 opt.      = {rows['best_f2'].mean():.4f} +- {rows['best_f2'].std(ddof=0):.4f}")
        print(f"  Recall opt.  = {rows['best_recall_f2'].mean() * 100:.1f}%  (fraude detectado)")
        print(f"  Precision opt. = {rows['best_precision_f2'].mean() * 100:.1f}%")
        print(f"  Accuracy     = {rows['test_acc'].mean():.4f} +- {rows['test_acc'].std(ddof=0):.4f}")
        print(f"  Umbral recomendado: {best_thr:.3f} (maximiza F2 promediado sobre semillas, beta=2)")


# ──────────────────────────────────────────────────────────────────────────────
# Big model comparison
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH = ROOT / "data" / "fraud_dataset.csv"


def _big_model_best_cm(beta: float = BETA) -> tuple[np.ndarray, float, float, float, float] | None:
    """Confusion matrix for big_model_fraud_probability at its own F2-optimal threshold.
    Returns (cm 2x2, threshold, recall, precision, f2) or None if unavailable.
    cm layout: [[TP, FN], [FP, TN]]
    """
    if not DATA_PATH.is_file():
        return None
    df = pd.read_csv(DATA_PATH)
    if "big_model_fraud_probability" not in df.columns or "flagged_fraud" not in df.columns:
        return None
    y_true = df["flagged_fraud"].to_numpy(int)
    y_prob = df["big_model_fraud_probability"].to_numpy(float)

    b2 = beta ** 2
    thresholds = np.linspace(0, 1, 400)
    best_thr, best_f2 = 0.5, -1.0
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f2 = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0
        if f2 > best_f2:
            best_f2, best_thr = f2, float(thr)

    pred = (y_prob >= best_thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta ** 2
    f2  = (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) > 0 else 0.0
    cm = np.array([[tp, fn], [fp, tn]], dtype=float)
    return cm, best_thr, rec, prec, f2


def plot_confusion_comparacion(
    split: pd.DataFrame,
    best_act: str,
    rec_tp: float,
    best_thr: float,
) -> None:
    """Matrices de confusión lado a lado: perceptrón vs big model."""
    rows = split[
        (split["activation"] == best_act) &
        np.isclose(split["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
    ]
    if rows.empty:
        print("  [comparacion] Sin datos del perceptrón.")
        return

    # — Perceptrón —
    n_test     = float(rows["n_test"].mean())
    fraud_rate = float(rows["fraud_rate_test"].mean())
    rec_p      = float(rows["best_recall_f2"].mean())
    prec_p     = float(rows["best_precision_f2"].mean())
    f2_p       = float(rows["best_f2"].mean())
    P  = round(n_test * fraud_rate)
    TP = round(rec_p * P)
    FP = round(TP / prec_p - TP) if prec_p > 0 else 0
    FN = P - TP
    TN = max(round(n_test) - P - FP, 0)
    cm_perc = np.array([[TP, FN], [FP, TN]], dtype=float)

    # — Big model en su propio umbral F2-óptimo —
    big = _big_model_best_cm()
    if big is None:
        print("  [comparacion] Dataset no encontrado para big model.")
        return
    cm_big, big_thr, big_rec, big_prec, f2_big = big

    def _draw_cm(ax: plt.Axes, cm: np.ndarray, title: str, thr: float) -> None:
        im = ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0)
        labels = [["VP", "FN"], ["FP", "VN"]]
        for r in range(2):
            for c in range(2):
                val = int(cm[r, c])
                color = "white" if cm[r, c] > cm.max() * 0.6 else STYLE["text_title"]
                ax.text(c, r, f"{labels[r][c]}\n{val:,}",
                        ha="center", va="center", fontsize=11,
                        fontweight="bold", color=color)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Fraude", "No fraude"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Fraude", "No fraude"])
        ax.set_xlabel("Predicción"); ax.set_ylabel("Clase real")
        ax.set_title(f"{title}\n(umbral = {thr:.3f})", fontsize=10)
        ax.set_facecolor(STYLE["axes_bg"])
        for spine in ax.spines.values():
            spine.set_color(STYLE["text_axis"])
        ax.tick_params(colors=STYLE["text_axis"])
        ax.xaxis.label.set_color(STYLE["text_axis"])
        ax.yaxis.label.set_color(STYLE["text_axis"])
        ax.title.set_color(STYLE["text_title"])

    with plt.rc_context(PLOT_RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(FIG_SIZE[0] * 1.05, FIG_SIZE[1] * 0.72),
        )
        _draw_cm(ax1, cm_perc, f"Perceptrón ({LABEL_ACT[best_act]})", best_thr)
        _draw_cm(ax2, cm_big,  "Big model", big_thr)

        # metrics annotation under each CM
        ax1.text(0.5, -0.22,
                 f"Recall={rec_p:.3f}  Precisión={prec_p:.3f}  F2={f2_p:.3f}",
                 transform=ax1.transAxes, ha="center", fontsize=8.5,
                 color=STYLE["text_axis"])
        ax2.text(0.5, -0.22,
                 f"Recall={big_rec:.3f}  Precisión={big_prec:.3f}  F2={f2_big:.3f}",
                 transform=ax2.transAxes, ha="center", fontsize=8.5,
                 color=STYLE["text_axis"])

        fig.suptitle(
            "Perceptrón vs Big model — cada uno en su umbral F2-óptimo (β=2)",
            fontsize=11,
        )
        fig.patch.set_facecolor(STYLE["figure_bg"])
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        _save(fig, "gen_study_comparacion_big_model.png")


# ──────────────────────────────────────────────────────────────────────────────
# CLI + main
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = ROOT / "configs" / "experiments_generalization.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estudio de generalizacion Q2a/Q2b/Q2c")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"Config JSON (default: {DEFAULT_CONFIG})")
    return p.parse_args()


def _lrs_from_config(cfg: dict) -> tuple[float | None, float | None]:
    """Extrae tanh_lr y logistic_lr de activation_lr_pairs. None si no aparece."""
    tanh_lr = logistic_lr = None
    for pair in cfg.get("activation_lr_pairs", []):
        act = _norm_act(pair.get("activation", ""))
        lr  = float(pair["lr"])
        if act == "tanh":
            tanh_lr = lr
        elif act == "logistic":
            logistic_lr = lr
    return tanh_lr, logistic_lr


def _test_per_from_config(cfg: dict) -> float:
    """test_per usado en Q2a (distribución, métricas, PR).

    Si ``base.q2a_test_per`` está definido, tiene prioridad.
    Si no, se usa el primer ``test_per`` no nulo del ``grid`` (orden del JSON),
    o 0.20 por defecto.
    """
    base = cfg.get("base") or {}
    if base.get("q2a_test_per") is not None:
        return float(base["q2a_test_per"])
    vals = [v for v in cfg.get("grid", {}).get("test_per", []) if v is not None]
    return float(vals[0]) if vals else 0.20


def main() -> None:
    args = parse_args()
    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config no encontrado: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    tanh_lr, logistic_lr = _lrs_from_config(cfg)
    test_per = _test_per_from_config(cfg)

    acts_info = []
    if tanh_lr is not None:
        acts_info.append(f"tanh lr={tanh_lr:g}")
    if logistic_lr is not None:
        acts_info.append(f"logistic lr={logistic_lr:g}")
    print(f"Config: {cfg_path.name}  |  {', '.join(acts_info) or 'todas las activaciones'}")
    print(f"  Q2a (distrib / métricas / PR): test_per = {test_per}"
          f"{'  (base.q2a_test_per)' if (cfg.get('base') or {}).get('q2a_test_per') is not None else '  (primer test_per no nulo del grid, o 0.20)'}")

    _, split, nosplit, roc = _load(tanh_lr, logistic_lr)

    if split.empty:
        raise SystemExit("Sin datos tras filtrar. Corre el experiment_runner con este config primero.")

    # Augmentar split con métricas F2-óptimas calculadas desde curvas P/R por semilla
    f2_df = _best_f2_by_seed(roc)
    if not f2_df.empty:
        split = split.merge(
            f2_df[["activation", "test_per", "seed",
                   "best_f2", "best_recall_f2", "best_precision_f2"]],
            on=["activation", "test_per", "seed"],
            how="left",
        )
    else:
        print("  [AVISO] No se pudo calcular best_f2 desde roc; usando best_f1 como fallback.")
        split["best_f2"]           = split["best_f1"]
        split["best_recall_f2"]    = split["best_recall"]
        split["best_precision_f2"] = split["best_precision"]

    # Determinar el mejor modelo primero (resuelve issue de umbral: best_thr disponible para todo)
    rec_tp   = _recommend_test_per(split)
    best_act, best_lr = _best_activation(split, rec_tp, tanh_lr, logistic_lr)
    best_thr = _recommend_threshold(roc, best_act, rec_tp)
    print(f"  Modelo seleccionado: {LABEL_ACT[best_act]} (lr={best_lr:g})  |  "
          f"rec_tp={rec_tp*100:.0f}%  |  umbral={best_thr:.3f}")

    # Q2a
    print("\n[Q2a] Metricas...")
    plot_q2a_distribucion(split, test_per)
    plot_q2a_metricas(split, test_per)
    plot_q2a_pr_curve(roc, test_per)

    # Q2b
    print("\n[Q2b] Generalizacion...")
    plot_q2b_auc_line(split, rec_tp)
    plot_q2b_auc_boxplots(split, rec_tp)
    _print_q2b(split, rec_tp)

    # Q2c
    print("\n[Q2c] Mejor modelo...")
    plot_q2c_metrics_bar(split, best_act, rec_tp)
    plot_q2c_umbral(roc, best_act, rec_tp, best_thr)
    plot_q2c_confusion_tabla(split, roc, best_act, best_lr, rec_tp, best_thr)
    _print_q2c(split, roc, best_act, best_lr, rec_tp, best_thr)

    print("\n[Comparacion] Big model vs Perceptrón...")
    plot_confusion_comparacion(split, best_act, rec_tp, best_thr)

    print("\nListo. Figuras en plots/ej1/")


if __name__ == "__main__":
    main()
