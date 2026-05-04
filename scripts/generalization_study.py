"""
generalization_study.py — Estudio de generalización (Q2a / Q2b / Q2c)

Genera 8 figuras respondiendo las tres preguntas del trabajo:
  Q2a (3 figs): distribución de clases, métricas vs baseline, curva PR
  Q2b (2 figs): AUC vs porcentaje de test, boxplots de AUC por porcentaje de test
  Q2c (3 figs): curva ROC, P/R/F1 vs umbral, matriz de confusión + tabla de métricas

Uso:
    python scripts/generalization_study.py [--config configs/experiments_generalization.json]
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

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _lr_match(series: pd.Series, lr: float) -> pd.Series:
    return np.isclose(series.astype(float), float(lr), rtol=0, atol=1e-11)


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


def _load(
    tanh_lr: float | None,
    logistic_lr: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga y filtra summary + roc.
    Retorna (summary_all, split_df, nosplit_df, roc_df).
    split_df   → no_split=False (train/test split runs)
    nosplit_df → no_split=True  (full-dataset runs, in-sample AUC)
    """
    summary = pd.read_csv(RESULTS / "linear_vs_nonlinear_summary.csv")
    roc_df  = pd.read_csv(RESULTS / "linear_vs_nonlinear_roc.csv")

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
        ax.set_title("Distribución de clases en el conjunto de datos")
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
        ("test_acc",             "Accuracy"),
        ("roc_auc",              "ROC-AUC"),
        ("best_f1",              "F1 (opt.)"),
        ("best_recall",          "Recall (opt.)"),
    ]
    activations = [a for a in ["tanh", "logistic"] if a in sub["activation"].values]
    n_metrics   = len(metrics)
    width       = 0.32
    gap         = 0.08

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.95, FIG_SIZE[1] * 0.85))

        for i, (col, label) in enumerate(metrics):
            for j, act in enumerate(activations):
                rows = sub[sub["activation"] == act]
                if rows.empty or col not in rows.columns:
                    continue
                vals = rows[col].dropna().values
                pos  = i + (j - 0.5) * (width + gap / 2)
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
        ax.set_title("Comparación de métricas — ambas activaciones vs clasificador trivial")

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

        present_acts = [a for a in ["tanh", "logistic"] if a in sub["activation"].values]
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
                    label=f"{LABEL_ACT[act]} (AUC-PR = {auc_pr:.3f})")
            ax.fill_between(recall_grid, np.maximum(mean - std, 0), mean + std,
                            color=COLORS_ACT[act], alpha=0.18, linewidth=0)

        # baseline: no-skill = fraud_rate
        # estimate from recall values: at threshold 0 recall=1, precision~fraud_rate
        try:
            sub_sum = roc[np.isclose(roc["test_per"].astype(float), test_per, rtol=0, atol=1e-9)]
            # at threshold close to 0, recall~1, precision~fraud_rate
            low_thr = sub_sum[sub_sum["threshold"] < 0.05]
            if not low_thr.empty:
                fr = float(low_thr["precision"].mean())
            else:
                fr = 0.11
        except Exception:
            fr = 0.11

        ax.axhline(fr, color="#7f8c8d", linestyle="--", linewidth=1.4,
                   label=f"Sin habilidad (tasa de fraude ≈ {fr * 100:.1f}%)")

        ax.set_xlabel("Recall (sensibilidad)")
        ax.set_ylabel("Precisión")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_title("Curva Precisión-Recall — métrica correcta para datos desbalanceados")
        ax.legend(fontsize=9, loc="upper right")

        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "gen_study_q2a_pr_curve.png")


# ──────────────────────────────────────────────────────────────────────────────
# Q2b — Generalización
# ──────────────────────────────────────────────────────────────────────────────

def _recommend_test_per(split: pd.DataFrame, gain_threshold: float = 0.002) -> float:
    """Devuelve el test_per recomendado.

    Recorre de mayor a menor test_per (de menos a más datos de entrenamiento).
    Cuando la ganancia en AUC al pasar al siguiente paso es menor que
    `gain_threshold`, el paso anterior ya es suficiente — no hace falta más datos.
    Devuelve el test_per en ese punto de inflexión.
    """
    tp_vals = sorted(split["test_per"].dropna().unique(), reverse=True)  # high → low

    if len(tp_vals) <= 1:
        return float(tp_vals[0]) if tp_vals else 0.20

    auc_per_tp: dict[float, float] = {}
    for tp in tp_vals:
        rows = split[np.isclose(split["test_per"].astype(float), tp, rtol=0, atol=1e-9)]
        auc_per_tp[float(tp)] = float(rows["roc_auc"].mean())

    # Walk from high test_per (little data) toward low (lots of data).
    # The first step where the gain from adding more training data drops below
    # the threshold is where we recommend stopping — the previous (higher) test_per.
    rec_tp = float(tp_vals[-1])  # default: use most training data
    for i in range(1, len(tp_vals)):
        tp_prev = tp_vals[i - 1]   # higher test_per (less training)
        tp_curr = tp_vals[i]       # lower test_per (more training)
        gain = auc_per_tp[tp_curr] - auc_per_tp[tp_prev]
        if gain < gain_threshold:
            rec_tp = float(tp_prev)  # diminishing returns beyond this point
            break

    return rec_tp


def _agg_by_tp(split: pd.DataFrame, activation: str) -> dict[float, tuple[float, float]]:
    """Mean ± std AUC per test_per for one activation."""
    out = {}
    for tp, grp in split[split["activation"] == activation].groupby("test_per"):
        out[float(tp)] = (float(grp["roc_auc"].mean()), float(grp["roc_auc"].std()))
    return out


def plot_q2b_auc_line(split: pd.DataFrame, nosplit: pd.DataFrame, rec_tp: float) -> None:
    """AUC vs porcentaje de test — líneas con banda de error.
    Incluye el punto en x=0 (sin particion, AUC en entrenamiento)."""
    present_acts = [a for a in ["tanh", "logistic"] if a in split["activation"].values]
    markers = {"tanh": "o", "logistic": "s"}

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.92, FIG_SIZE[1] * 0.85))

        all_aucs: list[float] = []
        for act in present_acts:
            agg = _agg_by_tp(split, act)
            if not agg:
                continue
            tps = sorted(agg)
            xs  = np.array([tp * 100 for tp in tps])
            ms  = np.array([agg[tp][0] for tp in tps])
            ss  = np.array([agg[tp][1] for tp in tps])

            # prepend no-split point at x=0 if available
            if not nosplit.empty:
                ns_rows = nosplit[nosplit["activation"] == act]["roc_auc"].dropna()
                if not ns_rows.empty:
                    xs = np.concatenate([[0.0], xs])
                    ms = np.concatenate([[ns_rows.mean()], ms])
                    ss = np.concatenate([[ns_rows.std()], ss])

            ax.plot(xs, ms, color=COLORS_ACT[act], marker=markers[act], markersize=6,
                    linewidth=2, label=LABEL_ACT[act])
            ax.fill_between(xs, np.maximum(ms - ss, 0), ms + ss,
                            color=COLORS_ACT[act], alpha=0.18, linewidth=0)
            all_aucs.extend(ms.tolist())

        # Mark x=0 region
        if not nosplit.empty:
            ax.axvline(0, color="#7f8c8d", linestyle=":", linewidth=1.0, alpha=0.7)
            ytext = min(all_aucs) if all_aucs else 0.85
            ax.text(0.4, ytext, "Sin\nparticion", fontsize=7, color="#7f8c8d",
                    va="bottom", ha="left")

        rec_pct = rec_tp * 100
        ax.axvline(rec_pct, color="#e67e22", linestyle="--", linewidth=1.5)
        if all_aucs:
            yrange = max(all_aucs) - min(all_aucs)
            ypos   = min(all_aucs) + yrange * 0.15
        else:
            ypos = 0.85
        ax.annotate(
            f"Recomendado:\n{rec_pct:.0f}% test",
            xy=(rec_pct, ypos),
            xytext=(14, 0),
            textcoords="offset points",
            fontsize=8.5, color="#c0392b", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        )

        ax.set_xlabel("Porcentaje de test (%)")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("AUC vs porcentaje de test — media +- std sobre semillas\n"
                     "(x=0: sin particion, AUC en entrenamiento)")
        ax.legend(fontsize=9)
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "gen_study_q2b_auc_vs_fraccion.png")


def plot_q2b_auc_boxplots(split: pd.DataFrame, nosplit: pd.DataFrame, rec_tp: float) -> None:
    """Boxplots de AUC por porcentaje de test.
    Incluye columna extra a la izquierda para el caso sin particion (x=0)."""
    tp_vals      = sorted(split["test_per"].dropna().unique())
    present_acts = [a for a in ["tanh", "logistic"] if a in split["activation"].values]
    has_nosplit  = not nosplit.empty and any(
        not nosplit[nosplit["activation"] == a]["roc_auc"].dropna().empty
        for a in present_acts
    )

    # Build ordered list: 0 (nosplit) + actual test_per values
    all_tp   = ([0.0] if has_nosplit else []) + list(tp_vals)
    x_labels = (["Sin\nparticion"] if has_nosplit else []) + [f"{tp*100:.0f}%" for tp in tp_vals]

    n_acts = len(present_acts)
    width  = 0.32 if n_acts > 1 else 0.5
    gap    = 0.08

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 1.05, FIG_SIZE[1] * 0.85))

        for i, tp in enumerate(all_tp):
            if tp == 0.0:
                rows = nosplit
                auc_col = "roc_auc"
            else:
                rows    = split[np.isclose(split["test_per"].astype(float), tp, rtol=0, atol=1e-9)]
                auc_col = "roc_auc"

            for j, act in enumerate(present_acts):
                vals = rows[rows["activation"] == act][auc_col].dropna().values
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
                    boxprops=dict(facecolor=COLORS_ACT[act],
                                  alpha=0.35 if tp == 0.0 else 0.55,
                                  edgecolor=COLORS_ACT[act],
                                  linestyle="--" if tp == 0.0 else "-"),
                )

        # Separator line between no-split and split columns
        if has_nosplit:
            ax.axvline(0.5, color="#7f8c8d", linestyle=":", linewidth=1.2, alpha=0.6)

        rec_i = next((i for i, tp in enumerate(all_tp) if tp > 0 and np.isclose(tp, rec_tp, atol=1e-9)), None)
        if rec_i is not None:
            ax.axvline(rec_i, color="#e67e22", linestyle="--", linewidth=1.5,
                       label=f"Recomendado: {rec_tp*100:.0f}% test")

        ax.set_xticks(range(len(all_tp)))
        ax.set_xticklabels(x_labels, fontsize=8.5)
        ax.set_xlabel("Porcentaje de test (%)")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Distribucion de AUC — sin particion vs distintos porcentajes de test\n"
                     "(caja punteada = AUC en entrenamiento, sin particion)")

        legend_handles = [
            mpatches.Patch(facecolor=COLORS_ACT[act], alpha=0.7, label=LABEL_ACT[act])
            for act in present_acts
        ]
        if has_nosplit:
            legend_handles.append(
                mpatches.Patch(facecolor="#aaaaaa", alpha=0.4,
                               label="Sin particion (AUC entrenamiento)",
                               linestyle="--", edgecolor="#555555")
            )
        if rec_i is not None:
            legend_handles.append(
                plt.Line2D([0], [0], color="#e67e22", linestyle="--",
                           label=f"Recomendado: {rec_tp*100:.0f}% test")
            )
        ax.legend(handles=legend_handles, fontsize=9)
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "gen_study_q2b_auc_boxplots.png")


def _print_q2b(split: pd.DataFrame, nosplit: pd.DataFrame, rec_tp: float) -> None:
    n_seeds = split["seed"].nunique()
    print("\n-- Q2b ----------------------------------------------------")
    print(f"Estrategia: particion estratificada por clase, repetida sobre {n_seeds} semillas.")
    if not nosplit.empty:
        print("  Caso sin particion incluido (AUC en entrenamiento, cota superior).")
        for act in ["tanh", "logistic"]:
            rows = nosplit[nosplit["activation"] == act]["roc_auc"].dropna()
            if not rows.empty:
                print(f"  Sin particion {LABEL_ACT[act]:10s} AUC = {rows.mean():.4f} +- {rows.std():.4f}")
    print(f"Porcentaje de test recomendado: {rec_tp*100:.0f}% (test_per = {rec_tp})")
    for act in ["tanh", "logistic"]:
        rows = split[
            (split["activation"] == act) &
            np.isclose(split["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
        ]["roc_auc"]
        if rows.empty:
            continue
        print(f"  {LABEL_ACT[act]:10s} AUC = {rows.mean():.4f} +- {rows.std():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Q2c — Mejor modelo
# ──────────────────────────────────────────────────────────────────────────────

def _best_activation(
    split: pd.DataFrame, rec_tp: float,
    tanh_lr: float | None, logistic_lr: float | None,
) -> tuple[str, float]:
    best_act, best_auc = None, -1.0
    for act in ["tanh", "logistic"]:
        rows = split[
            (split["activation"] == act) &
            np.isclose(split["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
        ]
        if rows.empty:
            continue
        m = float(rows["roc_auc"].mean())
        if m > best_auc:
            best_auc = m
            best_act = act
    if best_act is None:
        raise SystemExit("No se encontro ninguna activacion valida en los datos.")
    lr_map = {"tanh": tanh_lr, "logistic": logistic_lr}
    best_lr = lr_map.get(best_act) or float(
        split[split["activation"] == best_act]["lr"].iloc[0]
    )
    return best_act, best_lr


def _interp_roc_curves(roc: pd.DataFrame, activation: str, rec_tp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate ROC (FPR→TPR) over seeds, return (fpr_grid, mean_tpr, std_tpr)."""
    rows = roc[
        (roc["activation"] == activation) &
        np.isclose(roc["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
    ]
    fpr_grid = np.linspace(0, 1, 200)
    tprs: list[np.ndarray] = []
    for _, grp in rows.groupby("seed"):
        g = grp.sort_values("fpr")
        f = g["fpr"].values
        t = g["tpr"].values
        _, idx = np.unique(f, return_index=True)
        f, t = f[idx], t[idx]
        if len(f) < 2:
            continue
        tprs.append(np.interp(fpr_grid, f, t, left=0.0, right=1.0))
    if not tprs:
        return fpr_grid, np.zeros(200), np.zeros(200)
    mat = np.stack(tprs)
    return fpr_grid, mat.mean(0), mat.std(0)


def _recommend_threshold(roc: pd.DataFrame, best_act: str, rec_tp: float) -> float:
    """Umbral que maximiza F1 promediado sobre semillas."""
    rows = roc[
        (roc["activation"] == best_act) &
        np.isclose(roc["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
    ]
    thr_grid = np.linspace(0, 1, 200)
    f1s: list[np.ndarray] = []
    for _, grp in rows.groupby("seed"):
        g = grp.sort_values("threshold")
        t = g["threshold"].values
        p = g["precision"].values
        r = g["recall"].values
        f1 = 2 * p * r / np.where((p + r) == 0, 1, p + r)
        _, idx = np.unique(t, return_index=True)
        t, f1 = t[idx], f1[idx]
        if len(t) < 2:
            continue
        f1s.append(np.interp(thr_grid, t, f1, left=f1[0], right=f1[-1]))
    if not f1s:
        return 0.5
    mean_f1 = np.stack(f1s).mean(0)
    return float(thr_grid[np.argmax(mean_f1)])


def plot_q2c_roc(roc: pd.DataFrame, rec_tp: float, best_act: str, tanh_lr: float, logistic_lr: float) -> None:
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.78, FIG_SIZE[1] * 0.88))

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Azar (AUC = 0.500)")

        present_acts = [a for a in ["tanh", "logistic"] if a in roc["activation"].values]
        for act in present_acts:
            fpr_g, mean_tpr, std_tpr = _interp_roc_curves(roc, act, rec_tp)
            auc_val = float(np.trapezoid(mean_tpr, fpr_g))
            is_best  = act == best_act
            ls  = "-" if is_best else "--"
            lw  = 2.2 if is_best else 1.4
            alpha_fill = 0.12 if is_best else 0
            label = f"{LABEL_ACT[act]} — AUC = {auc_val:.3f}"
            if is_best:
                label += " [mejor]"
            ax.plot(fpr_g, mean_tpr, color=COLORS_ACT[act], linestyle=ls,
                    linewidth=lw, label=label)
            if is_best:
                ax.fill_between(fpr_g, 0, mean_tpr, color=COLORS_ACT[act], alpha=alpha_fill)

        ax.set_xlabel("Tasa de falsos positivos (FPR)")
        ax.set_ylabel("Tasa de verdaderos positivos (Recall)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title("Curva ROC — comparación de activaciones")
        ax.legend(fontsize=9, loc="lower right")
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "gen_study_q2c_roc.png")


def plot_q2c_umbral(roc: pd.DataFrame, best_act: str, rec_tp: float, best_thr: float) -> None:
    rows = roc[
        (roc["activation"] == best_act) &
        np.isclose(roc["test_per"].astype(float), rec_tp, rtol=0, atol=1e-9)
    ]
    thr_grid = np.linspace(0, 1, 200)

    prec_list, rec_list, f1_list = [], [], []
    for _, grp in rows.groupby("seed"):
        g = grp.sort_values("threshold")
        t  = g["threshold"].values
        pr = g["precision"].values
        rc = g["recall"].values
        f1 = 2 * pr * rc / np.where((pr + rc) == 0, 1, pr + rc)
        _, idx = np.unique(t, return_index=True)
        t, pr, rc, f1 = t[idx], pr[idx], rc[idx], f1[idx]
        if len(t) < 2:
            continue
        prec_list.append(np.interp(thr_grid, t, pr, left=pr[0], right=pr[-1]))
        rec_list.append(np.interp(thr_grid, t, rc, left=rc[0], right=rc[-1]))
        f1_list.append(np.interp(thr_grid, t, f1, left=f1[0], right=f1[-1]))

    if not prec_list:
        print("  [Q2c-umbral] Sin curvas disponibles.")
        return

    mean_p  = np.stack(prec_list).mean(0)
    mean_r  = np.stack(rec_list).mean(0)
    mean_f1 = np.stack(f1_list).mean(0)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 0.82, FIG_SIZE[1] * 0.88))

        ax.plot(thr_grid, mean_p,  color="#2980b9", linewidth=2,   label="Precisión")
        ax.plot(thr_grid, mean_r,  color="#27ae60", linewidth=2,   label="Recall")
        ax.plot(thr_grid, mean_f1, color="#c0392b", linewidth=2.2, label="F1")
        ax.fill_between(thr_grid, mean_p, mean_r, alpha=0.07, color="#7f8c8d", label="Zona de trade-off")

        ax.axvline(best_thr, color="#e67e22", linestyle="--", linewidth=1.8,
                   label=f"Umbral recomendado = {best_thr:.3f}")

        ax.set_xlabel("Umbral de decisión")
        ax.set_ylabel("Valor de métrica")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_title(f"Precisión / Recall / F1 vs umbral de decisión  [{LABEL_ACT[best_act]}]")
        ax.legend(fontsize=9, loc="center right")
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
    recall_thr = float(rows["recall_at_threshold"].mean())
    prec_thr   = float(rows["precision_at_threshold"].mean())
    roc_auc_m  = float(rows["roc_auc"].mean())
    roc_auc_s  = float(rows["roc_auc"].std())
    f1_m       = float(rows["best_f1"].mean())
    f1_s       = float(rows["best_f1"].std())
    fpr_thr    = float(rows["fpr_at_threshold"].mean())

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
        ax_cm.set_title(f"Matriz de confusión (umbral = {best_thr:.3f})")
        ax_cm.set_facecolor(STYLE["axes_bg"])

        # — Metrics table —
        ax_tbl.axis("off")
        table_data = [
            ("Modelo",           LABEL_ACT[best_act]),
            ("Learning rate",    f"{best_lr:g}"),
            ("ROC-AUC (media)",  f"{roc_auc_m:.4f} ± {roc_auc_s:.4f}"),
            ("F1 óptimo (media)",f"{f1_m:.4f} ± {f1_s:.4f}"),
            ("Precisión @ umbral", f"{prec_thr:.4f}"),
            ("Recall @ umbral",    f"{recall_thr:.4f}"),
            ("Umbral recomendado", f"{best_thr:.3f}"),
            ("Tasa de fraude",     f"{fraud_rate * 100:.1f}%"),
            ("Muestras test",      f"{int(n_test):,}"),
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
        ax_tbl.set_title("Resumen del modelo final", color=STYLE["text_title"])

        fig.suptitle(
            f"Q2c — Recomendación al cliente: {LABEL_ACT[best_act]}, umbral = {best_thr:.3f}",
            fontsize=11,
        )
        fig.patch.set_facecolor(STYLE["figure_bg"])
        for spine in ax_cm.spines.values():
            spine.set_color(STYLE["text_axis"])
        ax_cm.tick_params(colors=STYLE["text_axis"])
        ax_cm.xaxis.label.set_color(STYLE["text_axis"])
        ax_cm.yaxis.label.set_color(STYLE["text_axis"])
        ax_cm.title.set_color(STYLE["text_title"])
        ax_tbl.title.set_color(STYLE["text_title"])
        fig.tight_layout(rect=[0, 0, 1, 0.95])
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
        print(f"  ROC-AUC  = {rows['roc_auc'].mean():.4f} +- {rows['roc_auc'].std():.4f}")
        print(f"  F1 opt.  = {rows['best_f1'].mean():.4f} +- {rows['best_f1'].std():.4f}")
        print(f"  Umbral recomendado: {best_thr:.3f} (maximiza F1)")
        print(f"  Recall @ umbral: {rows['recall_at_threshold'].mean() * 100:.1f}%  (fraude detectado)")
        print(f"  FPR @ umbral:    {rows['fpr_at_threshold'].mean() * 100:.1f}%  (alarmas falsas)")


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
    """Primer test_per no-nulo del grid, o 0.20 si no hay."""
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

    _, split, nosplit, roc = _load(tanh_lr, logistic_lr)

    if split.empty:
        raise SystemExit("Sin datos tras filtrar. Corre el experiment_runner con este config primero.")

    # Q2a
    print("\n[Q2a] Metricas...")
    plot_q2a_distribucion(split, test_per)
    plot_q2a_metricas(split, test_per)
    plot_q2a_pr_curve(roc, test_per)

    # Q2b
    print("\n[Q2b] Generalizacion...")
    rec_tp = _recommend_test_per(split)
    plot_q2b_auc_line(split, nosplit, rec_tp)
    plot_q2b_auc_boxplots(split, nosplit, rec_tp)
    _print_q2b(split, nosplit, rec_tp)

    # Q2c
    print("\n[Q2c] Mejor modelo...")
    best_act, best_lr = _best_activation(split, rec_tp, tanh_lr, logistic_lr)
    best_thr = _recommend_threshold(roc, best_act, rec_tp)
    plot_q2c_roc(roc, rec_tp, best_act, tanh_lr, logistic_lr)
    plot_q2c_umbral(roc, best_act, rec_tp, best_thr)
    plot_q2c_confusion_tabla(split, roc, best_act, best_lr, rec_tp, best_thr)
    _print_q2c(split, roc, best_act, best_lr, rec_tp, best_thr)

    print("\nListo. 8 figuras en plots/ej1/")


if __name__ == "__main__":
    main()
