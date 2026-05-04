from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.style import (
    BOXPLOT_STYLE,
    FIG_DPI,
    FIG_SIZE,
    PAIRWISE_COLORS,
    PLOT_RC,
    SAVE_PAD_INCHES,
    STYLE,
)

RESULTS = ROOT / "results"
PLOTS   = ROOT / "plots" / "ej1"
PLOTS.mkdir(parents=True, exist_ok=True)

COLOR_LINEAR    = PAIRWISE_COLORS[0]["box_face"]   # "#4a90d9"
COLOR_NONLINEAR = PAIRWISE_COLORS[1]["box_face"]   # "#e67e22"
COLORS_MT = {"linear": COLOR_LINEAR, "non-linear": COLOR_NONLINEAR}
EDGE_MT   = {"linear": PAIRWISE_COLORS[0]["box_edge"],
             "non-linear": PAIRWISE_COLORS[1]["box_edge"]}

# Colores para activaciones
COLORS_ACT = {
    "logistic": "#8e44ad",
    "tanh":     "#27ae60",
    "relu":     "#c0392b",
}

LABEL_MT  = {"linear": "Lineal", "non-linear": "No lineal"}
LABEL_ACT = {"logistic": "Logística", "tanh": "Tanh", "relu": "ReLU"}

SHOW_ACTIVATIONS: list[str] | None = ["tanh", "logistic"]

MODEL_TYPES_ORDER = ("linear", "non-linear")


def _model_types_present(df: pd.DataFrame) -> list[str]:
    """Orden fijo linear → non-linear, restringido a los que aparecen en ``df``."""
    if df.empty or "model_type" not in df.columns:
        return []
    u = set(df["model_type"].dropna().astype(str).unique())
    return [mt for mt in MODEL_TYPES_ORDER if mt in u]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de estilo
# ─────────────────────────────────────────────────────────────────────────────

def _apply_style(fig: plt.Figure, *axes: plt.Axes) -> None:
    """Aplica el estilo visual del proyecto a la figura y los ejes dados."""
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


def _boxplot(ax: plt.Axes, data: list[list[float]], positions: list[int],
             colors_face: list[str], colors_edge: list[str]) -> None:
    """Dibuja boxplot con el estilo del proyecto."""
    if not any(d for d in data):
        return
    bp = ax.boxplot(
        data, positions=positions, patch_artist=True, widths=0.55,
        medianprops={"linewidth": 2.0, "color": BOXPLOT_STYLE["median_color"]},
        whiskerprops={"linewidth": 0.9, "color": BOXPLOT_STYLE["whisker_color"]},
        capprops={"linewidth": 0.9, "color": BOXPLOT_STYLE["cap_color"]},
        flierprops={"marker": "o", "markersize": 3,
                    "color": BOXPLOT_STYLE["flier_color"], "alpha": 0.5},
    )
    for patch, fc, ec in zip(bp["boxes"], colors_face, colors_edge):
        patch.set_facecolor(fc)
        patch.set_edgecolor(ec)
        patch.set_alpha(0.75)


def _remove_outliers_iqr(values: list[float], factor: float = 1.5) -> list[float]:
    """Elimina outliers usando las cercas de Tukey (factor * IQR)."""
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return []
    if len(finite) <= 2:
        return finite
    arr = np.asarray(finite, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = float(q3 - q1)
    if iqr <= 0:
        return list(finite)
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    filtered = [float(x) for x in arr if lo <= x <= hi]
    return filtered if filtered else [float(np.median(arr))]


def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS / name
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight",
                pad_inches=SAVE_PAD_INCHES,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  guardado: {path.name}")


def _best_lr(df: pd.DataFrame, model_type: str, activation: str,
             metric: str = "roc_auc") -> float:
    if model_type == "linear":
        sub = df[df["model_type"] == "linear"]
    else:
        sub = df[(df["model_type"] == model_type) & (df["activation"] == activation)]
    if sub.empty:
        raise ValueError(
            f"_best_lr: sin filas para model_type={model_type!r}, activation={activation!r}"
        )
    g = sub.groupby("lr")[metric].mean().dropna()
    if g.empty:
        raise ValueError(
            f"_best_lr: métrica {metric!r} vacía/NaN para model_type={model_type!r}"
        )
    return float(g.idxmax())


def _filter_best_lr(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Mantiene solo las filas del mejor LR por (model_type, activation)."""
    parts = []
    for mt in df["model_type"].unique():
        for act in df["activation"].unique():
            sub = df[(df["model_type"] == mt) & (df["activation"] == act)]
            if sub.empty:
                continue
            best = float(sub.groupby("lr")[metric].mean().idxmax())
            parts.append(sub[sub["lr"] == best])
    return pd.concat(parts) if parts else df.iloc[:0]


def _jitter_strip(ax: plt.Axes, positions: list[int],
                  data: list[list[float]], colors: list[str],
                  seed: int = 42) -> None:
    """Superpone puntos con jitter sobre un boxplot para mostrar los datos individuales."""
    rng = np.random.default_rng(seed)
    for pos, vals, col in zip(positions, data, colors):
        if not vals:
            continue
        xs = np.full(len(vals), pos) + rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(xs, vals, s=22, color=col,
                   edgecolors=BOXPLOT_STYLE["point_edge"], linewidths=0.4,
                   alpha=0.75, zorder=4)


def _seed_curves(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (xs, mean, std) de curvas agrupadas por seed."""
    seed_arrays = [
        grp.sort_values("epoch")["train_mse"].to_numpy()
        for _, grp in df.groupby("seed")
    ]
    if not seed_arrays:
        return np.array([]), np.array([]), np.array([])
    lengths = [len(a) for a in seed_arrays]
    n = min(lengths)
    mat  = np.array([a[:n] for a in seed_arrays])
    mean = mat.mean(axis=0)
    std  = mat.std(axis=0)
    return np.arange(1, n + 1), mean, std


def _prefer_full_dataset(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Prefer no_split=True; fallback to all rows if that subset is empty."""
    full = df[df["no_split"] == True]
    if full.empty:
        print(f"  aviso: no hay filas no_split=True en {source_name}; "
              "se usa la partición con menor test_per como aproximación.")
        return df
    return full


def _activations_for_nl_panels(df: pd.DataFrame, col: str = "activation") -> list:
    """Activaciones por columna de panel: excluye ``identity`` (solo usada en el lineal)."""
    raw = df[col].dropna().unique()
    acts = sorted(raw, key=lambda x: str(x).lower())
    acts = [a for a in acts if str(a).lower() != "identity"]
    if SHOW_ACTIVATIONS is not None:
        acts = [a for a in acts if a in SHOW_ACTIVATIONS]
    return acts


# =============================================================================
# Q1a + Q1b — Curvas de aprendizaje (dataset completo)
# Q1a: underfitting -> MSE se estabiliza en un valor alto
# Q1b: saturacion  -> la curva se aplana antes de llegar a MSE bajo
# =============================================================================

def plot_q1ab_curvas(summary: pd.DataFrame, curves: pd.DataFrame) -> None:
    print("Q1a+Q1b — curvas de aprendizaje ...")
    full = _prefer_full_dataset(summary, "summary")
    c    = _prefer_full_dataset(curves, "curves")
    data_label = "dataset completo" if full["no_split"].any() else "partición train/test"

    if not c["no_split"].any():
        min_tp = c["test_per"].dropna().min()
        c    = c[c["test_per"] == min_tp]
        full = full[full["test_per"] == min_tp]

    activations = _activations_for_nl_panels(full)
    if not activations:
        print("  aviso: no hay activaciones no lineales para Q1a+Q1b; se omite este gráfico.")
        return
    n = len(activations)
    mts = _model_types_present(full)

    # Pre-calcular curvas para determinar ylim y evitar recomputar en el loop de dibujo
    all_curve_data: dict[tuple, tuple] = {}
    for act in activations:
        for mt in mts:
            if mt == "linear":
                sub_full = full[full["model_type"] == "linear"]
            else:
                sub_full = full[(full["model_type"] == "non-linear") & (full["activation"] == act)]
            if sub_full.empty:
                all_curve_data[(act, mt)] = (np.nan, np.array([]), np.array([]), np.array([]), 0)
                continue
            best = float(sub_full.groupby("lr")["roc_auc"].mean().idxmax())
            if mt == "linear":
                sub = c[(c["model_type"] == "linear") & (c["lr"] == best)]
            else:
                sub = c[(c["activation"] == act) & (c["model_type"] == mt) & (c["lr"] == best)]
            xs, mean, std = _seed_curves(sub)
            n_seeds = int(sub["seed"].nunique()) if not sub.empty else 0
            all_curve_data[(act, mt)] = (best, xs, mean, std, n_seeds)

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, n, figsize=(FIG_SIZE[0], FIG_SIZE[1]), sharey=False)
        if n == 1:
            axes = [axes]

        for ax, act in zip(axes, activations):
            for mt in mts:
                best, xs, mean, std, n_seeds = all_curve_data[(act, mt)]
                if xs.size == 0:
                    continue
                ax.plot(xs, mean, color=COLORS_MT[mt], linewidth=2,
                        label=f"{LABEL_MT[mt]}  (lr={best}, n={n_seeds})")
                ax.fill_between(xs, np.maximum(mean - std, 1e-8), mean + std,
                                color=COLORS_MT[mt], alpha=0.18)

            ax.set_yscale("log")
            ax.set_title(f"Activaci\u00f3n: {LABEL_ACT.get(act, act)}", fontsize=11)
            ax.set_xlabel("\u00c9poca")
            if ax.lines:
                ax.legend(fontsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))

        axes[0].set_ylabel("MSE de entrenamiento  (escala log, media \u00b1 desv\u00edo, 10 semillas)")
        fig.suptitle(
            f"Q1a + Q1b  \u2014  Curvas de aprendizaje ({data_label}, mejor lr por modelo)\n"
            "Underfitting: MSE se mantiene alto tras convergencia  |  "
            "Saturaci\u00f3n: la curva se aplana sin descender",
            fontsize=10,
        )
        _apply_style(fig, *axes)
        fig.tight_layout()
        _save(fig, "q1ab_curvas_aprendizaje.png")


# =============================================================================
# Q1a + Q1b (suplementario) — Efecto del learning rate en el MSE final
# Muestra claramente underfitting para LR bajos y saturacion para el lineal
# =============================================================================

def plot_q1ab_lr(summary: pd.DataFrame) -> None:
    print("Q1a+Q1b — sensibilidad al learning rate ...")
    full = _prefer_full_dataset(summary, "summary")
    activations = _activations_for_nl_panels(full)
    lrs = sorted(full["lr"].dropna().unique())
    n = len(activations)
    has_linear = "linear" in _model_types_present(full)
    if n == 0 or len(lrs) == 0:
        print("  aviso: no hay activaciones o learning rates para Q1a+Q1b LR; se omite este gráfico.")
        return

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, n, figsize=(FIG_SIZE[0], FIG_SIZE[1]),
                                  sharey=True)
        if n == 1:
            axes = [axes]

        gap = 0.5
        w   = 2

        for ax, act in zip(axes, activations):
            af_l  = full[full["model_type"] == "linear"]
            af_nl = full[(full["model_type"] == "non-linear") & (full["activation"] == act)]
            positions_l, positions_nl = [], []
            data_l, data_nl = [], []
            tick_pos, tick_lbl = [], []

            for i, lr in enumerate(lrs):
                base = i * (w + gap)
                if has_linear:
                    positions_l.append(base)
                    positions_nl.append(base + 1)
                    tick_pos.append(base + 0.5)
                else:
                    positions_nl.append(base)
                    tick_pos.append(base)
                # IQR con factor=3 para eliminar solo divergencias extremas
                # _remove_outliers_iqr ya descarta Inf/NaN internamente
                if has_linear:
                    dl = _remove_outliers_iqr(
                        af_l[af_l["lr"] == lr]["final_train_mse"].dropna().tolist(), factor=3.0)
                    data_l.append([max(v, 1e-6) for v in dl if np.isfinite(v)])
                dnl = _remove_outliers_iqr(
                    af_nl[af_nl["lr"] == lr]["final_train_mse"].dropna().tolist(), factor=3.0)
                # Escala log requiere valores > 0 y finitos
                data_nl.append([max(v, 1e-6) for v in dnl if np.isfinite(v)])
                tick_lbl.append(f"{lr:.0e}" if lr < 0.01 else str(lr))

            if has_linear:
                _boxplot(ax, data_l,  positions_l,
                         [COLOR_LINEAR]    * len(lrs), [EDGE_MT["linear"]]    * len(lrs))
            _boxplot(ax, data_nl, positions_nl,
                     [COLOR_NONLINEAR] * len(lrs), [EDGE_MT["non-linear"]] * len(lrs))

            ax.set_yscale("log")
            ax.set_ylim(bottom=5e-4, top=20)   # foco en el rango informativo
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, fontsize=8)
            ax.set_title(f"Activaci\u00f3n: {LABEL_ACT.get(act, act)}", fontsize=11)
            ax.set_xlabel("Tasa de aprendizaje")

        from matplotlib.patches import Patch
        leg_handles = [Patch(facecolor=COLOR_NONLINEAR, label="No lineal")]
        if has_linear:
            leg_handles.insert(0, Patch(facecolor=COLOR_LINEAR, label="Lineal"))
        axes[-1].legend(handles=leg_handles, fontsize=9, loc="upper right")
        ref_mt = "linear" if has_linear else "non-linear"
        n_per_box = int(
            full[(full["model_type"] == ref_mt) & (full["lr"] == lrs[0])]["seed"].count()
        )
        axes[0].set_ylabel(f"MSE final de entrenamiento  (escala logar\u00edtmica, n={n_per_box} por caja)")
        lr_caption = (
            "LR bajo: ambos presentan underfitting  |  LR \u00f3ptimo: lineal satura (~5), no lineal converge (~0.01)"
            if has_linear
            else "LR bajo: underfitting  |  LR \u00f3ptimo: el no lineal converge a MSE bajo"
        )
        fig.suptitle(
            "Q1a + Q1b  \u2014  MSE final seg\u00fan tasa de aprendizaje (dataset completo, escala log)\n"
            + lr_caption,
            fontsize=10,
        )
        _apply_style(fig, *axes)
        fig.tight_layout()
        _save(fig, "q1ab_sensibilidad_lr.png")


# =============================================================================
# Q1c — Comparacion de capacidad de aprendizaje
# Decide que perceptron tiene mas potencial -> selecciónar para generalización
# =============================================================================

def _plot_q1c_panel(full: pd.DataFrame, metric: str, ylabel: str,
                    use_log: bool, ylim_fixed: tuple | None,
                    suptitle: str, filename: str) -> None:
    """Genera un panel 1×n_act para una metrica de Q1c."""
    from matplotlib.patches import Patch
    activations = _activations_for_nl_panels(full)
    if not activations:
        print(f"  aviso: sin activaciones no lineales; se omite {filename}.")
        return
    n_act = len(activations)
    mts = _model_types_present(full)
    if not mts:
        print(f"  aviso: sin tipos de modelo para Q1c; se omite {filename}.")
        return

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, n_act, figsize=FIG_SIZE, sharey=True)
        if n_act == 1:
            axes = [axes]

        for col, act in enumerate(activations):
            ax = axes[col]
            tick_labels, data_pair = [], []
            pos = list(range(1, len(mts) + 1))
            faces, edges = [], []
            for mt in mts:
                best = _best_lr(full, mt, act)
                if mt == "linear":
                    raw = full[
                        (full["model_type"] == "linear") & (full["lr"] == best)
                    ][metric].dropna().tolist()
                else:
                    raw = full[
                        (full["model_type"] == "non-linear") &
                        (full["activation"] == act) &
                        (full["lr"] == best)
                    ][metric].dropna().tolist()
                if use_log:
                    raw = [max(v, 1e-6) for v in raw]
                data_pair.append(raw)
                tick_labels.append(f"{LABEL_MT[mt]}\nlr={best}\nn={len(raw)}")
                faces.append(COLORS_MT[mt])
                edges.append(EDGE_MT[mt])

            _boxplot(ax, data_pair, pos, faces, edges)
            _jitter_strip(ax, pos, data_pair, faces)
            ax.set_xticks(pos)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_title(f"Activaci\u00f3n: {LABEL_ACT.get(act, act)}", fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if use_log:
                ax.set_yscale("log")
            elif ylim_fixed:
                ax.set_ylim(*ylim_fixed)

        leg = [Patch(facecolor=COLOR_NONLINEAR, label="No lineal")]
        if "linear" in mts:
            leg.insert(0, Patch(facecolor=COLOR_LINEAR, label="Lineal"))
        fig.legend(handles=leg, loc="upper right", fontsize=9)
        fig.suptitle(suptitle, fontsize=10)
        _apply_style(fig, *axes)
        fig.tight_layout()
        _save(fig, filename)


def plot_q1c(summary: pd.DataFrame) -> None:
    """Genera dos gráficos separados para Q1c: ROC-AUC y MSE final."""
    print("Q1c — capacidad de aprendizaje ...")
    full = _prefer_full_dataset(summary, "summary")
    if full.empty:
        print("  aviso: no hay datos para Q1c; se omiten estos gráficos.")
        return
    has_lin = "linear" in _model_types_present(full)

    _plot_q1c_panel(
        full,
        metric="train_roc_auc",
        ylabel="ROC-AUC de entrenamiento",
        use_log=False,
        ylim_fixed=None,
        suptitle=(
            "Q1c  \u2014  Capacidad de aprendizaje: ROC-AUC al mejor LR\n"
            + (
                "No lineal (log\u00edstica/ReLU) logra mayor AUC que lineal "
                "\u2192 se selecciona para generalizaci\u00f3n"
                if has_lin
                else "Modelos no lineales: AUC de entrenamiento al mejor LR por activaci\u00f3n"
            )
        ),
        filename="q1c_roc_auc.png",
    )

    _plot_q1c_panel(
        full,
        metric="final_train_mse",
        ylabel="MSE final de entrenamiento  (escala log)",
        use_log=True,
        ylim_fixed=None,
        suptitle=(
            "Q1c  \u2014  Capacidad de aprendizaje: MSE final al mejor LR  (escala log)\n"
            + (
                "No lineal alcanza MSE ~10\u00d7 menor que lineal "
                "\u2192 mayor capacidad de representaci\u00f3n"
                if has_lin
                else "MSE final de entrenamiento (mejor LR) por activaci\u00f3n no lineal"
            )
        ),
        filename="q1c_mse_final.png",
    )


# =============================================================================
# Q2a — Justificación de métricas de evaluación
# Muestra el desbalance de clases y por que accuracy sola es engañosa
# =============================================================================

def plot_q2a(summary: pd.DataFrame) -> None:
    print("Q2a — justificación de métricas ...")
    split = summary[summary["no_split"] == False]
    mts = _model_types_present(split)
    pos = list(range(1, len(mts) + 1))
    colors_b = [COLORS_MT[mt] for mt in mts]
    edges_b = [EDGE_MT[mt] for mt in mts]

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 3, figsize=(FIG_SIZE[0] * 1.1, FIG_SIZE[1]))

        # Panel 1: tasa de fraude -> desbalance de clases
        ax = axes[0]
        fr = split["fraud_rate_test"].dropna()
        ax.hist(fr, bins=20, color=COLOR_NONLINEAR, edgecolor="white", alpha=0.85)
        ax.axvline(fr.mean(), color=BOXPLOT_STYLE["median_color"],
                   linestyle="--", linewidth=1.8,
                   label=f"Media = {fr.mean():.1%}")
        ax.set_xlabel("Tasa de fraude en test")
        ax.set_ylabel("Cantidad de corridas")
        ax.set_title("Desbalance de clases\n(~11% fraude -> accuracy engaña)")
        ax.legend(fontsize=8)

        # Panel 2: accuracy vs ROC-AUC por tipo de modelo (solo mejor LR)
        split_best = _filter_best_lr(split)
        # Baseline calculado sobre el mismo subconjunto que los boxplots
        fr_best = split_best["fraud_rate_test"].dropna()
        baseline = 1 - (fr_best.mean() if not fr_best.empty else fr.mean())

        def _mt_tick_label(df_filt, mt: str) -> str:
            sub = df_filt[df_filt["model_type"] == mt]
            lrs = sorted(sub["lr"].unique())
            lr_str = "/".join(str(lr) for lr in lrs)
            return f"{LABEL_MT[mt]}\nlr={lr_str}\nn={len(sub)}"

        for ax, (metric, ylabel, ylim) in zip(axes[1:], [
            ("test_acc", "Accuracy (tolerancia 0.5)", (0.8, 1.02)),
            ("roc_auc",  "ROC-AUC (test)",            (0.8, 1.02)),
        ]):
            data = [split_best[split_best["model_type"] == mt][metric].dropna().tolist()
                    for mt in mts]
            _boxplot(ax, data, pos, colors_b, edges_b)
            ax.set_xticks(pos)
            ax.set_xticklabels([_mt_tick_label(split_best, mt) for mt in mts], fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel, fontsize=10)
            ax.set_ylim(*ylim)
            # Linea de referencia: baseline trivial para accuracy
            if "acc" in metric:
                ax.axhline(baseline, color=STYLE["grid"], linestyle="--",
                           linewidth=1.2,
                           label=f"Baseline trivial: {baseline:.1%}")
                ax.legend(fontsize=7)

        fig.suptitle(
            "Q2a  —  ¿Por qué usar ROC-AUC, F1, Precisión y Recall en lugar de accuracy?\n"
            "Con ~11% de fraude, predecir siempre 'no fraude' da >88% accuracy pero recall = 0",
            fontsize=10,
        )
        _apply_style(fig, *axes)
        fig.tight_layout()
        _save(fig, "q2a_justificacion_metricas.png")


# =============================================================================
# Q2b — Efecto del tamaño del conjunto de entrenamiento en la generalización
# Muestra como varia la performance con test_per -> revela la mejor partición
# =============================================================================

def _best_two_models(split: pd.DataFrame) -> tuple:
    """Devuelve (mt, act, lr) del mejor no-lineal y del mejor lineal (o None si no hay lineal)."""
    nl_mt, nl_act, nl_lr = (
        split[split["model_type"] == "non-linear"]
        .groupby(["model_type", "activation", "lr"])["roc_auc"]
        .mean().idxmax()
    )
    lin = split[split["model_type"] == "linear"]
    if lin.empty:
        return (nl_mt, nl_act, nl_lr), None
    l_act, l_lr = lin.groupby(["activation", "lr"])["roc_auc"].mean().idxmax()
    return (nl_mt, nl_act, nl_lr), ("linear", l_act, l_lr)


def plot_q2b_split(summary: pd.DataFrame) -> None:
    print("Q2b — generalización vs tamaño de partición ...")
    split = summary[summary["no_split"] == False]
    (nl_mt, nl_act, nl_lr), linear_best = _best_two_models(split)

    models = [(nl_mt, nl_act, nl_lr, COLOR_NONLINEAR, EDGE_MT["non-linear"])]
    if linear_best is not None:
        l_mt, l_act, l_lr = linear_best
        models.append((l_mt, l_act, l_lr, COLOR_LINEAR, EDGE_MT["linear"]))
    metrics = [("roc_auc", "ROC-AUC (test)"), ("best_f1", "F1 \u00f3ptimo (test)")]

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)

        for ax, (metric, ylabel) in zip(axes, metrics):
            for mt, act, lr, color, edge in models:
                sub = split[
                    (split["model_type"] == mt) &
                    (split["activation"] == act) &
                    (split["lr"] == lr)
                ]
                if sub.empty:
                    continue
                g    = sub.groupby("test_per")[metric]
                mean = g.mean()
                std  = g.std()
                xs   = mean.index.to_numpy() * 100
                n_s  = sub["seed"].nunique()
                ax.plot(xs, mean.values,
                        color=color, marker="o", markersize=6, linewidth=2,
                        label=f"{LABEL_MT[mt]} / {LABEL_ACT.get(act, act)}  "
                              f"lr={lr}, n={n_s}")
                ax.fill_between(xs,
                                mean.values - std.values,
                                mean.values + std.values,
                                color=color, alpha=0.12)

            ax.set_xlabel("Tama\u00f1o del conjunto de test  (%)")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel, fontsize=10)
            ax.legend(fontsize=8)
            tp_vals = split["test_per"].dropna().unique()
            if len(tp_vals) > 1:
                lo, hi = float(tp_vals.min() * 100), float(tp_vals.max() * 100)
                pad = max(1.0, (hi - lo) * 0.08)
                ax.set_xlim(lo - pad, hi + pad)

        fig.suptitle(
            "Q2b  \u2014  \u00bfC\u00f3mo afecta el tama\u00f1o del conjunto de entrenamiento a la generalizaci\u00f3n?\n"
            "Mayor test_per = menor entrenamiento; se comparan el mejor modelo no lineal y el mejor lineal",
            fontsize=10,
        )
        _apply_style(fig, *axes)
        fig.tight_layout()
        _save(fig, "q2b_generalizacion_vs_split.png")


# =============================================================================
# Q2b (suplementario) — Diagnóstico overfitting: MSE entrenamiento vs test
# Si MSE train << MSE test -> overfitting; si ambos altos -> underfitting
# =============================================================================

def plot_q2b_overfitting(summary: pd.DataFrame) -> None:
    print("Q2b — diagn\u00f3stico de overfitting ...")
    split = summary[summary["no_split"] == False]
    (nl_mt, nl_act, nl_lr), linear_best = _best_two_models(split)

    models = [
        (nl_mt, nl_act, nl_lr, COLOR_NONLINEAR,
         f"No lineal / {LABEL_ACT.get(nl_act, nl_act)},  lr={nl_lr}"),
    ]
    if linear_best is not None:
        l_mt, l_act, l_lr = linear_best
        models.append(
            (l_mt, l_act, l_lr, COLOR_LINEAR,
             f"Lineal / {LABEL_ACT.get(l_act, l_act)},  lr={l_lr}"),
        )

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        for mt, act, lr, color, base_lbl in models:
            sub = split[
                (split["model_type"] == mt) &
                (split["activation"] == act) &
                (split["lr"] == lr)
            ]
            if sub.empty:
                continue
            n_s = sub["seed"].nunique()

            g_train = sub.groupby("test_per")["final_train_mse"]
            g_test  = sub.groupby("test_per")["mse"]
            xs   = g_train.mean().index.to_numpy() * 100
            m_tr = g_train.mean().values
            m_te = g_test.mean().values
            s_te = g_test.std().values

            # lr y n solo en la leyenda, sin repetir en el titulo
            ax.plot(xs, m_tr, color=color, linestyle="--",
                    linewidth=1.8, marker="^", markersize=5,
                    label=f"{base_lbl},  n={n_s}  (train)")
            ax.plot(xs, m_te, color=color, linestyle="-",
                    linewidth=2.2, marker="o", markersize=6,
                    label=f"{base_lbl},  n={n_s}  (test)")
            ax.fill_between(xs, m_te - s_te, m_te + s_te,
                            color=color, alpha=0.12)

        ax.set_xlabel("Tama\u00f1o del conjunto de test  (%)")
        ax.set_ylabel("MSE  (media sobre semillas)")
        ax.legend(fontsize=8)
        tp_vals_ov = split["test_per"].dropna().unique()
        if len(tp_vals_ov) > 1:
            lo, hi = float(tp_vals_ov.min() * 100), float(tp_vals_ov.max() * 100)
            pad = max(1.0, (hi - lo) * 0.08)
            ax.set_xlim(lo - pad, hi + pad)

        fig.suptitle(
            "Q2b  \u2014  Diagn\u00f3stico de overfitting: MSE de entrenamiento vs test\n"
            "Mejores modelos seleccionados por ROC-AUC  \u2014  "
            "L\u00ednea discontinua = train  |  Continua = test  |  "
            "Brecha peque\u00f1a \u2192 sin overfitting",
            fontsize=10,
        )
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "q2b_overfitting_diagnostico.png")


# =============================================================================
# Q2c — Curva ROC del mejor modelo vs lineal
# =============================================================================

def plot_q2c_roc(roc_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("Q2c — curva ROC del mejor modelo ...")
    split = summary[summary["no_split"] == False]

    # Mejor configuracion no lineal (restringido a non-linear; Q1c ya eligió éste)
    best_mt, best_act, best_lr = (
        split[split["model_type"] == "non-linear"]
        .groupby(["model_type", "activation", "lr"])["roc_auc"]
        .mean().idxmax()
    )
    best_auc = float(
        split[
            (split["model_type"] == best_mt) &
            (split["activation"] == best_act) &
            (split["lr"] == best_lr)
        ]["roc_auc"].mean()
    )
    print(f"  mejor: {best_mt} / {best_act} / lr={best_lr}  "
          f"(AUC medio={best_auc:.4f})")

    lin_sub = split[split["model_type"] == "linear"]
    has_linear = not lin_sub.empty
    lin_act = lin_lr = None
    lin_auc = float("nan")
    if has_linear:
        lin_act, lin_lr = lin_sub.groupby(["activation", "lr"])["roc_auc"].mean().idxmax()
        lin_auc = float(lin_sub.groupby(["activation", "lr"])["roc_auc"].mean().max())

    common_fpr = np.linspace(0, 1, 200)

    def _mean_roc(model_type, act, lr):
        sub = roc_df[
            (roc_df["model_type"] == model_type) &
            (roc_df["activation"] == act) &
            (roc_df["lr"] == lr) &
            (roc_df["no_split"] == False)
        ]
        curves = []
        for _, grp in sub.groupby(["seed", "test_per"]):
            g = grp.sort_values("fpr")
            curves.append(
                np.interp(common_fpr, g["fpr"].to_numpy(), g["tpr"].to_numpy())
            )
        if not curves:
            return None, None
        mat = np.array(curves)
        return mat.mean(axis=0), mat.std(axis=0)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1,
                label="Clasificador aleatorio  (AUC = 0.5)")

        n_nl = split[(split["model_type"] == best_mt) & (split["activation"] == best_act) & (split["lr"] == best_lr)]["seed"].nunique()

        mean_nl, std_nl = _mean_roc(best_mt, best_act, best_lr)
        if mean_nl is not None:
            ax.plot(common_fpr, mean_nl, color=COLOR_NONLINEAR, linewidth=2.5,
                    label=(f"No lineal / {LABEL_ACT.get(best_act, best_act)} / "
                           f"lr={best_lr}, n={n_nl}  (AUC={best_auc:.4f})"))
            ax.fill_between(common_fpr, mean_nl - std_nl, mean_nl + std_nl,
                            color=COLOR_NONLINEAR, alpha=0.2)

        if has_linear and lin_act is not None and lin_lr is not None:
            n_l = split[
                (split["model_type"] == "linear")
                & (split["activation"] == lin_act)
                & (split["lr"] == lin_lr)
            ]["seed"].nunique()
            mean_l, std_l = _mean_roc("linear", lin_act, lin_lr)
            if mean_l is not None:
                ax.plot(common_fpr, mean_l, color=COLOR_LINEAR, linewidth=2,
                        linestyle="--",
                        label=(f"Lineal / {LABEL_ACT.get(lin_act, lin_act)} / "
                               f"lr={lin_lr}, n={n_l}  (AUC={lin_auc:.4f})"))

        ax.set_xlabel("Tasa de falsos positivos (FPR)")
        ax.set_ylabel("Tasa de verdaderos positivos (TPR / Recall)")
        roc_title = (
            "Q2c  —  Curva ROC: mejor modelo no lineal vs mejor lineal\n"
            "(promedio sobre semillas y particiones)"
            if has_linear
            else "Q2c  —  Curva ROC: mejor modelo no lineal\n"
                 "(promedio sobre semillas y particiones)"
        )
        ax.set_title(roc_title, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "q2c_curva_roc.png")


# =============================================================================
# Q2c — Recomendación del umbral de detección de fraude
# Muestra como varia Precision/Recall/F1 con el umbral y la distribución
# del umbral óptimo (max-F1) a través de distintas particiones y semillas
# =============================================================================

def plot_q2c_umbral(summary: pd.DataFrame, roc_df: pd.DataFrame) -> None:
    print("Q2c — recomendación de umbral ...")
    split = summary[summary["no_split"] == False]

    best_mt, best_act, best_lr = (
        split[split["model_type"] == "non-linear"]
        .groupby(["model_type", "activation", "lr"])["roc_auc"]
        .mean().idxmax()
    )
    best_runs = split[
        (split["model_type"] == best_mt) &
        (split["activation"] == best_act) &
        (split["lr"] == best_lr)
    ]

    common_thr = np.linspace(0, 1, 200)
    sub_roc = roc_df[
        (roc_df["model_type"] == best_mt) &
        (roc_df["activation"] == best_act) &
        (roc_df["lr"] == best_lr) &
        (roc_df["no_split"] == False)
    ]

    prec_list, rec_list, f1_list = [], [], []
    for _, grp in sub_roc.groupby(["seed", "test_per"]):
        g = grp.sort_values("threshold")
        p = np.interp(common_thr, g["threshold"].to_numpy(),
                      g["precision"].to_numpy(), left=0.0, right=0.0)
        r = np.interp(common_thr, g["threshold"].to_numpy(),
                      g["recall"].to_numpy(), left=1.0, right=0.0)
        denom = p + r
        f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f1)

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)

        # Panel 1: Precision / Recall / F1 vs umbral
        ax = axes[0]
        if prec_list:
            mp = np.array(prec_list).mean(axis=0)
            mr = np.array(rec_list).mean(axis=0)
            mf = np.array(f1_list).mean(axis=0)
            best_thr_idx = int(np.argmax(mf))
            best_thr_val = common_thr[best_thr_idx]

            ax.plot(common_thr, mp, color="#2980b9", linewidth=2,
                    label="Precision")
            ax.plot(common_thr, mr, color=BOXPLOT_STYLE["mean_color"],
                    linewidth=2, label="Recall")
            ax.plot(common_thr, mf,
                    color=BOXPLOT_STYLE["median_color"], linewidth=2.5,
                    label="F1")
            ax.axvline(best_thr_val, color=STYLE["text_title"],
                       linestyle="--", linewidth=1.8,
                       label=f"Umbral recomendado = {best_thr_val:.3f}")
            ax.fill_between(common_thr, mp, mr, alpha=0.06,
                            color="#aaaaaa", label="Zona de trade-off")

        ax.set_xlabel("Umbral de decisión")
        ax.set_ylabel("Métrica (media sobre semillas y particiones)")
        ax.set_title("Precision / Recall / F1 vs umbral\n"
                     "Linea vertical = umbral de max-F1 recomendado")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

        # Panel 2: distribución del umbral óptimo segun la partición
        ax = axes[1]
        test_pers = sorted(best_runs["test_per"].unique())
        tp_positions = list(range(1, len(test_pers) + 1))

        data_thr = [
            best_runs[best_runs["test_per"] == tp]["best_threshold"]
            .dropna().tolist()
            for tp in test_pers
        ]
        colors_f = [COLOR_NONLINEAR] * len(test_pers)
        colors_e = [EDGE_MT["non-linear"]] * len(test_pers)
        _boxplot(ax, data_thr, tp_positions, colors_f, colors_e)
        ax.set_xticks(tp_positions)
        ax.set_xticklabels([f"{tp*100:.0f}%" for tp in test_pers], fontsize=9)

        overall_mean = best_runs["best_threshold"].mean()
        ax.axhline(overall_mean,
                   color=BOXPLOT_STYLE["median_color"], linestyle="--",
                   linewidth=1.8,
                   label=f"Media global = {overall_mean:.3f}")
        ax.set_xlabel("Tamaño del conjunto de test")
        ax.set_ylabel("Umbral óptimo (max-F1)")
        ax.set_title("Estabilidad del umbral segun la partición\n"
                     "Robustez a traves de distintas divisiones train/test")
        ax.legend(fontsize=8)

        n_best = best_runs["seed"].nunique()
        fig.suptitle(
            f"Q2c  —  Recomendación de umbral de detección de fraude\n"
            f"Mejor modelo: {LABEL_MT[best_mt]} / {LABEL_ACT.get(best_act, best_act)} / "
            f"lr={best_lr}, n={n_best} semillas",
            fontsize=10,
        )
        _apply_style(fig, *axes)
        fig.tight_layout()
        _save(fig, "q2c_umbral_recomendado.png")


# =============================================================================
# Q2c (suplementario) — Matriz de confusion del mejor modelo en el umbral óptimo
# Presentación al cliente del modelo final
# =============================================================================

def plot_q2c_confusion(summary: pd.DataFrame) -> None:
    print("Q2c — matriz de confusion del mejor modelo ...")
    split = summary[summary["no_split"] == False]

    best_mt, best_act, best_lr = (
        split[split["model_type"] == "non-linear"]
        .groupby(["model_type", "activation", "lr"])["roc_auc"]
        .mean().idxmax()
    )
    best_runs = split[
        (split["model_type"] == best_mt) &
        (split["activation"] == best_act) &
        (split["lr"] == best_lr)
    ]

    # Aproximar la matriz de confusion a partir de las métricas
    n_test_avg   = best_runs["n_test"].mean()
    fr_avg       = best_runs["fraud_rate_test"].mean()
    prec_avg     = best_runs["best_precision"].mean()
    rec_avg      = best_runs["best_recall"].mean()
    thr_avg      = best_runs["best_threshold"].mean()
    auc_avg      = best_runs["roc_auc"].mean()
    f1_avg       = best_runs["best_f1"].mean()

    P  = n_test_avg * fr_avg
    N  = n_test_avg * (1 - fr_avg)
    TP = rec_avg * P
    FP = TP / prec_avg - TP if prec_avg > 0 else 0.0
    FN = P - TP
    TN = N - FP

    cm = np.array([[TP, FN], [FP, TN]])
    labels = [["VP", "FN"], ["FP", "VN"]]

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(FIG_SIZE[0] * 0.9, FIG_SIZE[1]),
                                  gridspec_kw={"width_ratios": [1.0, 1.2]})

        # Panel 1: heatmap de la matriz
        ax = axes[0]
        cmap = plt.get_cmap("Blues")
        im = ax.imshow(cm, cmap=cmap, aspect="auto", vmin=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        tick_names = ["Fraude (pred)", "No fraude (pred)"]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Fraude (pred)", "No fraude (pred)"],
                            fontsize=8, rotation=15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Fraude (real)", "No fraude (real)"], fontsize=8)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(j, i,
                        f"{labels[i][j]}\n{val:.0f}",
                        ha="center", va="center", fontsize=11,
                        color="white" if val > cm.max() * 0.5 else STYLE["text_title"],
                        fontweight="bold")
        ax.set_title("Matriz de confusion (media)\n"
                     f"umbral = {thr_avg:.3f}", fontsize=10)

        # Panel 2: resumen de métricas del modelo final
        ax = axes[1]
        ax.axis("off")
        fig.patch.set_facecolor(STYLE["figure_bg"])
        ax.set_facecolor(STYLE["axes_bg"])

        metric_rows = [
            ("Modelo",         f"{LABEL_MT[best_mt]} / {LABEL_ACT.get(best_act, best_act)}"),
            ("Tasa de aprendizaje", str(best_lr)),
            ("ROC-AUC",        f"{auc_avg:.4f}"),
            ("F1 (opt.)",      f"{f1_avg:.4f}"),
            ("Precision",      f"{prec_avg:.4f}"),
            ("Recall",         f"{rec_avg:.4f}"),
            ("Umbral opt.",    f"{thr_avg:.4f}"),
            ("Tasa de fraude", f"{fr_avg:.1%}"),
            ("Muestras test",  f"{n_test_avg:.0f}"),
        ]

        y = 0.95
        ax.text(0.5, 1.01, "Resumen del mejor modelo",
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=11, fontweight="bold",
                color=STYLE["text_title"])
        for label, value in metric_rows:
            ax.text(0.08, y, label, ha="left", va="top", transform=ax.transAxes,
                    fontsize=10, color=STYLE["text_axis"])
            ax.text(0.92, y, value, ha="right", va="top", transform=ax.transAxes,
                    fontsize=10, fontweight="bold", color=STYLE["text_title"])
            ax.plot([0.05, 0.95], [y - 0.01, y - 0.01], linewidth=0.4,
                    color=STYLE["grid"], transform=ax.transAxes,
                    clip_on=False)
            y -= 0.10

        n_best = best_runs["seed"].nunique()
        fig.suptitle(
            f"Q2c  —  Modelo final a presentar al cliente  "
            f"({LABEL_MT[best_mt]} / {LABEL_ACT.get(best_act, best_act)} / lr={best_lr}, n={n_best})\n"
            "Valores promediados sobre semillas y particiones",
            fontsize=10,
        )
        _apply_style(fig, axes[0])
        fig.tight_layout()
        _save(fig, "q2c_modelo_final.png")


# =============================================================================
# Q2 — Tanh vs Logística (mejores LR fijos): métricas, generalización, ROC, umbral
# =============================================================================

def _lr_match(series: pd.Series, lr: float) -> pd.Series:
    return np.isclose(series.astype(float), float(lr), rtol=0, atol=1e-11)


def _split_nl_tanh_logistic(
    summary: pd.DataFrame,
    tanh_lr: float,
    logistic_lr: float,
) -> pd.DataFrame:
    """Filas non-linear con (tanh, tanh_lr) o (logistic, logistic_lr)."""
    d = summary[(summary["no_split"] == False) & (summary["model_type"] == "non-linear")].copy()
    act = d["activation"].astype(str).str.lower().str.strip()
    m_tanh = (act == "tanh") & _lr_match(d["lr"], tanh_lr)
    m_log = (act == "logistic") & _lr_match(d["lr"], logistic_lr)
    out = d[m_tanh | m_log]
    return out


def _roc_nl_tanh_logistic(
    roc_df: pd.DataFrame,
    tanh_lr: float,
    logistic_lr: float,
) -> pd.DataFrame:
    d = roc_df[(roc_df["no_split"] == False) & (roc_df["model_type"] == "non-linear")].copy()
    act = d["activation"].astype(str).str.lower().str.strip()
    m_tanh = (act == "tanh") & _lr_match(d["lr"], tanh_lr)
    m_log = (act == "logistic") & _lr_match(d["lr"], logistic_lr)
    return d[m_tanh | m_log]


def plot_q2a_tanh_logistic(
    summary: pd.DataFrame,
    tanh_lr: float,
    logistic_lr: float,
) -> None:
    print("Q2a — Tanh vs Logística (métricas, LR fijos) ...")
    split = summary[summary["no_split"] == False]
    twin = _split_nl_tanh_logistic(summary, tanh_lr, logistic_lr)
    if twin.empty:
        print("  aviso: sin filas Tanh/Logística con los LR dados; se omite q2a_tanh_vs_logistica.")
        return

    acts_order = ("tanh", "logistic")
    lrs_map = {"tanh": tanh_lr, "logistic": logistic_lr}
    pos = list(range(1, len(acts_order) + 1))
    colors_b = [COLORS_ACT[a] for a in acts_order]
    edges_b = [PAIRWISE_COLORS[1]["box_edge"], PAIRWISE_COLORS[0]["box_edge"]]

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 3, figsize=(FIG_SIZE[0] * 1.1, FIG_SIZE[1]))

        ax = axes[0]
        fr = split["fraud_rate_test"].dropna()
        ax.hist(fr, bins=20, color=COLOR_NONLINEAR, edgecolor="white", alpha=0.85)
        ax.axvline(fr.mean(), color=BOXPLOT_STYLE["median_color"],
                   linestyle="--", linewidth=1.8,
                   label=f"Media = {fr.mean():.1%}")
        ax.set_xlabel("Tasa de fraude en test")
        ax.set_ylabel("Cantidad de corridas")
        ax.set_title("Desbalance de clases\n(~11% fraude -> accuracy engaña)")
        ax.legend(fontsize=8)

        fr_best = twin["fraud_rate_test"].dropna()
        baseline = 1 - (fr_best.mean() if not fr_best.empty else fr.mean())

        def _act_tick_label(act: str) -> str:
            sub = twin[
                (twin["activation"].astype(str).str.lower() == act)
                & _lr_match(twin["lr"], lrs_map[act])
            ]
            n = len(sub)
            return f"{LABEL_ACT[act]}\nlr={lrs_map[act]:g}\nn={n}"

        for ax, (metric, ylabel, ylim) in zip(axes[1:], [
            ("test_acc", "Accuracy (tolerancia 0.5)", (0.8, 1.02)),
            ("roc_auc",  "ROC-AUC (test)",            (0.8, 1.02)),
        ]):
            data = []
            for act in acts_order:
                sub = twin[
                    (twin["activation"].astype(str).str.lower() == act)
                    & _lr_match(twin["lr"], lrs_map[act])
                ]
                data.append(sub[metric].dropna().tolist())
            _boxplot(ax, data, pos, colors_b, edges_b)
            ax.set_xticks(pos)
            ax.set_xticklabels([_act_tick_label(a) for a in acts_order], fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel, fontsize=10)
            ax.set_ylim(*ylim)
            if "acc" in metric:
                ax.axhline(baseline, color=STYLE["grid"], linestyle="--",
                           linewidth=1.2,
                           label=f"Baseline trivial: {baseline:.1%}")
                ax.legend(fontsize=7)

        fig.suptitle(
            "Q2a  —  Métricas (Tanh vs Logística, mejores LR)\n"
            "ROC-AUC vs accuracy con desbalance de clases",
            fontsize=10,
        )
        _apply_style(fig, *axes)
        fig.tight_layout()
        _save(fig, "q2a_tanh_vs_logistica_metricas.png")


def plot_q2b_split_tanh_logistic(
    summary: pd.DataFrame,
    tanh_lr: float,
    logistic_lr: float,
) -> None:
    print("Q2b — generalización vs split (Tanh vs Logística) ...")
    split = _split_nl_tanh_logistic(summary, tanh_lr, logistic_lr)
    split = split[split["test_per"].notna()]
    if split.empty:
        print("  aviso: sin datos con test_per definido; se omite q2b_generalizacion_vs_split_tanh_logistica.")
        return

    models = [
        ("tanh", tanh_lr, COLORS_ACT["tanh"]),
        ("logistic", logistic_lr, COLORS_ACT["logistic"]),
    ]
    metrics = [("roc_auc", "ROC-AUC (test)"), ("best_f1", "F1 óptimo (test)")]

    tp_u = np.array(sorted(split["test_per"].unique()), dtype=float)
    x_lo, x_hi = float((tp_u * 100).min()), float((tp_u * 100).max())
    pad = max(1.0, (x_hi - x_lo) * 0.08)

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)

        for ax, (metric, ylabel) in zip(axes, metrics):
            for act, lr, color in models:
                sub = split[
                    (split["activation"].astype(str).str.lower() == act)
                    & _lr_match(split["lr"], lr)
                ]
                if sub.empty:
                    continue
                g = sub.groupby("test_per")[metric]
                mean = g.mean()
                std = g.std()
                xs = mean.index.to_numpy() * 100
                n_s = sub["seed"].nunique()
                ax.plot(xs, mean.values,
                        color=color, marker="o", markersize=6, linewidth=2,
                        label=f"{LABEL_ACT[act]}  lr={lr:g}, n={n_s}")
                ax.fill_between(xs,
                                mean.values - std.values,
                                mean.values + std.values,
                                color=color, alpha=0.12)

            ax.set_xlabel("Tamaño del conjunto de test  (%)")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel, fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xlim(x_lo - pad, x_hi + pad)

        fig.suptitle(
            "Q2b  —  Generalización vs tamaño del test (Tanh vs Logística, LR fijos)\n"
            "Mayor test_per implica menos datos de entrenamiento",
            fontsize=10,
        )
        _apply_style(fig, *axes)
        fig.tight_layout()
        _save(fig, "q2b_generalizacion_vs_split_tanh_logistica.png")


def plot_q2b_overfitting_tanh_logistic(
    summary: pd.DataFrame,
    tanh_lr: float,
    logistic_lr: float,
) -> None:
    print("Q2b — overfitting Tanh vs Logística ...")
    split = _split_nl_tanh_logistic(summary, tanh_lr, logistic_lr)
    split = split[split["test_per"].notna()]
    if split.empty:
        print("  aviso: sin datos con test_per; se omite q2b_overfitting_tanh_logistica.")
        return

    models = [
        ("tanh", tanh_lr, COLORS_ACT["tanh"],
         f"Tanh, lr={tanh_lr:g}"),
        ("logistic", logistic_lr, COLORS_ACT["logistic"],
         f"Logística, lr={logistic_lr:g}"),
    ]

    tp_u = np.array(sorted(split["test_per"].unique()), dtype=float)
    x_lo, x_hi = float((tp_u * 100).min()), float((tp_u * 100).max())
    pad = max(1.0, (x_hi - x_lo) * 0.08)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        for act, lr, color, base_lbl in models:
            sub = split[
                (split["activation"].astype(str).str.lower() == act)
                & _lr_match(split["lr"], lr)
            ]
            if sub.empty:
                continue
            n_s = sub["seed"].nunique()
            g_train = sub.groupby("test_per")["final_train_mse"]
            g_test = sub.groupby("test_per")["mse"]
            xs = g_train.mean().index.to_numpy() * 100
            m_tr = g_train.mean().values
            m_te = g_test.mean().values
            s_te = g_test.std().values

            ax.plot(xs, m_tr, color=color, linestyle="--",
                    linewidth=1.8, marker="^", markersize=5,
                    label=f"{base_lbl}, n={n_s} (train)")
            ax.plot(xs, m_te, color=color, linestyle="-",
                    linewidth=2.2, marker="o", markersize=6,
                    label=f"{base_lbl}, n={n_s} (test)")
            ax.fill_between(xs, m_te - s_te, m_te + s_te,
                            color=color, alpha=0.12)

        ax.set_xlabel("Tamaño del conjunto de test  (%)")
        ax.set_ylabel("MSE  (media sobre semillas)")
        ax.legend(fontsize=8)
        ax.set_xlim(x_lo - pad, x_hi + pad)

        fig.suptitle(
            "Q2b  —  MSE train vs test (Tanh vs Logística)\n"
            "Discontinua = train  |  Continua = test",
            fontsize=10,
        )
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "q2b_overfitting_tanh_logistica.png")


def plot_q2c_roc_tanh_logistic(
    roc_df: pd.DataFrame,
    summary: pd.DataFrame,
    tanh_lr: float,
    logistic_lr: float,
) -> None:
    print("Q2c — curva ROC Tanh vs Logística ...")
    split = _split_nl_tanh_logistic(summary, tanh_lr, logistic_lr)
    roc_f = _roc_nl_tanh_logistic(roc_df, tanh_lr, logistic_lr)
    if split.empty or roc_f.empty:
        print("  aviso: sin datos ROC/summary; se omite q2c_curva_roc_tanh_logistica.")
        return

    common_fpr = np.linspace(0, 1, 200)

    def _mean_roc(act: str, lr: float):
        sub = roc_f[
            (roc_f["activation"].astype(str).str.lower() == act)
            & _lr_match(roc_f["lr"], lr)
        ]
        curves = []
        for _, grp in sub.groupby(["seed", "test_per"]):
            g = grp.sort_values("fpr")
            curves.append(
                np.interp(common_fpr, g["fpr"].to_numpy(), g["tpr"].to_numpy())
            )
        if not curves:
            return None, None
        mat = np.array(curves)
        return mat.mean(axis=0), mat.std(axis=0)

    models = [
        ("tanh", tanh_lr, COLORS_ACT["tanh"]),
        ("logistic", logistic_lr, COLORS_ACT["logistic"]),
    ]

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1,
                label="Clasificador aleatorio  (AUC = 0.5)")

        for act, lr, color in models:
            mean_c, std_c = _mean_roc(act, lr)
            if mean_c is None:
                continue
            sub_s = split[
                (split["activation"].astype(str).str.lower() == act)
                & _lr_match(split["lr"], lr)
            ]
            auc_m = float(sub_s["roc_auc"].mean())
            n_s = sub_s["seed"].nunique()
            ax.plot(common_fpr, mean_c, color=color, linewidth=2.5,
                    label=f"{LABEL_ACT[act]}  lr={lr:g}, n={n_s}  (AUC={auc_m:.4f})")
            ax.fill_between(common_fpr, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.2)

        ax.set_xlabel("Tasa de falsos positivos (FPR)")
        ax.set_ylabel("Tasa de verdaderos positivos (TPR / Recall)")
        ax.set_title(
            "Q2c  —  ROC: Tanh vs Logística (promedio sobre semillas y particiones)",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "q2c_curva_roc_tanh_logistica.png")


def plot_q2c_umbral_tanh_logistic(
    summary: pd.DataFrame,
    roc_df: pd.DataFrame,
    tanh_lr: float,
    logistic_lr: float,
) -> None:
    print("Q2c — umbral Tanh vs Logística ...")
    split = _split_nl_tanh_logistic(summary, tanh_lr, logistic_lr)
    roc_f = _roc_nl_tanh_logistic(roc_df, tanh_lr, logistic_lr)
    split_t = split[split["test_per"].notna()]
    if split.empty or roc_f.empty:
        print("  aviso: sin datos; se omite q2c_umbral_tanh_logistica.")
        return

    common_thr = np.linspace(0, 1, 200)

    def _curves_for(act: str, lr: float):
        sub_roc = roc_f[
            (roc_f["activation"].astype(str).str.lower() == act)
            & _lr_match(roc_f["lr"], lr)
            & (roc_f["no_split"] == False)
        ]
        prec_list, rec_list, f1_list = [], [], []
        for _, grp in sub_roc.groupby(["seed", "test_per"]):
            g = grp.sort_values("threshold")
            p = np.interp(common_thr, g["threshold"].to_numpy(),
                          g["precision"].to_numpy(), left=0.0, right=0.0)
            r = np.interp(common_thr, g["threshold"].to_numpy(),
                          g["recall"].to_numpy(), left=1.0, right=0.0)
            denom = p + r
            f1 = np.divide(
                2 * p * r, denom,
                out=np.zeros_like(denom, dtype=float),
                where=denom > 0,
            )
            prec_list.append(p)
            rec_list.append(r)
            f1_list.append(f1)
        if not prec_list:
            return None, None, None, None
        mp = np.array(prec_list).mean(axis=0)
        mr = np.array(rec_list).mean(axis=0)
        mf = np.array(f1_list).mean(axis=0)
        best_thr = float(common_thr[int(np.argmax(mf))])
        return mp, mr, mf, best_thr

    models = [
        ("tanh", tanh_lr, COLORS_ACT["tanh"]),
        ("logistic", logistic_lr, COLORS_ACT["logistic"]),
    ]

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, 2, figsize=(FIG_SIZE[0] * 1.15, FIG_SIZE[1] * 1.25))

        for col, (act, lr, color) in enumerate(models):
            mp, mr, mf, best_thr = _curves_for(act, lr)
            ax = axes[0, col]
            if mp is not None:
                ax.plot(common_thr, mp, color="#2980b9", linewidth=2, label="Precision")
                ax.plot(common_thr, mr, color=BOXPLOT_STYLE["mean_color"],
                        linewidth=2, label="Recall")
                ax.plot(common_thr, mf,
                        color=BOXPLOT_STYLE["median_color"], linewidth=2.5,
                        label="F1")
                ax.axvline(best_thr, color=STYLE["text_title"],
                           linestyle="--", linewidth=1.8,
                           label=f"Umbral max-F1 = {best_thr:.3f}")
            ax.set_xlabel("Umbral de decisión")
            ax.set_ylabel("Métrica (media sobre semillas y particiones)")
            ax.set_title(f"{LABEL_ACT[act]}  (lr={lr:g})\nPrecision / Recall / F1 vs umbral")
            ax.legend(fontsize=7)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)

            ax2 = axes[1, col]
            best_runs = split_t[
                (split_t["activation"].astype(str).str.lower() == act)
                & _lr_match(split_t["lr"], lr)
            ]
            test_pers = sorted(best_runs["test_per"].dropna().unique())
            if test_pers:
                tp_positions = list(range(1, len(test_pers) + 1))
                data_thr = [
                    best_runs[best_runs["test_per"] == tp]["best_threshold"]
                    .dropna().tolist()
                    for tp in test_pers
                ]
                colors_f = [color] * len(test_pers)
                colors_e = [PAIRWISE_COLORS[1]["box_edge"]] * len(test_pers)
                _boxplot(ax2, data_thr, tp_positions, colors_f, colors_e)
                ax2.set_xticks(tp_positions)
                ax2.set_xticklabels([f"{tp * 100:.0f}%" for tp in test_pers], fontsize=9)
                om = best_runs["best_threshold"].mean()
                ax2.axhline(om, color=BOXPLOT_STYLE["median_color"], linestyle="--",
                            linewidth=1.8, label=f"Media global = {om:.3f}")
                ax2.legend(fontsize=7)
            ax2.set_xlabel("Tamaño del conjunto de test")
            ax2.set_ylabel("Umbral óptimo (max-F1)")
            ax2.set_title("Estabilidad del umbral vs partición")

        fig.suptitle(
            "Q2c  —  Umbral de detección (Tanh vs Logística, LR fijos)",
            fontsize=10,
        )
        _apply_style(fig, axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
        fig.tight_layout()
        _save(fig, "q2c_umbral_tanh_logistica.png")


# =============================================================================
# Entry point
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera gráficos del ejercicio 1 a partir de los CSV en results/."
    )
    p.add_argument(
        "--no-linear",
        action="store_true",
        help="Excluye corridas lineales de los gráficos (p. ej. si el runner se usó con --no-linear).",
    )
    p.add_argument(
        "--q2-tanh-lr",
        type=float,
        default=3e-5,
        metavar="LR",
        help="LR del modelo Tanh para figuras Q2 Tanh vs Logística (default: 3e-5).",
    )
    p.add_argument(
        "--q2-logistic-lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="LR de Logística para figuras Q2 Tanh vs Logística (default: 1e-4).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("Cargando resultados ...")
    summary = pd.read_csv(RESULTS / "linear_vs_nonlinear_summary.csv")

    print("Cargando curvas de aprendizaje (puede tardar unos segundos) ...")
    curves = pd.read_csv(RESULTS / "linear_vs_nonlinear_curves.csv")

    print("Cargando datos ROC ...")
    roc_df = pd.read_csv(RESULTS / "linear_vs_nonlinear_roc.csv")

    if args.no_linear:
        summary = summary[summary["model_type"] != "linear"].copy()
        curves = curves[curves["model_type"] != "linear"].copy()
        roc_df = roc_df[roc_df["model_type"] != "linear"].copy()
        print("Modo --no-linear: filas lineales excluidas de summary, curves y roc.")

    print(f"\nGenerando gráficos en: {PLOTS}\n")

    plot_q1ab_curvas(summary, curves)
    plot_q1ab_lr(summary)
    plot_q1c(summary)
    plot_q2a(summary)
    plot_q2b_split(summary)
    plot_q2b_overfitting(summary)
    plot_q2c_roc(roc_df, summary)
    plot_q2c_umbral(summary, roc_df)
    plot_q2c_confusion(summary)

    plot_q2a_tanh_logistic(summary, args.q2_tanh_lr, args.q2_logistic_lr)
    plot_q2b_split_tanh_logistic(summary, args.q2_tanh_lr, args.q2_logistic_lr)
    plot_q2b_overfitting_tanh_logistic(summary, args.q2_tanh_lr, args.q2_logistic_lr)
    plot_q2c_roc_tanh_logistic(roc_df, summary, args.q2_tanh_lr, args.q2_logistic_lr)
    plot_q2c_umbral_tanh_logistic(summary, roc_df, args.q2_tanh_lr, args.q2_logistic_lr)


if __name__ == "__main__":
    main()
