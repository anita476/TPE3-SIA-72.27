from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.style import FIG_DPI, FIG_SIZE, PLOT_RC, SAVE_PAD_INCHES, STYLE

DEFAULT_SUMMARY = ROOT / "results" / "linear_vs_nonlinear_summary.csv"
DEFAULT_OUTPUT = ROOT / "plots" / "ej1" / "comparacion_tanh_vs_logistica.png"
DEFAULT_OUTPUT_MSE = ROOT / "plots" / "ej1" / "comparacion_tanh_vs_logistica_mse.png"
DEFAULT_OUTPUT_FACTORES = ROOT / "plots" / "ej1" / "comparacion_tanh_vs_logistica_factores_roc.png"

COLORS_ACT = {
    "logistic": "#8e44ad",
    "tanh": "#27ae60",
}
LABEL_ACT = {"logistic": "Logística", "tanh": "Tanh"}
ACTIVATION_ORDER = ("logistic", "tanh")


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


def _normalize_activation(s: str) -> str:
    a = str(s).strip().lower()
    if a in ("logistics", "sigmoid"):
        return "logistic"
    return a


def _filter_test_per(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    m = mode.strip().lower()
    if m == "all":
        return df
    if m in ("null", "none", "nan"):
        return df[df["test_per"].isna()]
    val = float(mode)
    return df[np.isclose(df["test_per"].astype(float), val, rtol=0, atol=1e-9)]


def _prepare_frame(summary: pd.DataFrame, test_per_mode: str) -> pd.DataFrame:
    df = summary.copy()
    if "model_type" not in df.columns or "activation" not in df.columns:
        raise SystemExit("El CSV no tiene columnas model_type / activation.")

    df["activation"] = df["activation"].map(_normalize_activation)
    df = df[df["model_type"].astype(str) == "non-linear"]
    df = df[df["activation"].isin(["tanh", "logistic"])]

    if "no_split" not in df.columns:
        raise SystemExit("Falta la columna no_split en el CSV.")
    ns = df["no_split"].map(lambda x: str(x).strip().lower() in ("false", "0"))
    df = df[ns.astype(bool)]

    df = _filter_test_per(df, test_per_mode)

    for col in ("lr", "seed", "roc_auc", "best_f1", "train_roc_auc", "final_train_mse"):
        if col not in df.columns:
            raise SystemExit(f"Falta la columna requerida: {col}")

    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (activation, lr) con mean y std sobre todas las filas restantes."""
    g = df.groupby(["activation", "lr"], sort=False)
    agg_kw: dict = {
        "roc_auc_mean": ("roc_auc", "mean"),
        "roc_auc_std": ("roc_auc", "std"),
        "best_f1_mean": ("best_f1", "mean"),
        "best_f1_std": ("best_f1", "std"),
        "train_roc_auc_mean": ("train_roc_auc", "mean"),
        "train_roc_auc_std": ("train_roc_auc", "std"),
        "final_train_mse_mean": ("final_train_mse", "mean"),
        "final_train_mse_std": ("final_train_mse", "std"),
    }
    if "recall_at_threshold" in df.columns:
        agg_kw["recall_at_threshold_mean"] = ("recall_at_threshold", "mean")
        agg_kw["recall_at_threshold_std"] = ("recall_at_threshold", "std")
    if "fpr_at_threshold" in df.columns:
        agg_kw["fpr_at_threshold_mean"] = ("fpr_at_threshold", "mean")
        agg_kw["fpr_at_threshold_std"] = ("fpr_at_threshold", "std")
    out = g.agg(**agg_kw).reset_index()
    for c in list(out.columns):
        if c.endswith("_std"):
            out[c] = out[c].fillna(0.0)
    return out


def _curve_for_activation(agg: pd.DataFrame, act: str) -> pd.DataFrame:
    sub = agg[agg["activation"] == act].sort_values("lr")
    return sub


def _best_lr_for_activation(
    agg: pd.DataFrame,
    activation: str,
    mean_col: str,
    *,
    minimize: bool = False,
) -> tuple[float, float] | None:
    """Extremo de ``mean_col`` por activación: max (default) o min (``minimize=True``). Empates → menor LR."""
    sub = agg[agg["activation"] == activation].dropna(subset=[mean_col])
    if sub.empty:
        return None
    if minimize:
        mx = float(sub[mean_col].min())
        tied = sub[np.isclose(sub[mean_col], mx, rtol=1e-12, atol=1e-15)]
    else:
        mx = float(sub[mean_col].max())
        tied = sub[np.isclose(sub[mean_col], mx, rtol=0, atol=1e-9)]
    row = tied.sort_values("lr").iloc[0]
    return float(row["lr"]), float(row[mean_col])


def _report_best_lrs_roc_test(agg: pd.DataFrame) -> str:
    """Mejor LR por activación según ROC-AUC en test (solo consola)."""
    lines: list[str] = ["--- Mejor LR por activación (ROC-AUC en test, media) ---"]

    for act in ("tanh", "logistic"):
        got = _best_lr_for_activation(agg, act, "roc_auc_mean", minimize=False)
        lab = LABEL_ACT[act]
        if got is None:
            lines.append(f"  {lab}: (sin datos)")
            continue
        lr_b, v = got
        lines.append(f"  {lab}:  lr = {lr_b:g}    (AUC medio ~ {v:.6f})")

    return "\n".join(lines).strip()


def _report_best_lrs_final_train_mse(agg: pd.DataFrame) -> str:
    """Mejor LR por activación según MSE final de entrenamiento (menor es mejor)."""
    lines: list[str] = ["--- Mejor LR por activación (MSE final train, media; menor es mejor) ---"]

    for act in ("tanh", "logistic"):
        got = _best_lr_for_activation(agg, act, "final_train_mse_mean", minimize=True)
        lab = LABEL_ACT[act]
        if got is None:
            lines.append(f"  {lab}: (sin datos)")
            continue
        lr_b, v = got
        lines.append(f"  {lab}:  lr = {lr_b:g}    (MSE medio ~ {v:.6f})")

    return "\n".join(lines).strip()


def _plot_panel(
    ax: plt.Axes,
    agg: pd.DataFrame,
    mean_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    log_y: bool,
    *,
    show_xlabel: bool = True,
) -> None:
    for act in ACTIVATION_ORDER:
        sub = _curve_for_activation(agg, act)
        if sub.empty:
            continue
        xs = sub["lr"].to_numpy(dtype=float)
        m = sub[mean_col].to_numpy(dtype=float)
        s = sub[std_col].to_numpy(dtype=float)
        color = COLORS_ACT[act]
        label = LABEL_ACT[act]
        ax.semilogx(xs, m, color=color, marker="o", markersize=5, linewidth=2, label=label)
        lo = np.clip(m - s, 1e-8 if log_y else -np.inf, None)
        hi = m + s
        if log_y:
            lo = np.maximum(lo, 1e-8)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)
    if show_xlabel:
        ax.set_xlabel("Learning rate (escala log)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if log_y:
        ax.set_yscale("log")


def _mark_optima_on_panel(
    ax: plt.Axes,
    agg: pd.DataFrame,
    mean_col: str,
    *,
    minimize: bool,
) -> None:
    """Marca con estrella y etiqueta el mejor LR de cada activación para la métrica de ese panel."""
    offsets = {
        "logistic": (12, 14),
        "tanh":     (12, -22),
    }
    for act in ACTIVATION_ORDER:
        got = _best_lr_for_activation(agg, act, mean_col, minimize=minimize)
        if got is None:
            continue
        lr_b, y_b = got
        color = COLORS_ACT[act]
        ax.scatter(
            [lr_b],
            [y_b],
            s=200,
            marker="*",
            c=color,
            edgecolors="#2c3e50",
            linewidths=0.9,
            zorder=8,
        )
        ox, oy = offsets[act]
        ax.annotate(
            f"{LABEL_ACT[act]}\nlr={lr_b:g}",
            xy=(lr_b, y_b),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.92),
            zorder=9,
        )


def _plot_factores_roc_figure(
    agg: pd.DataFrame,
    df: pd.DataFrame,
    out_path: Path,
    threshold_val: float | None,
) -> None:
    """2×2: AUC test, AUC train, recall y FPR en test (umbral fijo del experimento)."""
    need = ("recall_at_threshold_mean", "fpr_at_threshold_mean")
    if not all(c in agg.columns for c in need):
        return

    thr_txt = f"{threshold_val:g}" if threshold_val is not None else "config"

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, 2, figsize=(FIG_SIZE[0] * 1.05, FIG_SIZE[1] * 1.15))
        ax00, ax01 = axes[0]
        ax10, ax11 = axes[1]

        _plot_panel(
            ax00, agg, "roc_auc_mean", "roc_auc_std",
            "ROC-AUC (test)",
            "ROC-AUC en test",
            log_y=False,
        )
        _mark_optima_on_panel(ax00, agg, "roc_auc_mean", minimize=False)

        _plot_panel(
            ax01, agg, "train_roc_auc_mean", "train_roc_auc_std",
            "ROC-AUC (train)",
            "ROC-AUC en entrenamiento",
            log_y=False,
        )
        _mark_optima_on_panel(ax01, agg, "train_roc_auc_mean", minimize=False)

        _plot_panel(
            ax10, agg, "recall_at_threshold_mean", "recall_at_threshold_std",
            "Recall / TPR (test)",
            f"Recall en test  (umbral = {thr_txt})",
            log_y=False,
        )
        _mark_optima_on_panel(ax10, agg, "recall_at_threshold_mean", minimize=False)

        _plot_panel(
            ax11, agg, "fpr_at_threshold_mean", "fpr_at_threshold_std",
            "FPR (test)",
            f"Tasa de falsos positivos en test  (umbral = {thr_txt})",
            log_y=False,
        )
        _mark_optima_on_panel(ax11, agg, "fpr_at_threshold_mean", minimize=True)

        for ax in (ax00, ax01, ax10, ax11):
            ax.legend(fontsize=9, loc="best")
            ax.set_xlim(left=df["lr"].min() * 0.85, right=df["lr"].max() * 1.15)

        _apply_style(fig, ax00, ax01, ax10, ax11)
        fig.tight_layout(rect=[0, 0.03, 1, 0.99])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=FIG_DPI,
            bbox_inches="tight",
            pad_inches=SAVE_PAD_INCHES,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        print(f"Guardado (factores ROC): {out_path}")


def _print_factores_best_lrs(agg: pd.DataFrame) -> None:
    if "recall_at_threshold_mean" not in agg.columns:
        return
    print("--- Mejor LR (factores ROC / umbral fijo) ---")
    for title, col, minimize in (
        ("ROC-AUC test", "roc_auc_mean", False),
        ("ROC-AUC train", "train_roc_auc_mean", False),
        ("Recall test @ umbral", "recall_at_threshold_mean", False),
        ("FPR test @ umbral (menor es mejor)", "fpr_at_threshold_mean", True),
    ):
        print(f"  [{title}]")
        for act in ("tanh", "logistic"):
            got = _best_lr_for_activation(agg, act, col, minimize=minimize)
            lab = LABEL_ACT[act]
            if got is None:
                print(f"    {lab}: —")
            else:
                lr_b, v = got
                print(f"    {lab}: lr = {lr_b:g}  (media ~ {v:.6f})")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Figuras Tanh vs Logística: ROC-AUC en test (PNG principal), "
            "MSE final train en otro PNG con escala lineal, y opcionalmente factores ROC."
        )
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"CSV de resumen (default: {DEFAULT_SUMMARY})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PNG de ROC-AUC en test (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--output-mse",
        type=Path,
        dest="output_mse",
        default=DEFAULT_OUTPUT_MSE,
        help=f"PNG de MSE final train, escala lineal (default: {DEFAULT_OUTPUT_MSE})",
    )
    p.add_argument(
        "--test-per",
        type=str,
        default="0.2",
        metavar="VAL",
        help='Partición test: "0.2", "null" (dataset completo en CSV), o "all" (promediar todos los test_per).',
    )
    p.add_argument(
        "--output-factores",
        type=Path,
        default=DEFAULT_OUTPUT_FACTORES,
        help=(
            "Segunda figura: AUC test + AUC train + recall + FPR @ umbral "
            f"(default: {DEFAULT_OUTPUT_FACTORES})."
        ),
    )
    p.add_argument(
        "--skip-factores",
        action="store_true",
        help="No generar la figura de factores ROC (aunque el CSV tenga las columnas).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary.resolve()
    if not summary_path.is_file():
        raise SystemExit(f"No se encontró el summary: {summary_path}")

    summary = pd.read_csv(summary_path)
    df = _prepare_frame(summary, args.test_per)

    if df.empty:
        raise SystemExit(
            "No quedaron filas tras filtrar (non-linear, tanh/logistic, no_split=False, test_per)."
        )

    present = set(df["activation"].unique())
    for need in ("tanh", "logistic"):
        if need not in present:
            raise SystemExit(
                f"Falta la activación {need!r} en los datos filtrados. Presentes: {sorted(present)}"
            )

    if df["lr"].nunique() == 0:
        raise SystemExit("No hay valores de lr en el subconjunto filtrado.")

    agg = _aggregate(df)

    print(_report_best_lrs_roc_test(agg))
    print()
    print(_report_best_lrs_final_train_mse(agg))
    print()

    lr_xlim = (df["lr"].min() * 0.85, df["lr"].max() * 1.15)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(1, 1, figsize=(FIG_SIZE[0], FIG_SIZE[1] * 0.82))

        _plot_panel(
            ax, agg, "roc_auc_mean", "roc_auc_std",
            "ROC-AUC (test)",
            "ROC-AUC en test",
            log_y=False,
        )
        _mark_optima_on_panel(ax, agg, "roc_auc_mean", minimize=False)

        ax.legend(fontsize=10, loc="best")
        ax.set_xlim(left=lr_xlim[0], right=lr_xlim[1])

        _apply_style(fig, ax)
        fig.tight_layout(rect=[0, 0.03, 1, 0.99])

        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out,
            dpi=FIG_DPI,
            bbox_inches="tight",
            pad_inches=SAVE_PAD_INCHES,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        print(f"Guardado (ROC-AUC): {out}")

    with plt.rc_context(PLOT_RC):
        fig_m, ax_m = plt.subplots(1, 1, figsize=(FIG_SIZE[0], FIG_SIZE[1] * 0.82))

        _plot_panel(
            ax_m, agg, "final_train_mse_mean", "final_train_mse_std",
            "MSE final (train)",
            "MSE final de entrenamiento",
            log_y=False,
        )
        _mark_optima_on_panel(ax_m, agg, "final_train_mse_mean", minimize=True)

        ax_m.legend(fontsize=10, loc="best")
        ax_m.set_xlim(left=lr_xlim[0], right=lr_xlim[1])

        _apply_style(fig_m, ax_m)
        fig_m.tight_layout(rect=[0, 0.03, 1, 0.99])

        out_mse = args.output_mse.resolve()
        out_mse.parent.mkdir(parents=True, exist_ok=True)
        fig_m.savefig(
            out_mse,
            dpi=FIG_DPI,
            bbox_inches="tight",
            pad_inches=SAVE_PAD_INCHES,
            facecolor=fig_m.get_facecolor(),
        )
        plt.close(fig_m)
        print(f"Guardado (MSE train): {out_mse}")

    thr_val: float | None = None
    if "threshold" in df.columns and df["threshold"].notna().any():
        thr_val = float(df["threshold"].dropna().iloc[0])

    if args.skip_factores:
        return
    if "fpr_at_threshold" not in df.columns or "recall_at_threshold" not in df.columns:
        print(
            "Aviso: el CSV no tiene columnas `fpr_at_threshold` / `recall_at_threshold`. "
            "Volvé a correr el experiment_runner con el código actualizado y regenerá el summary."
        )
        return

    _print_factores_best_lrs(agg)
    _plot_factores_roc_figure(
        agg, df, args.output_factores.resolve(), thr_val,
    )


if __name__ == "__main__":
    main()
