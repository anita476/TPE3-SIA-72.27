from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    from matplotlib.lines import Line2D
except ImportError as e:
    raise SystemExit(
        "Requires matplotlib and numpy. Install with: pip install matplotlib numpy pandas"
    ) from e

try:
    from scipy import stats as _scipy_stats

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
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

REQUIRED_METRIC_COLUMNS = ("test_acc",)
OPTIONAL_METRIC_COLUMNS = (
    "train_acc",
    "mae",
    "mse",
    "final_train_mse",
    "epochs_completed",
    "elapsed_seconds",
    # dataset stats
    "fraud_rate_train",
    "fraud_rate_test",
    # binary fraud metrics — test set
    "roc_auc",
    "best_f1",
    "best_precision",
    "best_recall",
    "best_threshold",
    "f1_at_half",
    "precision_at_half",
    "recall_at_half",
    # binary fraud metrics — train set (for overfitting diagnosis)
    "train_roc_auc",
    "train_f1_at_half",
    "train_precision_at_half",
    "train_recall_at_half",
)
DERIVED_METRIC_COLUMNS = ("rmse", "test_acc_per_second")
INTEGER_METRICS = frozenset({"epochs_completed", "confusion_bins_config", "seed"})

METRIC_YLABEL: dict[str, str] = {
    "test_acc":               "Test accuracy (tolerance)",
    "train_acc":              "Train accuracy (tolerance)",
    "mae":                    "Mean absolute error (test)",
    "mse":                    "Mean squared error (test)",
    "rmse":                   "Root mean squared error (test)",
    "final_train_mse":        "Final MSE (train)",
    "epochs_completed":       "Epochs completed",
    "elapsed_seconds":        "Training time (s)",
    "test_acc_per_second":    "Test accuracy per second (1/s)",
    # dataset stats
    "fraud_rate_train":       "Fraud rate (train set)",
    "fraud_rate_test":        "Fraud rate (test set)",
    # binary metrics — test
    "roc_auc":                "ROC-AUC (test)",
    "best_f1":                "Best F1 (test, optimal threshold)",
    "best_precision":         "Precision at optimal threshold (test)",
    "best_recall":            "Recall at optimal threshold (test)",
    "best_threshold":         "Optimal threshold (max F1)",
    "f1_at_half":             "F1 at threshold=0.5 (test)",
    "precision_at_half":      "Precision at threshold=0.5 (test)",
    "recall_at_half":         "Recall at threshold=0.5 (test)",
    # binary metrics — train (overfitting diagnosis)
    "train_roc_auc":          "ROC-AUC (train)",
    "train_f1_at_half":       "F1 at threshold=0.5 (train)",
    "train_precision_at_half": "Precision at threshold=0.5 (train)",
    "train_recall_at_half":   "Recall at threshold=0.5 (train)",
}

PARAM_LABEL: dict[str, str] = {
    "name":                  "Experiment",
    "data":                  "Dataset",
    "model_type":            "Model type",
    "lr":                    "Learning rate",
    "epochs":                "Epochs (max)",
    "epsilon":               "Epsilon (stop threshold)",
    "tolerance":             "Tolerance",
    "activation":            "Activation",
    "beta":                  "Beta",
    "test_per":              "Test fraction",
    "normalize":             "Normalisation",
    "no_split":              "No split (full dataset)",
    "seed":                  "Seed",
    "confusion_mode":        "Confusion matrix mode",
    "confusion_bins_config": "Number of bins",
    "run_id":                "Run ID",
}


def metric_ylabel(metric: str) -> str:
    return METRIC_YLABEL.get(metric, metric.replace("_", " "))


def param_xlabel(param: str) -> str:
    return PARAM_LABEL.get(param, param.replace("_", " "))


def _safe_filename_part(s: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.UNICODE)
    return s.strip("_") or "out"


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def parse_float_cell(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    return _try_float(raw)


def parse_int_cell(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def sort_category_labels(labels: list[str]) -> list[str]:
    if not labels:
        return labels
    if all(_try_float(s) is not None for s in labels):
        return sorted(labels, key=lambda s: float(s))
    return sorted(labels)


def unique_non_null(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v is None or v == "":
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def load_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("CSV has no header row.")
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    return rows, fieldnames


def validate_columns(fieldnames: list[str]) -> None:
    missing = [c for c in REQUIRED_METRIC_COLUMNS if c not in fieldnames]
    if missing:
        raise SystemExit(
            f"Missing required columns: {missing}. Found: {fieldnames}"
        )


def apply_filters(
    rows: list[dict[str, str]],
    filters: dict[str, list[str]],
) -> list[dict[str, str]]:
    if not filters:
        return rows
    return [
        r
        for r in rows
        if all(r.get(k, "") in allowed for k, allowed in filters.items())
    ]


def compute_derived_metrics(
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    new_fields = list(fieldnames)
    for col in ("rmse", "test_acc_per_second"):
        if col not in new_fields:
            new_fields.append(col)
    updated: list[dict[str, str]] = []
    for row in rows:
        r = dict(row)
        mse = parse_float_cell(r.get("mse"))
        if mse is not None and mse >= 0:
            r["rmse"] = str(math.sqrt(mse))
        else:
            r["rmse"] = ""
        elapsed = parse_float_cell(r.get("elapsed_seconds"))
        ta = parse_float_cell(r.get("test_acc"))
        if elapsed is not None and elapsed > 0 and ta is not None:
            r["test_acc_per_second"] = str(ta / elapsed)
        else:
            r["test_acc_per_second"] = ""
        updated.append(r)
    return updated, new_fields


def resolve_active_metrics(
    requested: list[str] | None,
    fieldnames: list[str],
) -> list[str]:
    all_metrics = list(REQUIRED_METRIC_COLUMNS)
    for m in OPTIONAL_METRIC_COLUMNS:
        if m in fieldnames:
            all_metrics.append(m)
    for m in DERIVED_METRIC_COLUMNS:
        if m in fieldnames:
            all_metrics.append(m)
    if not requested:
        return all_metrics
    unknown = [m for m in requested if m not in all_metrics]
    if unknown:
        raise SystemExit(f"Unknown --y-axis: {unknown}. Available: {all_metrics}")
    return [m for m in all_metrics if m in requested]


SKIP_DEFAULT_PARAMS = frozenset({"run_id", "confusion_meta_json"})


def all_param_columns(fieldnames: list[str]) -> list[str]:
    metric_cols = (
        set(REQUIRED_METRIC_COLUMNS)
        | set(OPTIONAL_METRIC_COLUMNS)
        | set(DERIVED_METRIC_COLUMNS)
    )
    skip = metric_cols | {"confusion_meta_json"}
    return [c for c in fieldnames if c not in skip]


def param_columns_default(fieldnames: list[str]) -> list[str]:
    """Exclude run_id from the default X-axis columns (one category per run)."""
    return [c for c in all_param_columns(fieldnames) if c not in SKIP_DEFAULT_PARAMS]


def resolve_x_axis_columns(
    requested: list[str] | None,
    fieldnames: list[str],
) -> list[str]:
    all_params = all_param_columns(fieldnames)
    if not requested:
        return param_columns_default(fieldnames)
    unknown = [c for c in requested if c not in all_params]
    if unknown:
        raise SystemExit(f"Unknown --x-axis: {unknown}. Plottable: {all_params}")
    bad = [
        c
        for c in requested
        if c
        in set(REQUIRED_METRIC_COLUMNS)
        | set(OPTIONAL_METRIC_COLUMNS)
        | set(DERIVED_METRIC_COLUMNS)
    ]
    if bad:
        raise SystemExit(f"--x-axis cannot be a metric column: {bad}")
    return list(requested)


def remove_outliers_iqr(values: list[float], factor: float = 1.5) -> list[float]:
    """Tukey fences per category; keeps original list if too few points or zero IQR."""
    if len(values) <= 2:
        return list(values)
    arr = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = float(q3 - q1)
    if iqr <= 0:
        return list(values)
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    out = [float(x) for x in arr if lo <= x <= hi]
    if not out and len(values) >= 1:
        return [float(np.median(arr))]
    return out


def filter_plot_groups_iqr(
    data: list[list[float]],
    *,
    factor: float,
) -> tuple[list[list[float]], int]:
    """Apply IQR filtering independently to each group's values."""
    new_data: list[list[float]] = []
    dropped = 0
    for vals in data:
        before = len(vals)
        filt = remove_outliers_iqr(vals, factor)
        dropped += before - len(filt)
        new_data.append(filt)
    return new_data, dropped


def collect_groups(
    rows: list[dict[str, str]],
    param: str,
    metric: str,
) -> tuple[list[list[float]], list[str]]:
    by_val: dict[str, list[float]] = {}
    for r in rows:
        key = r.get(param)
        if key is None or key == "":
            continue
        raw = r.get(metric)
        if metric in INTEGER_METRICS:
            v = parse_int_cell(raw)
            fv = float(v) if v is not None else None
        else:
            fv = parse_float_cell(raw)
        if fv is None:
            continue
        by_val.setdefault(key, []).append(fv)

    labels = sort_category_labels(list(by_val.keys()))
    data = [by_val[lab] for lab in labels]
    return data, labels


class GroupStats(NamedTuple):
    mean: float
    std: float
    ci_lo: float
    ci_hi: float
    n: int


def compute_stats(values: list[float], ci_level: float = 0.95) -> GroupStats | None:
    if not values:
        return None
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n > 1 and _SCIPY_AVAILABLE:
        lo, hi = _scipy_stats.t.interval(ci_level, df=n - 1, loc=mean, scale=std / math.sqrt(n))
        lo, hi = float(lo), float(hi)
    elif n > 1:
        z = 1.96 if abs(ci_level - 0.95) < 1e-6 else 2.576
        margin = z * std / math.sqrt(n)
        lo, hi = mean - margin, mean + margin
    else:
        lo, hi = mean, mean
    return GroupStats(mean=mean, std=std, ci_lo=lo, ci_hi=hi, n=n)


def _setup_axes_style(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])
    ax.grid(
        axis="y",
        which="major",
        linestyle="-",
        linewidth=0.6,
        alpha=0.55,
        color=STYLE["grid"],
        zorder=0,
    )
    ax.minorticks_on()
    ax.grid(
        axis="y",
        which="minor",
        linestyle=":",
        linewidth=0.4,
        alpha=0.45,
        color=STYLE["grid_minor"],
        zorder=0,
    )
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="both", colors=STYLE["text_axis"])
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(STYLE["text_axis"])


def _apply_boxplot_style(
    bp: dict,
    face_color: str = BOXPLOT_STYLE["box_face"],
    edge_color: str = BOXPLOT_STYLE["box_edge"],
) -> None:
    for box in bp["boxes"]:
        box.set_facecolor(face_color)
        box.set_edgecolor(edge_color)
        box.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color(BOXPLOT_STYLE["median_color"])
        median.set_linewidth(2.0)
    for whisker in bp["whiskers"]:
        whisker.set_color(BOXPLOT_STYLE["whisker_color"])
        whisker.set_linewidth(0.9)
    for cap in bp["caps"]:
        cap.set_color(BOXPLOT_STYLE["cap_color"])
        cap.set_linewidth(0.9)


def _add_jitter(
    ax: plt.Axes,
    positions: list[int],
    data: list[list[float]],
    rng: np.random.Generator,
) -> None:
    for i, vals in enumerate(data):
        if not vals:
            continue
        ax.scatter(
            np.full(len(vals), positions[i]),
            vals,
            s=20,
            color=BOXPLOT_STYLE["point_color"],
            edgecolors=BOXPLOT_STYLE["point_edge"],
            linewidths=0.5,
            alpha=0.85,
            zorder=3,
        )


def _add_mean_markers(
    ax: plt.Axes,
    positions: list[int],
    data: list[list[float]],
) -> None:
    for i, vals in enumerate(data):
        if not vals:
            continue
        mean_val = float(np.mean(vals))
        ax.plot(
            positions[i],
            mean_val,
            marker="D",
            markersize=7,
            color=BOXPLOT_STYLE["mean_color"],
            markeredgecolor="#1a7a44",
            markeredgewidth=0.8,
            zorder=5,
            label="_nolegend_",
        )


def _add_stats_annotations(
    ax: plt.Axes,
    positions: list[int],
    data: list[list[float]],
    ci_level: float,
    y_min: float,
    y_range: float,
) -> None:
    for i, vals in enumerate(data):
        if not vals:
            continue
        s = compute_stats(vals, ci_level)
        if s is None:
            continue
        ci_pct = int(round(ci_level * 100))
        q3 = float(np.percentile(vals, 75))
        iqr = q3 - float(np.percentile(vals, 25))
        cap = min(max(vals), q3 + 1.5 * iqr)
        text_y = cap + y_range * 0.03
        text = (
            f"n={s.n}  media={s.mean:.4g}\n"
            f"desv.={s.std:.4g}  IC{ci_pct}%:[{s.ci_lo:.4g},{s.ci_hi:.4g}]"
        )
        ax.text(
            positions[i],
            text_y,
            text,
            ha="center",
            va="bottom",
            fontsize=6,
            color=STYLE["stats_text"],
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.75,
                edgecolor=STYLE["grid"],
                linewidth=0.5,
            ),
            zorder=6,
        )


def _set_y_formatter(ax: plt.Axes, metric: str) -> None:
    if metric in INTEGER_METRICS:
        ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=False))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.6g}"))


def plot_all_methods(
    data_per_config: list[list[float]],
    labels: list[str],
    title: str,
    ylabel: str,
    metric: str,
    xlabel: str | None = None,
    show_mean: bool = False,
    add_stats: bool = False,
    ci_level: float = 0.95,
) -> plt.Figure:
    n = len(labels)
    fig_w = max(11.0, 0.9 * n + 3.0)
    fig_h = FIG_SIZE[1]

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=FIG_DPI)
    _setup_axes_style(fig, ax)

    if n == 0 or all(len(d) == 0 for d in data_per_config):
        ax.text(
            0.5,
            0.5,
            "Sin datos",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=STYLE["text_axis"],
        )
        ax.set_title(title, fontsize=13, fontweight="600", color=STYLE["text_title"], pad=14)
        if xlabel:
            ax.set_xlabel(xlabel, color=STYLE["text_axis"])
        fig.subplots_adjust(left=0.09, right=0.9, top=0.88, bottom=0.14)
        return fig

    positions = list(range(1, n + 1))
    bp = ax.boxplot(
        data_per_config,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        zorder=2,
    )
    _apply_boxplot_style(bp)

    rng = np.random.default_rng(42)
    _add_jitter(ax, positions, data_per_config, rng)

    if show_mean:
        _add_mean_markers(ax, positions, data_per_config)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    if xlabel:
        ax.set_xlabel(xlabel, color=STYLE["text_axis"])
    ax.set_ylabel(ylabel, color=STYLE["text_axis"])
    _set_y_formatter(ax, metric)

    all_vals = [v for d in data_per_config for v in d]
    y_min = min(all_vals) if all_vals else 0.0
    y_max = max(all_vals) if all_vals else 1.0
    y_range = y_max - y_min or 1.0

    if add_stats:
        _add_stats_annotations(ax, positions, data_per_config, ci_level, y_min, y_range)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BOXPLOT_STYLE["median_color"],
            linewidth=1.8,
            label="Mediana",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=BOXPLOT_STYLE["point_color"],
            markersize=6,
            label="Observaciones",
        ),
    ]
    if show_mean:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor=BOXPLOT_STYLE["mean_color"],
                markersize=7,
                label="Media",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.8,
        edgecolor=STYLE["grid"],
    )

    ax.set_title(title, fontsize=13, fontweight="600", color=STYLE["text_title"], pad=14)

    fig.subplots_adjust(left=0.09, right=0.93, top=0.88, bottom=0.22)
    return fig


def plot_pairwise_comparison(
    data_a: list[float],
    data_b: list[float],
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    metric: str,
    xlabel: str | None = None,
    show_mean: bool = False,
    add_stats: bool = False,
    ci_level: float = 0.95,
) -> plt.Figure:
    fig_h = FIG_SIZE[1]

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(5.5, fig_h), dpi=FIG_DPI)
    _setup_axes_style(fig, ax)

    positions = [1, 1.6]
    data = [data_a, data_b]
    colors = PAIRWISE_COLORS

    for idx, (vals, pos, clr) in enumerate(zip(data, positions, colors, strict=True)):
        if not vals:
            continue
        bp = ax.boxplot(
            [vals],
            positions=[pos],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            zorder=2,
        )
        _apply_boxplot_style(bp, face_color=clr["box_face"], edge_color=clr["box_edge"])

    rng = np.random.default_rng(42)
    point_colors = [
        (PAIRWISE_COLORS[0]["box_face"], PAIRWISE_COLORS[0]["box_edge"]),
        (PAIRWISE_COLORS[1]["box_face"], PAIRWISE_COLORS[1]["box_edge"]),
    ]
    for i, (vals, pos) in enumerate(zip(data, positions, strict=True)):
        if not vals:
            continue
        ax.scatter(
            np.full(len(vals), pos),
            vals,
            s=20,
            color=point_colors[i][0],
            edgecolors=point_colors[i][1],
            linewidths=0.5,
            alpha=0.80,
            zorder=3,
        )

    if show_mean:
        _add_mean_markers(ax, positions, data)

    ax.set_xticks(positions)
    ax.set_xticklabels([label_a, label_b], rotation=20, ha="right")
    ax.set_xlim(min(positions) - 0.5, max(positions) + 0.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=STYLE["text_axis"])
    ax.set_ylabel(ylabel, color=STYLE["text_axis"])
    _set_y_formatter(ax, metric)

    all_vals = [v for d in data for v in d]
    y_min = min(all_vals) if all_vals else 0.0
    y_max = max(all_vals) if all_vals else 1.0
    y_range_val = y_max - y_min or 1.0

    if add_stats:
        _add_stats_annotations(ax, positions, data, ci_level, y_min, y_range_val)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BOXPLOT_STYLE["median_color"],
            linewidth=1.8,
            label="Mediana",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PAIRWISE_COLORS[0]["box_face"],
            markersize=7,
            label=label_a,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PAIRWISE_COLORS[1]["box_face"],
            markersize=7,
            label=label_b,
        ),
    ]
    if show_mean:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor=BOXPLOT_STYLE["mean_color"],
                markersize=7,
                label="Media",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.8,
        edgecolor=STYLE["grid"],
    )

    ax.set_title(title, fontsize=12, fontweight="600", color=STYLE["text_title"], pad=14)

    fig.subplots_adjust(left=0.11, right=0.93, top=0.88, bottom=0.22)
    return fig


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
        facecolor=fig.get_facecolor(),
        format=path.suffix.lstrip(".").lower() or "png",
    )


def output_basename(
    metric: str,
    param: str,
    label_a: str | None = None,
    label_b: str | None = None,
) -> str:
    core = f"linear_vs_nonlinear_boxplot_{metric}_by_{param}"
    if label_a is not None and label_b is not None:
        core += f"__pair_{_safe_filename_part(label_a)}_vs_{_safe_filename_part(label_b)}"
    return f"{_safe_filename_part(core)}.png"


def build_title_linear_vs_nonlinear(
    param: str,
    metric: str,
    label_a: str | None = None,
    label_b: str | None = None,
    n_labels: int = 0,
) -> str:
    ylabel = metric_ylabel(metric)
    if n_labels == 1 and label_a is not None:
        body = f"{label_a} — {ylabel}"
    elif label_a is not None and label_b is not None:
        body = f"{label_a} vs. {label_b} — {ylabel}"
    else:
        body = f"By {param_xlabel(param).lower()} — {ylabel}"
    return f"Linear vs non-linear: {body}"


def run_compare_from_summary(
    csv_path: Path,
    out_dir: Path,
    filters: dict[str, list[str]] | None = None,
    x_axis_columns: list[str] | None = None,
    y_axis_metrics: list[str] | None = None,
    pairwise: bool = False,
    show_mean: bool = False,
    add_stats: bool = False,
    ci_level: float = 0.95,
    *,
    drop_outliers: str = "none",
    outlier_iqr_factor: float = 1.5,
) -> list[Path]:
    rows, fieldnames = load_rows(csv_path)
    validate_columns(fieldnames)

    if filters:
        rows = apply_filters(rows, filters)
        if not rows:
            print(f"Warning: no rows left after filters {filters}")
            return []

    rows, fieldnames = compute_derived_metrics(rows, fieldnames)
    params = resolve_x_axis_columns(x_axis_columns, fieldnames)
    metrics = resolve_active_metrics(y_axis_metrics, fieldnames)

    if not rows:
        print("Warning: CSV has no data rows.")
        return []

    written: list[Path] = []
    plt.ioff()

    for param in params:
        labels_all = sort_category_labels(unique_non_null([r.get(param) for r in rows]))
        if len(labels_all) < 1:
            continue

        for metric in metrics:
            data, labels = collect_groups(rows, param, metric)
            if len(labels) < 1 or all(len(d) == 0 for d in data):
                continue

            dropped_points = 0
            if drop_outliers == "iqr":
                data, dropped_points = filter_plot_groups_iqr(
                    data, factor=outlier_iqr_factor
                )
                if dropped_points:
                    print(
                        f"  [{metric} × {param}] dropped {dropped_points} value(s) "
                        f"(IQR factor={outlier_iqr_factor})."
                    )

            n_groups = len(labels)
            ylabel = metric_ylabel(metric)

            title_suffix = ""
            if drop_outliers == "iqr" and dropped_points:
                title_suffix = "\n(outliers removed: IQR)"

            if n_groups == 1:
                xlabel = None
                title = (
                    build_title_linear_vs_nonlinear(param, metric, label_a=labels[0], n_labels=1)
                    + title_suffix
                )
            elif n_groups == 2:
                xlabel = None
                title = (
                    build_title_linear_vs_nonlinear(
                        param, metric, label_a=labels[0], label_b=labels[1], n_labels=2
                    )
                    + title_suffix
                )
            else:
                xlabel = param_xlabel(param)
                title = (
                    build_title_linear_vs_nonlinear(
                        param, metric, n_labels=n_groups
                    )
                    + title_suffix
                )

            fig = plot_all_methods(
                data,
                labels,
                title,
                ylabel,
                metric,
                xlabel=xlabel,
                show_mean=show_mean,
                add_stats=add_stats,
                ci_level=ci_level,
            )
            dest = out_dir / output_basename(metric, param)
            save_figure(fig, dest)
            plt.close(fig)
            written.append(dest)
            print(f"Saved: {dest}")

            if pairwise:
                for la, lb in itertools.combinations(labels, 2):
                    idx_a = labels.index(la)
                    idx_b = labels.index(lb)
                    da, db = data[idx_a], data[idx_b]
                    if not da or not db:
                        continue
                    pw_title = (
                        build_title_linear_vs_nonlinear(
                            param, metric, label_a=la, label_b=lb, n_labels=2
                        )
                        + title_suffix
                    )
                    pw_fig = plot_pairwise_comparison(
                        da,
                        db,
                        la,
                        lb,
                        title=pw_title,
                        ylabel=ylabel,
                        metric=metric,
                        xlabel=xlabel,
                        show_mean=show_mean,
                        add_stats=add_stats,
                        ci_level=ci_level,
                    )
                    pw_dest = out_dir / output_basename(metric, param, la, lb)
                    save_figure(pw_fig, pw_dest)
                    plt.close(pw_fig)
                    written.append(pw_dest)
                    print(f"Saved: {pw_dest}")

    return written


def _parse_filter(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--filter must be KEY=VALUE, got: {raw!r}"
        )
    k, _, v = raw.partition("=")
    return k.strip(), v.strip()


# --- Confusion heatmaps (figure text in Spanish) ---


def apply_filters_dataframe(
    df: pd.DataFrame, filters: dict[str, list[str]]
) -> pd.DataFrame:
    out = df
    for k, vals in filters.items():
        if k not in out.columns:
            continue
        vs = [str(v) for v in vals]
        out = out[out[k].astype(str).isin(vs)]
    return out


# Group confusion runs by every config column except run_id, model_type, seed, and metric outputs
# so that --average-seeds (default) only averages across seeds, not across lr/activation/etc.
CONFUSION_GROUP_EXCLUDE: frozenset[str] = frozenset(
    {
        "run_id",
        "model_type",
        "seed",
        "train_acc",
        "test_acc",
        "mae",
        "mse",
        "final_train_mse",
        "epochs_completed",
        "elapsed_seconds",
        "confusion_meta_json",
    }
)


def _confusion_group_column_names(df: pd.DataFrame) -> list[str]:
    """Group by name, data, and any other config column that actually varies (excl. metrics, seed)."""
    candidates = [c for c in df.columns if c not in CONFUSION_GROUP_EXCLUDE]
    must = [c for c in ("name", "data") if c in df.columns]
    varying = sorted(
        c
        for c in candidates
        if c not in must and df[c].astype(str).nunique(dropna=False) > 1
    )
    return must + varying


def _long_to_matrix(sub: pd.DataFrame) -> np.ndarray:
    if sub.empty:
        return np.zeros((1, 1), dtype=float)
    ni = int(sub["i"].max()) + 1
    nj = int(sub["j"].max()) + 1
    m = np.zeros((ni, nj), dtype=float)
    for _, r in sub.iterrows():
        m[int(r["i"]), int(r["j"])] = float(r["count"])
    return m


def _confusion_meta_labels(meta: dict) -> tuple[list[str], list[str]]:
    mode = meta.get("mode", "")
    if mode == "discrete":
        labels = [float(x) for x in meta.get("labels", [])]
        s = [f"{x:.4g}" for x in labels]
        return s, s
    if mode == "binned":
        edges = [float(x) for x in meta.get("edges", [])]
        if len(edges) < 2:
            return [str(i) for i in range(10)], [str(i) for i in range(10)]
        pair = [f"{edges[i]:.3g}–{edges[i + 1]:.3g}" for i in range(len(edges) - 1)]
        return pair, pair
    return [], []


def _confusion_pick_meta(summary: pd.DataFrame, row_keys: dict) -> dict:
    m = summary
    for k, v in row_keys.items():
        if k in m.columns:
            m = m[m[k].astype(str) == str(v)]
    if m.empty:
        return {"mode": "unknown"}
    raw = m.iloc[0].get("confusion_meta_json", "")
    if not raw or (isinstance(raw, float) and math.isnan(raw)):
        return {"mode": "unknown"}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"mode": "unknown"}


def aggregate_confusion_matrix(
    conf_df: pd.DataFrame,
    *,
    average_seeds: bool,
) -> np.ndarray:
    run_ids = sorted(conf_df["run_id"].astype(str).unique())
    if not average_seeds:
        return _long_to_matrix(conf_df[conf_df["run_id"].astype(str) == run_ids[0]])
    mats = [
        _long_to_matrix(conf_df[conf_df["run_id"].astype(str) == rid])
        for rid in run_ids
    ]
    if not mats:
        return np.zeros((1, 1), dtype=float)
    shapes = {m.shape for m in mats}
    if len(shapes) > 1:
        raise ValueError(f"Inconsistent confusion matrix shapes in group: {shapes}")
    return np.mean(np.stack(mats, axis=0), axis=0)


def plot_confusion_figure(
    matrices: list[tuple[str, np.ndarray]],
    title: str,
    tick_labels: list[str],
    suptitle_meta: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    n = len(matrices)
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 6), 4.8), dpi=FIG_DPI)
    if n == 1:
        axes = np.array([axes])
    fig.patch.set_facecolor(STYLE["figure_bg"])

    vals = [float(m.max()) for _, m in matrices if m.size]
    auto_max = max(vals) if vals else 1.0
    v_min = 0.0 if vmin is None else vmin
    v_max = max(auto_max, 1e-6) if vmax is None else vmax

    for ax, (subtitle, cm) in zip(axes, matrices, strict=True):
        ax.set_facecolor(STYLE["axes_bg"])
        for spine in ax.spines.values():
            spine.set_color(STYLE["text_axis"])
        im = ax.imshow(cm, cmap="Blues", vmin=v_min, vmax=v_max, aspect="auto")
        thr = (v_min + v_max) / 2
        ni, nj = cm.shape
        for i in range(ni):
            for j in range(nj):
                val = cm[i, j]
                txt = f"{val:.1f}" if val != int(val) else f"{int(val)}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val > thr else "black",
                )
        ax.set_title(subtitle, fontsize=10, color=STYLE["text_title"])
        ax.set_xlabel("Predicho", color=STYLE["text_axis"])
        ax.set_ylabel("Verdadero", color=STYLE["text_axis"])
        if tick_labels and len(tick_labels) == max(ni, nj):
            ax.set_xticks(range(nj))
            ax.set_yticks(range(ni))
            ax.set_xticklabels(tick_labels[:nj], rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(tick_labels[:ni], fontsize=6)
        else:
            ax.set_xticks(range(nj))
            ax.set_yticks(range(ni))
        fig.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle(
        title + (f"\n{suptitle_meta}" if suptitle_meta else ""),
        fontsize=12,
        fontweight="600",
        color=STYLE["text_title"],
        y=1.02,
    )
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.96)
    return fig


def _group_dict_from_key(gcols: list[str], group_key: object) -> dict[str, object]:
    if len(gcols) == 1:
        return {gcols[0]: group_key}
    tup = group_key if isinstance(group_key, tuple) else (group_key,)
    return dict(zip(gcols, tup, strict=True))


def run_confusion_figures(
    summary: pd.DataFrame,
    confusion: pd.DataFrame,
    out_dir: Path,
    *,
    average_seeds: bool,
    vmin: float | None,
    vmax: float | None,
) -> list[Path]:
    """
    One PNG per distinct experiment configuration (all columns except metrics / run_id /
    model_type / seed). Within each figure, linear vs non-linear panels; with default
    average_seeds, matrices are averaged only across seeds for that configuration.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.ioff()
    written: list[Path] = []

    gcols = _confusion_group_column_names(summary)
    if not gcols:
        gcols = [c for c in ("name", "data") if c in summary.columns]

    for group_key, sub_sum in summary.groupby(gcols, sort=False):
        gdict = _group_dict_from_key(gcols, group_key)

        name = str(sub_sum["name"].iloc[0])
        data_bn = str(sub_sum["data"].iloc[0])

        mats: list[tuple[str, np.ndarray]] = []
        metas: list[str] = []

        for mt in ("linear", "non-linear"):
            rows_meta = sub_sum[sub_sum["model_type"].astype(str) == mt]
            if rows_meta.empty:
                continue
            run_ids = rows_meta["run_id"].astype(str).tolist()
            cf = confusion[confusion["run_id"].astype(str).isin(run_ids)]
            if cf.empty:
                continue
            try:
                cm = aggregate_confusion_matrix(cf, average_seeds=average_seeds)
            except ValueError as e:
                print(f"Warning [{name}/{data_bn}/{mt}]: {e}")
                continue

            meta = _confusion_pick_meta(sub_sum, {"model_type": mt})
            mats.append((mt, cm))
            mode = meta.get("mode", "?")
            nb = meta.get("n_bins", "")
            metas.append(f"{mode}" + (f", bins={nb}" if nb != "" else ""))

        if len(mats) < 1:
            continue

        meta_str = metas[0] if metas else ""
        raw_meta = sub_sum["confusion_meta_json"].iloc[0]
        if pd.isna(raw_meta):
            meta0 = {}
        else:
            try:
                meta0 = json.loads(str(raw_meta))
            except json.JSONDecodeError:
                meta0 = {}
        _, tick_labels = _confusion_meta_labels(meta0)

        extra_bits = [
            f"{k}={gdict[k]}"
            for k in sorted(gdict.keys())
            if k not in ("name", "data")
        ]
        sub_title = " · ".join(extra_bits) if extra_bits else ""
        title = (
            f"Linear vs non-linear — Confusion matrix (test): {name} · {data_bn}"
            + (f"\n{sub_title}" if sub_title else "")
        )

        fig = plot_confusion_figure(
            mats,
            title,
            tick_labels,
            meta_str,
            vmin=vmin,
            vmax=vmax,
        )

        fname_parts = [_safe_filename_part(name), _safe_filename_part(data_bn.replace(".", "_"))]
        for k in sorted(gdict.keys()):
            if k in ("name", "data"):
                continue
            fname_parts.append(_safe_filename_part(f"{k}_{gdict[k]}"))
        dest = out_dir / f"linear_vs_nonlinear_confusion__{'__'.join(fname_parts)}.png"
        save_figure(fig, dest)
        plt.close(fig)
        written.append(dest)
        print(f"Saved: {dest}")

    return written


def _add_compare_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "compare",
        help="Boxplots from linear_vs_nonlinear_summary.csv.",
    )
    p.add_argument("--csv", type=Path, required=True, help="Path to linear_vs_nonlinear_summary.csv")
    p.add_argument("--out", type=Path, required=True, help="Output directory for PNGs")
    p.add_argument(
        "--x-axis",
        action="append",
        dest="x_axes",
        metavar="COL",
        help="Column for X axis (repeat for multiple). Default: all hyperparameter columns.",
    )
    p.add_argument(
        "--y-axis",
        action="append",
        dest="y_axes",
        metavar="METRIC",
        help="Metric on Y (repeat). Default: all available metrics.",
    )
    p.add_argument(
        "--filter",
        action="append",
        dest="filters",
        metavar="KEY=VALUE",
        type=_parse_filter,
        help="Filter rows, e.g. --filter data=and_data.csv --filter lr=0.1",
    )
    p.add_argument(
        "--pairwise",
        action="store_true",
        help="Also emit pairwise two-group boxplots for every X value pair.",
    )
    p.add_argument("--show-mean", action="store_true")
    p.add_argument("--add-stats", action="store_true")
    p.add_argument(
        "--ci",
        type=float,
        default=0.95,
        metavar="LEVEL",
        help="Confidence level for annotation (default: 0.95)",
    )
    p.add_argument(
        "--list-columns",
        action="store_true",
        help="Print plottable column names and exit.",
    )
    p.add_argument(
        "--drop-outliers",
        choices=("none", "iqr"),
        default="none",
        help="Remove extreme points per category before plotting (IQR / Tukey fences).",
    )
    p.add_argument(
        "--outlier-iqr-factor",
        type=float,
        default=1.5,
        metavar="K",
        help="IQR multiplier for --drop-outliers iqr (default: 1.5).",
    )
    return p


# ---------------------------------------------------------------------------
# Subcommand: curves  (per-epoch learning curves)
# ---------------------------------------------------------------------------

_CURVES_MODEL_COLORS = {"linear": "#4a90d9", "non-linear": "#e67e22"}
_CURVES_DEFAULT_COLS = ["name", "activation", "lr", "test_per"]


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def run_curves_figures(
    curves_csv: Path,
    out_dir: Path,
    filters: dict[str, list[str]] | None = None,
    group_by: list[str] | None = None,
    smooth: int = 0,
) -> list[Path]:
    """Generate one learning-curve figure per combination of `group_by` columns."""
    if not curves_csv.is_file():
        raise SystemExit(f"Curves CSV not found: {curves_csv}")

    df = pd.read_csv(curves_csv)
    if df.empty:
        print("Curves CSV is empty.")
        return []

    if filters:
        for k, vals in filters.items():
            if k in df.columns:
                df = df[df[k].astype(str).isin([str(v) for v in vals])]
    if df.empty:
        print("No data after applying filters.")
        return []

    group_cols = group_by or [c for c in _CURVES_DEFAULT_COLS if c in df.columns]
    group_cols = [c for c in group_cols if c in df.columns]

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    groups = df.groupby(group_cols, dropna=False) if group_cols else [(("all",), df)]
    for key, gdf in groups:
        key_str = "_".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))

        with plt.rc_context(PLOT_RC):
            fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
        fig.patch.set_facecolor(STYLE["figure_bg"])
        ax.set_facecolor(STYLE["axes_bg"])
        ax.grid(True, color=STYLE["grid"], linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        any_plotted = False
        for model_type, color in _CURVES_MODEL_COLORS.items():
            sub = gdf[gdf["model_type"] == model_type]
            if sub.empty:
                continue

            # Promediar sobre semillas: interpolar a la longitud más corta
            seed_curves: list[np.ndarray] = []
            for _, seed_df in sub.groupby("seed", dropna=False):
                arr = seed_df.sort_values("epoch")["train_mse"].to_numpy(dtype=float)
                if len(arr) > 0:
                    seed_curves.append(arr)

            if not seed_curves:
                continue

            min_len = min(len(c) for c in seed_curves)
            mat = np.array([c[:min_len] for c in seed_curves])
            mean = mat.mean(axis=0)
            std  = mat.std(axis=0)
            xs   = np.arange(1, min_len + 1)

            if smooth > 1:
                mean = _moving_average(mean, smooth)
                std  = _moving_average(std,  smooth)
                xs   = xs[:len(mean)]

            label = f"{model_type} (n={len(seed_curves)} seeds)"
            ax.plot(xs, mean, color=color, linewidth=1.8, label=label, zorder=3)
            if len(seed_curves) > 1:
                ax.fill_between(
                    xs, mean - std, mean + std,
                    color=color, alpha=0.18, zorder=2,
                )
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        title_parts = [f"{c}={k}" for c, k in zip(group_cols, key if isinstance(key, tuple) else (key,))]
        title = "Learning curves — " + ", ".join(title_parts)
        ax.set_title(title, color=STYLE["text_title"], pad=8)
        ax.set_xlabel("Epoch", color=STYLE["text_axis"])
        ax.set_ylabel("Train MSE (mean ± std over seeds)", color=STYLE["text_axis"])
        ax.tick_params(colors=STYLE["text_axis"])
        ax.legend(fontsize=9)
        fig.tight_layout(pad=1.5)

        safe_key = re.sub(r"[^\w.\-]+", "_", key_str).strip("_") or "group"
        dest = out_dir / f"curves_{safe_key}.png"
        save_figure(fig, dest)
        plt.close(fig)
        written.append(dest)
        print(f"Saved: {dest}")

    return written


def _add_curves_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "curves",
        help="Learning curves (MSE per epoch) from linear_vs_nonlinear_curves.csv.",
    )
    p.add_argument(
        "--curves-csv",
        type=Path,
        default=ROOT / "results" / "linear_vs_nonlinear_curves.csv",
        help="Path to curves CSV (default: results/linear_vs_nonlinear_curves.csv)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory for PNGs")
    p.add_argument(
        "--filter",
        action="append",
        dest="filters",
        metavar="KEY=VALUE",
        type=_parse_filter,
        help="Filter rows, e.g. --filter activation=logistic --filter no_split=True",
    )
    p.add_argument(
        "--group-by",
        action="append",
        dest="group_by",
        metavar="COL",
        help="Grouping columns (repeat for multiple). Default: name activation lr test_per",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=0,
        metavar="W",
        help="Moving-average window for curve smoothing (0 = no smoothing).",
    )
    return p


# ---------------------------------------------------------------------------
# Subcommand: roc  (ROC curves and threshold analysis)
# ---------------------------------------------------------------------------

_ROC_GROUP_COLS = ["name", "activation", "lr", "test_per"]


def run_roc_figures(
    roc_csv: Path,
    out_dir: Path,
    filters: dict[str, list[str]] | None = None,
    group_by: list[str] | None = None,
) -> list[Path]:
    """Generate ROC curves averaged over seeds, one figure per group."""
    if not roc_csv.is_file():
        raise SystemExit(f"ROC CSV not found: {roc_csv}")

    df = pd.read_csv(roc_csv)
    if df.empty:
        print("ROC CSV is empty.")
        return []

    if filters:
        for k, vals in filters.items():
            if k in df.columns:
                df = df[df[k].astype(str).isin([str(v) for v in vals])]
    if df.empty:
        print("No data after applying filters.")
        return []

    group_cols = group_by or [c for c in _ROC_GROUP_COLS if c in df.columns]
    group_cols = [c for c in group_cols if c in df.columns]

    # Grid de FPR común para interpolar y promediar curvas entre semillas
    fpr_grid = np.linspace(0.0, 1.0, 300)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    groups = df.groupby(group_cols, dropna=False) if group_cols else [(("all",), df)]
    for key, gdf in groups:
        key_str = "_".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))

        with plt.rc_context(PLOT_RC):
            fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
        fig.patch.set_facecolor(STYLE["figure_bg"])
        ax.set_facecolor(STYLE["axes_bg"])
        ax.plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=1, label="Clasificador aleatorio")
        ax.grid(True, color=STYLE["grid"], linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        any_plotted = False
        for model_type, color in _CURVES_MODEL_COLORS.items():
            sub = gdf[gdf["model_type"] == model_type]
            if sub.empty:
                continue

            # Interpolar cada semilla a fpr_grid y apilar
            tpr_interps: list[np.ndarray] = []
            for _, seed_df in sub.groupby("seed", dropna=False):
                sdf = seed_df.sort_values("fpr")
                fpr_s = sdf["fpr"].to_numpy(dtype=float)
                tpr_s = sdf["tpr"].to_numpy(dtype=float)
                if len(fpr_s) < 2:
                    continue
                tpr_interp = np.interp(fpr_grid, fpr_s, tpr_s)
                tpr_interps.append(tpr_interp)

            if not tpr_interps:
                continue

            mat  = np.array(tpr_interps)
            mean = mat.mean(axis=0)
            std  = mat.std(axis=0)
            auc_mean = float(np.sum((fpr_grid[1:] - fpr_grid[:-1]) * (mean[1:] + mean[:-1]) / 2.0))

            label = f"{model_type}  AUC={auc_mean:.3f} (n={len(tpr_interps)} seeds)"
            ax.plot(fpr_grid, mean, color=color, linewidth=2.0, label=label, zorder=3)
            if len(tpr_interps) > 1:
                ax.fill_between(
                    fpr_grid, mean - std, mean + std,
                    color=color, alpha=0.15, zorder=2,
                )
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        title_parts = [f"{c}={k}" for c, k in zip(group_cols, key if isinstance(key, tuple) else (key,))]
        title = "ROC Curve — " + ", ".join(title_parts)
        ax.set_title(title, color=STYLE["text_title"], pad=8)
        ax.set_xlabel("FPR (False Positive Rate)", color=STYLE["text_axis"])
        ax.set_ylabel("TPR (True Positive Rate / Recall)", color=STYLE["text_axis"])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(colors=STYLE["text_axis"])
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout(pad=1.5)

        safe_key = re.sub(r"[^\w.\-]+", "_", key_str).strip("_") or "group"
        dest = out_dir / f"roc_{safe_key}.png"
        save_figure(fig, dest)
        plt.close(fig)
        written.append(dest)
        print(f"Saved: {dest}")

    return written


def _add_roc_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "roc",
        help="ROC curves for threshold analysis from linear_vs_nonlinear_roc.csv.",
    )
    p.add_argument(
        "--roc-csv",
        type=Path,
        default=ROOT / "results" / "linear_vs_nonlinear_roc.csv",
        help="Path to ROC CSV (default: results/linear_vs_nonlinear_roc.csv)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory for PNGs")
    p.add_argument(
        "--filter",
        action="append",
        dest="filters",
        metavar="KEY=VALUE",
        type=_parse_filter,
        help="Filter rows, e.g. --filter model_type=non-linear --filter activation=logistic",
    )
    p.add_argument(
        "--group-by",
        action="append",
        dest="group_by",
        metavar="COL",
        help="Grouping columns (repeat for multiple). Default: name activation lr test_per",
    )
    return p


def _add_confusion_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "confusion",
        help="Confusion matrix heatmaps from linear_vs_nonlinear_confusion_runs.csv.",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "results" / "linear_vs_nonlinear_summary.csv",
        help="Summary CSV (for filters and confusion_meta_json)",
    )
    p.add_argument(
        "--confusion-csv",
        type=Path,
        default=ROOT / "results" / "linear_vs_nonlinear_confusion_runs.csv",
        help="Long-format confusion counts (i, j, count) per run_id"
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory for PNGs")
    p.add_argument(
        "--filter",
        action="append",
        dest="filters",
        metavar="KEY=VALUE",
        type=_parse_filter,
        help="Filter on summary columns, e.g. --filter name=and",
    )
    p.set_defaults(average_seeds=True)
    p.add_argument(
        "--no-average-seeds",
        action="store_false",
        dest="average_seeds",
        help="Do not average: use the first run_id in each group",
    )
    p.add_argument("--vmin", type=float, default=None, help="Color scale minimum")
    p.add_argument("--vmax", type=float, default=None, help="Color scale maximum")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Linear vs non-linear plots: boxplot comparisons, confusion heatmaps, "
            "learning curves (curves) o curvas ROC (roc)."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_compare_parser(sub)
    _add_confusion_parser(sub)
    _add_curves_parser(sub)
    _add_roc_parser(sub)
    args = parser.parse_args()

    if args.command == "compare":
        csv_path = args.csv.resolve()
        if not csv_path.is_file():
            print(f"File not found: {csv_path}", file=sys.stderr)
            sys.exit(1)

        if args.list_columns:
            rows, fieldnames = load_rows(csv_path)
            validate_columns(fieldnames)
            rows, fieldnames = compute_derived_metrics(rows, fieldnames)
            print("Hyperparameter columns for --x-axis (default batch omits run_id):")
            for c in param_columns_default(fieldnames):
                print(f"  {c}")
            print("\nMetrics for --y-axis:")
            for m in _all_metric_names_for_list():
                print(f"  {m}")
            return

        if not _SCIPY_AVAILABLE:
            print("Note: scipy not installed; some CI features may be limited.")

        out_dir = args.out.resolve()
        filters: dict[str, list[str]] = {}
        for k, v in args.filters or []:
            filters.setdefault(k, []).append(v)

        if args.outlier_iqr_factor <= 0:
            print("--outlier-iqr-factor must be > 0", file=sys.stderr)
            sys.exit(1)

        paths = run_compare_from_summary(
            csv_path,
            out_dir,
            filters=filters or None,
            x_axis_columns=args.x_axes,
            y_axis_metrics=args.y_axes,
            pairwise=args.pairwise,
            show_mean=args.show_mean,
            add_stats=args.add_stats,
            ci_level=args.ci,
            drop_outliers=args.drop_outliers,
            outlier_iqr_factor=args.outlier_iqr_factor,
        )

        if paths:
            print(f"\nTotal: {len(paths)} figure(s) written to {out_dir}")
        else:
            print("No figures written (check filters and columns).")
        return

    if args.command == "curves":
        filters: dict[str, list[str]] = {}
        for k, v in args.filters or []:
            filters.setdefault(k, []).append(v)
        written = run_curves_figures(
            args.curves_csv.resolve(),
            args.out.resolve(),
            filters=filters or None,
            group_by=args.group_by,
            smooth=args.smooth,
        )
        if not written:
            print("No figures written. Check filters and CSV path.")
        else:
            print(f"Total: {len(written)} figure(s) written to {args.out.resolve()}")
        return

    if args.command == "roc":
        filters: dict[str, list[str]] = {}
        for k, v in args.filters or []:
            filters.setdefault(k, []).append(v)
        written = run_roc_figures(
            args.roc_csv.resolve(),
            args.out.resolve(),
            filters=filters or None,
            group_by=args.group_by,
        )
        if not written:
            print("No figures written. Check filters and CSV path.")
        else:
            print(f"Total: {len(written)} figure(s) written to {args.out.resolve()}")
        return

    # confusion
    summary_path = args.summary_csv.resolve()
    confusion_path = args.confusion_csv.resolve()
    if not summary_path.is_file():
        print(f"File not found: {summary_path}", file=sys.stderr)
        sys.exit(1)
    if not confusion_path.is_file():
        print(f"File not found: {confusion_path}", file=sys.stderr)
        sys.exit(1)

    summary = pd.read_csv(summary_path)
    confusion = pd.read_csv(confusion_path)

    filters: dict[str, list[str]] = {}
    for k, v in args.filters or []:
        filters.setdefault(k, []).append(v)

    if filters:
        summary = apply_filters_dataframe(summary, filters)
        allowed = set(summary["run_id"].astype(str))
        confusion = confusion[confusion["run_id"].astype(str).isin(allowed)]

    if summary.empty or confusion.empty:
        print("No data after filters.")
        sys.exit(0)

    out_dir = args.out.resolve()
    written = run_confusion_figures(
        summary,
        confusion,
        out_dir,
        average_seeds=args.average_seeds,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    if not written:
        print("No figures written. Try a wider --filter or check CSVs.")
    else:
        print(f"Total: {len(written)} figure(s) written to {out_dir}")


def _all_metric_names_for_list() -> list[str]:
    """Full metric catalog for --list-columns (may not all exist in this CSV yet)."""
    return (
        list(REQUIRED_METRIC_COLUMNS)
        + list(OPTIONAL_METRIC_COLUMNS)
        + list(DERIVED_METRIC_COLUMNS)
    )


if __name__ == "__main__":
    main()
