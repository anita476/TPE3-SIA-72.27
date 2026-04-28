from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main import build_perceptron, load_data
from utils.confusion_from_regression import (
    bin_range_from_train,
    confusion_matrix_binned,
    confusion_matrix_discrete,
    flatten_confusion_rows,
    resolve_confusion_mode,
)
from utils.fraud_metrics import (
    find_best_threshold,
    metrics_at_threshold,
    roc_auc,
    roc_curve_points,
)
from utils.normalizers import standard_scale_apply, standard_scale_params
from utils.test_data_split import stratified_split_regression

ROOT = _ROOT
DEFAULT_CONFIG = ROOT / "configs" / "linear_vs_nonlinear_fraud.json"

RESULTS_DIR   = os.path.join("results")
SUMMARY_CSV   = os.path.join(RESULTS_DIR, "linear_vs_nonlinear_summary.csv")
CONFUSION_CSV = os.path.join(RESULTS_DIR, "linear_vs_nonlinear_confusion_runs.csv")
CURVES_CSV    = os.path.join(RESULTS_DIR, "linear_vs_nonlinear_curves.csv")
ROC_CSV       = os.path.join(RESULTS_DIR, "linear_vs_nonlinear_roc.csv")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_data_path(p: str | Path) -> str:
    pp = Path(p)
    if not pp.is_absolute():
        pp = ROOT / pp
    return str(pp)


def _split_seeds(merged: dict) -> list[dict]:
    """Expand a `seeds` list or normalise a scalar `seed` into one dict per seed."""
    m = dict(merged)
    seeds_val = m.pop("seeds", None)
    if seeds_val is not None:
        if isinstance(seeds_val, (int, float)):
            seeds_val = [int(seeds_val)]
        return [{**m, "seed": int(s)} for s in seeds_val]
    if "seed" in m:
        return [{**m, "seed": int(m["seed"])}]
    return [{**m, "seed": 1}]


def experiment_bases_from_config(cfg: dict) -> list[dict]:
    """Build a flat list of experiment dicts from a JSON config.

    Merges `base` with the Cartesian product of `grid` (each grid value must be
    a list).  Each output dict has a scalar `seed` key.
    """
    base = dict(cfg.get("base", {}))
    if cfg.get("runs"):
        raise SystemExit(
            "Config key 'runs' is not supported. Use only 'base' and 'grid' (see README)."
        )
    grid_cfg = cfg.get("grid") or {}

    out: list[dict] = []
    if grid_cfg:
        keys = list(grid_cfg.keys())
        for combo in itertools.product(*[grid_cfg[k] for k in keys]):
            merged = {**base, **dict(zip(keys, combo, strict=True))}
            out.extend(_split_seeds(merged))
        return out

    out.extend(_split_seeds(base))
    return out


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def _make_run_id(base: dict, model_type: str, seed: int) -> str:
    data_bn = os.path.basename(str(base["data"]))
    act = str(base.get("activation", "tanh")).replace(".", "_")
    if "test_per" in base and base["test_per"] is None:
        tp = "full"
    elif "test_per" in base:
        tp = str(base["test_per"]).replace(".", "_")
    else:
        tp = str(0.2).replace(".", "_")
    return (
        f"{base['name']}__{model_type}__s{seed}__tp{tp}__lr{base['lr']}__"
        f"ep{base['epochs']}__eps{base['epsilon']}__act_{act}__{data_bn.replace('.', '_')}"
    )


# ---------------------------------------------------------------------------
# Regression-style metrics (tolerance-based accuracy for continuous targets)
# ---------------------------------------------------------------------------

def _metrics_float(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float,
) -> tuple[float, float, float]:
    """Return (tolerance-accuracy, MAE, MSE) for continuous predictions."""
    matches = np.abs(y_pred - y_true) < tolerance
    acc = float(np.mean(matches)) if len(y_true) else 0.0
    mae = float(np.mean(np.abs(y_pred - y_true))) if len(y_true) else 0.0
    mse = float(np.mean((y_pred - y_true) ** 2)) if len(y_true) else 0.0
    return acc, mae, mse


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single(job: tuple[dict, str]) -> dict:
    """Train one perceptron (linear or non-linear) on the fraud dataset.

    Returns a dict with keys: summary, cm_rows, curve_rows, roc_rows.
    """
    base, model_type = job
    seed      = int(base["seed"])
    data_path = _resolve_data_path(base["data"])
    label_col = str(base.get("label", "label"))
    tolerance = float(base["tolerance"])

    tp_raw   = base.get("test_per", 0.2)
    no_split = bool(base.get("no_split", False)) or tp_raw is None

    X, y = load_data(data_path, label_col)
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    # Binary labels for stratification (works with both hard {0,1} and soft targets)
    y_stratify = (y >= 0.5).astype(int)

    if no_split:
        if base.get("normalize") == "standard":
            mean, std = standard_scale_params(X)
            X = standard_scale_apply(X, mean, std)
        X_train = X_test = X
        y_train = y_test = y
    else:
        # Stratified split: preserves fraud/non-fraud ratio in both partitions
        X_train, X_test, y_train, y_test = stratified_split_regression(
            X, y, y_stratify, val_size=float(tp_raw), random_state=seed
        )
        if base.get("normalize") == "standard":
            mean, std = standard_scale_params(X_train)
            X_train = standard_scale_apply(X_train, mean, std)
            X_test  = standard_scale_apply(X_test,  mean, std)

    perceptron = build_perceptron(
        "non-linear" if model_type == "non-linear" else "linear",
        base["lr"],
        base["epochs"],
        base["epsilon"],
        seed,
        base.get("activation", "tanh"),
        base.get("beta", 1.0),
        None,       # layers (unused for simple perceptrons)
        "random",   # initializer
        "online",   # training_mode
        1,          # batch_size
        "sgd",      # optimizer
    )

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        perceptron.fit(X_train, y_train)
    elapsed_seconds = time.perf_counter() - t0

    train_preds = perceptron.predict(X_train)
    test_preds  = perceptron.predict(X_test)

    train_acc, _, _       = _metrics_float(y_train, train_preds, tolerance)
    test_acc, mae, mse    = _metrics_float(y_test,  test_preds,  tolerance)
    final_train_mse       = float(np.mean((train_preds - y_train) ** 2)) if len(y_train) else 0.0
    epochs_completed      = int(getattr(perceptron, "epochs_run_", base["epochs"]))

    # Dataset size and class balance info
    n_train          = len(y_train)
    n_test           = len(y_test)
    fraud_rate_train = float(np.mean(y_stratify[(y >= 0.5) if no_split else np.ones(len(y), dtype=bool)]))
    # Recompute per-split fraud rates correctly
    fraud_rate_train = float(np.mean((y_train >= 0.5).astype(int)))
    fraud_rate_test  = float(np.mean((y_test  >= 0.5).astype(int)))

    # ------------------------------------------------------------------
    # Binary classification metrics (fraud detection)
    # ------------------------------------------------------------------
    roc_auc_val = float("nan")
    best_thr = best_f1 = best_prec = best_rec = float("nan")
    f1_half = prec_half = rec_half = float("nan")
    train_roc_auc_val = float("nan")
    train_f1_half = train_prec_half = train_rec_half = float("nan")

    if n_test > 0:
        roc_auc_val = roc_auc(y_test, test_preds)

        # Best threshold on test (maximises F1)
        best      = find_best_threshold(y_test, test_preds, metric="f1")
        best_thr  = best["threshold"]
        best_f1   = best["f1"]
        best_prec = best["precision"]
        best_rec  = best["recall"]

        # Fixed threshold = 0.5 (natural midpoint for logistic and tanh with labels in [0, 1])
        half      = metrics_at_threshold(y_test, test_preds, threshold=0.5)
        f1_half   = half["f1"]
        prec_half = half["precision"]
        rec_half  = half["recall"]

    if n_train > 0:
        train_roc_auc_val = roc_auc(y_train, train_preds)
        tr_half            = metrics_at_threshold(y_train, train_preds, threshold=0.5)
        train_f1_half      = tr_half["f1"]
        train_prec_half    = tr_half["precision"]
        train_rec_half     = tr_half["recall"]

    # ------------------------------------------------------------------
    # Confusion matrix (discrete 2x2 for binary labels, binned otherwise)
    # ------------------------------------------------------------------
    cm_mode = resolve_confusion_mode(
        y_train,
        str(base.get("confusion_mode", "auto")),
        discrete_max_unique=15,
    )
    n_bins = int(base.get("confusion_bins", 10))
    ref    = np.concatenate([y_train, y_test])

    if cm_mode == "discrete":
        cm, labels = confusion_matrix_discrete(y_test, test_preds, ref)
        meta = {"mode": "discrete", "labels": [float(x) for x in labels]}
    else:
        lo, hi     = bin_range_from_train(y_train, y_test)
        cm, edges  = confusion_matrix_binned(y_test, test_preds, n_bins, lo, hi)
        meta = {"mode": "binned", "n_bins": n_bins, "edges": [float(x) for x in edges]}

    run_id   = _make_run_id(base, model_type, seed)
    data_bn  = os.path.basename(data_path)
    act_name = str(base.get("activation", "tanh"))
    tp_val   = float(tp_raw) if tp_raw is not None else math.nan

    def _r(v: float) -> float:
        return round(v, 6) if not math.isnan(v) else math.nan

    # ------------------------------------------------------------------
    # Summary row
    # ------------------------------------------------------------------
    summary_row = {
        # --- experiment config ---
        "run_id":               run_id,
        "name":                 base["name"],
        "data":                 data_bn,
        "model_type":           model_type,
        "lr":                   base["lr"],
        "epochs":               base["epochs"],
        "epsilon":              base["epsilon"],
        "tolerance":            tolerance,
        "activation":           act_name,
        "beta":                 base.get("beta", 1.0),
        "test_per":             tp_val,
        "normalize":            base.get("normalize", "none"),
        "no_split":             no_split,
        "seed":                 seed,
        "confusion_mode":       cm_mode,
        "confusion_bins_config": n_bins,
        # --- dataset stats ---
        "n_train":              n_train,
        "n_test":               n_test,
        "fraud_rate_train":     round(fraud_rate_train, 6),
        "fraud_rate_test":      round(fraud_rate_test,  6),
        # --- regression metrics ---
        "train_acc":            round(train_acc,     6),
        "test_acc":             round(test_acc,      6),
        "mae":                  round(mae,           6),
        "mse":                  round(mse,           6),
        "final_train_mse":      round(final_train_mse, 6),
        "epochs_completed":     epochs_completed,
        "elapsed_seconds":      round(elapsed_seconds, 6),
        # --- binary fraud metrics (test set) ---
        "roc_auc":              _r(roc_auc_val),
        "best_threshold":       _r(best_thr),
        "best_f1":              _r(best_f1),
        "best_precision":       _r(best_prec),
        "best_recall":          _r(best_rec),
        "f1_at_half":           _r(f1_half),
        "precision_at_half":    _r(prec_half),
        "recall_at_half":       _r(rec_half),
        # --- binary fraud metrics (train set — for overfitting diagnosis) ---
        "train_roc_auc":        _r(train_roc_auc_val),
        "train_f1_at_half":     _r(train_f1_half),
        "train_precision_at_half": _r(train_prec_half),
        "train_recall_at_half": _r(train_rec_half),
        # --- metadata ---
        "confusion_meta_json":  json.dumps(meta, ensure_ascii=False),
    }

    # ------------------------------------------------------------------
    # Confusion matrix rows
    # ------------------------------------------------------------------
    cm_rows: list[dict] = []
    for row in flatten_confusion_rows(run_id, cm):
        row["name"]           = base["name"]
        row["data"]           = data_bn
        row["model_type"]     = model_type
        row["seed"]           = seed
        row["confusion_mode"] = cm_mode
        cm_rows.append(row)

    # ------------------------------------------------------------------
    # Learning curve rows (one row per epoch)
    # ------------------------------------------------------------------
    curve_rows: list[dict] = []
    for epoch_idx, mse_val in enumerate(getattr(perceptron, "train_mse_history_", [])):
        curve_rows.append({
            "run_id":     run_id,
            "model_type": model_type,
            "name":       base["name"],
            "activation": act_name,
            "lr":         base["lr"],
            "seed":       seed,
            "test_per":   tp_val,
            "no_split":   no_split,
            "epoch":      epoch_idx + 1,
            "train_mse":  round(float(mse_val), 8),
        })

    # ------------------------------------------------------------------
    # ROC curve rows (one row per threshold point)
    # ------------------------------------------------------------------
    roc_rows: list[dict] = []
    if n_test > 0:
        fpr_arr, tpr_arr, thr_arr = roc_curve_points(y_test, test_preds, n_thresholds=100)
        y_test_bin = (y_test >= 0.5).astype(int)
        for thr_val, fpr_val, tpr_val in zip(thr_arr, fpr_arr, tpr_arr):
            pred_bin = (test_preds >= thr_val).astype(int)
            tp_c = int(np.sum((y_test_bin == 1) & (pred_bin == 1)))
            fp_c = int(np.sum((y_test_bin == 0) & (pred_bin == 1)))
            fn_c = int(np.sum((y_test_bin == 1) & (pred_bin == 0)))
            prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            rec  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            roc_rows.append({
                "run_id":     run_id,
                "model_type": model_type,
                "name":       base["name"],
                "activation": act_name,
                "lr":         base["lr"],
                "seed":       seed,
                "test_per":   tp_val,
                "no_split":   no_split,
                "threshold":  round(float(thr_val), 6),
                "fpr":        round(float(fpr_val), 6),
                "tpr":        round(float(tpr_val), 6),
                "precision":  round(float(prec),    6),
                "recall":     round(float(rec),     6),
            })

    return {
        "summary":    summary_row,
        "cm_rows":    cm_rows,
        "curve_rows": curve_rows,
        "roc_rows":   roc_rows,
    }


# ---------------------------------------------------------------------------
# Job expansion
# ---------------------------------------------------------------------------

def expand_jobs(bases: list[dict]) -> list[tuple[dict, str]]:
    """Produce one (base, model_type) job per base config × {linear, non-linear}."""
    jobs: list[tuple[dict, str]] = []
    for b in bases:
        for mt in ("linear", "non-linear"):
            jobs.append((b, mt))
    return jobs


def _worker(job: tuple[dict, str]) -> dict:
    return run_single(job)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run linear vs non-linear perceptron experiments from a JSON config."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to JSON config (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job count and exit without training.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel worker processes (default: 1 = serial). Capped by job count.",
    )
    return p.parse_args()


def _append_csv(path: str, df_new: pd.DataFrame) -> None:
    """Append rows to an existing CSV, or create it if it does not exist."""
    if os.path.isfile(path):
        prev   = pd.read_csv(path)
        df_new = pd.concat([prev, df_new], ignore_index=True, sort=False)
    df_new.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    experiment_bases = experiment_bases_from_config(cfg)
    jobs = expand_jobs(experiment_bases)

    print(
        f"Config loaded: {len(experiment_bases)} grid row(s) -> {len(jobs)} jobs "
        f"(linear + non-linear per row)."
    )

    if args.dry_run:
        print("Dry run: no training performed.")
        return

    if not jobs:
        print("No jobs to run (empty base + grid).")
        return

    n_workers = min(len(jobs), args.workers)
    print(f"Running {len(jobs)} jobs on {n_workers} worker(s)...")

    if n_workers <= 1:
        out = [_worker(j) for j in jobs]
    else:
        with mp.Pool(n_workers) as pool:
            out = pool.map(_worker, jobs)

    summary_rows = [o["summary"]    for o in out]
    all_cm:     list[dict] = [row for o in out for row in o["cm_rows"]]
    all_curves: list[dict] = [row for o in out for row in o["curve_rows"]]
    all_roc:    list[dict] = [row for o in out for row in o["roc_rows"]]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_s = pd.DataFrame(summary_rows)
    sort_cols = [c for c in ("name", "data", "model_type", "activation", "test_per", "lr", "seed")
                 if c in df_s.columns]
    if sort_cols:
        df_s = df_s.sort_values(sort_cols, kind="stable")

    _append_csv(SUMMARY_CSV,   df_s)
    _append_csv(CONFUSION_CSV, pd.DataFrame(all_cm))
    if all_curves:
        _append_csv(CURVES_CSV, pd.DataFrame(all_curves))
    if all_roc:
        _append_csv(ROC_CSV, pd.DataFrame(all_roc))

    print(f"Appended {len(summary_rows)} rows         -> {SUMMARY_CSV}")
    print(f"Appended {len(all_cm)} confusion cells  -> {CONFUSION_CSV}")
    if all_curves:
        print(f"Appended {len(all_curves)} curve points    -> {CURVES_CSV}")
    if all_roc:
        print(f"Appended {len(all_roc)} ROC points       -> {ROC_CSV}")


if __name__ == "__main__":
    main()
