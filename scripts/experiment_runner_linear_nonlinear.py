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
DEFAULT_CONFIG = ROOT / "configs" / "lr_exploration_tanh_logistic.json"

RESULTS_DIR        = str(ROOT / "results")
SUMMARY_CSV        = str(Path(RESULTS_DIR) / "linear_vs_nonlinear_summary.csv")
CONFUSION_CSV      = str(Path(RESULTS_DIR) / "linear_vs_nonlinear_confusion_runs.csv")
CURVES_CSV         = str(Path(RESULTS_DIR) / "linear_vs_nonlinear_curves.csv")
ROC_CSV            = str(Path(RESULTS_DIR) / "linear_vs_nonlinear_roc.csv")
FITTING_CURVES_CSV = str(Path(RESULTS_DIR) / "linear_vs_nonlinear_fitting_curves.csv")


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


def _drop_cols_from_config(cfg: dict) -> list[str]:
    """Optional top-level ``drop`` key: list of feature column names (same as CLI ``--drop``)."""
    raw = cfg.get("drop")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise SystemExit("Config key 'drop' must be a JSON array of strings.")
    out = []
    for i, x in enumerate(raw):
        if not isinstance(x, str) or not x.strip():
            raise SystemExit(f"Config 'drop' entry at index {i} must be a non-empty string.")
        out.append(x.strip())
    return out


def _merge_drop_cols(cfg: dict, cli_drop: list[str]) -> list[str]:
    """Union of config ``drop`` and CLI ``--drop`` (order preserved, no duplicates)."""
    merged = _drop_cols_from_config(cfg) + list(cli_drop)
    return list(dict.fromkeys(merged))


def experiment_bases_from_config(cfg: dict) -> list[dict]:
    """Build a flat list of experiment dicts from a JSON config.

    Merges ``base`` with the Cartesian product of ``grid`` (each grid value must be
    a list). Each output dict has a scalar ``seed`` key.

    Alternatively, set ``activation_lr_pairs`` (list of ``{"activation": ..., "lr": ...}``)
    to assign one learning rate per activation; ``grid`` must then omit ``activation``
    and ``lr``. Each pair is crossed with the Cartesian product of the remaining
    ``grid`` keys.
    """
    base = dict(cfg.get("base", {}))
    if cfg.get("runs"):
        raise SystemExit(
            "Config key 'runs' is not supported. Use 'base' + 'grid' "
            "or 'activation_lr_pairs' (see configs)."
        )

    pairs_raw = cfg.get("activation_lr_pairs")
    grid_cfg = cfg.get("grid") or {}

    if pairs_raw is not None:
        if not isinstance(pairs_raw, list) or not pairs_raw:
            raise SystemExit("'activation_lr_pairs' must be a non-empty JSON array.")
        if "activation" in grid_cfg or "lr" in grid_cfg:
            raise SystemExit(
                "With 'activation_lr_pairs', remove 'activation' and 'lr' from 'grid'."
            )
        parsed_pairs: list[tuple[str, float]] = []
        for i, row in enumerate(pairs_raw):
            if isinstance(row, dict):
                if "activation" not in row or "lr" not in row:
                    raise SystemExit(
                        f"activation_lr_pairs[{i}] needs 'activation' and 'lr' keys."
                    )
                parsed_pairs.append((str(row["activation"]), float(row["lr"])))
            elif isinstance(row, (list, tuple)) and len(row) == 2:
                parsed_pairs.append((str(row[0]), float(row[1])))
            else:
                raise SystemExit(
                    f"activation_lr_pairs[{i}] must be an object or [activation, lr] pair."
                )

        out_pairs: list[dict] = []
        keys = list(grid_cfg.keys())
        val_lists = [grid_cfg[k] for k in keys]
        for act, lr in parsed_pairs:
            for combo in itertools.product(*val_lists):
                merged = {**base, **dict(zip(keys, combo, strict=True)), "activation": act, "lr": lr}
                out_pairs.extend(_split_seeds(merged))
        return out_pairs

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

def _to_probability(predictions: np.ndarray) -> np.ndarray:
    """Clip raw perceptron output to [0, 1] so --threshold is comparable across all model types."""
    return np.clip(predictions, 0.0, 1.0)


def _metrics_float(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Return (threshold-accuracy, MAE, MSE) for continuous predictions.

    y_pred_prob must already be clipped to [0, 1] via _to_probability().
    Accuracy = fraction of samples where the binary decision matches the true label.
    """
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    y_bin = (y_pred_prob >= threshold).astype(int)
    y_true_bin = (y_true >= 0.5).astype(int)
    acc = float(np.mean(y_bin == y_true_bin))
    mae = float(np.mean(np.abs(y_pred_prob - y_true)))
    mse = float(np.mean((y_pred_prob - y_true) ** 2))
    return acc, mae, mse


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single(job: tuple[dict, str], drop_cols: list[str] = []) -> dict:
    """Train one perceptron (linear or non-linear) on the fraud dataset.

    Returns a dict with keys: summary, cm_rows, curve_rows, roc_rows,
    fitting_curve_row.
    """
    base, model_type = job
    seed      = int(base["seed"])
    data_path = _resolve_data_path(base["data"])
    label_col = str(base.get("label", "label"))
    threshold = float(base["threshold"])

    tp_raw   = base.get("test_per", 0.2)
    no_split = bool(base.get("no_split", False)) or tp_raw is None

    X, y = load_data(data_path, label_col,drop_cols)
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

    act_name = str(base.get("activation", "tanh"))

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        perceptron.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    elapsed_seconds = time.perf_counter() - t0

    train_preds = perceptron.predict(X_train)
    test_preds  = perceptron.predict(X_test)

    if model_type == "non-linear" and act_name == "tanh":
        train_preds = (train_preds + 1) / 2
        test_preds  = (test_preds  + 1) / 2

    # Clip all outputs to [0,1] so threshold has identical meaning across model types
    train_preds = _to_probability(train_preds)
    test_preds  = _to_probability(test_preds)

    train_acc, _, _       = _metrics_float(y_train, train_preds, threshold)
    test_acc, mae, mse    = _metrics_float(y_test,  test_preds,  threshold)
    final_train_mse       = float(np.mean((train_preds - y_train) ** 2)) if len(y_train) else 0.0
    final_test_mse        = float(np.mean((test_preds  - y_test)  ** 2)) if len(y_test)  else 0.0
    epochs_completed      = int(getattr(perceptron, "epochs_run_", base["epochs"]))

    # Dataset size and class balance info
    n_train          = len(y_train)
    n_test           = len(y_test)
    fraud_rate_train = float(np.mean((y_train >= 0.5).astype(int)))
    fraud_rate_test  = float(np.mean((y_test >= 0.5).astype(int)))

    # ------------------------------------------------------------------
    # Binary classification metrics (fraud detection)
    # ------------------------------------------------------------------
    roc_auc_val = float("nan")
    best_thr = best_f1 = best_prec = best_rec = float("nan")
    f1_half = prec_half = rec_half = fpr_half = float("nan")
    train_roc_auc_val = float("nan")
    train_f1_half = train_prec_half = train_rec_half = train_fpr_half = float("nan")

    if n_test > 0:
        roc_auc_val = roc_auc(y_test, test_preds)

        # Best threshold on test (maximises F1)
        best      = find_best_threshold(y_test, test_preds, metric="f1")
        best_thr  = best["threshold"]
        best_f1   = best["f1"]
        best_prec = best["precision"]
        best_rec  = best["recall"]

        # Metrics at the configured threshold (comparable across model types)
        half      = metrics_at_threshold(y_test, test_preds, threshold=threshold)
        f1_half   = half["f1"]
        prec_half = half["precision"]
        rec_half  = half["recall"]
        fpr_half  = half["fpr"]

    if n_train > 0:
        train_roc_auc_val = roc_auc(y_train, train_preds)
        tr_half            = metrics_at_threshold(y_train, train_preds, threshold=threshold)
        train_f1_half      = tr_half["f1"]
        train_prec_half    = tr_half["precision"]
        train_rec_half     = tr_half["recall"]
        train_fpr_half     = tr_half["fpr"]

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
        "threshold":            threshold,
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
        "f1_at_threshold":           _r(f1_half),
        "precision_at_threshold":    _r(prec_half),
        "recall_at_threshold":       _r(rec_half),
        "fpr_at_threshold":          _r(fpr_half),
        # --- binary fraud metrics (train set — for overfitting diagnosis) ---
        "train_roc_auc":        _r(train_roc_auc_val),
        "train_f1_at_threshold":     _r(train_f1_half),
        "train_precision_at_threshold": _r(train_prec_half),
        "train_recall_at_threshold": _r(train_rec_half),
        "train_fpr_at_threshold":    _r(train_fpr_half),
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
    # Now includes test_mse alongside train_mse for over/undertraining plots.
    # ------------------------------------------------------------------
    train_mse_hist = list(getattr(perceptron, "train_mse_history_", []))
    test_mse_hist  = list(getattr(perceptron, "test_mse_history_",  []))

    curve_rows: list[dict] = []
    for epoch_idx, mse_val in enumerate(train_mse_hist):
        row: dict = {
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
            # test_mse is present when X_val was passed (i.e. no_split is False)
            "test_mse":   round(float(test_mse_hist[epoch_idx]), 8)
                          if epoch_idx < len(test_mse_hist) else None,
        }
        curve_rows.append(row)

    # ------------------------------------------------------------------
    # Fitting-curve row (one row per run, for over/underfitting vs LR plot)
    #
    # Columns:
    #   model_type, activation, lr, seed, epochs_completed,
    #   final_train_mse, final_test_mse,   <- MSE-based gap
    #   train_roc_auc, roc_auc,            <- AUC-based gap
    #   train_f1_at_half, f1_at_half,      <- F1-based gap
    #   gap_mse  = final_test_mse - final_train_mse   (>0 underfit on test / generalization error)
    #   gap_auc  = train_roc_auc - roc_auc            (>0 overfit in AUC)
    #   gap_f1   = train_f1_at_half - f1_at_half      (>0 overfit in F1)
    # ------------------------------------------------------------------
    gap_mse = (
        round(final_test_mse - final_train_mse, 8)
        if not (math.isnan(final_test_mse) or math.isnan(final_train_mse))
        else math.nan
    )
    gap_auc = (
        _r(train_roc_auc_val - roc_auc_val)
        if not (math.isnan(train_roc_auc_val) or math.isnan(roc_auc_val))
        else math.nan
    )
    gap_f1 = (
        _r(train_f1_half - f1_half)
        if not (math.isnan(train_f1_half) or math.isnan(f1_half))
        else math.nan
    )

    fitting_curve_row: dict = {
        "run_id":            run_id,
        "name":              base["name"],
        "data":              data_bn,
        "model_type":        model_type,
        "activation":        act_name,
        "lr":                base["lr"],
        "epochs":            base["epochs"],
        "epsilon":           base["epsilon"],
        "seed":              seed,
        "test_per":          tp_val,
        "no_split":          no_split,
        "epochs_completed":  epochs_completed,
        # --- absolute MSE ---
        "final_train_mse":   round(final_train_mse, 8),
        "final_test_mse":    round(final_test_mse,  8),
        # --- absolute AUC ---
        "train_roc_auc":     _r(train_roc_auc_val),
        "test_roc_auc":      _r(roc_auc_val),
        # --- absolute F1 ---
        "train_f1":          _r(train_f1_half),
        "test_f1":           _r(f1_half),
        # --- gaps (positive = model generalizes worse on test = tends toward overfitting) ---
        "gap_mse":           gap_mse,   # test_mse - train_mse
        "gap_auc":           gap_auc,   # train_auc - test_auc
        "gap_f1":            gap_f1,    # train_f1  - test_f1
    }

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

    print("Finished job")
    return {
        "summary":           summary_row,
        "cm_rows":           cm_rows,
        "curve_rows":        curve_rows,
        "fitting_curve_row": fitting_curve_row,
        "roc_rows":          roc_rows,
    }


# ---------------------------------------------------------------------------
# Job expansion
# ---------------------------------------------------------------------------

def _linear_job_signature(base: dict) -> tuple:
    """Keys that affect linear training; activation is ignored (linear has no activation)."""
    def _atom(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return float(v) if isinstance(v, float) else int(v)
        return str(v)

    parts = []
    for k in sorted(base.keys()):
        if k == "activation":
            continue
        parts.append((k, _atom(base[k])))
    return tuple(parts)


def expand_jobs(bases: list[dict], no_linear: bool = False) -> list[tuple[dict, str]]:
    """One non-linear job per base row; one linear job per distinct config ignoring ``activation``."""
    jobs: list[tuple[dict, str]] = []
    seen_linear: set[tuple] = set()
    for b in bases:
        if not no_linear:
            sig = _linear_job_signature(b)
            if sig not in seen_linear:
                seen_linear.add(sig)
                linear_base = dict(b)
                linear_base["activation"] = "identity"
                jobs.append((linear_base, "linear"))
        jobs.append((b, "non-linear"))
    return jobs


def _worker(job: tuple[dict, str], drop_cols : list[str]) -> dict:
    return run_single(job,drop_cols)


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
    p.add_argument(
        "--drop",
        type=str,
        nargs="*",
        default=[],
        help="Extra columns to drop (merged with optional top-level 'drop' array in the JSON config).",
    )
    p.add_argument(
        "--no-linear",
        action="store_true",
        help="Skip all linear perceptron jobs; only non-linear runs are executed.",
    )
    return p.parse_args()


def _write_result_csv(path: str, df_new: pd.DataFrame) -> None:
    """Write one results CSV (full replace for this run)."""
    df_new.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    drop_cols = _merge_drop_cols(cfg, args.drop)
    if drop_cols:
        print(f"Feature columns dropped before training: {drop_cols}")

    experiment_bases = experiment_bases_from_config(cfg)
    jobs = expand_jobs(experiment_bases, no_linear=args.no_linear)

    mode = "non-linear only" if args.no_linear else "linear + non-linear per row"
    print(
        f"Config loaded: {len(experiment_bases)} grid row(s) -> {len(jobs)} jobs ({mode})."
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
        out = [_worker(j, drop_cols) for j in jobs]
    else:
        import functools
        worker_fn = functools.partial(_worker, drop_cols=drop_cols)
        with mp.Pool(n_workers) as pool:
            out = pool.map(worker_fn, jobs)

    summary_rows        = [o["summary"]           for o in out]
    all_cm:     list[dict] = [row for o in out for row in o["cm_rows"]]
    all_curves: list[dict] = [row for o in out for row in o["curve_rows"]]
    all_fitting: list[dict] = [o["fitting_curve_row"] for o in out]
    all_roc:    list[dict] = [row for o in out for row in o["roc_rows"]]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for stale in (
        SUMMARY_CSV,
        CONFUSION_CSV,
        CURVES_CSV,
        ROC_CSV,
        FITTING_CURVES_CSV,
    ):
        if os.path.isfile(stale):
            os.remove(stale)

    df_s = pd.DataFrame(summary_rows)
    sort_cols = [c for c in ("name", "data", "model_type", "activation", "test_per", "lr", "seed")
                 if c in df_s.columns]
    if sort_cols:
        df_s = df_s.sort_values(sort_cols, kind="stable")

    _write_result_csv(SUMMARY_CSV, df_s)
    _write_result_csv(CONFUSION_CSV, pd.DataFrame(all_cm))
    _write_result_csv(FITTING_CURVES_CSV, pd.DataFrame(all_fitting))

    if all_curves:
        _write_result_csv(CURVES_CSV, pd.DataFrame(all_curves))
    if all_roc:
        _write_result_csv(ROC_CSV, pd.DataFrame(all_roc))

    print(f"Wrote {len(summary_rows)} summary rows -> {SUMMARY_CSV}")
    print(f"Wrote {len(all_cm)} confusion cells   -> {CONFUSION_CSV}")
    print(f"Wrote {len(all_fitting)} fitting rows    -> {FITTING_CURVES_CSV}")
    if all_curves:
        print(f"Wrote {len(all_curves)} curve points    -> {CURVES_CSV}")
    if all_roc:
        print(f"Wrote {len(all_roc)} ROC points        -> {ROC_CSV}")


if __name__ == "__main__":
    main()