from __future__ import annotations

import argparse
import math
import contextlib
import io
import itertools
import json
import os
import multiprocessing as mp
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
from utils.normalizers import standard_scale_apply, standard_scale_params
from utils.test_data_split import test_data_split

ROOT = _ROOT
DEFAULT_CONFIG = ROOT / "configs" / "linear_vs_nonlinear_fraud.json"

RESULTS_DIR = os.path.join("results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "linear_vs_nonlinear_summary.csv")
CONFUSION_CSV = os.path.join(RESULTS_DIR, "linear_vs_nonlinear_confusion_runs.csv")


def _resolve_data_path(p: str | Path) -> str:
    pp = Path(p)
    if not pp.is_absolute():
        pp = ROOT / pp
    return str(pp)


def _split_seeds(merged: dict) -> list[dict]:
    """Expand `seeds` list or normalize scalar `seed` into one dict per seed."""
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
    """
    Build a flat list of experiment dicts, each with scalar `seed` and no `seeds` key.

    Merges `base` with the cartesian product of `grid` (each grid value must be a list).
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


def _make_run_id(base: dict, model_type: str, seed: int) -> str:
    data_bn = os.path.basename(str(base["data"]))
    act = str(base.get("activation", "tanh")).replace(".", "_")
    if "test_per" in base and base["test_per"] is None:
        tp = "full"
    elif "test_per" in base:
        tp = str(base["test_per"]).replace(".", "_")
    else:
        tp = str(0.2).replace(".", "_")
    safe = (
        f"{base['name']}__{model_type}__s{seed}__tp{tp}__lr{base['lr']}__"
        f"ep{base['epochs']}__eps{base['epsilon']}__act_{act}__{data_bn.replace('.', '_')}"
    )
    return safe


def _metrics_float(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float,
) -> tuple[float, float, float]:
    matches = np.abs(y_pred - y_true) < tolerance
    acc = float(np.mean(matches)) if len(y_true) else 0.0
    mae = float(np.mean(np.abs(y_pred - y_true))) if len(y_true) else 0.0
    mse = float(np.mean((y_pred - y_true) ** 2)) if len(y_true) else 0.0
    return acc, mae, mse


def run_single(job: tuple[dict, str]) -> tuple[dict, list[dict]]:
    """
    job = (experiment_base_with_scalar_seed, model_type)
    model_type: 'linear' | 'non-linear'
    """
    base, model_type = job
    seed = int(base["seed"])
    data_path = _resolve_data_path(base["data"])
    label_col = str(base.get("label", "label"))
    tolerance = float(base["tolerance"])
    if "test_per" in base:
        tp_raw = base["test_per"]
    else:
        tp_raw = 0.2
    # JSON `null` en el grid: sin train/test, todo el set (evaluación en el mismo data que el fit)
    no_split = bool(base.get("no_split", False)) or tp_raw is None

    X, y = load_data(data_path, label_col)
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    if no_split:
        if base.get("normalize") == "standard":
            mean, std = standard_scale_params(X)
            X = standard_scale_apply(X, mean, std)
        X_train = X_test = X
        y_train = y_test = y
    else:
        X_train, X_test, y_train, y_test = test_data_split(
            X, y, test_size=float(tp_raw), random_state=seed
        )
        if base.get("normalize") == "standard":
            mean, std = standard_scale_params(X_train)
            X_train = standard_scale_apply(X_train, mean, std)
            X_test = standard_scale_apply(X_test, mean, std)

    type_p = "non-linear" if model_type == "non-linear" else "linear"
    perceptron = build_perceptron(
        type_p,
        base["lr"],
        base["epochs"],
        base["epsilon"],
        seed,
        base.get("activation", "tanh"),
        base.get("beta", 1.0),
        None,
        "random",
        "online",
        1,
        "sgd",
    )

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        perceptron.fit(X_train, y_train)
    elapsed_seconds = time.perf_counter() - t0

    train_preds = perceptron.predict(X_train)
    test_preds = perceptron.predict(X_test)

    train_acc, _, _ = _metrics_float(y_train, train_preds, tolerance)
    test_acc, mae, mse = _metrics_float(y_test, test_preds, tolerance)
    final_train_mse = (
        float(np.mean((train_preds - y_train) ** 2)) if len(y_train) else 0.0
    )
    epochs_completed = int(getattr(perceptron, "epochs_run_", base["epochs"]))

    cm_mode = resolve_confusion_mode(
        y_train,
        str(base.get("confusion_mode", "auto")),
        discrete_max_unique=15,
    )
    n_bins = int(base.get("confusion_bins", 10))

    ref = np.concatenate([y_train, y_test])
    if cm_mode == "discrete":
        cm, labels = confusion_matrix_discrete(y_test, test_preds, ref)
        meta = {"mode": "discrete", "labels": [float(x) for x in labels]}
    else:
        lo, hi = bin_range_from_train(y_train, y_test)
        cm, edges = confusion_matrix_binned(y_test, test_preds, n_bins, lo, hi)
        meta = {
            "mode": "binned",
            "n_bins": n_bins,
            "edges": [float(x) for x in edges],
        }

    run_id = _make_run_id(base, model_type, seed)
    data_bn = os.path.basename(data_path)

    summary_row = {
        "run_id": run_id,
        "name": base["name"],
        "data": data_bn,
        "model_type": model_type,
        "lr": base["lr"],
        "epochs": base["epochs"],
        "epsilon": base["epsilon"],
        "tolerance": tolerance,
        "activation": base.get("activation", "tanh"),
        "beta": base.get("beta", 1.0),
        "test_per": (float(tp_raw) if tp_raw is not None else math.nan),
        "normalize": base.get("normalize", "none"),
        "no_split": no_split,
        "seed": seed,
        "confusion_mode": cm_mode,
        "confusion_bins_config": n_bins,
        "train_acc": round(train_acc, 6),
        "test_acc": round(test_acc, 6),
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "final_train_mse": round(final_train_mse, 6),
        "epochs_completed": epochs_completed,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "confusion_meta_json": json.dumps(meta, ensure_ascii=False),
    }

    cm_rows: list[dict] = []
    for row in flatten_confusion_rows(run_id, cm):
        row["name"] = base["name"]
        row["data"] = data_bn
        row["model_type"] = model_type
        row["seed"] = seed
        row["confusion_mode"] = cm_mode
        cm_rows.append(row)

    return summary_row, cm_rows


def expand_jobs(bases: list[dict]) -> list[tuple[dict, str]]:
    jobs: list[tuple[dict, str]] = []
    for b in bases:
        for mt in ("linear", "non-linear"):
            jobs.append((b, mt))
    return jobs


def _worker(job: tuple[dict, str]):
    return run_single(job)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run linear vs non-linear perceptron batches from a JSON config"
        )
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
        help="Print job counts and exit without training",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel worker processes (default: 1; serial). Capped by job count.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    EXPERIMENT_BASES = experiment_bases_from_config(cfg)
    jobs = expand_jobs(EXPERIMENT_BASES)

    print(
        f"Loaded {len(EXPERIMENT_BASES)} grid row(s) -> {len(jobs)} jobs "
        f"(linear + non-linear for each row)."
    )

    if args.dry_run:
        print("Dry run: no training performed.")
        return

    if not jobs:
        print("No jobs to run (empty base+grid).")
        return

    n_workers = min(len(jobs), args.workers)
    print(f"Running {len(jobs)} jobs on {n_workers} worker(s)...")

    if n_workers <= 1:
        out = [_worker(j) for j in jobs]
    else:
        with mp.Pool(n_workers) as pool:
            out = pool.map(_worker, jobs)

    summary_rows = [o[0] for o in out]
    all_cm: list[dict] = []
    for o in out:
        all_cm.extend(o[1])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_s = pd.DataFrame(summary_rows)
    _sort_summary = [
        c
        for c in (
            "name",
            "data",
            "model_type",
            "activation",
            "test_per",
            "lr",
            "seed",
        )
        if c in df_s.columns
    ]
    if _sort_summary:
        df_s = df_s.sort_values(_sort_summary, kind="stable")
    df_c = pd.DataFrame(all_cm)

    if os.path.isfile(SUMMARY_CSV):
        prev = pd.read_csv(SUMMARY_CSV)
        df_s = pd.concat([prev, df_s], ignore_index=True, sort=False)
    df_s.to_csv(SUMMARY_CSV, index=False)

    if os.path.isfile(CONFUSION_CSV):
        prev_c = pd.read_csv(CONFUSION_CSV)
        df_c = pd.concat([prev_c, df_c], ignore_index=True, sort=False)
    df_c.to_csv(CONFUSION_CSV, index=False)

    print(f"Appended {len(summary_rows)} rows to {SUMMARY_CSV}")
    print(f"Appended {len(all_cm)} confusion cells to {CONFUSION_CSV}")


if __name__ == "__main__":
    main()
