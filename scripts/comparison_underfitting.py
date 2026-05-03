"""
run_comparison.py
=================
Runs a linear vs. non-linear perceptron comparison from a JSON config file.
Trains on the FULL dataset — no train/val split. This is intentional:
the goal here is to study learning capacity and underfitting, not generalisation.

Usage:
    python run_comparison.py --config linear_vs_nonlinear_fraud.json

Output CSVs (written to results/):
    <name>_linear_curves.csv       — one row per (seed, lr, epoch)
    <name>_nonlinear_curves.csv    — one row per (seed, lr, activation, epoch)
    <name>_linear_recall.csv       — one row per (seed, lr, epoch)  [recall]
    <name>_nonlinear_recall.csv    — one row per (seed, lr, activation, epoch) [recall]

Columns MSE/BCE CSVs : model, seed, lr, [activation,] epoch, train_mse, train_bce
Columns recall CSVs  : model, seed, lr, [activation,] epoch, train_recall

Label scaling:
    tanh output range is (-1, 1), so when activation='tanh' the labels are
    scaled from {0,1} → {-1,1} before training.  Predictions are mapped back
    to [0,1] before MSE, BCE and recall are computed so all metrics stay
    comparable across activations.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perceptrons.SimpleLinearPerceptron import SimpleLinearPerceptron
from perceptrons.SimpleNonLinearPerceptron import SimpleNonLinearPerceptron


# ── helpers ─────────────────────────────────────────────────────────────────

def bce_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def recall_score(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Recall for the positive class. Returns 0.0 if no positive samples exist."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def normalize(X: np.ndarray, method: str) -> np.ndarray:
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        return X
    return scaler.fit_transform(X)


def label_scale_for_activation(y: np.ndarray, activation: str) -> np.ndarray:
    if activation == "tanh":
        return 2.0 * y - 1.0
    return y


def tanh_pred_to_prob(y_pred: np.ndarray) -> np.ndarray:
    return (y_pred + 1.0) / 2.0


def load_data(path: str, label: str, drop_cols: list[str] = []) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if label not in df.columns:
        raise ValueError(
            f"Label column '{label}' not found. "
            f"Available: {list(df.columns)}"
        )
    cols_to_drop = [label] + [c for c in drop_cols if c in df.columns]
    y = df[label].values.astype(float)
    X = (
        df.drop(columns=cols_to_drop)
        .select_dtypes(include=[np.number])
        .values.astype(float)
    )
    return X, y


# ── training loop ────────────────────────────────────────────────────────────

def fit_and_record(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    epsilon: float,
    activation: str = "logistic",
) -> list[dict]:
    """
    Trains the model epoch by epoch, recording train MSE, BCE and recall
    after every epoch.

    y is always expected in {0, 1}.
    For tanh, labels are scaled to {-1, +1} internally; predictions are mapped
    back to [0, 1] before computing any metric.
    """
    y_train = label_scale_for_activation(y, activation)
    n       = len(y_train)
    model._initialize_parameters(X.shape[1])

    rows = []
    for epoch in range(epochs):
        indices = model.rng.permutation(n)
        for i in indices:
            x_i, y_i = X[i], y_train[i]
            pred  = model._predict_single(x_i)
            error = y_i - pred

            if hasattr(model, "g_prime"):        # non-linear
                delta = error * model.g_prime(pred, model.beta)
            else:                                 # linear
                delta = error

            model.weights += model.learning_rate * delta * x_i
            model.bias    += model.learning_rate * delta

        raw_pred = model.predict(X)

        # convergence check on scaled-label SSE
        sse = float(np.sum((y_train - raw_pred) ** 2))

        # map back to [0,1] for comparable metrics
        prob_pred = tanh_pred_to_prob(raw_pred) if activation == "tanh" else raw_pred

        tr_mse    = mse(y, prob_pred)
        tr_bce    = bce_from_predictions(y, prob_pred)
        tr_recall = recall_score(y, prob_pred)

        rows.append({
            "epoch":        epoch + 1,
            "train_mse":    tr_mse,
            "train_bce":    tr_bce,
            "train_recall": tr_recall,
        })

        if sse < epsilon:
            print(f"      ✓ converged at epoch {epoch + 1}  (SSE={sse:.4f})  recall={tr_recall:.4f}")
            break

    return rows


# ── parallel worker functions ────────────────────────────────────────────────

def _run_linear_job(args_tuple):
    seed, lr, X, y, epochs, epsilon = args_tuple
    model = SimpleLinearPerceptron(
        learning_rate=lr,
        epochs=epochs,
        epsilon=epsilon,
        seed=seed,
    )
    t0      = time.time()
    rows    = fit_and_record(model, X, y, epochs, epsilon, activation="logistic")
    elapsed = time.time() - t0
    return seed, lr, rows, elapsed


def _run_nonlinear_job(args_tuple):
    seed, lr, activation, beta, X, y, epochs, epsilon = args_tuple
    model = SimpleNonLinearPerceptron(
        learning_rate=lr,
        epochs=epochs,
        epsilon=epsilon,
        seed=seed,
        activation=activation,
        beta=beta,
    )
    t0      = time.time()
    rows    = fit_and_record(model, X, y, epochs, epsilon, activation=activation)
    elapsed = time.time() - t0
    return seed, lr, activation, rows, elapsed


# ── main ─────────────────────────────────────────────────────────────────────

def run(config_path: str, outpath: str, workers: int, drop_cols: list[str] = []) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    base = cfg["base"]
    grid = cfg["grid"]

    lr_linear    = grid.get("lr_linear",    grid["lr"])
    lr_nonlinear = grid.get("lr_nonlinear", grid["lr"])
    seeds        = grid.get("seed",         [1])
    activations  = grid.get("activation",   ["logistic"])

    epochs  = base["epochs"]
    epsilon = base["epsilon"]
    norm    = base.get("normalize", "standard")
    beta    = base.get("beta", 1.0)

    print(f"Loading data: {base['data']}")
    X, y = load_data(base["data"], base["label"], drop_cols)
    X    = normalize(X, norm)
    print(
        f"  {X.shape[0]} samples | {X.shape[1]} features | "
        f"label='{base['label']}' | fraud rate={y.mean()*100:.2f}%\n"
    )

    out_dir = Path(outpath)
    out_dir.mkdir(exist_ok=True)
    name = base.get("name", "experiment")

    # Output file paths
    linear_csv         = out_dir / f"{name}_linear_curves.csv"
    nonlinear_csv      = out_dir / f"{name}_nonlinear_curves.csv"
    linear_recall_csv  = out_dir / f"{name}_linear_recall.csv"
    nonlinear_recall_csv = out_dir / f"{name}_nonlinear_recall.csv"

    total_lin = len(seeds) * len(lr_linear)
    total_nln = len(seeds) * len(lr_nonlinear) * len(activations)
    max_workers = min(workers, os.cpu_count() or 1)

    print("=" * 60)
    print(f"  Training on FULL dataset (no val split)")
    print(f"  LINEAR experiments    : {total_lin}")
    print(f"  NON-LINEAR experiments: {total_nln}")
    print(f"  Workers               : {max_workers}")
    print("=" * 60)

    # ── LINEAR ──────────────────────────────────────────────────────────────
    print("\n── Linear perceptron ───────────────────────────────────────────")
    lin_fields        = ["model", "seed", "lr", "epoch", "train_mse", "train_bce"]
    lin_recall_fields = ["model", "seed", "lr", "epoch", "train_recall"]
    lin_jobs = [
        (seed, lr, X, y, epochs, epsilon)
        for seed, lr in product(seeds, lr_linear)
    ]

    lin_results: dict[tuple, list] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_linear_job, job): job for job in lin_jobs}
        done = 0
        for future in as_completed(futures):
            seed, lr, rows, elapsed = future.result()
            done += 1
            final_recall = rows[-1]["train_recall"]
            print(
                f"  [{done}/{total_lin}] seed={seed}  lr={lr}"
                f"  epochs={len(rows)}  recall={final_recall:.4f}  time={elapsed:.1f}s"
            )
            lin_results[(seed, lr)] = rows

    # Write MSE/BCE CSV
    with open(linear_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=lin_fields)
        writer.writeheader()
        for (seed, lr), rows in lin_results.items():
            for r in rows:
                writer.writerow({
                    "model":     "linear",
                    "seed":      seed,
                    "lr":        lr,
                    "epoch":     r["epoch"],
                    "train_mse": r["train_mse"],
                    "train_bce": r["train_bce"],
                })
    print(f"\n  ✓ Saved MSE/BCE → {linear_csv}")

    # Write recall CSV
    with open(linear_recall_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=lin_recall_fields)
        writer.writeheader()
        for (seed, lr), rows in lin_results.items():
            for r in rows:
                writer.writerow({
                    "model":        "linear",
                    "seed":         seed,
                    "lr":           lr,
                    "epoch":        r["epoch"],
                    "train_recall": r["train_recall"],
                })
    print(f"  ✓ Saved recall   → {linear_recall_csv}")

    # ── NON-LINEAR ───────────────────────────────────────────────────────────
    print("\n── Non-linear perceptron ───────────────────────────────────────")
    nln_fields        = ["model", "seed", "lr", "activation", "epoch", "train_mse", "train_bce"]
    nln_recall_fields = ["model", "seed", "lr", "activation", "epoch", "train_recall"]
    nln_jobs = [
        (seed, lr, activation, beta, X, y, epochs, epsilon)
        for seed, lr, activation in product(seeds, lr_nonlinear, activations)
    ]

    nln_results: dict[tuple, list] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_nonlinear_job, job): job for job in nln_jobs}
        done = 0
        for future in as_completed(futures):
            seed, lr, activation, rows, elapsed = future.result()
            done += 1
            final_recall = rows[-1]["train_recall"]
            print(
                f"  [{done}/{total_nln}] seed={seed}  lr={lr}  act={activation}"
                f"  epochs={len(rows)}  recall={final_recall:.4f}  time={elapsed:.1f}s"
            )
            nln_results[(seed, lr, activation)] = rows

    # Write MSE/BCE CSV
    with open(nonlinear_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=nln_fields)
        writer.writeheader()
        for (seed, lr, activation), rows in nln_results.items():
            for r in rows:
                writer.writerow({
                    "model":      "nonlinear",
                    "seed":       seed,
                    "lr":         lr,
                    "activation": activation,
                    "epoch":      r["epoch"],
                    "train_mse":  r["train_mse"],
                    "train_bce":  r["train_bce"],
                })
    print(f"\n  ✓ Saved MSE/BCE → {nonlinear_csv}")

    # Write recall CSV
    with open(nonlinear_recall_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=nln_recall_fields)
        writer.writeheader()
        for (seed, lr, activation), rows in nln_results.items():
            for r in rows:
                writer.writerow({
                    "model":        "nonlinear",
                    "seed":         seed,
                    "lr":           lr,
                    "activation":   activation,
                    "epoch":        r["epoch"],
                    "train_recall": r["train_recall"],
                })
    print(f"  ✓ Saved recall   → {nonlinear_recall_csv}")
    print("\nAll done.")


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear vs non-linear perceptron comparison (full-dataset training)."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to JSON config (e.g. linear_vs_nonlinear_fraud.json)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1)"
    )
    parser.add_argument(
        "--drop", type=str, nargs="*", default=[],
        help="Column names to drop from features (e.g. --drop col1 col2)"
    )
    parser.add_argument("--outpath", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    run(args.config, outpath=args.outpath, workers=args.workers, drop_cols=args.drop)