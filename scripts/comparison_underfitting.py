"""
run_comparison.py
=================
Runs a linear vs. non-linear perceptron comparison from a JSON config file.
Trains on the FULL dataset — no train/val split. This is intentional:
the goal here is to study learning capacity and underfitting, not generalisation.

Usage:
    python run_comparison.py --config linear_vs_nonlinear_fraud.json

Output CSVs (written to results/):
    <name>_linear_curves.csv     — one row per (seed, lr, epoch)
    <name>_nonlinear_curves.csv  — one row per (seed, lr, activation, epoch)

Columns: model, seed, lr, [activation,] epoch, train_mse, train_bce

Label scaling:
    tanh output range is (-1, 1), so when activation='tanh' the labels are
    scaled from {0,1} → {-1,1} before training.  Predictions are mapped back
    to [0,1] before MSE and BCE are computed so all metrics stay comparable
    across activations.  Linear and other activations always train on {0,1}.

To use different learning rates per model type, add to your config grid:
    "lr_linear":    [0.001, 0.01]
    "lr_nonlinear": [0.0001, 0.001, 0.01]
If only "lr" is present it is shared by both.
"""

import argparse
import csv
import json
import sys
import time
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
    """
    Binary cross-entropy. Clips predictions to (eps, 1-eps) so log(0) never fires.
    For the linear model, outputs outside [0,1] get clamped — BCE will be large,
    which is precisely the point of keeping it in the comparison.
    """
    eps = 1e-12
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def normalize(X: np.ndarray, method: str) -> np.ndarray:
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        return X
    return scaler.fit_transform(X)


def label_scale_for_activation(y: np.ndarray, activation: str) -> np.ndarray:
    """
    tanh saturates at ±1, so targets must live in (-1, 1) not {0, 1}.
    Map  0 → -1  and  1 → +1  :   y_scaled = 2*y - 1
    All other activations expect targets in [0, 1] — return y unchanged.
    """
    if activation == "tanh":
        return 2.0 * y - 1.0
    return y


def tanh_pred_to_prob(y_pred: np.ndarray) -> np.ndarray:
    """
    Invert the label scaling so predictions land back in [0, 1].
    p = (pred + 1) / 2
    This lets MSE and BCE be computed on the original probability scale
    regardless of which activation was used during training.
    """
    return (y_pred + 1.0) / 2.0


def load_data(path: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if label not in df.columns:
        raise ValueError(
            f"Label column '{label}' not found. "
            f"Available: {list(df.columns)}"
        )
    y = df[label].values.astype(float)
    X = df.drop(columns=[label]).select_dtypes(include=[np.number]).values.astype(float)
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
    Replicates the SGD loop from both perceptron classes and records
    train MSE and BCE after every epoch, without touching the originals.

    y is expected in {0, 1}.  When activation='tanh', labels are scaled to
    {-1, +1} for training and predictions are mapped back to [0, 1] before
    metrics are computed — so MSE and BCE are always on the probability scale.

    Returns one dict per epoch actually run.
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

        # convergence check on scaled-label SSE — mirrors original perceptron behaviour
        sse = float(np.sum((y_train - raw_pred) ** 2))

        # map predictions back to [0,1] for comparable, activation-agnostic metrics
        prob_pred = tanh_pred_to_prob(raw_pred) if activation == "tanh" else raw_pred

        tr_mse = mse(y, prob_pred)
        tr_bce = bce_from_predictions(y, prob_pred)

        rows.append({
            "epoch":     epoch + 1,
            "train_mse": tr_mse,
            "train_bce": tr_bce,
        })

        if sse < epsilon:
            print(f"      ✓ converged at epoch {epoch + 1}  (SSE={sse:.4f})")
            break


    return rows


# ── runner ───────────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
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
    X, y = load_data(base["data"], base["label"])
    X    = normalize(X, norm)
    print(
        f"  {X.shape[0]} samples | {X.shape[1]} features | "
        f"label='{base['label']}' | fraud rate={y.mean()*100:.2f}%\n"
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    name = base.get("name", "experiment")

    linear_csv    = out_dir / f"{name}_linear_curves.csv"
    nonlinear_csv = out_dir / f"{name}_nonlinear_curves.csv"

    total_lin = len(seeds) * len(lr_linear)
    total_nln = len(seeds) * len(lr_nonlinear) * len(activations)

    print("=" * 60)
    print(f"  Training on FULL dataset (no val split)")
    print(f"  LINEAR experiments    : {total_lin}")
    print(f"  NON-LINEAR experiments: {total_nln}")
    print("=" * 60)

    # ── LINEAR ──────────────────────────────────────────────────────────────
    print("\n── Linear perceptron ───────────────────────────────────────────")
    lin_fields = ["model", "seed", "lr", "epoch", "train_mse", "train_bce"]

    with open(linear_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=lin_fields)
        writer.writeheader()

        for done, (seed, lr) in enumerate(product(seeds, lr_linear), start=1):
            print(f"  [{done}/{total_lin}] seed={seed}  lr={lr}")

            model = SimpleLinearPerceptron(
                learning_rate=lr,
                epochs=epochs,
                epsilon=epsilon,
                seed=seed,
            )

            t0   = time.time()
            rows = fit_and_record(model, X, y, epochs, epsilon, activation="logistic")
            print(f"      epochs={len(rows)}  time={time.time()-t0:.1f}s")

            for r in rows:
                writer.writerow({
                    "model":     "linear",
                    "seed":      seed,
                    "lr":        lr,
                    "epoch":     r["epoch"],
                    "train_mse": r["train_mse"],
                    "train_bce": r["train_bce"],
                })

    print(f"\n  ✓ Saved → {linear_csv}")

    # ── NON-LINEAR ───────────────────────────────────────────────────────────
    print("\n── Non-linear perceptron ───────────────────────────────────────")
    nln_fields = ["model", "seed", "lr", "activation", "epoch", "train_mse", "train_bce"]

    with open(nonlinear_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=nln_fields)
        writer.writeheader()

        for done, (seed, lr, activation) in enumerate(
            product(seeds, lr_nonlinear, activations), start=1
        ):
            print(f"  [{done}/{total_nln}] seed={seed}  lr={lr}  act={activation}")

            model = SimpleNonLinearPerceptron(
                learning_rate=lr,
                epochs=epochs,
                epsilon=epsilon,
                seed=seed,
                activation=activation,
                beta=beta,
            )

            t0   = time.time()
            rows = fit_and_record(model, X, y, epochs, epsilon, activation=activation)
            print(f"      epochs={len(rows)}  time={time.time()-t0:.1f}s")

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

    print(f"\n  ✓ Saved → {nonlinear_csv}")
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
    args = parser.parse_args()
    run(args.config)