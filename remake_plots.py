"""
Reload saved models from results/models/ and regenerate all plots
without retraining.

Usage:
    python3 remake_plots.py                          # all configs in results/models/
    python3 remake_plots.py --configs xavier random  # specific config folders
"""
import os
import glob
import argparse
import numpy as np

from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from datasets.digit_dataset_loader import load_digits
from utils.test_data_split import stratified_split
from utils.metrics import compute_metrics
from utils.visualization import (
    print_summary,
    plot_accuracy_bars, plot_val_accuracy, plot_overfitting_diagnosis,
    plot_perclass_heatmap,
)

MODELS_DIR = "results/models"
TRAIN_PATH = "data/digits.csv"
SEED       = 1   # must match experiment_runner_digits.py


def _build_result(path, name, X_train, train_labels, X_val, val_labels):
    print(f"  Loading {name}/{os.path.basename(path)} ← {path}")
    mlp = MultiLayerPerceptron.load(path)

    train_preds = np.argmax(mlp.predict(X_train), axis=1)
    val_preds   = np.argmax(mlp.predict(X_val),   axis=1)
    train_acc   = float(np.mean(train_preds == train_labels))
    val_acc     = float(np.mean(val_preds   == val_labels))

    train_metrics = compute_metrics(train_labels, train_preds)
    val_metrics   = compute_metrics(val_labels,   val_preds)

    has_history  = hasattr(mlp, "errors_") and mlp.errors_
    train_loss        = [e / len(X_train) for e in mlp.errors_]     if has_history else []
    val_loss          = [e / len(X_val)   for e in mlp.val_errors_] if has_history else []
    val_accs_ep       = mlp.val_accuracies_                          if has_history else []
    train_accs_ep     = mlp.train_accuracies_                        if has_history else []

    best_val_acc = float(np.max(val_accs_ep)) if val_accs_ep else val_acc

    return {
        "name":               name,
        "config":             {"layers": list(mlp.layers)},
        "train_acc":          train_acc,
        "best_val_acc":       best_val_acc,
        "macro_f1":           val_metrics["macro_f1"],
        "train_loss":         train_loss,
        "val_loss":           val_loss,
        "val_acc":            val_accs_ep,
        "train_acc_per_epoch": train_accs_ep,
        "train_metrics":      train_metrics,
        "test_metrics":       val_metrics,
        "config_id":          name,  # folder name is the config identity
    }


def main():
    parser = argparse.ArgumentParser(description="Regenerate plots from saved models")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config folder names to load (e.g. xavier random). Default: all.")
    args = parser.parse_args()

    folders = args.configs or (
        sorted(os.listdir(MODELS_DIR)) if os.path.isdir(MODELS_DIR) else []
    )
    pairs = []
    for folder in folders:
        folder_path = os.path.join(MODELS_DIR, folder)
        if os.path.isdir(folder_path):
            for f in sorted(glob.glob(os.path.join(folder_path, "*.npz"))):
                pairs.append((folder, f))
        else:
            print(f"Warning: config folder not found: {folder_path}")

    if not pairs:
        print(f"No models found in {MODELS_DIR}/. Train first with experiment_runner_digits.py.")
        return
    print(f"Found {len(pairs)} model file(s) across {len({p[0] for p in pairs})} config(s).")

    print("Loading data and recreating val split...")
    X_all, all_labels = load_digits(TRAIN_PATH)
    X_train, X_val, train_labels, val_labels = stratified_split(
        X_all, all_labels, val_size=0.2, random_state=SEED
    )
    print(f"  train: {len(X_train)}  val: {len(X_val)}")

    results = [_build_result(p, name, X_train, train_labels, X_val, val_labels)
               for name, p in pairs]
    results.sort(key=lambda r: r["name"])

    print("\n=== SUMMARY ===")
    print_summary(results)
    plot_accuracy_bars(results)
    plot_perclass_heatmap(results)

    if any(r["val_acc"] for r in results):
        plot_val_accuracy(results)
        plot_overfitting_diagnosis(results)
    else:
        print("(No training history in saved models — curve plots skipped)")


if __name__ == "__main__":
    main()
