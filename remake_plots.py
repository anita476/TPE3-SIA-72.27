"""
Reload saved models from results/models/ and regenerate all plots
without retraining.

Usage:
    python3 remake_plots.py                        # all models in results/models/
    python3 remake_plots.py --models results/models/xavier_s_1.npz xavier_s_2.npz
"""
import os
import re
import glob
import argparse
import numpy as np

from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from datasets.digit_dataset_loader import load_digits, encode_one_hot
from utils.test_data_split import stratified_split
from utils.metrics import compute_metrics, epochs_to_threshold
from utils.visualization import (
    print_summary, print_perclass_summary,
    plot_loss_curves, plot_accuracy_bars, plot_convergence,
    plot_confusion_matrices, plot_perclass_heatmap,
)

MODELS_DIR = "results/models"
TRAIN_PATH = "datasets/digits.csv"
SEED       = 1   # must match experiment_runner_digits.py


def _name_from_path(path):
    """'results/models/xavier_s_1.npz' → 'xavier s=1'"""
    stem = os.path.splitext(os.path.basename(path))[0]
    # reverse the safe_name substitution: underscores back to spaces,
    # then fix 's_N' → 's=N'
    name = stem.replace("_", " ")
    name = re.sub(r"\bs (\d+)$", r"s=\1", name)
    return name


def _build_result(path, X_train, train_labels, X_val, val_labels, y_train, y_val):
    name = _name_from_path(path)
    print(f"  Loading {name} ← {path}")
    mlp = MultiLayerPerceptron.load(path)

    train_preds = np.argmax(mlp.predict(X_train), axis=1)
    val_preds   = np.argmax(mlp.predict(X_val),   axis=1)
    train_acc   = float(np.mean(train_preds == train_labels))
    val_acc     = float(np.mean(val_preds   == val_labels))

    train_metrics = compute_metrics(train_labels, train_preds)
    val_metrics   = compute_metrics(val_labels,   val_preds)

    # Epoch-level history (available only if model was saved after fit())
    has_history   = hasattr(mlp, "errors_") and mlp.errors_
    train_loss    = [e / len(X_train) for e in mlp.errors_]       if has_history else []
    val_loss      = [e / len(X_val)   for e in mlp.val_errors_]   if has_history else []
    val_accs_ep   = mlp.val_accuracies_                            if has_history else []

    best_epoch    = int(np.argmax(val_accs_ep)) + 1 if val_accs_ep else None
    best_val_acc  = float(np.max(val_accs_ep))      if val_accs_ep else val_acc

    return {
        "name":               name,
        "config":             {},
        "train_acc":          train_acc,
        "test_acc":           val_acc,
        "best_test_acc":      best_val_acc,
        "best_epoch":         best_epoch,
        "gap":                train_acc - val_acc,
        "epochs_to_80":       epochs_to_threshold(val_accs_ep, 0.80),
        "epochs_to_85":       epochs_to_threshold(val_accs_ep, 0.85),
        "final_train_loss":   train_loss[-1] if train_loss else None,
        "final_test_loss":    val_loss[-1]   if val_loss   else None,
        "train_loss":         train_loss,
        "test_loss":          val_loss,
        "test_acc_per_epoch": val_accs_ep,
        "train_metrics":      train_metrics,
        "test_metrics":       val_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Regenerate plots from saved models")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific .npz files to load. Default: all in results/models/")
    args = parser.parse_args()

    paths = args.models or sorted(glob.glob(os.path.join(MODELS_DIR, "*.npz")))
    if not paths:
        print(f"No models found in {MODELS_DIR}/. Train first with experiment_runner_digits.py.")
        return
    print(f"Found {len(paths)} model(s).")

    print("Loading data and recreating val split...")
    X_all, all_labels = load_digits(TRAIN_PATH)
    X_train, X_val, train_labels, val_labels = stratified_split(
        X_all, all_labels, val_size=0.2, random_state=SEED
    )
    y_train = encode_one_hot(train_labels, 10)
    y_val   = encode_one_hot(val_labels,   10)
    print(f"  train: {len(X_train)}  val: {len(X_val)}")

    results = [_build_result(p, X_train, train_labels, X_val, val_labels, y_train, y_val)
               for p in paths]
    results.sort(key=lambda r: r["name"])

    print("\n=== SUMMARY ===")
    print_summary(results)
    print_perclass_summary(results)
    plot_accuracy_bars(results)
    plot_confusion_matrices(results)
    plot_perclass_heatmap(results)

    if any(r["train_loss"] for r in results):
        plot_loss_curves(results)
        plot_convergence(results)
    else:
        print("(No training history in saved models — loss/convergence plots skipped)")


if __name__ == "__main__":
    main()
