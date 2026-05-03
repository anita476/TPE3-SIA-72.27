import argparse

import numpy as np
import pandas as pd

from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from perceptrons.SimpleLinearPerceptron import SimpleLinearPerceptron
from perceptrons.SimpleNonLinearPerceptron import SimpleNonLinearPerceptron
from perceptrons.SimpleStepPerceptron import SimpleStepPerceptron
from utils.normalizers import standard_scale_apply, standard_scale_params
from utils.test_data_split import test_data_split


def build_perceptron(type, lr, epochs, epsilon, seed, activation='tanh', beta=1.0, layers=None, initializer="random", training_mode="online", batch_size=1, optimizer="sgd"):
    if type == "simple-step":
        return SimpleStepPerceptron(lr, epochs, seed)
    elif type == "linear":
        return SimpleLinearPerceptron(lr, epochs, epsilon, seed)
    elif type == "non-linear":
        return SimpleNonLinearPerceptron(lr, epochs, epsilon, seed, activation, beta)
    elif type == "multilayer":
        return MultiLayerPerceptron(layers, lr, epochs, epsilon, seed, beta, activation, initializer, training_mode, batch_size, optimizer)
    raise Exception("Unknown perceptron type")


def parse_arguments() -> argparse.Namespace:
    arguments = argparse.ArgumentParser(
        description="Run a perceptron training + evaluation cycle"
    )
    arguments.add_argument("--lr", type=float, default=0.01, help="Perceptron learning rate")
    arguments.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    arguments.add_argument("--data", type=str, required=True, help="Path to CSV file. Required")
    arguments.add_argument("--type_p", type=str, required=True, choices=["simple-step", "linear", "non-linear", "multilayer"], help="Perceptron type to use")
    arguments.add_argument("--epsilon", type=float, default=0.001, help="Perceptron epsilon value (threshold). Default is 0.001")
    arguments.add_argument("--tolerance", type=float, default=0.5, help="Tolerance for interpreting predictions as correct (for linear and non-linear perceptrons). Default is 0.5")
    arguments.add_argument("--test_per", type=float, default=0.2, help="Fraction for test split in decimals (for example 0.2 = 20%%)")
    arguments.add_argument("--seed", type=int, default=1, help="Random seed")
    arguments.add_argument("--activation", type=str, default="tanh", choices=["tanh", "logistic", "relu"], help="Activation function for non-linear and multilayer perceptrons. Default is tanh")
    arguments.add_argument("--beta", type=float, default=1.0, help="Beta scaling parameter for non-linear perceptron. Default is 1.0")
    arguments.add_argument("--no_split", action="store_true", help="Skip train/test split, evaluate on full dataset")
    arguments.add_argument("--layers", type=int, nargs="+", default=[2, 2, 1], help="Layer sizes for multilayer perceptron (e.g. --layers 2 2 1)")
    arguments.add_argument("--initializer", type=str, default="random", choices=["random", "xavier", "xavier_n"], help="Weight initializer for multilayer perceptron. Default is random")
    arguments.add_argument("--training_mode", type=str, default="online", choices=["online", "minibatch"], help="Weight update mode for multilayer perceptron. Default is online")
    arguments.add_argument("--batch_size", type=int, default=1, help="Mini-batch size for multilayer perceptron when --training_mode minibatch, or for --optimizer sgd")
    arguments.add_argument("--optimizer", type=str, default="sgd", choices=["gd", "sgd", "rmsprop", "adam"], help="Optimizer for multilayer perceptron. gd uses the full dataset in one update per epoch; sgd uses online or mini-batch updates. Default is sgd")
    arguments.add_argument("--label", type=str, default="label", help="Label column name to drop in data loading")
    arguments.add_argument("--drop", type=str, nargs="*", default=[], help="Column names to drop from features before training (e.g. --drop col1 col2)")
    arguments.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "standard"],
        help="Feature scaling: 'standard' = zero mean / unit variance (fit on train only when split). "
             "Use on real-world CSVs with mixed feature scales to avoid NaN weights.",
    )
    return arguments.parse_args()


def load_data(path: str, label_col: str, drop_cols: list[str] = []):
    df = pd.read_csv(path)
    cols_to_drop = [label_col] + [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop).values
    y = df[label_col].values
    return X, y


# ── metrics ───────────────────────────────────────────────────────────────────

def _confusion(y_true: np.ndarray, y_pred_binary: np.ndarray):
    """Return (TP, FP, FN, TN)."""
    tp = int(np.sum((y_pred_binary == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred_binary == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred_binary == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred_binary == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def _derived(tp, fp, fn, tn):
    precision   = tp / (tp + fp)           if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn)           if (tp + fn) > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
    specificity = tn / (tn + fp)           if (tn + fp) > 0 else 0.0
    fpr         = fp / (fp + tn)           if (fp + tn) > 0 else 0.0
    return precision, recall, f1, specificity, fpr


def _print_confusion(tp, fp, fn, tn):
    w = max(len(str(v)) for v in (tp, fp, fn, tn)) + 2
    w = max(w, 8)
    print("\n  Confusion Matrix:")
    print(f"                  {'Pred 0':>{w}}   {'Pred 1':>{w}}")
    print(f"  Actual 0 (legit)  {tn:{w}}   {fp:{w}}")
    print(f"  Actual 1 (fraud)  {fn:{w}}   {tp:{w}}")


def _print_table(accuracy, correct, total, precision, recall, f1, specificity, fpr,
                 mae=None, mse=None):
    print(f"\n  {'Metric':<26}  {'Value':>10}  Note")
    print(f"  {'─'*26}  {'─'*10}  {'─'*38}")
    print(f"  {'Accuracy':<26}  {accuracy*100:>9.2f}%  ({correct}/{total})")
    print(f"  {'Precision':<26}  {precision*100:>9.2f}%  of predicted fraud, how many real")
    print(f"  {'Recall / Sensitivity':<26}  {recall*100:>9.2f}%  of real fraud, how many caught")
    print(f"  {'Specificity':<26}  {specificity*100:>9.2f}%  of real legit, how many correct")
    print(f"  {'F1 Score':<26}  {f1:>10.4f}")
    print(f"  {'False Positive Rate':<26}  {fpr*100:>9.2f}%  legit flagged as fraud")
    if mae is not None:
        print(f"  {'MAE':<26}  {mae:>10.4f}")
    if mse is not None:
        print(f"  {'MSE':<26}  {mse:>10.4f}")


def print_metrics(y_true: np.ndarray, predictions: np.ndarray, tolerance: float):
    """Full metrics for continuous-output perceptrons (linear / non-linear / multilayer)."""
    total    = len(y_true)
    matches  = np.abs(predictions - y_true) < tolerance
    correct  = int(np.sum(matches))
    accuracy = correct / total
    mae      = float(np.mean(np.abs(predictions - y_true)))
    mse      = float(np.mean((predictions - y_true) ** 2))

    # binarise: prediction is "fraud" when output is closer to 1 than to 0
    y_bin = (np.abs(predictions - 1) < tolerance).astype(int)
    tp, fp, fn, tn = _confusion(y_true, y_bin)
    precision, recall, f1, specificity, fpr = _derived(tp, fp, fn, tn)

    _print_confusion(tp, fp, fn, tn)
    _print_table(accuracy, correct, total, precision, recall, f1,
                 specificity, fpr, mae=mae, mse=mse)

    actual_fraud = int(np.sum(y_true == 1))
    print(f"\n  Fraud in test set  : {actual_fraud}/{total} ({actual_fraud/total*100:.1f}%)")
    print(f"  Fraud caught  (TP) : {tp}/{actual_fraud} ({recall*100:.1f}%)")
    print(f"  Fraud missed  (FN) : {fn}/{actual_fraud} ({fn/actual_fraud*100:.1f}%)  ← false negatives")
    print(f"  False alarms  (FP) : {fp} legitimate transactions flagged as fraud")


def print_metrics_step(y_true: np.ndarray, predictions: np.ndarray):
    """Full metrics for hard-threshold (step) perceptrons."""
    total    = len(y_true)
    correct  = int(np.sum(predictions == y_true))
    accuracy = correct / total

    tp, fp, fn, tn = _confusion(y_true, predictions.astype(int))
    precision, recall, f1, specificity, fpr = _derived(tp, fp, fn, tn)

    _print_confusion(tp, fp, fn, tn)
    _print_table(accuracy, correct, total, precision, recall, f1, specificity, fpr)

    actual_fraud = int(np.sum(y_true == 1))
    if actual_fraud > 0:
        print(f"\n  Fraud in test set  : {actual_fraud}/{total} ({actual_fraud/total*100:.1f}%)")
        print(f"  Fraud caught  (TP) : {tp}/{actual_fraud} ({recall*100:.1f}%)")
        print(f"  Fraud missed  (FN) : {fn}/{actual_fraud} ({fn/actual_fraud*100:.1f}%)  ← false negatives")
        print(f"  False alarms  (FP) : {fp} legitimate transactions flagged as fraud")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_arguments()
    print(f"Running perceptron {args.type_p} with {args.epochs} epochs and learning rate of {args.lr}\n")

    X, y = load_data(args.data, args.label, args.drop)
    X = X.astype(np.float64, copy=False)

    perceptron = build_perceptron(
        args.type_p, args.lr, args.epochs, args.epsilon, args.seed,
        args.activation, args.beta, args.layers, args.initializer,
        args.training_mode, args.batch_size, args.optimizer,
    )

    if args.no_split:
        print("Running with no split\n")
        if args.normalize == "standard":
            mean, std = standard_scale_params(X)
            X = standard_scale_apply(X, mean, std)
            print("Applied standard scaling (mean/std from full dataset).\n")
        perceptron.fit(X, y)
        predictions = perceptron.predict(X)
        y_test = y
    else:
        X_train, X_test, y_train, y_test = test_data_split(
            X, y, test_size=args.test_per, random_state=args.seed
        )
        if args.normalize == "standard":
            mean, std = standard_scale_params(X_train)
            X_train = standard_scale_apply(X_train, mean, std)
            X_test  = standard_scale_apply(X_test, mean, std)
            print("Applied standard scaling (mean/std from training split only).\n")
        perceptron.fit(X_train, y_train)
        print(f"Learned weights: {perceptron.weights}")
        bias = getattr(perceptron, "bias", None)
        if bias is None:
            bias = getattr(perceptron, "biases", None)
        print(f"Learned bias: {bias}")
        predictions = perceptron.predict(X_test)

    total = len(y_test)
    if total == 0:
        print("Warning: test set is empty (dataset too small for the given --test_per).")
        print("Consider using --no_split or increasing --test_per.")
        return

    if args.type_p in ("linear", "non-linear", "multilayer"):
        tolerance = args.tolerance
        print(f"\nResults on test set ({total} samples):")
        for i, (pred, expected) in enumerate(zip(predictions, y_test, strict=True)):
            match = "OK" if abs(pred - expected) < tolerance else "X"
            print(f"  sample {i + 1}: predicted={pred:.4f}  expected={int(expected)}  {match}")

        print(f"\n── Metrics (tolerance={tolerance}) " + "─" * 42)
        print_metrics(y_test, predictions, tolerance)

    else:  # simple-step
        print(f"\nResults on test set ({total} samples):")
        for i, (pred, expected) in enumerate(zip(predictions, y_test, strict=True)):
            match = "OK" if pred == expected else "X"
            print(f"  sample {i + 1}: predicted={int(pred)}  expected={int(expected)}  {match}")

        print("\n── Metrics " + "─" * 50)
        print_metrics_step(y_test, predictions)


if __name__ == "__main__":
    main()