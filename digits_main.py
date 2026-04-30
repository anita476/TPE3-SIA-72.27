import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from digit_dataset_loader import load_dataset
from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix, plot_loss_curve, plot_per_class_metrics

DIGITS_TRAIN_PATH = "data/digits.csv"
DIGITS_TEST_PATH = "data/digits_test.csv"
OFF_TARGET_BY_ACTIVATION = {"logistic": 0.0, "tanh": -1.0}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an MLP on the digit dataset"
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[784, 100, 10])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--activation", type=str, default="tanh", choices=["tanh", "logistic"]
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--initializer",
        type=str,
        default="xavier",
        choices=["random", "xavier", "xavier_n"],
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="online",
        choices=["online", "minibatch"],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["gd", "sgd", "rmsprop", "adam"],
        help="Optimizer. gd uses the full dataset in one update per epoch; sgd uses online or mini-batch updates.",
    )
    return parser.parse_args()


def load_digits(path):
    df = load_dataset(path)
    return np.stack(df["image"].to_numpy()), df["label"].to_numpy(dtype=np.int64)


def encode_digit_targets(labels, n_outputs, activation="tanh"):
    if activation not in OFF_TARGET_BY_ACTIVATION:
        raise ValueError("activation must be 'tanh' or 'logistic'")

    labels = labels.astype(int)
    targets = np.full((len(labels), n_outputs), OFF_TARGET_BY_ACTIVATION[activation])
    targets[np.arange(len(labels)), labels] = 1.0
    return targets


def csv_path(prefix, args, timestamp):
    layers = "_".join(str(layer) for layer in args.layers)
    lr = f"{args.lr:g}".replace("-", "neg_").replace(".", "_")
    batch = (
        f"_{args.training_mode}_bs_{args.batch_size}"
        if args.training_mode == "minibatch"
        else "_online"
    )
    optimizer = f"_{args.optimizer}"
    return Path(
        f"{prefix}_{layers}_lr_{lr}_{args.epochs}_epoch"
        f"{batch}{optimizer}_{timestamp:%Y%m%d_%H%M%S}.csv"
    )


def train_with_epoch_log(
    perceptron, args, X_train, y_train, train_labels, X_test, y_test, test_labels
):
    epoch_log = []
    n_train, n_test = len(X_train), len(X_test)

    for epoch in range(1, args.epochs + 1):
        perceptron.train_epoch(X_train, y_train)

        learn_predictions = perceptron.predict(X_train)
        test_predictions = perceptron.predict(X_test)
        learn_error = 0.5 * np.sum((y_train - learn_predictions) ** 2)
        test_error = 0.5 * np.sum((y_test - test_predictions) ** 2)
        learn_accuracy = float(
            np.mean(np.argmax(learn_predictions, axis=1) == train_labels)
        )
        test_accuracy = float(np.mean(np.argmax(test_predictions, axis=1) == test_labels))

        perceptron.errors_.append(learn_error)
        perceptron.val_errors_.append(test_error)
        perceptron.val_accuracies_.append(test_accuracy)

        epoch_log.append(
            {
                "epoch": epoch,
                "learn_error": learn_error,
                "test_error": test_error,
                "learn_error_avg": learn_error / n_train if n_train else np.nan,
                "test_error_avg": test_error / n_test if n_test else np.nan,
                "learn_accuracy_percent": learn_accuracy * 100,
                "test_accuracy_percent": test_accuracy * 100,
            }
        )

        print(
            f"Epoch {epoch}: learn error = {learn_error:.4f}  "
            f"test error = {test_error:.4f}  learn acc = {learn_accuracy * 100:.1f}%  "
            f"test acc = {test_accuracy * 100:.1f}%"
        )

        if learn_error < args.epsilon:
            print(f"Converged at epoch {epoch}")
            break

    return epoch_log


def save_epoch_log(epoch_log, metadata, results_path):
    pd.DataFrame(epoch_log).assign(**metadata).to_csv(results_path, index=False)
    print(f"Learn/test epoch log saved to: {results_path}")


def save_classification_results(predictions, expected_labels, metadata, results_path):
    predicted_labels = np.argmax(predictions, axis=1)
    total = len(expected_labels)
    correct = int(np.sum(predicted_labels == expected_labels)) if total else 0
    accuracy = correct / total if total else 0.0

    results = pd.DataFrame(
        {
            "sample": np.arange(1, total + 1),
            "predicted": predicted_labels.astype(int),
            "expected": expected_labels.astype(int),
            "match": np.where(predicted_labels == expected_labels, "OK", "X"),
        }
    ).assign(**metadata)
    results.to_csv(results_path, index=False)

    print(f"\nResults on test set ({total} samples)")
    if total == 0:
        print(f"Test set is empty ({DIGITS_TEST_PATH}).")
    print(f"Sample evaluation saved to: {results_path}")
    print(f"Accuracy: {correct}/{total} = {accuracy * 100:.1f}%")

    return predicted_labels, accuracy


def build_metadata(args, run_started_at, run_finished_at, epochs_trained):
    return {
        "layers": "-".join(str(layer) for layer in args.layers),
        "activation": args.activation,
        "initializer": args.initializer,
        "training_mode": args.training_mode,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "epochs_requested": args.epochs,
        "epochs_trained": epochs_trained,
        "run_started_at": run_started_at.isoformat(timespec="seconds"),
        "run_finished_at": run_finished_at.isoformat(timespec="seconds"),
    }


def main():
    run_started_at = datetime.now().astimezone()
    args = parse_arguments()

    print("Loading data...")
    X_train, train_labels = load_digits(DIGITS_TRAIN_PATH)
    X_test, test_labels = load_digits(DIGITS_TEST_PATH)
    y_train = encode_digit_targets(train_labels, args.layers[-1], args.activation)
    y_test = encode_digit_targets(test_labels, args.layers[-1], args.activation)

    print(
        f"Training MLP {args.layers} with {args.optimizer}, lr={args.lr}, "
        f"epochs={args.epochs}, init={args.initializer}, "
        f"mode={args.training_mode}, batch_size={args.batch_size}\n"
    )

    perceptron = MultiLayerPerceptron(
        args.layers,
        args.lr,
        args.epochs,
        args.epsilon,
        args.seed,
        args.beta,
        args.activation,
        args.initializer,
        args.training_mode,
        args.batch_size,
        args.optimizer,
    )
    perceptron.errors_ = []
    perceptron.val_errors_ = []
    perceptron.val_accuracies_ = []

    epoch_log = train_with_epoch_log(
        perceptron, args, X_train, y_train, train_labels, X_test, y_test, test_labels
    )
    predictions = perceptron.predict(X_test)
    train_predictions = perceptron.predict(X_train)
    train_predicted_labels = np.argmax(train_predictions, axis=1)
    train_accuracy = float(np.mean(train_predicted_labels == train_labels))
    run_finished_at = datetime.now().astimezone()

    metadata = build_metadata(args, run_started_at, run_finished_at, len(epoch_log))
    results_path = csv_path("digits_eval_results", args, run_started_at)
    epoch_log_path = csv_path("digits_learn_test_log", args, run_started_at)

    save_epoch_log(epoch_log, metadata, epoch_log_path)
    test_predicted_labels, test_accuracy = save_classification_results(
        predictions, test_labels, metadata, results_path
    )

    metrics = compute_metrics(test_labels, test_predicted_labels, n_classes=args.layers[-1])

    print(f"\nTrain accuracy: {train_accuracy * 100:.1f}%")
    print(f"Test accuracy:  {test_accuracy * 100:.1f}%")
    print(f"Macro F1:       {metrics['macro_f1'] * 100:.1f}%")
    print(f"Min class F1:   {metrics['min_class_f1'] * 100:.1f}%")

    train_loss = [e / len(X_train) for e in perceptron.errors_]
    test_loss = [e / len(X_test) for e in perceptron.val_errors_]
    plot_loss_curve(train_loss, val_errors=test_loss)
    plot_confusion_matrix(metrics["confusion_matrix"])
    plot_per_class_metrics(metrics)


if __name__ == "__main__":
    main()
