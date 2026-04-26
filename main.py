import argparse

import numpy as np
import pandas as pd

from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from perceptrons.SimpleLinearPerceptron import SimpleLinearPerceptron
from perceptrons.SimpleNonLinearPerceptron import SimpleNonLinearPerceptron
from perceptrons.SimpleStepPerceptron import SimpleStepPerceptron
from datasets.digit_dataset_loader import encode_one_hot
from utils.test_data_split import test_data_split
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix, plot_loss_curve, plot_per_class_metrics


def build_perceptron(type, lr, epochs, epsilon, seed, activation='tanh', beta=1.0, layers=None, initializer="random", training_mode="online", batch_size=1):
    if type == "simple-step":
        return SimpleStepPerceptron(lr, epochs, seed)
    elif type == "linear":
        return SimpleLinearPerceptron(lr, epochs, epsilon, seed)
    elif type == "non-linear":
        return SimpleNonLinearPerceptron(lr, epochs, epsilon, seed, activation, beta)
    elif type == "multilayer":
        return MultiLayerPerceptron(layers, lr, epochs, epsilon, seed, beta, activation, initializer, training_mode, batch_size)
    raise Exception("Unknown perceptron type")


def parse_arguments()-> argparse.Namespace:
    arguments = argparse.ArgumentParser(
        description="Run a perceptron training + evaluation cycle"
    )
    arguments.add_argument("--lr", type=float, default=0.01, help="Perceptron learning rate")
    arguments.add_argument("--epochs", type=int, default=100, help= "Number of epochs")
    arguments.add_argument("--data", type=str, required=True, help="Path to CSV file. Required")
    arguments.add_argument("--type_p", type=str, required=True, choices=["simple-step", "linear", "non-linear", "multilayer"], help="Perceptron type to use")
    arguments.add_argument("--epsilon", type=float, default=0.001, help="Perceptron epsilon value (threshold). Default is 0.001")
    arguments.add_argument("--tolerance", type=float, default=0.5, help="Tolerance for interpreting predictions as correct (for linear and non-linear perceptrons). Default is 0.5")
    arguments.add_argument("--test_per", type=float, default=0.2, help="Fraction for test split in decimals (for example 0.2 = 20%%)")
    arguments.add_argument("--seed", type=int, default=1, help="Random seed")
    arguments.add_argument("--activation", type=str, default="tanh", choices=["tanh", "logistic"], help="Activation function for non-linear and multilayer perceptrons. Default is tanh")
    arguments.add_argument("--beta", type=float, default=1.0, help="Beta scaling parameter for non-linear perceptron. Default is 1.0")
    arguments.add_argument("--no_split", action="store_true", help="Skip train/test split, evaluate on full dataset")
    arguments.add_argument("--layers", type=int, nargs="+", default=[2, 2, 1], help="Layer sizes for multilayer perceptron (e.g. --layers 2 2 1)")
    arguments.add_argument("--initializer", type=str, default="random", choices=["random", "xavier", "xavier_n"], help="Weight initializer for multilayer perceptron. Default is random")
    arguments.add_argument("--training_mode", type=str, default="online", choices=["online", "minibatch"], help="Weight update mode for multilayer perceptron. Default is online")
    arguments.add_argument("--batch_size", type=int, default=1, help="Mini-batch size for multilayer perceptron when --training_mode minibatch")
    return arguments.parse_args()


def load_data(path: str):
    from datasets.digit_dataset_loader import load_digits
    df = pd.read_csv(path, nrows=0)
    if "image" in df.columns:
        return load_digits(path)
    df = pd.read_csv(path)
    return df.drop(columns=["label"]).values, df["label"].values


def _encode_if_multiclass(y, type_p, layers):
    """One-hot encode integer labels for multilayer networks with >1 output."""
    if type_p == "multilayer" and layers[-1] > 1:
        return encode_one_hot(y.astype(int), layers[-1])
    return y


def main():
    args = parse_arguments()
    print(f"Running perceptron {args.type_p} with {args.epochs} epochs and learning rate of {args.lr}\n")

    X, y = load_data(args.data)

    perceptron = build_perceptron(args.type_p, args.lr, args.epochs, args.epsilon, args.seed, args.activation, args.beta, args.layers, args.initializer, args.training_mode, args.batch_size)

    if args.no_split:
        print("Running with no split\n")
        perceptron.fit(X, _encode_if_multiclass(y, args.type_p, args.layers))
        predictions = perceptron.predict(X)
        y_test = y
    else:
        X_train, X_test, y_train, y_test = test_data_split(
            X, y, test_size=args.test_per, random_state=args.seed
        )
        perceptron.fit(X_train, _encode_if_multiclass(y_train, args.type_p, args.layers))
        print(f"Learned weights: {perceptron.weights}")
        bias = getattr(perceptron, "bias", None)
        if bias is None:
            bias = getattr(perceptron, "biases", None)
        print(f"Learned bias: {bias}")
        predictions = perceptron.predict(X_test)

    total = len(y_test)

    if args.type_p == "linear" or args.type_p == "non-linear" or args.type_p == "multilayer":
        # because we are doing floats exact match is always off, add a level of tolerance to interpret a result as "correct"
        if total == 0:
            print("Warning: test set is empty (dataset too small for the given --test_per).")
            print("Consider using --no_split or increasing --test_per.")
        else:
            tolerance = args.tolerance
            matches = np.abs(predictions - y_test) < tolerance
            correct = np.sum(matches)
            accuracy = correct / total
            mae = np.mean(np.abs(predictions - y_test))
            mse = np.mean((predictions - y_test) ** 2)

            print(f"\nResults on test set ({total} samples):")
            for i, (pred, expected) in enumerate(zip(predictions, y_test, strict=True)):
                match = "OK" if abs(pred - expected) < tolerance else "X"
                print(f"  sample {i + 1}: predicted={pred:.2f}  expected={expected}  {match}")

            print(f"\nAccuracy (tolerance={tolerance}): {correct}/{total} = {accuracy * 100:.1f}%")
            print(f"MAE: {mae:.4f}")
            print(f"MSE: {mse:.4f}")
    else:
        if total == 0:
            print("Warning: test set is empty (dataset too small for the given --test_per).")
            print("Consider using --no_split or increasing --test_per.")
        else:
            correct = np.sum(predictions == y_test)
            accuracy = correct / total

            print(f"\nResults on test set ({total} samples):")
            for i, (pred, expected) in enumerate(zip(predictions, y_test, strict=True)):
                match = "OK" if pred == expected else "X"
                print(f"  sample {i + 1}: predicted={int(pred)}  expected={int(expected)}  {match}")

            print(f"\nAccuracy: {correct}/{total} = {accuracy * 100:.1f}%")
            
    # Loss curve for any perceptron that tracks training error 
    errors = getattr(perceptron, "errors_", None)
    if errors:
        plot_loss_curve(errors)
    
    # Confusion matrix and per-class metrics for multilayer
    if args.type_p == "multilayer" and predictions.ndim == 2:
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = y_test.astype(int)
        n_classes = predictions.shape[1]
        metrics = compute_metrics(true_labels, pred_labels, n_classes=n_classes)
        plot_confusion_matrix(metrics["confusion_matrix"])
        plot_per_class_metrics(metrics)


if __name__ == "__main__":
    main()
