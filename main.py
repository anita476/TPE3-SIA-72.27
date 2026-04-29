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
    arguments.add_argument("--activation", type=str, default="tanh", choices=["tanh", "logistic", "relu"], help="Activation function for non-linear and multilayer perceptrons. Default is tanh")
    arguments.add_argument("--beta", type=float, default=1.0, help="Beta scaling parameter for non-linear perceptron. Default is 1.0")
    arguments.add_argument("--no_split", action="store_true", help="Skip train/test split, evaluate on full dataset")
    arguments.add_argument("--layers", type=int, nargs="+", default=[2, 2, 1], help="Layer sizes for multilayer perceptron (e.g. --layers 2 2 1)")
    arguments.add_argument("--initializer", type=str, default="random", choices=["random", "xavier", "xavier_n"], help="Weight initializer for multilayer perceptron. Default is random")
    arguments.add_argument("--training_mode", type=str, default="online", choices=["online", "minibatch"], help="Weight update mode for multilayer perceptron. Default is online")
    arguments.add_argument("--batch_size", type=int, default=1, help="Mini-batch size for multilayer perceptron when --training_mode minibatch, or for --optimizer sgd")
    arguments.add_argument("--optimizer", type=str, default="sgd", choices=["gd", "sgd", "rmsprop", "adam"], help="Optimizer for multilayer perceptron. gd uses the full dataset in one update per epoch; sgd uses online or mini-batch updates. Default is sgd")
    arguments.add_argument("--label", type=str, default="label",help= "Label column name to drop in data loading")
    arguments.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "standard"],
        help="Feature scaling: 'standard' = zero mean / unit variance (fit on train only when split). "
        "Use on real-world CSVs with mixed feature scales to avoid NaN weights.",
    )
    return arguments.parse_args()


def load_data(path: str, label_col: str):
    df = pd.read_csv(path)
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    return X, y


def main():
    args = parse_arguments()
    print(f"Running perceptron {args.type_p} with {args.epochs} epochs and learning rate of {args.lr}\n")

    X, y = load_data(args.data,args.label)
    X = X.astype(np.float64, copy=False)

    perceptron = build_perceptron(args.type_p, args.lr, args.epochs, args.epsilon, args.seed, args.activation, args.beta, args.layers, args.initializer, args.training_mode, args.batch_size, args.optimizer)

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
            X_test = standard_scale_apply(X_test, mean, std)
            print("Applied standard scaling (mean/std from training split only).\n")
        perceptron.fit(X_train, y_train)
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

if __name__ == "__main__":
    main()
