import ast

import numpy as np
import pandas as pd
from main import build_perceptron, parse_arguments


DIGITS_TEST_PATH = "datasets/digits_test.csv"
DIGIS_TRAIN_PATH = "datasets/digits.csv"
EVAL_RESULTS_PATH = "digits_eval_results.csv"

def parse_digits_from_csv(filename):
    df = pd.read_csv(filename)

    images = df["image"].apply(lambda s: np.array(ast.literal_eval(s), dtype=np.float32))
    X = np.stack(images.to_numpy())
    Y = df["label"].to_numpy(dtype=np.int64)
    return X, Y

def encode_digit_targets(labels, n_outputs):
    labels = labels.astype(int)
    targets = np.full((len(labels), n_outputs), -1.0)
    targets[np.arange(len(labels)), labels] = 1.0
    return targets

def print_classification_results(predictions, expected_labels):
    predicted_labels = np.argmax(predictions, axis=1)
    total = len(expected_labels)
    if total == 0:
        print(f"Test set is empty ({DIGITS_TEST_PATH}).")
        return

    correct = np.sum(predicted_labels == expected_labels)
    accuracy = correct / total

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as results_file:
        results_file.write("sample,predicted,expected,match\n")
        for i, (predicted, expected) in enumerate(zip(predicted_labels, expected_labels)):
            match = "OK" if predicted == expected else "X"
            results_file.write(f"{i + 1},{int(predicted)},{int(expected)},{match}\n")

    print(f"\nResults on test set ({total} samples)")
    print(f"Sample evaluation saved to: {EVAL_RESULTS_PATH}")
    print(f"Accuracy: {correct}/{total} = {accuracy * 100:.1f}%")


def main():
    args = parse_arguments()
    X_train, train_labels = parse_digits_from_csv(DIGIS_TRAIN_PATH)
    X_test, test_labels = parse_digits_from_csv(DIGITS_TEST_PATH)

    if args.type_p != "multilayer":
        raise ValueError("digits_main.py only supports --type_p multilayer")

    y_train = encode_digit_targets(train_labels, args.layers[-1])

    print(f"Running digit perceptron {args.type_p} with {args.epochs} epochs, learning rate {args.lr}, layers {args.layers}\n")

    perceptron = build_perceptron( args.type_p, args.lr, args.epochs, args.epsilon, args.seed, args.activation, args.beta, args.layers)
    perceptron.fit(X_train, y_train)
    predictions = perceptron.predict(X_test)

    print_classification_results(predictions, test_labels)


if __name__ == "__main__":
    main()
