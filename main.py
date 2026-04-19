import argparse
import numpy as np
import pandas as pd

from perceptrons.SimpleLinearPerceptron import SimpleLinearPerceptron
from perceptrons.SimpleStepPerceptron import SimpleStepPerceptron
from utils.test_data_split import test_data_split

def build_perceptron(type, lr, epochs, epsilon,seed):
    if type == "simple-step":
        return SimpleStepPerceptron(lr, epochs, seed)
    elif type == "linear":
        return SimpleLinearPerceptron(lr,epochs,epsilon,seed)

def parse_arguments()-> argparse.Namespace:
    arguments = argparse.ArgumentParser(
        description="Run a perceptron training + evaluation cycle"
    )
    arguments.add_argument("--lr", type=float, default=0.01, help="Perceptron learning rate")
    arguments.add_argument("--epochs", type=int, default=100, help= "Number of epochs")
    arguments.add_argument("--data", type=str, required=True, help="Path to CSV file. Required")
    arguments.add_argument("--type", type=str, required=True,
                        choices=["simple-step", "linear"],
                        help="Perceptron type to use")
    arguments.add_argument("--epsilon",type=float, default=0.001, help="Perceptron epsilon value (threshold). Default is 0.001")
    arguments.add_argument("--test_per", type=float, default=0.2, help="Fraction for test split in decimals (for example 0.2 = 20%%)")
    arguments.add_argument("--seed", type=int,default=42, help="Random seed")
    arguments.add_argument("--no_split", action="store_true", help="Skip train/test split, evaluate on full dataset")
    return arguments.parse_args()

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y


def main():
    args = parse_arguments()
    print(f"Running perceptron {str(args.type)} with {int(args.epochs)} epochs and learning rate of {float(args.lr)}\n")

    X, y = load_data(args.data)


    perceptron = build_perceptron(args.type,args.lr,args.epochs, args.epsilon,args.seed)

    if args.no_split:
        print(f"Running with no split\n")
        perceptron.fit(X, y)
        predictions = perceptron.predict(X)
        y_test = y
    else:
        # train of a percentage of data
        X_train, X_test, y_train, y_test = test_data_split(
            X, y, test_size=args.test_per, random_state=args.seed  # seed
        )
        perceptron.fit(X_train, y_train)
        # add this after fit() in main.py
        print(f"Learned weights: {perceptron.weights}")
        print(f"Learned bias: {perceptron.bias}")
        predictions = perceptron.predict(X_test)


    # results
    accuracy = np.mean(predictions == y_test)
    correct = np.sum(predictions == y_test)
    total = len(y_test)

    if args.type == "linear":
        tolerance = 0.5
        matches = np.abs(predictions - y_test) < tolerance
        correct = np.sum(matches)
        accuracy = correct / total
        mae = np.mean(np.abs(predictions - y_test))
        mse = np.mean((predictions - y_test) ** 2)

        print(f"\nResults on test set ({total} samples):")
        for i, (pred, expected) in enumerate(zip(predictions, y_test)):
            match = "✓" if abs(pred - expected) < tolerance else "✗"
            print(f"  sample {i + 1}: predicted={pred:.2f}  expected={expected}  {match}")

        print(f"\nAccuracy (tolerance={tolerance}): {correct}/{total} = {accuracy * 100:.1f}%")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")

    else:
        # step perceptron - exact match is fine
        correct = np.sum(predictions == y_test)
        accuracy = correct / total

        print(f"\nResults on test set ({total} samples):")
        for i, (pred, expected) in enumerate(zip(predictions, y_test)):
            match = "✓" if pred == expected else "✗"
            print(f"  sample {i + 1}: predicted={int(pred)}  expected={int(expected)}  {match}")

        print(f"\nAccuracy: {correct}/{total} = {accuracy * 100:.1f}%")



if __name__ == "__main__":
    main()