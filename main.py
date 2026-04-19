import argparse
import numpy as np
import pandas as pd

from perceptrons.SimpleStepPerceptron import SimpleStepPerceptron
from utils.test_data_split import test_data_split

def parse_arguments()-> argparse.Namespace:
    arguments = argparse.ArgumentParser(
        description="Run a perceptron training + evaluation cycle"
    )
    arguments.add_argument("--lr", type=float, default=0.01, help="Perceptron learning rate")
    arguments.add_argument("--epochs", type=int, default=100, help= "Number of epochs")
    arguments.add_argument("--data", type=str, required=True, help="Path to CSV file. Required")
    arguments.add_argument("--type", type=str, required=True,
                        choices=["simple-step"],
                        help="Perceptron type to use")
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

    X, y = load_data(args.data)

    # missing build perceptron @todo
    perceptron = SimpleStepPerceptron(args.lr,args.epochs,args.seed)

    if args.no_split:
        perceptron.fit(X, y)
        predictions = perceptron.predict(X)
        y_test = y
    else:
        # train of a percentage of data
        X_train, X_test, y_train, y_test = test_data_split(
            X, y, test_size=args.test_per, random_state=args.seed  # seed
        )
        perceptron.fit(X_train, y_train)
        predictions = perceptron.predict(X_test)


    # results
    accuracy = np.mean(predictions == y_test)
    correct = np.sum(predictions == y_test)
    total = len(y_test)

    print(f"Results on test set ({total} samples):")
    for i, (pred, expected) in enumerate(zip(predictions, y_test)):
        match = "✓" if pred == expected else "✗"
        print(f"  sample {i + 1}: predicted={int(pred)}  expected={int(expected)}  {match}")

    print(f"\nAccuracy: {correct}/{total} = {accuracy * 100:.1f}%")





if __name__ == "__main__":
    main()