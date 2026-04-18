import argparse
import numpy as np
import pandas as pd

from perceptrons import SimpleStepPerceptron
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

    return arguments.parse_args()

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y


def main():
    args = parse_arguments()

    X, y = load_data(args.data)

    X_train, X_test, y_train, y_test = test_data_split(
        X, y, test_size=args.test_size, random_state=args.seed #seed
    )

    # missing build perceptron @todo
    perceptron = SimpleStepPerceptron(args.lr,args.epochs)








