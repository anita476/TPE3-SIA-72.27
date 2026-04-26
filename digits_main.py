import argparse
import numpy as np

from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from datasets.digit_dataset_loader import load_digits, encode_one_hot
from utils.metrics import compute_metrics
from utils.visualization import plot_loss_curve, plot_confusion_matrix, plot_per_class_metrics

TRAIN_PATH = "datasets/digits.csv"
TEST_PATH  = "datasets/digits_test.csv"


def parse_arguments():
    p = argparse.ArgumentParser(description="Train and evaluate an MLP on the digit dataset")
    p.add_argument("--layers",      type=int,   nargs="+", default=[784, 100, 10])
    p.add_argument("--lr",          type=float, default=0.01)
    p.add_argument("--epochs",      type=int,   default=250)
    p.add_argument("--epsilon",     type=float, default=1e-6)
    p.add_argument("--seed",        type=int,   default=1)
    p.add_argument("--activation",  type=str,   default="tanh",   choices=["tanh", "logistic"])
    p.add_argument("--beta",        type=float, default=1.0)
    p.add_argument("--initializer", type=str,   default="xavier", choices=["random", "xavier", "xavier_n"])
    p.add_argument("--optimizer",   type=str,   default="sgd",    choices=["sgd", "rmsprop", "adam"])
    return p.parse_args()


def main():
    args = parse_arguments()

    print("Loading data...")
    X_train, train_labels = load_digits(TRAIN_PATH)
    X_test,  test_labels  = load_digits(TEST_PATH)

    n_classes = args.layers[-1]
    y_train = encode_one_hot(train_labels, n_classes)
    y_test  = encode_one_hot(test_labels,  n_classes)

    print(f"Training MLP {args.layers} — lr={args.lr}, epochs={args.epochs}, init={args.initializer}\n")
    mlp = MultiLayerPerceptron(
        args.layers, args.lr, args.epochs, args.epsilon,
        args.seed, args.beta, args.activation, args.initializer,
        optimizer=args.optimizer,
    )
    mlp.fit(X_train, y_train, X_val=X_test, y_val=y_test, val_labels=test_labels)

    train_preds = np.argmax(mlp.predict(X_train), axis=1)
    test_preds  = np.argmax(mlp.predict(X_test),  axis=1)
    train_acc = np.mean(train_preds == train_labels)
    test_acc  = np.mean(test_preds  == test_labels)

    metrics = compute_metrics(test_labels, test_preds, n_classes=n_classes)

    print(f"\nTrain accuracy: {train_acc * 100:.1f}%")
    print(f"Test accuracy:  {test_acc * 100:.1f}%")
    print(f"Macro F1:       {metrics['macro_f1'] * 100:.1f}%")
    print(f"Min class F1:   {metrics['min_class_f1'] * 100:.1f}%")

    train_loss = [e / len(X_train) for e in mlp.errors_]
    test_loss  = [e / len(X_test)  for e in mlp.val_errors_]

    plot_loss_curve(train_loss, val_errors=test_loss)
    plot_confusion_matrix(metrics["confusion_matrix"])
    plot_per_class_metrics(metrics)


if __name__ == "__main__":
    main()
