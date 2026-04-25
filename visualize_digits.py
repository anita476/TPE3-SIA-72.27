import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron

TRAIN_PATH = "datasets/digits.csv"
TEST_PATH  = "datasets/digits_test.csv"
LAYERS     = [784, 10, 10]
LR         = 0.1
EPOCHS     = 500
EPSILON    = 0.01
SEED       = 1
BETA       = 1.0


def load(path):
    df = pd.read_csv(path)
    images = df["image"].apply(lambda s: np.array(ast.literal_eval(s), dtype=np.float32))
    X = np.stack(images.to_numpy())
    y_labels = df["label"].to_numpy(dtype=np.int64)
    return X, y_labels


def encode(labels, n_outputs):
    targets = np.full((len(labels), n_outputs), -1.0)
    targets[np.arange(len(labels)), labels] = 1.0
    return targets


def plot_loss_curve(train_errors, n_train, val_errors=None, n_val=None):
    plt.figure()
    epochs = range(1, len(train_errors) + 1)
    plt.plot(epochs, [e / n_train for e in train_errors], label="Train")
    if val_errors and n_val:
        plt.plot(epochs, [e / n_val for e in val_errors], label="Test")
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Error (MSE)")
    plt.title("Loss curve")
    plt.tight_layout()
    plt.savefig("plots/plot_loss_curve.png")
    print("Saved: plots/plot_loss_curve.png")


def plot_confusion_matrix(y_true, y_pred, title, filename, n_classes=10):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.max() * 0.5 else "black")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")


def main():
    print("Loading data...")
    X_train, train_labels = load(TRAIN_PATH)
    X_test,  test_labels  = load(TEST_PATH)
    y_train = encode(train_labels, LAYERS[-1])
    y_test  = encode(test_labels,  LAYERS[-1])

    print(f"Training MLP {LAYERS} for up to {EPOCHS} epochs...")
    mlp = MultiLayerPerceptron(LAYERS, LR, EPOCHS, EPSILON, SEED, BETA)
    mlp.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    train_predictions = mlp.predict(X_train)
    train_predicted_labels = np.argmax(train_predictions, axis=1)
    train_accuracy = np.mean(train_predicted_labels == train_labels)
    print(f"\nTrain accuracy: {train_accuracy * 100:.1f}%")

    test_predictions = mlp.predict(X_test)
    test_predicted_labels = np.argmax(test_predictions, axis=1)
    test_accuracy = np.mean(test_predicted_labels == test_labels)
    print(f"Test accuracy:  {test_accuracy * 100:.1f}%")

    plot_loss_curve(mlp.errors_, len(X_train), mlp.val_errors_, len(X_test))
    plot_confusion_matrix(train_labels, train_predicted_labels, "Confusion matrix (train)", "plots/plot_confusion_matrix_train.png")
    plot_confusion_matrix(test_labels, test_predicted_labels, "Confusion matrix (test)", "plots/plot_confusion_matrix_test.png")


if __name__ == "__main__":
    main()
