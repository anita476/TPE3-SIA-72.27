import numpy as np

from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron


def run_and_evaluate():
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=float)
    y = np.array([1, 1, -1, -1], dtype=float)

    mlp = MultiLayerPerceptron(layers=[2, 2, 1], learning_rate=0.1, epochs=5000, epsilon=0.01, seed=1)

    mlp.fit(X, y)

    predictions = mlp.predict(X)
    classified = np.where(predictions >= 0, 1, -1)
    accuracy = np.mean(classified == y)
    mse = np.mean((predictions - y) ** 2)

    print("XOR multilayer perceptron")
    for x, expected, predicted, output in zip(X, y, classified, predictions):
        print(f"input={x.astype(int).tolist()} expected={int(expected)} predicted={int(predicted)} output={output:.4f}")
    print(f"accuracy={accuracy * 100:.1f}%")
    print(f"mse={mse:.4f}")

    assert accuracy == 1.0
    assert mse < 0.01


if __name__ == "__main__":
    run_and_evaluate()
