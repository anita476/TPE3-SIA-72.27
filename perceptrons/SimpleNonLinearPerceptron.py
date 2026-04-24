import numpy as np
from perceptrons.Perceptron import Perceptron


def _tanh(h, beta=1.0):
    return np.tanh(beta * h)

def _tanh_derivative(g_h, beta=1.0):
    return beta * (1 - g_h**2)

def _logistic(h, beta=1.0):
    return 1.0 / (1.0 + np.exp(-2 * beta * h))

def _logistic_derivative(g_h, beta=1.0):
    return 2 * beta * g_h * (1 - g_h)

ACTIVATIONS = {
    'tanh': (_tanh, _tanh_derivative),       # output in [-1, 1]
    'logistic': (_logistic, _logistic_derivative),  # output in [0, 1]
}


class SimpleNonLinearPerceptron(Perceptron):
    def __init__(self, learning_rate, epochs, epsilon, seed, activation='tanh', beta=1.0):
        super().__init__(learning_rate, epochs, seed)
        self.epsilon = epsilon
        self.beta = beta

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(ACTIVATIONS.keys())}"
            )
        self.g, self.g_prime = ACTIVATIONS[activation]

    def _activation(self, value):
        # called by base class _predict_single
        return self.g(value, self.beta)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        for epoch in range(self.epochs):
            indices = self.rng.permutation(n_samples)
            for i in indices:
                x_i = X[i]
                y_i = y[i]

                prediction = self._predict_single(x_i)          # g(h), computed once
                error = y_i - prediction
                delta = error * self.g_prime(prediction, self.beta)  # reuses g(h)
                self.weights += self.learning_rate * delta * x_i
                self.bias += self.learning_rate * delta

            total_error = self._total_error(X, y)
            print(f"Epoch {epoch + 1}: total error = {total_error:.4f}")
            if total_error < self.epsilon:
                print(f"Converged at epoch {epoch + 1}")
                break

    def _total_error(self, X, y):
        return np.sum((y - self.predict(X)) ** 2)
