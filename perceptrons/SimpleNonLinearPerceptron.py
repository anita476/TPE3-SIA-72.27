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

def _relu(h, beta=1.0):
    return np.maximum(0.0, h)

def _relu_derivative(g_h, beta=1.0):
    return (g_h > 0).astype(float)

ACTIVATIONS = {
    'tanh':     (_tanh,     _tanh_derivative),
    'logistic': (_logistic, _logistic_derivative),
    'relu':     (_relu,     _relu_derivative),
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
        return self.g(value, self.beta)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X, y         — training data
        X_val, y_val — optional validation data for test MSE tracking.
                       If provided, test_mse_history_ is populated each epoch.
        """
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        self.train_mse_history_: list[float] = []
        self.test_mse_history_:  list[float] = []

        compute_val = X_val is not None and y_val is not None

        for epoch in range(self.epochs):
            indices = self.rng.permutation(n_samples)
            for i in indices:
                x_i = X[i]
                y_i = y[i]

                prediction = self._predict_single(x_i)
                error = y_i - prediction
                delta = error * self.g_prime(prediction, self.beta)
                self.weights += self.learning_rate * delta * x_i
                self.bias    += self.learning_rate * delta

            train_mse = self._total_error(X, y) / n_samples
            self.train_mse_history_.append(train_mse)

            if compute_val:
                val_mse = self._total_error(X_val, y_val) / len(y_val)
                self.test_mse_history_.append(val_mse)

            total_error = train_mse * n_samples
            print(f"Epoch {epoch + 1}: total error = {total_error:.4f}")
            if total_error < self.epsilon:
                print(f"Converged at epoch {epoch + 1}")
                self.epochs_run_ = epoch + 1
                break
        else:
            self.epochs_run_ = self.epochs

    def _total_error(self, X, y):
        return np.sum((y - self.predict(X)) ** 2)