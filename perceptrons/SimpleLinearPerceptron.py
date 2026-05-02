import numpy as np
from perceptrons.Perceptron import Perceptron


def _activation_identity(value):
    return value


class SimpleLinearPerceptron(Perceptron):
    def __init__(self, learning_rate, epochs, epsilon, seed):
        super().__init__(learning_rate, epochs, seed)
        self.epsilon = epsilon

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X, y         training data
        X_val, y_val optional validation data for test MSE tracking.
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
                self.weights += self.learning_rate * error * x_i
                self.bias    += self.learning_rate * error

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
        predictions = self.predict(X)
        return np.sum((y - predictions) ** 2)

    def _activation(self, value):
        return _activation_identity(value)