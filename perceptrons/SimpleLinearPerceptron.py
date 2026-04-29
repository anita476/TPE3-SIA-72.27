import numpy as np
from perceptrons.Perceptron import Perceptron


def _activation_identity(value):
    return value


class SimpleLinearPerceptron(Perceptron):
    def __init__(self, learning_rate, epochs, epsilon, seed):
        super().__init__(learning_rate, epochs, seed)
        self.epsilon = epsilon


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize weights and bias to random values
        self._initialize_parameters(n_features)
        self.train_mse_history_: list[float] = []

        for epoch in range(self.epochs):
            indices = self.rng.permutation(n_samples)
            for i in indices:
                x_i = X[i]
                y_i = y[i]

                prediction = self._predict_single(x_i)
                error = y_i - prediction
                # theta(h) = h  =>  theta'(h) = 1, so the gradient simplifies to just the error
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error

            total_error = self._total_error(X, y)
            self.train_mse_history_.append(total_error / n_samples)
            print(f"Epoch {epoch + 1}: total error = {total_error:.4f}")
            if total_error < self.epsilon:
                print(f"Converged at epoch {epoch + 1}")
                self.epochs_run_ = epoch + 1
                break
        else:
            self.epochs_run_ = self.epochs


    def _total_error(self,X, y):
        predictions = self.predict(X)
        return np.sum((y - predictions) ** 2)


    def _activation(self, value):
        return _activation_identity(value)
