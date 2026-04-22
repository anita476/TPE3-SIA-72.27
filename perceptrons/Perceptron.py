import numpy as np


class Perceptron:
    def __init__(self, learning_rate, epochs, seed):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights = None
        self.bias = None

    def _initialize_parameters(self, n_features):
        np.random.seed(self.seed)
        self.weights = np.random.random_sample(n_features) * 2 - 1
        self.bias = np.random.random_sample() * 2 - 1

    def _linear_output(self, x):
        return np.dot(x, self.weights) + self.bias

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        return self._activation(self._linear_output(x))

    def _activation(self, value):
        raise NotImplementedError("Subclasses must implement _activation().")
