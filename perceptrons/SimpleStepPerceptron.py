import numpy as np


def _activation(value):
    return 1 if value >= 0 else -1


class SimpleStepPerceptron:
    def __init__(self, learning_rate, epochs,seed):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize weights and bias to zero
        np.random.seed(self.seed)
        self.weights = np.random.random_sample(n_features)
        self.bias = np.random.random_sample(1)

        for epoch in range(self.epochs):
            errors = 0
            for x_i, y_i in zip(X, y):
                prediction = self._predict_single(x_i)
                error = y_i - prediction


                self.weights += self.learning_rate * error * x_i
                self.bias    += self.learning_rate * error
                if error != 0:
                    errors+=1
                print(f"Epoch {epoch + 1}: {errors} errors")

                # stop early if no errors
            if errors == 0:
                    print(f"Converged at epoch: {epoch + 1}")
                    break

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return _activation(linear_output)

