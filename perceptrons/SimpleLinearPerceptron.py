import numpy as np


def _activation_identity(value):
    return (value)


class SimpleLinearPerceptron:
    def __init__(self, learning_rate, epochs, epsilon,seed):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.seed = seed
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize weights and bias to random values
        np.random.seed(self.seed)
        self.weights = np.random.random_sample(n_features) * 2 - 1
        self.bias = np.random.random_sample() * 2 - 1
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                x_i = X[i]
                y_i = y[i]

            # for x_i, y_i in zip(X, y):
                prediction = self._predict_single(x_i)
                error = y_i - prediction # y - output -> if != 0 theres an error..
                # Because theta(h) = h, theta'(h) = 1 (linear), so the update is not affected
                self.weights += (self.learning_rate * error * x_i)
                self.bias += self.learning_rate * error

            total_error = self._total_error(X,y)
            print(f"Epoch {epoch + 1}: total error = {total_error:.4f}")
            if total_error < self.epsilon:
                print(f"Converged at epoch {epoch + 1}")
                break


    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _total_error(self,X, y):
        predictions = self.predict(X)
        return np.sum((y - predictions) ** 2)

    def _predict_single(self, x):
        # calculate the weighted sum....
        linear_output = np.dot(x, self.weights) + self.bias
        # compute activation...
        return _activation_identity(linear_output)





