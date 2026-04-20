import numpy as np


def _activation_simple(value):
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

        # initialize weights and bias to randoms
        np.random.seed(self.seed)
        # self.weights = np.random.random_sample(n_features)
        self.weights = np.random.random_sample(n_features) * 2 - 1 # entre -1 y 1
        #self.bias = np.random.random_sample()
        self.bias = np.random.random_sample() * 2 - 1

        for epoch in range(self.epochs):
            # elegir patrones en orden aleatorio dentro de la epoca 
            # apunte dice "elegir un patron al azar entre 1 y p"
            indices = np.random.permutation(n_samples)
            errors = 0
            
            for i in indices:
                x_i = X[i]
                y_i = y[i]
            
                prediction = self._predict_single(x_i)
                error = y_i - prediction
            
                # UPDATE WEIGHTS IF THERES AN ERROR
                if prediction != y_i:  # only update when O^μ ≠ ζ^μ, not strictly necessary
                    self.weights += 2 * self.learning_rate * y_i * x_i
                    self.bias += 2 * self.learning_rate * y_i
                
                if error != 0:
                    errors += 1
            
            print(f"Epoch {epoch + 1}: {errors} errors")
                    
            # stop early if no errors
            if errors == 0:
                    print(f"Converged at epoch: {epoch + 1}")
                    break

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return _activation_simple(linear_output)

