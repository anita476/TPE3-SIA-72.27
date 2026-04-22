import numpy as np
from perceptrons.Perceptron import Perceptron

def _activation_tanh(value):
    return np.tanh(value)

# uses tan previous result (IT MUST NOT BE CALLED WITHOUT THE PREVIOUS RESULT) to calculate the derivative
def _activation_derivative(value):
    return 1 - value ** 2


class SimpleNonLinearPerceptron(Perceptron):
    # Non linear tanh perceptron. Must be used with data in which "y" has been normalized to the range [-1, 1]
    def __init__(self, learning_rate, epochs, epsilon, seed):
        super().__init__(learning_rate, epochs, seed)
        self.epsilon = epsilon


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize weights and bias to random values
        self._initialize_parameters(n_features)
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                x_i = X[i]
                y_i = y[i]

                prediction = self._predict_single(x_i)
                error = y_i - prediction
                # Δw = η · (y-o) · g'(z) · x, g = tanh; g'(z) = 1 - tanh²(z) = 1 - prediction²
                delta = error * self._activation_derivative(prediction)
                self.weights += (self.learning_rate * delta * x_i)
                self.bias += self.learning_rate * delta

            total_error = self._total_error(X,y)
            print(f"Epoch {epoch + 1}: total error = {total_error:.4f}")
            if total_error < self.epsilon:
                print(f"Converged at epoch {epoch + 1}")
                break


    def _total_error(self,X, y):
        predictions = self.predict(X)
        return np.sum((y - predictions) ** 2)

    def _activation(self, value):
        return _activation_tanh(value)
    
    def _activation_derivative(self, value):
        return _activation_derivative(value)
