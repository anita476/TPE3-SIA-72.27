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


def _recall_score(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_pred_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


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
        self.activation_name = activation
        self.g, self.g_prime = ACTIVATIONS[activation]

    def _activation(self, value):
        return self.g(value, self.beta)

    def _pred_to_prob(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Map raw predictions to [0, 1] probability space
        """
        if self.activation_name == 'tanh':
            return (y_pred + 1.0) / 2.0
        return y_pred

    def fit(self, X, y, X_val=None, y_val=None):
        """
        For tanh, labels are internally scaled to {-1, +1} during training.
        All metrics (MSE, recall) are computed on the original {0, 1} scale.
        """
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        self.train_mse_history_:    list[float] = []
        self.test_mse_history_:     list[float] = []
        self.train_recall_history_: list[float] = []
        self.test_recall_history_:  list[float] = []

        # Scale labels for tanh
        y_train = (2.0 * y - 1.0) if self.activation_name == 'tanh' else y

        compute_val = X_val is not None and y_val is not None

        for epoch in range(self.epochs):
            indices = self.rng.permutation(n_samples)
            for i in indices:
                x_i = X[i]
                y_i = y_train[i]

                prediction = self._predict_single(x_i)
                error      = y_i - prediction
                delta      = error * self.g_prime(prediction, self.beta)
                self.weights += self.learning_rate * delta * x_i
                self.bias    += self.learning_rate * delta

            # Raw predictions (may be in (-1,1) for tanh)
            raw_pred  = self.predict(X)
            prob_pred = self._pred_to_prob(raw_pred)   # always in [0, 1]

            train_mse    = float(np.mean((y - prob_pred) ** 2))
            train_recall = _recall_score(y, prob_pred)

            self.train_mse_history_.append(train_mse)
            self.train_recall_history_.append(train_recall)

            if compute_val:
                raw_val    = self.predict(X_val)
                prob_val   = self._pred_to_prob(raw_val)
                val_mse    = float(np.mean((y_val - prob_val) ** 2))
                val_recall = _recall_score(y_val, prob_val)
                self.test_mse_history_.append(val_mse)
                self.test_recall_history_.append(val_recall)

            total_error = float(np.sum((y_train - raw_pred) ** 2))
            print(f"Epoch {epoch + 1}: total error = {total_error:.4f}  recall = {train_recall:.4f}")
            if total_error < self.epsilon:
                print(f"Converged at epoch {epoch + 1}")
                self.epochs_run_ = epoch + 1
                break
        else:
            self.epochs_run_ = self.epochs

    def _total_error(self, X, y):
        return np.sum((y - self.predict(X)) ** 2)