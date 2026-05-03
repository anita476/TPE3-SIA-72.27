import numpy as np
from perceptrons.Perceptron import Perceptron


def _activation_identity(value):
    return value


def _recall_score(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> float:
    """
    Recall for the positive class (fraud = 1).
    recall = TP / (TP + FN)
    Returns 0.0 if there are no positive samples.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


class SimpleLinearPerceptron(Perceptron):
    def __init__(self, learning_rate, epochs, epsilon, seed):
        super().__init__(learning_rate, epochs, seed)
        self.epsilon = epsilon

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X, y         training data
        X_val, y_val optional validation data for test MSE / recall tracking.
                     If provided, test_mse_history_ and test_recall_history_
                     are populated each epoch.
        """
        n_samples, n_features = X.shape

        self._initialize_parameters(n_features)
        self.train_mse_history_:    list[float] = []
        self.test_mse_history_:     list[float] = []
        self.train_recall_history_: list[float] = []
        self.test_recall_history_:  list[float] = []

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

            train_pred = self.predict(X)
            train_mse  = self._total_error(X, y) / n_samples
            train_recall = _recall_score(y, train_pred)

            self.train_mse_history_.append(train_mse)
            self.train_recall_history_.append(train_recall)

            if compute_val:
                val_pred   = self.predict(X_val)
                val_mse    = self._total_error(X_val, y_val) / len(y_val)
                val_recall = _recall_score(y_val, val_pred)
                self.test_mse_history_.append(val_mse)
                self.test_recall_history_.append(val_recall)

            total_error = train_mse * n_samples
            print(f"Epoch {epoch + 1}: total error = {total_error:.4f}  recall = {train_recall:.4f}")
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