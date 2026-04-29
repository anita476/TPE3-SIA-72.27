import json
import numpy as np
from utils.initializers import build_initializer
from utils.optimizers import build_optimizer


def _tanh(h, beta=1.0):
    return np.tanh(beta * h)

# Calculates tanh again so it uses h and not g(h).
def _tanh_derivative(h, beta=1.0):
    return beta * (1 - np.tanh(beta * h)**2)

def _logistic(h, beta=1.0):
    return 1.0 / (1.0 + np.exp(-2 * beta * h))

def _logistic_derivative(h, beta=1.0):
    g_h = _logistic(h, beta)
    return 2 * beta * g_h * (1 - g_h)

ACTIVATIONS = {
    "tanh": (_tanh, _tanh_derivative),
    "logistic": (_logistic, _logistic_derivative),
}
TRAINING_MODES = {"online", "minibatch"}

class MultiLayerPerceptron:
    """Feed-forward multilayer perceptron trained with backpropagation."""

    def __init__(self, layers, learning_rate, epochs, epsilon, seed, beta=1.0, activation="tanh", initializer="random", training_mode="online", batch_size=1, optimizer="sgd", weight_decay=0.0, patience=0):
        if len(layers) < 2:
            raise ValueError("layers must contain at least input and output sizes")
        if activation not in ACTIVATIONS:
            activation = "tanh"
        if training_mode not in TRAINING_MODES:
            raise ValueError(f"training_mode must be one of {sorted(TRAINING_MODES)}")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        # layers=[2, 2, 1] => 2 inputs, 2 hidden neurons, 1 output.
        self.layers = list(layers)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.beta = beta
        self.activation = activation
        self.g, self.g_prime = ACTIVATIONS[activation]
        self.rng = np.random.default_rng(seed)
        self.training_mode = training_mode
        self.batch_size = int(batch_size)
        self.weight_decay = weight_decay
        self.patience = patience
        self._optimizer_name = optimizer
        self.optimizer = build_optimizer(optimizer, learning_rate)
        init = build_initializer(initializer)

        # One weight matrix per connection between consecutive layers.
        # Shape: (fan_out, fan_in) so that:
        #   single sample: h = W @ x
        #   batch:         H = X @ W.T  (X shape: n × fan_in)
        self.weights = [
            init.init_weights(self.layers[l - 1], self.layers[l], self.rng)
            for l in range(1, len(self.layers))
        ]

        # One bias vector per layer (input layer has no bias).
        self.biases = [
            init.init_biases(self.layers[l], self.rng)
            for l in range(1, len(self.layers))
        ]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_batch(self, X):
        """Vectorized forward pass for a batch X of shape (n, fan_in).

        Returns:
            activations     – list of (n, layer_size) arrays; [0] is the input.
            pre_activations – list of (n, layer_size) arrays; h values before g().
        """
        activations = [np.asarray(X, dtype=float)]
        pre_activations = []
        for W, b in zip(self.weights, self.biases):
            h = activations[-1] @ W.T + b   # (n, fan_out)
            pre_activations.append(h)
            activations.append(self.g(h, self.beta))
        return activations, pre_activations

    def _forward(self, x):
        """Single-sample forward pass (kept for compute_saliency)."""
        activations = [np.asarray(x, dtype=float)]
        pre_activations = []
        for W, b in zip(self.weights, self.biases):
            h = W @ activations[-1] + b
            pre_activations.append(h)
            activations.append(self.g(h, self.beta))
        return activations, pre_activations

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _effective_batch_size(self, n_samples):
        if self._optimizer_name == "gd":
            return max(1, n_samples)
        return 1 if self.training_mode == "online" else self.batch_size

    def _train_batch_epoch(self, X, y, indices):
        effective_batch_size = self._effective_batch_size(len(indices))

        for start in range(0, len(indices), effective_batch_size):
            batch_idx = indices[start:start + effective_batch_size]
            X_batch = X[batch_idx]   # (n, fan_in)
            y_batch = y[batch_idx]   # (n, output_size)
            n = len(batch_idx)

            activations, pre_activations = self._forward_batch(X_batch)

            # --- Vectorized backward pass ---
            dW = [None] * len(self.weights)
            db = [None] * len(self.biases)

            # Output layer: δ = (ŷ − y) ⊙ g'(h),  shape (n, output_size)
            delta = (activations[-1] - y_batch) * self.g_prime(pre_activations[-1], self.beta)
            # dW = (1/n) δᵀ A_prev + λW,  shape (fan_out, fan_in) = W shape
            dW[-1] = delta.T @ activations[-2] / n + self.weight_decay * self.weights[-1]
            db[-1] = delta.mean(axis=0)

            # Hidden layers: propagate δ backwards
            for l in range(len(self.weights) - 2, -1, -1):
                delta = (delta @ self.weights[l + 1]) * self.g_prime(pre_activations[l], self.beta)
                dW[l] = delta.T @ activations[l] / n + self.weight_decay * self.weights[l]
                db[l] = delta.mean(axis=0)

            self.optimizer.update(self.weights, dW, self.biases, db)

    def train_epoch(self, X, y):
        indices = self.rng.permutation(X.shape[0])
        self._train_batch_epoch(X, y, indices)

    def fit(self, X, y, X_val=None, y_val=None, val_labels=None, train_labels=None, name=""):
        self.errors_ = []
        self.val_errors_ = []
        self.val_accuracies_ = []
        self.train_accuracies_ = []
        best_val_error = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.train_epoch(X, y)

            total_error = self._total_error(X, y)
            self.errors_.append(total_error)

            if train_labels is not None:
                train_preds = np.argmax(self.predict(X), axis=1)
                self.train_accuracies_.append(float(np.mean(train_preds == train_labels)))

            if X_val is not None and y_val is not None:
                val_error = self._total_error(X_val, y_val)
                self.val_errors_.append(val_error)

                if val_labels is not None:
                    val_preds = np.argmax(self.predict(X_val), axis=1)
                    self.val_accuracies_.append(np.mean(val_preds == val_labels))

                print(f"[{name}] Epoch {epoch + 1}: train error = {total_error:.4f}  validation error = {val_error:.4f}")

                if self.patience > 0:
                    if val_error < best_val_error:
                        best_val_error = val_error
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(f"[{name}] Early stopping at epoch {epoch + 1}")
                        break
            else:
                print(f"[{name}] Epoch {epoch + 1}: total error = {total_error:.4f}")

            if total_error < self.epsilon:
                print(f"[{name}] Converged at epoch {epoch + 1}")
                break

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X):
        """Vectorized predict for a batch X of shape (n, fan_in)."""
        a = np.asarray(X, dtype=float)
        for W, b in zip(self.weights, self.biases):
            a = self.g(a @ W.T + b, self.beta)
        # Single-output networks: return flat vector instead of (n, 1).
        return a.flatten() if a.shape[1] == 1 else a

    def _total_error(self, X, y):
        predictions = self.predict(X)
        return 0.5 * np.sum((y - predictions) ** 2)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Save weights, biases, and hyperparameters to a .npz file."""
        config = {
            "layers":        self.layers,
            "learning_rate": self.learning_rate,
            "epochs":        self.epochs,
            "epsilon":       self.epsilon,
            "beta":          self.beta,
            "activation":    self.activation,
            "training_mode": self.training_mode,
            "batch_size":    self.batch_size,
            "weight_decay":  self.weight_decay,
            "patience":      self.patience,
            "optimizer":     self._optimizer_name,
        }
        arrays = {f"W{i}": W for i, W in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        arrays["_config"] = np.array(json.dumps(config))
        # Include training history if fit() has been called
        if hasattr(self, "errors_"):
            arrays["_train_loss"]  = np.array(self.errors_)
            arrays["_val_loss"]    = np.array(self.val_errors_)
            arrays["_val_acc"]     = np.array(self.val_accuracies_)
            arrays["_train_acc"]   = np.array(self.train_accuracies_)
        np.savez(path, **arrays)
        print(f"Model saved → {path}.npz")

    @classmethod
    def load(cls, path, seed=0):
        """Load a model from a .npz file produced by save()."""
        p = path if path.endswith(".npz") else path + ".npz"
        data = np.load(p, allow_pickle=True)
        config = json.loads(str(data["_config"]))
        model = cls(
            layers=config["layers"],
            learning_rate=config["learning_rate"],
            epochs=config["epochs"],
            epsilon=config["epsilon"],
            seed=seed,
            beta=config["beta"],
            activation=config["activation"],
            training_mode=config.get("training_mode", "online"),
            batch_size=config.get("batch_size", 1),
            optimizer=config.get("optimizer", "sgd"),
            weight_decay=config.get("weight_decay", 0.0),
            patience=config.get("patience", 0),
        )
        n = len(config["layers"]) - 1
        model.weights = [data[f"W{i}"] for i in range(n)]
        model.biases  = [data[f"b{i}"] for i in range(n)]
        # Restore training history if present
        if "_train_loss" in data:
            model.errors_           = data["_train_loss"].tolist()
            model.val_errors_       = data["_val_loss"].tolist()
            model.val_accuracies_   = data["_val_acc"].tolist()
            model.train_accuracies_ = data["_train_acc"].tolist() if "_train_acc" in data else []
        print(f"Model loaded ← {p}")
        return model
