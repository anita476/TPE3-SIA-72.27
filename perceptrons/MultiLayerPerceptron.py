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

    def __init__(self, layers, learning_rate, epochs, epsilon, seed, beta=1.0, activation="tanh", initializer="random", training_mode="online", batch_size=1, optimizer="sgd"):
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

        self.optimizer = build_optimizer(optimizer, learning_rate)
        init = build_initializer(initializer)

        # One weight matrix per connection between consecutive layers.
        # Matrix shape: (current_layer_size, previous_layer_size).
        self.weights = [
            init.init_weights(self.layers[l - 1], self.layers[l], self.rng)
            for l in range(1, len(self.layers))
        ]

        # One bias vector per layer (input layer has no bias).
        self.biases = [
            init.init_biases(self.layers[l], self.rng)
            for l in range(1, len(self.layers))
        ]

    def _forward(self, x):
        """Run one sample through the network and keep intermediate values."""
        # input layer => later entries are layer last after first iteration.
        activations = [np.asarray(x, dtype=float)]
        # neuron values before applying the activation function.
        pre_activations = []

        for W, b in zip(self.weights, self.biases):
            # Weighted sum for the current layer: h= W*a_previous +b.
            # should work exactly as using the auggmented matrix for the calculation
            # @ = dot product
            h = W @ activations[-1] + b
            pre_activations.append(h)

            # apply activation and send it to the next layer.
            activations.append(self.g(h, self.beta))

        return activations, pre_activations

    def _backward(self, activations, pre_activations, y_true):
        """Compute weight and bias gradients with backpropagation."""
        # Scalar targets will behave like vectors.
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        L_weights = len(self.weights)
        L_biases = len(self.biases)

        # dW and db will store gradients for every weight and bias of the neural network.
        dW = [None] * L_weights
        db = [None] * L_biases

        # output layer
        # (-1)(ζᵢ − Oᵢ) = (activations[-1] - y_true)
        # θ'(hᵢ) = _tanh_derivative(...)
        # pre_activations[-1] = h^M
        delta = (activations[-1]-y_true) * self.g_prime(pre_activations[-1], self.beta)
        # Weight gradient for a layer is delta outer-product previous activation.
        # δᵢ​⋅Vⱼ^(M-1)
        # Vⱼ^(M-1) = activations[-2]
        dW[-1] = np.outer(delta, activations[-2])
        db[-1] = delta

        # hidden layers (from last one to first)
        for l in range(L_weights - 2, -1, -1):
            # Move the next layer's error backward, then scale by activation slope here.
            # Delta is from the layer on top
            # θ'(hⱼ^(l)) evaluated in pre-activación from this layer = pre_activations[l]
            # ∑ᵢ​δᵢ​Wᵢⱼ = self.weights[l + 1].T @ delta
            # θ'(hᵢ^m) = _tanh_derivative(pre_activations[l], self.beta)
            # Vⱼ^(m-1) = activations[l]
            # δᵢ^m​⋅Vⱼ^(m-1)
            delta = (self.weights[l + 1].T @ delta) * self.g_prime(pre_activations[l], self.beta)
            dW[l] = np.outer(delta, activations[l])
            db[l] = delta

        return dW, db

    def _apply_update(self, dW, db):
        self.optimizer.update(self.weights, dW, self.biases, db)

    def _zero_gradients(self):
        return [np.zeros_like(W) for W in self.weights], [np.zeros_like(b) for b in self.biases]

    def _train_batch_epoch(self, X, y, indices):
        effective_batch_size = 1 if self.training_mode == "online" else self.batch_size

        for start in range(0, len(indices), effective_batch_size):
            batch_indices = indices[start:start + effective_batch_size]
            batch_dW, batch_db = self._zero_gradients()

            for i in batch_indices:
                activations, pre_acts = self._forward(X[i])
                dW, db = self._backward(activations, pre_acts, y[i])
                for acc, grad in zip(batch_dW, dW):
                    acc += grad
                for acc, grad in zip(batch_db, db):
                    acc += grad

            current_batch_size = len(batch_indices)
            mean_dW = [grad / current_batch_size for grad in batch_dW]
            mean_db = [grad / current_batch_size for grad in batch_db]
            self._apply_update(mean_dW, mean_db)

    def train_epoch(self, X, y):
        n_samples = X.shape[0]
        indices = self.rng.permutation(n_samples)
        self._train_batch_epoch(X, y, indices)

    def fit(self, X, y, X_val=None, y_val=None, val_labels=None, name=""):
        self.errors_ = []
        self.val_errors_ = []
        self.val_accuracies_ = []
        for epoch in range(self.epochs):
            self.train_epoch(X, y)

            total_error = self._total_error(X, y)
            self.errors_.append(total_error)

            if X_val is not None and y_val is not None:
                val_error = self._total_error(X_val, y_val)
                self.val_errors_.append(val_error)

                if val_labels is not None:
                    val_preds = np.argmax(self.predict(X_val), axis=1)
                    self.val_accuracies_.append(np.mean(val_preds == val_labels))

                print(f"[{name}] Epoch {epoch + 1}: train error = {total_error:.4f}  test error = {val_error:.4f}")
            else:
                print(f"[{name}] Epoch {epoch + 1}: total error = {total_error:.4f}")

            if total_error < self.epsilon:
                print(f"[{name}] Converged at epoch {epoch + 1}")
                break

    def predict(self, X):
        # For each sample, run a forward pass and keep only the final activation.
        outputs = np.array([self._forward(x)[0][-1] for x in X])

        # Single-output networks return a flat vector instead of shape (n, 1).
        return outputs.flatten() if outputs.shape[1] == 1 else outputs

    def _total_error(self, X, y):
        predictions = self.predict(X)
        return 0.5 * np.sum((y - predictions) ** 2)
