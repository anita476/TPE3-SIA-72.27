import numpy as np
from perceptrons.initializers import build_initializer


def _tanh(h, beta=1.0):
    return np.tanh(beta * h)

#Calculates tan again so it uses h and not g(h)
def _tanh_derivative(h, beta=1.0):
    return beta * (1 - np.tanh(beta * h)**2)

class MultiLayerPerceptron:
    """Feed-forward multilayer perceptron trained with backpropagation."""

    def __init__(self, layers, learning_rate, epochs, epsilon, seed, beta=1.0,
                 initializer="random"):
        if len(layers) < 2:
            raise ValueError("layers must contain at least input and output sizes")

        # layers=[2, 2, 1] => 2 inputs, 2 hidden neurons, 1 output.
        self.layers = list(layers)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.beta = beta
        self.rng = np.random.default_rng(seed)

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
        # neuron values before applying tanh.
        pre_activations = []

        for W, b in zip(self.weights, self.biases):
            # Weighted sum for the current layer: h= W*a_previous +b.
            # should work exactly as using the auggmented matrix for the calculation
            # @ = dot product
            h = W @ activations[-1] + b
            pre_activations.append(h)

            # apply tanh activation and send it to the next layer.
            activations.append(_tanh(h, self.beta))

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
        delta = (activations[-1]-y_true) * _tanh_derivative(pre_activations[-1], self.beta)
        # Weight gradient for a layer is delta outer-product previous activation.
        # δᵢ​⋅Vⱼ^(M-1)
        # Vⱼ^(M-1) = activations[-2]
        dW[-1] = np.outer(delta, activations[-2])
        db[-1] = delta

        # hidden layers (from last one to first)
        for l in range(L_weights - 2, -1, -1):
            # Move the next layer's error backward, then scale by tanh slope here.
            # Delta is from the layer on top
            # θ'(hⱼ^(l)) evaluated in pre-activación from this layer = pre_activations[l]
            # ∑ᵢ​δᵢ​Wᵢⱼ = self.weights[l + 1].T @ delta
            # θ'(hᵢ^m) = _tanh_derivative(pre_activations[l], self.beta)
            # Vⱼ^(m-1) = activations[l]
            # δᵢ^m​⋅Vⱼ^(m-1)
            delta = (self.weights[l + 1].T @ delta) * _tanh_derivative(pre_activations[l], self.beta)
            dW[l] = np.outer(delta, activations[l])
            db[l] = delta

        return dW, db

    def fit(self, X, y, X_val=None, y_val=None, val_labels=None, name=""):
        n_samples = X.shape[0]
        self.errors_ = []
        self.val_errors_ = []
        self.val_accuracies_ = []
        for epoch in range(self.epochs):
            indices = self.rng.permutation(n_samples)

            for i in indices:
                activations, pre_acts = self._forward(X[i])
                dW, db = self._backward(activations, pre_acts, y[i])
                for W, dw, b, d in zip(self.weights, dW, self.biases, db):
                    W -= self.learning_rate * dw
                    b -= self.learning_rate * d

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
