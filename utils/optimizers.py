import numpy as np


class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, weights, dW, biases, db):
        for W, dw, b, d in zip(weights, dW, biases, db):
            W -= self.learning_rate * dw
            b -= self.learning_rate * d


class RMSProp:
    # v_W = beta * v_W + (1-beta) * dW^2
    # v_b = beta * v_b + (1-beta) * db^2
    # W   = W - lr * dW / (sqrt(v_W) + eps)
    # b   = b - lr * db / (sqrt(v_b) + eps)
    def __init__(self, learning_rate, beta=0.9, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps
        self._v_W = None
        self._v_b = None

    def update(self, weights, dW, biases, db):
        if self._v_W is None:
            self._v_W = [np.zeros_like(W) for W in weights]
            self._v_b = [np.zeros_like(b) for b in biases]

        for i, (W, dw, b, d) in enumerate(zip(weights, dW, biases, db)):
            self._v_W[i] = self.beta * self._v_W[i] + (1 - self.beta) * dw ** 2
            self._v_b[i] = self.beta * self._v_b[i] + (1 - self.beta) * d  ** 2
            W -= self.learning_rate / (np.sqrt(self._v_W[i]) + self.eps) * dw
            b -= self.learning_rate / (np.sqrt(self._v_b[i]) + self.eps) * d


# "Good default settings for the tested ml problems are alpha=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10^-8" (https://arxiv.org/pdf/1412.6980)
class Adam:
    # m_W = beta1 * m_W + (1-beta1) * dW        (mean of gradients)
    # v_W = beta2 * v_W + (1-beta2) * dW^2      (mean of squared gradients)
    # m_W_hat = m_W / (1 - beta1^t)             (bias correction)
    # v_W_hat = v_W / (1 - beta2^t)
    # W = W - lr * m_W_hat / (sqrt(v_W_hat) + eps)
    # same for biases
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m_W = None
        self._v_W = None
        self._m_b = None
        self._v_b = None
        self._t = 0

    def update(self, weights, dW, biases, db):
        if self._m_W is None:
            self._m_W = [np.zeros_like(W) for W in weights]
            self._v_W = [np.zeros_like(W) for W in weights]
            self._m_b = [np.zeros_like(b) for b in biases]
            self._v_b = [np.zeros_like(b) for b in biases]

        self._t += 1
        for i, (W, dw, b, d) in enumerate(zip(weights, dW, biases, db)):
            self._m_W[i] = self.beta1 * self._m_W[i] + (1 - self.beta1) * dw
            self._v_W[i] = self.beta2 * self._v_W[i] + (1 - self.beta2) * dw ** 2
            self._m_b[i] = self.beta1 * self._m_b[i] + (1 - self.beta1) * d
            self._v_b[i] = self.beta2 * self._v_b[i] + (1 - self.beta2) * d  ** 2

            m_W_hat = self._m_W[i] / (1 - self.beta1 ** self._t)
            v_W_hat = self._v_W[i] / (1 - self.beta2 ** self._t)
            m_b_hat = self._m_b[i] / (1 - self.beta1 ** self._t)
            v_b_hat = self._v_b[i] / (1 - self.beta2 ** self._t)

            W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


def build_optimizer(name, learning_rate):
    if name in {"gd", "sgd"}:
        return SGD(learning_rate)
    elif name == "rmsprop":
        return RMSProp(learning_rate)
    elif name == "adam":
        return Adam(learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")
