import numpy as np


# Good initialization keeps signal variance constant across layers,
# avoiding vanishing (tanh saturates -> gradients == 0) or exploding gradients.
#
# Interface:
#   init_weights(fan_in, fan_out, rng) → ndarray (fan_out, fan_in)
#   init_biases(fan_out, rng)          → ndarray (fan_out,)  [always zeros]


class RandomUniform:
    """
    W ~ U(-limit, +limit), fixed range regardless of layer size.
    Problem: large fan_in causes tanh saturation from epoch 1.
    Use for: tiny networks or as a naive baseline.
    """

    def __init__(self, limit=1.0):
        self.limit = limit

    def init_weights(self, fan_in, fan_out, rng):
        return rng.uniform(-self.limit, self.limit, (fan_out, fan_in))

    def init_biases(self, fan_out, rng):
        return np.zeros(fan_out)


class Xavier:
    """
    Scales weights by layer size so variance
    is preserved forward and backward:  Var(W) = 2 / (fan_in + fan_out).

    Uniform: W ~ U(-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out)))
    Normal:  W ~ N(0, √(2/(fan_in+fan_out)))

    Use for: tanh, sigmoid (near-linear activations).
    Avoid:   ReLU (use He instead).
    """

    def __init__(self, mode="uniform"):
        if mode not in ("uniform", "normal"):
            raise ValueError("mode must be 'uniform' or 'normal'")
        self.mode = mode

    def init_weights(self, fan_in, fan_out, rng):
        if self.mode == "uniform":
            limit = np.sqrt(6 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, (fan_out, fan_in))
        std = np.sqrt(2 / (fan_in + fan_out))
        return rng.normal(0, std, (fan_out, fan_in))

    def init_biases(self, fan_out, rng):
        return np.zeros(fan_out)


def build_initializer(name, **kwargs):
    """
    "random"   -> RandomUniform  (limit=1.0)
    "xavier"   -> Xavier uniform  <- use with tanh
    "xavier_n" -> Xavier normal
    """
    if name == "random":
        return RandomUniform(**kwargs)
    elif name == "xavier":
        return Xavier(mode="uniform", **kwargs)
    elif name == "xavier_n":
        return Xavier(mode="normal", **kwargs)
    raise ValueError(f"Unknown initializer: '{name}'. Choose from: random, xavier, xavier_n")
