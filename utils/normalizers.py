import numpy as np

_STD_EPS = 1e-12


def standard_scale_params(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std per column of X (fit reference). Std below eps is set to 1 to avoid division by zero."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < _STD_EPS, 1.0, std)
    return mean, std


def standard_scale_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply (X - mean) / std with broadcast row-wise."""
    return (X - mean) / std
