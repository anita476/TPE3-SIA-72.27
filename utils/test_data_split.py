import numpy as np


def test_data_split(X, y, test_size=0.2, random_state=None):
    """Random train/test split (no stratification)."""
    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    indices = rng.permutation(n_samples)

    n_test = int(n_samples * test_size)
    test_indices  = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def stratified_split(X, y_labels, val_size=0.2, random_state=None):
    """Split X preserving the class distribution in both halves."""
    rng = np.random.default_rng(random_state)
    train_idx, val_idx = [], []
    for cls in np.unique(y_labels):
        cls_idx = np.where(y_labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_size))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    return X[train_idx], X[val_idx], y_labels[train_idx], y_labels[val_idx]


def stratified_split_regression(
    X: np.ndarray,
    y: np.ndarray,
    stratify_by: np.ndarray,
    val_size: float = 0.2,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split where targets `y` may be continuous.

    Unlike `stratified_split`, the returned `y` arrays preserve the original
    (possibly continuous) targets, while class balance is enforced using
    `stratify_by` (a binary or discrete integer array).

    Useful for Knowledge Distillation where training targets are soft probability
    scores from a teacher model but the dataset may be class-imbalanced.
    """
    rng = np.random.default_rng(random_state)
    train_idx, val_idx = [], []
    for cls in np.unique(stratify_by):
        cls_idx = np.where(stratify_by == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_size))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]