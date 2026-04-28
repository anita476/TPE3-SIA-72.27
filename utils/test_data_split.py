import numpy as np


def test_data_split(X, y, test_size=0.2, random_state=None):
    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    indices = rng.permutation(n_samples)  # shuffle indices

    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]  # array[start(0):end] slices the list. w return first n_test
    train_indices = indices[n_test:] # we return last n_test indices

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