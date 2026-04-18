import numpy as np

def test_data_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.random.permutation(n_samples)  # shuffle indices

    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]  # array[start(0):end] slices the list. w return first n_test
    train_indices = indices[n_test:] # we return last n_test indices

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]