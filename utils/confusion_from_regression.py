from __future__ import annotations

from typing import Literal

import numpy as np


def resolve_confusion_mode(
    y_train: np.ndarray,
    mode: str,
    discrete_max_unique: int = 15,
) -> Literal["discrete", "binned"]:
    if mode == "discrete":
        return "discrete"
    if mode == "binned":
        return "binned"
    if mode != "auto":
        raise ValueError(f"Unknown confusion mode: {mode}")
    u = np.unique(y_train)
    return "discrete" if len(u) <= discrete_max_unique else "binned"


def _nearest_class_index(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map each scalar to index of nearest value in `classes` (1d sorted)."""
    classes = np.asarray(classes, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1, 1)
    c = classes.reshape(1, -1)
    return np.abs(v - c).argmin(axis=1).astype(int)


def confusion_matrix_discrete(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rows = true class index, cols = predicted class index (nearest label in reference set).

    `y_reference` is typically concatenation of train+test labels to define the label set.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    uniques = np.sort(np.unique(np.asarray(y_reference, dtype=float).ravel()))
    n = len(uniques)
    cm = np.zeros((n, n), dtype=int)
    ti = _nearest_class_index(y_true, uniques)
    pi = _nearest_class_index(y_pred, uniques)
    for i, j in zip(ti, pi, strict=True):
        cm[i, j] += 1
    return cm, uniques


def bin_range_from_train(
    y_train: np.ndarray,
    y_eval: np.ndarray | None = None,
) -> tuple[float, float]:
    yt = np.asarray(y_train, dtype=float).ravel()
    lo, hi = float(yt.min()), float(yt.max())
    if y_eval is not None:
        ye = np.asarray(y_eval, dtype=float).ravel()
        lo = min(lo, float(ye.min()))
        hi = max(hi, float(ye.max()))
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi


def confusion_matrix_binned(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Both true and predicted values are assigned to bin indices 0..n_bins-1 using the same edges on [lo, hi].
    Returns (cm, edges) where edges has length n_bins + 1.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    edges = np.linspace(lo, hi, int(n_bins) + 1, dtype=float)
    cm = np.zeros((n_bins, n_bins), dtype=int)

    def to_bin(y: np.ndarray) -> np.ndarray:
        span = hi - lo
        if span <= 0:
            return np.zeros(len(y), dtype=int)
        yc = np.clip(y, lo, hi)
        idx = np.floor((yc - lo) / span * n_bins).astype(int)
        # Include right endpoint in last bin
        idx = np.clip(idx, 0, n_bins - 1)
        idx = np.where(y >= hi, n_bins - 1, idx)
        return idx

    ti = to_bin(y_true)
    pi = to_bin(y_pred)
    for i, j in zip(ti, pi, strict=True):
        cm[i, j] += 1
    return cm, edges


def flatten_confusion_rows(
    run_id: str,
    cm: np.ndarray,
) -> list[dict[str, int | str]]:
    """One CSV row per cell (i, j, count)."""
    rows: list[dict[str, int | str]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            rows.append(
                {
                    "run_id": run_id,
                    "i": i,
                    "j": j,
                    "count": int(cm[i, j]),
                }
            )
    return rows
