from __future__ import annotations

import numpy as np


def binarize(y: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a continuous array to binary {0, 1} using `threshold`."""
    return (np.asarray(y, dtype=float) >= threshold).astype(int)


def precision_recall_f1(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
) -> tuple[float, float, float]:
    """Precision, recall, and F1 for the positive class (1).

    Returns (0, 0, 0) when there are no positive predictions or no positive
    ground-truth samples, instead of raising a division-by-zero error.
    """
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    return float(precision), float(recall), float(f1)


def roc_curve_points(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve points by sweeping `n_thresholds` decision thresholds.

    Returns (fpr, tpr, thresholds) including the corner points (0, 0) and (1, 1).
    y_true is binarized at 0.5 internally.
    """
    y_true_bin = binarize(y_true)
    p = int(y_true_bin.sum())
    n = len(y_true_bin) - p

    if p == 0 or n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    score_min = float(y_scores.min())
    score_max = float(y_scores.max())
    # Sweep from high to low so the curve goes from (0,0) to (1,1)
    thresholds = np.linspace(score_max + 1e-9, score_min - 1e-9, n_thresholds)

    fprs, tprs = [], []
    for thr in thresholds:
        pred = (y_scores >= thr).astype(int)
        tp = int(np.sum((y_true_bin == 1) & (pred == 1)))
        fp = int(np.sum((y_true_bin == 0) & (pred == 1)))
        fprs.append(fp / n)
        tprs.append(tp / p)

    fpr = np.array([0.0] + fprs + [1.0])
    tpr = np.array([0.0] + tprs + [1.0])
    thr_arr = np.concatenate([
        [thresholds[0] + 1e-9],
        thresholds,
        [thresholds[-1] - 1e-9],
    ])
    return fpr, tpr, thr_arr


def roc_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 200,
) -> float:
    """Area under the ROC curve (trapezoidal rule)."""
    fpr, tpr, _ = roc_curve_points(y_true, y_scores, n_thresholds)
    order = np.argsort(fpr)
    x, y = fpr[order], tpr[order]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def find_best_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 200,
) -> dict:
    """Sweep thresholds and return the one maximising `metric`.

    metric: 'f1' | 'recall' | 'precision'
    Returns a dict with keys: threshold, precision, recall, f1.
    """
    y_true_bin = binarize(y_true)
    thresholds = np.linspace(float(y_scores.min()), float(y_scores.max()), n_thresholds)

    best_val = -1.0
    mid = float(thresholds[len(thresholds) // 2])
    best: dict = {"threshold": mid, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for thr in thresholds:
        pred = (y_scores >= thr).astype(int)
        p, r, f = precision_recall_f1(y_true_bin, pred)
        val = {"f1": f, "recall": r, "precision": p}.get(metric, f)
        if val > best_val:
            best_val = val
            best = {"threshold": float(thr), "precision": p, "recall": r, "f1": f}

    return best


def metrics_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> dict:
    """Compute binary metrics at a specific decision threshold."""
    y_true_bin = binarize(y_true)
    pred = (y_scores >= threshold).astype(int)
    p, r, f = precision_recall_f1(y_true_bin, pred)
    return {"threshold": threshold, "precision": p, "recall": r, "f1": f}
