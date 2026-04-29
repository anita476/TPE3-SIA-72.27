import json
import hashlib
import numpy as np


def compute_config_id(config):
    """8-char hash of hyperparameters (excludes name, seed, seeds).

    Two configs with identical architecture/optimizer/training settings
    produce the same id regardless of their name or random seed.
    """
    excluded = {"name", "seed", "seeds"}
    canonical = {k: v for k, v in sorted(config.items()) if k not in excluded}
    h = hashlib.md5(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    return h[:8]


def compute_metrics(y_true, y_pred, n_classes=10):
    """
    Builds a confusion matrix and derives per-class and macro metrics.

    Confusion matrix C[i,j] = number of samples of true class i predicted as class j.
    From it, for each class k:
        TP[k] = C[k,k]
        FP[k] = sum(C[:,k]) - C[k,k]   (predicted k but were something else)
        FN[k] = sum(C[k,:]) - C[k,k]   (were k but predicted something else)

        Precision[k] = TP / (TP + FP)  — of all predicted k, how many are truly k
        Recall[k]    = TP / (TP + FN)  — of all true k, how many did we catch
        F1[k]        = 2 * P * R / (P + R)

    Macro averages treat every class equally (simple mean across classes).
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    precision = np.zeros(n_classes)
    recall    = np.zeros(n_classes)
    f1        = np.zeros(n_classes)

    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        precision[k] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[k]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom        = precision[k] + recall[k]
        f1[k]        = 2 * precision[k] * recall[k] / denom if denom > 0 else 0.0

    return {
        "confusion_matrix": cm,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "support":          cm.sum(axis=1),
        "macro_precision":  precision.mean(),
        "macro_recall":     recall.mean(),
        "macro_f1":         f1.mean(),
        "min_class_f1":     f1.min(),
    }




