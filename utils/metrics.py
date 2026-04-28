import os
import numpy as np
import pandas as pd


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


def epochs_to_threshold(val_accuracies, threshold):
    """Return the first epoch (1-indexed) where val accuracy >= threshold, or None."""
    for i, acc in enumerate(val_accuracies):
        if acc >= threshold:
            return i + 1
    return None


def save_perclass_csv(results, perclass_csv):
    """Append per-class precision/recall/F1 for every experiment to a CSV."""
    os.makedirs(os.path.dirname(perclass_csv), exist_ok=True)
    rows = []
    for r in results:
        cfg  = r["config"]
        base = {
            "name":        r["name"],
            "layers":      str(cfg.get("layers", "")),
            "lr":          cfg.get("lr", ""),
            "initializer": cfg.get("initializer", "random"),
        }
        for split, metrics in [("train", r["train_metrics"]), ("test", r["test_metrics"])]:
            for k in range(len(metrics["f1"])):
                rows.append({
                    **base,
                    "set":       split,
                    "class":     k,
                    "precision": round(metrics["precision"][k], 4),
                    "recall":    round(metrics["recall"][k],    4),
                    "f1":        round(metrics["f1"][k],        4),
                    "support":   int(metrics["support"][k]),
                })
    pd.DataFrame(rows).to_csv(perclass_csv, mode="a", header=not os.path.exists(perclass_csv), index=False)
    print(f"Per-class metrics saved → {perclass_csv}")
