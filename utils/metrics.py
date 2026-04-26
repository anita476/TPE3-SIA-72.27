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


def save_results_csvs(results, summary_csv, curves_csv, perclass_csv):
    """Append experiment results to three CSV files (summary, per-epoch curves, per-class metrics)."""
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)

    summary_rows  = []
    curve_rows    = []
    perclass_rows = []

    for r in results:
        cfg = r["config"]
        base = {
            "name":              r["name"],
            "layers":            str(cfg["layers"]),
            "lr":                cfg["lr"],
            "epochs_configured": cfg["epochs"],
            "initializer":       cfg.get("initializer", "random"),
            "beta":              cfg.get("beta", 1.0),
        }
        tm = r["test_metrics"]
        summary_rows.append({
            **base,
            "train_acc":        round(r["train_acc"],        4),
            "test_acc":         round(r["test_acc"],         4),
            "best_test_acc":    round(r["best_test_acc"],    4),
            "best_epoch":       r["best_epoch"],
            "gap":              round(r["gap"],              4),
            "macro_precision":  round(tm["macro_precision"], 4),
            "macro_recall":     round(tm["macro_recall"],    4),
            "macro_f1":         round(tm["macro_f1"],        4),
            "min_class_f1":     round(tm["min_class_f1"],    4),
            "epochs_to_80":     r["epochs_to_80"],
            "epochs_to_85":     r["epochs_to_85"],
            "final_train_loss": round(r["final_train_loss"], 6) if r["final_train_loss"] else None,
            "final_test_loss":  round(r["final_test_loss"],  6) if r["final_test_loss"]  else None,
        })
        for epoch, (trl, tel, acc) in enumerate(
            zip(r["train_loss"], r["test_loss"], r["test_acc_per_epoch"]), start=1
        ):
            curve_rows.append({
                **base,
                "epoch":      epoch,
                "train_loss": round(trl, 6),
                "test_loss":  round(tel, 6),
                "test_acc":   round(acc, 4),
            })
        for split, metrics in [("train", r["train_metrics"]), ("test", tm)]:
            for k in range(len(metrics["f1"])):
                perclass_rows.append({
                    **base,
                    "set":       split,
                    "class":     k,
                    "precision": round(metrics["precision"][k], 4),
                    "recall":    round(metrics["recall"][k],    4),
                    "f1":        round(metrics["f1"][k],        4),
                    "support":   int(metrics["support"][k]),
                })

    pd.DataFrame(summary_rows).to_csv(summary_csv,  mode="a", header=not os.path.exists(summary_csv),  index=False)
    pd.DataFrame(curve_rows).to_csv(curves_csv,      mode="a", header=not os.path.exists(curves_csv),   index=False)
    pd.DataFrame(perclass_rows).to_csv(perclass_csv, mode="a", header=not os.path.exists(perclass_csv), index=False)
    print(f"Results saved to {os.path.dirname(summary_csv)}/")
