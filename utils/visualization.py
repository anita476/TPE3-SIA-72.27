import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

PLOTS_DIR = "plots"


def _save(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Multi-experiment plots ────────────────────────────────────────────────────

def plot_loss_curves(results):
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        epochs = range(1, len(r["train_loss"]) + 1)
        ax_train.plot(epochs, r["train_loss"], label=r["name"])
        ax_test.plot(epochs,  r["test_loss"],  label=r["name"])
    for ax, title in [(ax_train, "Train loss"), (ax_test, "Test loss")]:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "loss_curves.png")


def plot_accuracy_bars(results):
    names      = [r["name"]      for r in results]
    train_accs = [r["train_acc"] for r in results]
    test_accs  = [r["test_acc"]  for r in results]
    x     = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
    ax.bar(x - width / 2, train_accs, width, label="Train")
    ax.bar(x + width / 2, test_accs,  width, label="Test")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs test accuracy per experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend()
    for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
        ax.text(i - width / 2, tr + 0.01, f"{tr*100:.1f}%", ha="center", fontsize=7)
        ax.text(i + width / 2, te + 0.01, f"{te*100:.1f}%", ha="center", fontsize=7)
    fig.tight_layout()
    _save(fig, "accuracy_bars.png")


def plot_convergence(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        accs = r["test_acc_per_epoch"]
        if accs:
            ax.plot(range(1, len(accs) + 1), [a * 100 for a in accs], label=r["name"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Test accuracy over epochs")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "convergence.png")


def plot_confusion_matrices(results):
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm        = r["test_metrics"]["confusion_matrix"]
        im        = ax.imshow(cm, cmap="Blues")
        threshold = cm.max() / 2
        ax.set_title(r["name"], fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > threshold else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


def plot_per_class_f1(results):
    """Heatmap: rows = digit classes, columns = experiments. Color = F1 score."""
    names  = [r["name"] for r in results]
    f1_mat = np.array([r["test_metrics"]["f1"] for r in results]).T  # (10, n_exp)

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 6))
    cmap = LinearSegmentedColormap.from_list("f1", ["#d73027", "#fee08b", "#1a9850"])
    im   = ax.imshow(f1_mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Digit {k}" for k in range(10)])
    ax.set_title("Per-class F1 score (test set)")

    for i in range(10):
        for j in range(len(names)):
            val = f1_mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if val < 0.4 or val > 0.85 else "black")

    fig.colorbar(im, ax=ax, label="F1 score")
    fig.tight_layout()
    _save(fig, "per_class_f1.png")


def plot_summary_table(results):
    """Color-coded visual summary table of all key metrics."""
    cols   = ["Test", "Best", "MacroF1", "MinF1", "Gap", "BEpoch", "E→80%", "E→85%"]
    names  = [r["name"] for r in results]

    # Build numeric matrix for color mapping (per column)
    def _val(r, col):
        tm = r["test_metrics"]
        return {
            "Test":    r["test_acc"],
            "Best":    r["best_test_acc"],
            "MacroF1": tm["macro_f1"],
            "MinF1":   tm["min_class_f1"],
            "Gap":     r["gap"],
            "BEpoch":  r["best_epoch"],
            "E→80%":   r["epochs_to_80"] or 9999,
            "E→85%":   r["epochs_to_85"] or 9999,
        }[col]

    def _label(r, col):
        tm = r["test_metrics"]
        e80 = str(r["epochs_to_80"]) if r["epochs_to_80"] else "-"
        e85 = str(r["epochs_to_85"]) if r["epochs_to_85"] else "-"
        return {
            "Test":    f"{r['test_acc']*100:.1f}%",
            "Best":    f"{r['best_test_acc']*100:.1f}%",
            "MacroF1": f"{tm['macro_f1']*100:.1f}%",
            "MinF1":   f"{tm['min_class_f1']*100:.1f}%",
            "Gap":     f"{r['gap']*100:.1f}pp",
            "BEpoch":  str(r["best_epoch"]),
            "E→80%":   e80,
            "E→85%":   e85,
        }[col]

    n_rows = len(results)
    n_cols = len(cols)
    # Higher = better for these; lower = better for gap/BEpoch/E→X
    higher_better = {"Test", "Best", "MacroF1", "MinF1"}

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.5), max(3, n_rows * 0.55 + 1)))
    ax.axis("off")

    # Column header row + data rows
    table_data  = [[_label(r, c) for c in cols] for r in results]
    table_data  = [[r["name"]] + row for r, row in zip(results, table_data)]
    col_labels  = ["Experiment"] + cols

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    # Color header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Color data cells by column
    green = np.array([26, 152, 80]) / 255
    red   = np.array([215, 48, 39]) / 255
    for j, col in enumerate(cols, start=1):
        vals = [_val(r, col) for r in results]
        vmin, vmax = min(vals), max(vals)
        for i, v in enumerate(vals, start=1):
            t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            if col not in higher_better:
                t = 1 - t   # invert: lower is greener
            color = tuple(red * (1 - t) + green * t) + (0.3,)
            tbl[i, j].set_facecolor(color)

    ax.set_title("Experiment summary", fontsize=11, pad=12)
    fig.tight_layout()
    _save(fig, "summary_table.png")


# ── Console output ───────────────────────────────────────────────────────────

def print_summary(results):
    """Print a formatted summary table of experiment results to stdout."""
    header = (
        f"{'Experiment':<32} {'Train':>7} {'Test':>7} {'Best':>7}"
        f" {'BEpoch':>7} {'Gap':>6} {'MacroF1':>8} {'MinF1':>6} {'E→80%':>6} {'E→85%':>6}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        e80 = str(r["epochs_to_80"]) if r["epochs_to_80"] else "-"
        e85 = str(r["epochs_to_85"]) if r["epochs_to_85"] else "-"
        tm  = r["test_metrics"]
        print(
            f"{r['name']:<32}"
            f" {r['train_acc']*100:>6.1f}%"
            f" {r['test_acc']*100:>6.1f}%"
            f" {r['best_test_acc']*100:>6.1f}%"
            f" {str(r['best_epoch']):>7}"
            f" {r['gap']*100:>5.1f}pp"
            f" {tm['macro_f1']*100:>7.1f}%"
            f" {tm['min_class_f1']*100:>5.1f}%"
            f" {e80:>6}"
            f" {e85:>6}"
        )
    print("=" * len(header))


# ── Single-model plots (for main.py) ─────────────────────────────────────────

def plot_confusion_matrix(cm, title="Confusion matrix", filename="confusion_matrix.png"):
    n_classes = cm.shape[0]
    fig, ax   = plt.subplots(figsize=(7, 6))
    im        = ax.imshow(cm, cmap="Blues")
    threshold = cm.max() / 2
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im)
    fig.tight_layout()
    _save(fig, filename)


def plot_loss_curve(errors, val_errors=None, title="Training loss", filename="loss_curve.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(errors) + 1)
    ax.plot(epochs, errors, label="Train")
    if val_errors:
        ax.plot(range(1, len(val_errors) + 1), val_errors, label="Test")
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, filename)


def plot_per_class_metrics(metrics, title="Per-class metrics", filename="per_class_metrics.png"):
    """Bar chart of precision, recall, F1 per class for a single model."""
    n_classes = len(metrics["f1"])
    x         = np.arange(n_classes)
    width     = 0.25
    fig, ax   = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, metrics["precision"], width, label="Precision")
    ax.bar(x,         metrics["recall"],    width, label="Recall")
    ax.bar(x + width, metrics["f1"],        width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in range(n_classes)])
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.axhline(metrics["macro_f1"], color="gray", linestyle="--",
               linewidth=1, label=f"MacroF1={metrics['macro_f1']:.2f}")
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)
