import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PLOTS_DIR = "plots"


def _group_results(results):
    """Group results that share the same base config name.

    A result named "xavier s=2" is grouped under "xavier" (strips trailing ' s=<N>').
    Results without that suffix form single-element groups (std = 0).

    Returns a list of (group_name, [result, ...]) in original order.
    """
    groups = {}
    order = []
    for r in results:
        base = re.sub(r"\s+s=\d+$", "", r["name"])
        if base not in groups:
            groups[base] = []
            order.append(base)
        groups[base].append(r)
    return [(name, groups[name]) for name in order]


def _save(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Multi-experiment plots ────────────────────────────────────────────────────

def plot_loss_curves(results):
    groups = _group_results(results)
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 5))
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, runs) in enumerate(groups):
        color = prop_cycle[i % len(prop_cycle)]
        for key, ax in [("train_loss", ax_train), ("test_loss", ax_test)]:
            curves = [r[key] for r in runs]
            n_epochs = min(len(c) for c in curves)
            mat = np.array([c[:n_epochs] for c in curves])   # (n_seeds, epochs)
            mean = mat.mean(axis=0)
            std  = mat.std(axis=0)
            xs   = range(1, n_epochs + 1)
            ax.plot(xs, mean, color=color, label=name)
            if len(runs) > 1:
                ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)

    for ax, title in [(ax_train, "Train loss"), (ax_test, "Test loss")]:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "loss_curves.png")


def plot_accuracy_bars(results):
    groups = _group_results(results)
    names       = [name for name, _ in groups]
    train_means = [np.mean([r["train_acc"] for r in runs]) for _, runs in groups]
    train_stds  = [np.std( [r["train_acc"] for r in runs]) for _, runs in groups]
    test_means  = [np.mean([r["test_acc"]  for r in runs]) for _, runs in groups]
    test_stds   = [np.std( [r["test_acc"]  for r in runs]) for _, runs in groups]

    x     = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
    ax.bar(x - width / 2, train_means, width, yerr=train_stds, capsize=4, label="Train")
    ax.bar(x + width / 2, test_means,  width, yerr=test_stds,  capsize=4, label="Test")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs test accuracy per experiment (mean ± std across seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    all_vals = train_means + test_means
    top = min(3, max(all_vals) + 0.08)   # headroom for labels
    ax.set_ylim(max(0, min(all_vals) - 0.1), top)
    ax.legend()
    for i, (tr, te) in enumerate(zip(train_means, test_means)):
        ax.text(i - width / 2, tr + 0.01, f"{tr*100:.1f}%", ha="center", fontsize=7)
        ax.text(i + width / 2, te + 0.01, f"{te*100:.1f}%", ha="center", fontsize=7)
    fig.tight_layout()
    _save(fig, "accuracy_bars.png")


def plot_convergence(results):
    groups = _group_results(results)
    fig, ax = plt.subplots(figsize=(10, 5))
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, runs) in enumerate(groups):
        color  = prop_cycle[i % len(prop_cycle)]
        curves = [r["test_acc_per_epoch"] for r in runs if r["test_acc_per_epoch"]]
        if not curves:
            continue
        n_epochs = min(len(c) for c in curves)
        mat  = np.array([c[:n_epochs] for c in curves]) * 100   # percent
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        xs   = range(1, n_epochs + 1)
        ax.plot(xs, mean, color=color, label=name)
        if len(runs) > 1:
            ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Test accuracy over epochs (mean ± std across seeds)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "convergence.png")


def plot_confusion_matrices(results):
    """One confusion matrix per config group, averaged across seeds."""
    groups = _group_results(results)
    n   = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (name, runs) in zip(axes, groups):
        # Sum raw counts across seeds, then normalize each row to [0, 1]
        # Row i sums to 1 → each cell = fraction of true class i predicted as j
        cm_sum  = np.sum([r["test_metrics"]["confusion_matrix"] for r in runs], axis=0).astype(float)
        row_sum = cm_sum.sum(axis=1, keepdims=True)
        cm_norm = cm_sum / np.maximum(row_sum, 1)   # avoids div-by-zero for empty classes

        n_seeds = len(runs)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{name}" + (f"  (Σ {n_seeds} seeds)" if n_seeds > 1 else ""), fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        for i in range(10):
            for j in range(10):
                v = cm_norm[i, j]
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


def plot_perclass_heatmap(results):
    """Figure B: per-class metrics heatmap, one subplot per config group.

    Layout: 10 rows (digits 0-9) × 3 columns (Precision, Recall, F1).
    Each cell shows the mean value across seeds.
    The figure caption reports the maximum std observed (hybrid approach).
    """
    groups    = _group_results(results)
    n_groups  = len(groups)
    metrics   = ["precision", "recall", "f1"]
    col_labels = ["Precision", "Recall", "F1"]
    row_labels = [str(k) if k != 8 else "8 (absent)" for k in range(10)]
    cmap = LinearSegmentedColormap.from_list("metrics", ["#d73027", "#fee08b", "#1a9850"])

    fig, axes = plt.subplots(1, n_groups, figsize=(4.5 * n_groups, 5.5))
    if n_groups == 1:
        axes = [axes]

    max_std_overall = 0.0

    for ax, (name, runs) in zip(axes, groups):
        mean_mat = np.zeros((10, 3))
        std_mat  = np.zeros((10, 3))
        for j, metric in enumerate(metrics):
            for k in range(10):
                vals = [r["test_metrics"][metric][k] for r in runs]
                mean_mat[k, j] = np.mean(vals)
                std_mat[k, j]  = np.std(vals)

        max_std_overall = max(max_std_overall, std_mat.max())

        im = ax.imshow(mean_mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(3))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(10))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(name, fontsize=9)

        for k in range(10):
            for j in range(3):
                v = mean_mat[k, j]
                ax.text(j, k, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.35 or v > 0.80 else "black")

        fig.colorbar(im, ax=ax, shrink=0.85, label="Score")

    std_note = f"Max std across all cells: ±{max_std_overall:.3f}" if max_std_overall > 0 else ""
    fig.suptitle(
        "Per-class Precision / Recall / F1  (test set)"
        + (f"\n{std_note}" if std_note else ""),
        fontsize=10
    )
    fig.tight_layout()
    _save(fig, "perclass_heatmap.png")


# ── Console output ───────────────────────────────────────────────────────────

def print_summary(results):
    """Print a formatted summary table of experiment results to stdout.

    When multiple results share a base config name (differ only by ' s=N' suffix),
    they are collapsed into one row showing mean ± std.
    """
    groups = _group_results(results)
    header = (
        f"{'Experiment':<32} {'Train':>12} {'Test':>12} {'Best':>12}"
        f" {'Gap':>6} {'MacroF1':>8} {'E→80%':>5} {'E→85%':>5}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, runs in groups:
        def _fmt(vals, pct=True):
            m = np.mean(vals) * (100 if pct else 1)
            s = np.std(vals)  * (100 if pct else 1)
            return f"{m:5.1f}±{s:.1f}%" if len(runs) > 1 else f"{m:5.1f}%      "

        train_str = _fmt([r["train_acc"]     for r in runs])
        test_str  = _fmt([r["test_acc"]      for r in runs])
        best_str  = _fmt([r["best_test_acc"] for r in runs])
        gap_str   = _fmt([r["gap"]           for r in runs])
        f1_str    = _fmt([r["test_metrics"]["macro_f1"] for r in runs])

        e80_vals = [r["epochs_to_80"] for r in runs if r["epochs_to_80"]]
        e85_vals = [r["epochs_to_85"] for r in runs if r["epochs_to_85"]]
        e80 = f"{np.mean(e80_vals):.0f}" if e80_vals else "-"
        e85 = f"{np.mean(e85_vals):.0f}" if e85_vals else "-"

        print(
            f"{name:<32}"
            f" {train_str:>12}"
            f" {test_str:>12}"
            f" {best_str:>12}"
            f" {gap_str:>6}"
            f" {f1_str:>8}"
            f" {e80:>6}"
            f" {e85:>6}"
        )
    print("=" * len(header))


def print_perclass_summary(results):
    """Per-class precision / recall / F1 table, one block per config group.

    Rows: digits 0-9 + Macro-avg.
    Columns: Precision, Recall, F1  (mean ± std across seeds).
    """
    groups = _group_results(results)
    header = f"  {'Dígito':>6}  {'Precision':>13}  {'Recall':>13}  {'F1':>13}"
    sep    = "  " + "-" * (len(header) - 2)

    for name, runs in groups:
        print(f"\n{'─' * len(header)}")
        print(f"  {name}")
        print(f"{'─' * len(header)}")
        print(header)
        print(sep)

        n_classes = len(runs[0]["test_metrics"]["f1"])
        for k in range(n_classes):
            p_vals = [r["test_metrics"]["precision"][k] for r in runs]
            r_vals = [r["test_metrics"]["recall"][k]    for r in runs]
            f_vals = [r["test_metrics"]["f1"][k]        for r in runs]

            def fmt(vals):
                m, s = np.mean(vals), np.std(vals)
                return f"{m:.2f} ± {s:.2f}" if len(runs) > 1 else f"{m:.2f}      "

            digit = str(k) if k != 8 else "8 (absent)"
            print(f"  {digit:>10}  {fmt(p_vals):>13}  {fmt(r_vals):>13}  {fmt(f_vals):>13}")

        print(sep)
        # Macro-avg
        mp = [r["test_metrics"]["macro_precision"] for r in runs]
        mr = [r["test_metrics"]["macro_recall"]    for r in runs]
        mf = [r["test_metrics"]["macro_f1"]        for r in runs]

        def fmt(vals):
            m, s = np.mean(vals), np.std(vals)
            return f"{m:.2f} ± {s:.2f}" if len(runs) > 1 else f"{m:.2f}      "

        print(f"  {'Macro-avg':>10}  {fmt(mp):>13}  {fmt(mr):>13}  {fmt(mf):>13}")
        print(f"{'─' * len(header)}")




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


def plot_sample_predictions(X, true_labels, pred_labels, raw_outputs, n=16,
                            filename="sample_predictions.png"):
    """
    Grid of n random test samples showing:
      - the 28x28 image
      - true label vs predicted label (green border = correct, red = wrong)
      - bar chart of all 10 output activations so you can see what the
        network 'thought' about every digit, not just the winner
    """
    rng     = np.random.default_rng(0)
    indices = rng.choice(len(X), size=min(n, len(X)), replace=False)
    cols    = 4
    rows    = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.8))
    axes = axes.reshape(rows, cols * 2)

    for idx, sample_i in enumerate(indices):
        row, col = divmod(idx, cols)
        ax_img  = axes[row, col * 2]
        ax_bar  = axes[row, col * 2 + 1]

        image  = X[sample_i].reshape(28, 28)
        true   = int(true_labels[sample_i])
        pred   = int(pred_labels[sample_i])
        output = raw_outputs[sample_i]  # 10 raw activations

        # ── image panel ──────────────────────────────────────────────────
        ax_img.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax_img.set_title(f"true={true}  pred={pred}", fontsize=8,
                         color="green" if true == pred else "red")
        ax_img.axis("off")
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor("green" if true == pred else "red")

        # ── activation bar chart ─────────────────────────────────────────
        colors = ["steelblue"] * 10
        colors[pred]  = "red" if true != pred else "green"
        colors[true]  = "green"
        ax_bar.bar(range(10), output, color=colors)
        ax_bar.set_xticks(range(10))
        ax_bar.set_xticklabels(range(10), fontsize=6)
        ax_bar.tick_params(axis="y", labelsize=6)
        ax_bar.set_ylim(-1.05, 1.05)  # tanh range
        ax_bar.axhline(0, color="gray", linewidth=0.5)
        ax_bar.set_title("activations", fontsize=7)

    # hide any unused slots
    for idx in range(len(indices), rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")

    fig.suptitle(
        "Each pair: digit image (left) + network output activations (right)\n"
        "Green bar = correct class  |  Red bar = what it wrongly picked",
        fontsize=9
    )
    fig.tight_layout()
    _save(fig, filename)


def plot_mistakes(X, true_labels, pred_labels, raw_outputs, n=16,
                  filename="mistakes.png"):
    """
    Grid of n misclassified samples showing the image, what it truly is,
    what the network predicted, and the full output activations.
    """
    wrong   = np.where(true_labels != pred_labels)[0]
    if len(wrong) == 0:
        print("No mistakes to show — perfect accuracy on this set.")
        return
    rng     = np.random.default_rng(0)
    indices = rng.choice(wrong, size=min(n, len(wrong)), replace=False)
    cols    = 4
    rows    = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.8))
    axes = axes.reshape(rows, cols * 2)

    for idx, sample_i in enumerate(indices):
        row, col = divmod(idx, cols)
        ax_img  = axes[row, col * 2]
        ax_bar  = axes[row, col * 2 + 1]

        image  = X[sample_i].reshape(28, 28)
        true   = int(true_labels[sample_i])
        pred   = int(pred_labels[sample_i])
        output = raw_outputs[sample_i]

        ax_img.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax_img.set_title(f"true={true}  pred={pred}", fontsize=8, color="red")
        ax_img.axis("off")
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor("red")

        colors        = ["lightgray"] * 10
        colors[true]  = "green"
        colors[pred]  = "red"
        ax_bar.bar(range(10), output, color=colors)
        ax_bar.set_xticks(range(10))
        ax_bar.set_xticklabels(range(10), fontsize=6)
        ax_bar.tick_params(axis="y", labelsize=6)
        ax_bar.set_ylim(-1.05, 1.05)
        ax_bar.axhline(0, color="gray", linewidth=0.5)
        ax_bar.set_title("activations", fontsize=7)

    for idx in range(len(indices), rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")

    total_wrong = len(wrong)
    fig.suptitle(
        f"Misclassified samples ({total_wrong} total wrong)\n"
        "Green bar = true class  |  Red bar = what the network predicted",
        fontsize=9
    )
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
