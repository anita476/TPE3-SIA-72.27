import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils.style import STYLE, PLOT_RC, FIG_DPI, SAVE_PAD_INCHES, PAIRWISE_COLORS

PLOTS_DIR = "plots"

# Ordered color cycle aligned with the project palette
COLOR_CYCLE = [
    "#4a90d9",  # blue   (PAIRWISE_COLORS[0])
    "#e67e22",  # orange (PAIRWISE_COLORS[1])
    "#27ae60",  # green  (tanh color from plot_ej1)
    "#8e44ad",  # purple (logistic color)
    "#c0392b",  # red
    "#16a085",  # teal
]

# Fixed semantic colors for train / validation split
COLOR_TRAIN = PAIRWISE_COLORS[0]["box_face"]   # "#4a90d9"
COLOR_VAL   = PAIRWISE_COLORS[1]["box_face"]   # "#e67e22"


def _apply_style(fig, *axes):
    """Apply project visual style (warm bg, grid, spine/tick colors)."""
    fig.patch.set_facecolor(STYLE["figure_bg"])
    for ax in axes:
        ax.set_facecolor(STYLE["axes_bg"])
        ax.grid(axis="y", which="major", linestyle="-", linewidth=0.6,
                alpha=0.55, color=STYLE["grid"], zorder=0)
        ax.minorticks_on()
        ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.4,
                alpha=0.45, color=STYLE["grid_minor"], zorder=0)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.tick_params(axis="both", colors=STYLE["text_axis"])
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(STYLE["text_axis"])
        ax.title.set_color(STYLE["text_title"])
        ax.xaxis.label.set_color(STYLE["text_axis"])
        ax.yaxis.label.set_color(STYLE["text_axis"])


def _group_results(results):
    """Group results by name. Runs with the same name are treated as different
    seeds of the same config.

    Returns a list of (name, [result, ...]) in original order.
    """
    groups = {}
    order  = []
    for r in results:
        key = r["name"]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    return [(key, groups[key]) for key in order]


def _save(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=FIG_DPI,
                pad_inches=SAVE_PAD_INCHES, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")


# ── Multi-experiment plots ────────────────────────────────────────────────────

def plot_learning_curves(results):
    """One figure, two panels: losses (left) and val accuracy (right).

    Each config group gets one color. Train loss = solid, val loss = dashed.
    Std band across seeds is shown when > 1 seed. Answers:
      - Is the model overfitting? (train loss << val loss)
      - Is it converging? (val accuracy flattening)
      - Is it unstable? (jagged curves)
    """
    groups = _group_results(results)
    with plt.rc_context(PLOT_RC):
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.5))

        for i, (name, runs) in enumerate(groups):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]

            # ── losses ──────────────────────────────────────────────────────────
            for key, ls in [("train_loss", "-"), ("val_loss", "--")]:
                curves = [r[key] for r in runs if r[key]]
                if not curves:
                    continue
                n = min(len(c) for c in curves)
                mat  = np.array([c[:n] for c in curves])
                mean = mat.mean(axis=0)
                xs   = range(1, n + 1)
                label = f"{name} ({'train' if ls == '-' else 'val'})"
                ax_loss.plot(xs, mean, color=color, linestyle=ls, label=label, linewidth=1.5)
                if len(runs) > 1:
                    std = mat.std(axis=0)
                    ax_loss.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)

            # ── val accuracy ─────────────────────────────────────────────────────
            curves = [r["val_acc"] for r in runs if r.get("val_acc")]
            if curves:
                n    = min(len(c) for c in curves)
                mat  = np.array([c[:n] for c in curves]) * 100
                mean = mat.mean(axis=0)
                xs   = range(1, n + 1)
                ax_acc.plot(xs, mean, color=color, label=name, linewidth=1.5)
                if len(runs) > 1:
                    std = mat.std(axis=0)
                    ax_acc.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)

        ax_loss.set_title("Loss per epoch  (— train,  -- val)")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=7)

        ax_acc.set_title("Validation accuracy per epoch")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend(fontsize=7)

        _apply_style(fig, ax_loss, ax_acc)
        fig.tight_layout()
        _save(fig, "learning_curves.png")


def plot_val_accuracy(results):
    """All configs on one axes: val accuracy per epoch. Answers: which learns best and fastest?"""
    groups = _group_results(results)
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(9, 5))

        for i, (name, runs) in enumerate(groups):
            color  = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            curves = [r["val_acc"] for r in runs if r.get("val_acc")]
            if not curves:
                continue
            n    = min(len(c) for c in curves)
            mat  = np.array([c[:n] for c in curves]) * 100
            mean = mat.mean(axis=0)
            xs   = range(1, n + 1)
            ax.plot(xs, mean, label=name, color=color, linewidth=2)
            if len(runs) > 1:
                std = mat.std(axis=0)
                ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_title("Validation accuracy: which configuration learns best?")
        ax.legend(fontsize=9)
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "val_accuracy.png")


def plot_overfitting_diagnosis(results):
    """One figure per config: train (dashed △) vs val (solid ○) accuracy per epoch."""
    groups = _group_results(results)

    for name, runs in groups:
        with plt.rc_context(PLOT_RC):
            fig, ax = plt.subplots(figsize=(8, 4.5))

            for curves_key, label, color, ls, marker in [
                ("train_acc_per_epoch", "Train",      COLOR_TRAIN, "--", "^"),
                ("val_acc",             "Validation",  COLOR_VAL,   "-",  "o"),
            ]:
                curves = [r.get(curves_key) for r in runs]
                curves = [c for c in curves if c]
                if not curves:
                    continue
                n    = min(len(c) for c in curves)
                mat  = np.array([c[:n] for c in curves]) * 100
                mean = mat.mean(axis=0)
                xs   = list(range(1, n + 1))
                # Marker every ~10% of epochs so it's readable
                step = max(1, n // 10)
                ax.plot(xs, mean, label=label, color=color, linestyle=ls,
                        linewidth=2, marker=marker, markevery=step, markersize=5)
                if len(runs) > 1:
                    std = mat.std(axis=0)
                    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{name}  —  train vs validation\n"
                         "Dashed △ = train  |  Solid ○ = validation  |  "
                         "Gap → overfitting")
            ax.legend(fontsize=9)
            _apply_style(fig, ax)
            fig.tight_layout()
            safe = name.replace(" ", "_").replace("/", "-")
            _save(fig, f"overfitting_{safe}.png")


def plot_accuracy_bars(results):
    """Bar chart: train vs best-val accuracy per config group (mean ± std across seeds)."""
    groups      = _group_results(results)
    names       = [name for name, _ in groups]
    train_means = [np.mean([r["train_acc"]    for r in runs]) for _, runs in groups]
    train_stds  = [np.std( [r["train_acc"]    for r in runs]) for _, runs in groups]
    val_means   = [np.mean([r["best_val_acc"] for r in runs]) for _, runs in groups]
    val_stds    = [np.std( [r["best_val_acc"] for r in runs]) for _, runs in groups]

    x     = np.arange(len(names))
    width = 0.35
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
        ax.bar(x - width / 2, train_means, width, yerr=train_stds, capsize=4,
               label="Train", color=COLOR_TRAIN, alpha=0.85)
        ax.bar(x + width / 2, val_means,   width, yerr=val_stds,   capsize=4,
               label="Validation", color=COLOR_VAL, alpha=0.85)
        ax.set_ylabel("Accuracy")
        ax.set_title("Train vs validation accuracy (mean ± std across seeds)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        all_vals = train_means + val_means
        ax.set_ylim(max(0, min(all_vals) - 0.05), min(1.05, max(all_vals) + 0.08))
        ax.legend()
        for i, (tr, va) in enumerate(zip(train_means, val_means)):
            ax.text(i - width / 2, tr + 0.005, f"{tr*100:.1f}%", ha="center", fontsize=7,
                    color=STYLE["text_title"])
            ax.text(i + width / 2, va + 0.005, f"{va*100:.1f}%", ha="center", fontsize=7,
                    color=STYLE["text_title"])
        _apply_style(fig, ax)
        fig.tight_layout()
        _save(fig, "accuracy_bars.png")


# ── Console output ───────────────────────────────────────────────────────────

def print_summary(results):
    """Print a formatted summary table of experiment results to stdout.

    When multiple results share a base config name (differ only by ' s=N' suffix),
    they are collapsed into one row showing mean ± std.
    """
    def _params(layers):
        return sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers) - 1))

    groups = _group_results(results)
    header = f"{'Experiment':<28} {'Params':>8} {'Val acc':>13} {'Macro-F1':>13}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, runs in groups:
        def _fmt(vals, pct=True):
            m = np.mean(vals) * (100 if pct else 1)
            s = np.std(vals)  * (100 if pct else 1)
            return f"{m:5.1f}±{s:.1f}%" if len(runs) > 1 else f"{m:5.1f}%"

        layers  = runs[0]["config"].get("layers", [])
        params  = f"{_params(layers):,}" if layers else "-"
        acc_str = _fmt([r["best_val_acc"] for r in runs])
        f1_str  = _fmt([r["macro_f1"] for r in runs])

        print(f"{name:<28} {params:>8} {acc_str:>13} {f1_str:>13}")
    print("=" * len(header))




# ── Single-model plots (for main.py) ─────────────────────────────────────────

def plot_confusion_matrix(cm, title="Confusion matrix", filename="confusion_matrix.png"):
    n_classes = cm.shape[0]
    with plt.rc_context(PLOT_RC):
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
        fig.patch.set_facecolor(STYLE["figure_bg"])
        ax.set_facecolor(STYLE["axes_bg"])
        ax.tick_params(axis="both", colors=STYLE["text_axis"])
        ax.title.set_color(STYLE["text_title"])
        ax.xaxis.label.set_color(STYLE["text_axis"])
        ax.yaxis.label.set_color(STYLE["text_axis"])
        for spine in ax.spines.values():
            spine.set_color(STYLE["text_axis"])
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

    with plt.rc_context(PLOT_RC):
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
            ax.set_title(name, fontsize=9, color=STYLE["text_title"])
            ax.tick_params(axis="both", colors=STYLE["text_axis"])
            for spine in ax.spines.values():
                spine.set_color(STYLE["text_axis"])

            for k in range(10):
                for j in range(3):
                    v = mean_mat[k, j]
                    ax.text(j, k, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v < 0.35 or v > 0.80 else "black")

            fig.colorbar(im, ax=ax, shrink=0.85, label="Score")

        std_note = f"Max std across all cells: ±{max_std_overall:.3f}" if max_std_overall > 0 else ""
        fig.patch.set_facecolor(STYLE["figure_bg"])
        fig.suptitle(
            "Per-class Precision / Recall / F1  (validation set)"
            + (f"\n{std_note}" if std_note else ""),
            fontsize=10, color=STYLE["text_title"]
        )
        fig.tight_layout()
        _save(fig, "perclass_heatmap.png")

