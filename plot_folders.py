"""
Grafica val accuracy (curva por epoca) y barras Train/Val/F1 a partir de
carpetas de resultados pasadas como argumento, preservando el orden.

Uso:
    python plot_folders.py results/models/[784,100,10] results/models/[784,200,10]
    python plot_folders.py --root results/models [784,100,10] [784,200,10]

Cada carpeta puede contener uno o varios seed_*.npz; se promedian.
"""
import os
import glob
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from digit_dataset_loader import load_dataset
from utils.test_data_split import stratified_split
from utils.metrics import compute_metrics
from utils.style import STYLE, PLOT_RC, FIG_DPI, SAVE_PAD_INCHES
from utils.visualization import COLOR_CYCLE, COLOR_TRAIN, COLOR_VAL, _apply_style

TRAIN_PATH = "datasets/digits.csv"
SEED = 1
PLOTS_DIR = "plots"
COLOR_F1 = "#90ee90"  # light green


def load_digits(path):
    df = load_dataset(path)
    return np.stack(df["image"].to_numpy()), df["label"].to_numpy(dtype=np.int64)


def collect_folder(folder, X_train, train_labels, X_val, val_labels):
    npz_files = sorted(glob.glob(os.path.join(glob.escape(folder), "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {folder}")

    val_curves = []
    train_accs = []
    val_accs = []
    f1s = []
    for f in npz_files:
        mlp = MultiLayerPerceptron.load(f[:-4])
        if hasattr(mlp, "val_accuracies_") and len(mlp.val_accuracies_):
            val_curves.append(np.asarray(mlp.val_accuracies_, dtype=float))
        train_preds = np.argmax(mlp.predict(X_train), axis=1)
        val_preds = np.argmax(mlp.predict(X_val), axis=1)
        train_accs.append(float(np.mean(train_preds == train_labels)))
        val_accs.append(float(np.mean(val_preds == val_labels)))
        f1s.append(compute_metrics(val_labels, val_preds)["macro_f1"])

    if val_curves:
        n = min(len(c) for c in val_curves)
        mat = np.stack([c[:n] for c in val_curves])
        val_curve_mean = mat.mean(axis=0)
        val_curve_std = mat.std(axis=0)
    else:
        val_curve_mean = val_curve_std = None

    return {
        "val_curve_mean": val_curve_mean,
        "val_curve_std": val_curve_std,
        "train_mean": float(np.mean(train_accs)),
        "train_std": float(np.std(train_accs)),
        "val_mean": float(np.mean(val_accs)),
        "val_std": float(np.std(val_accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "n_seeds": len(npz_files),
    }


def _save_fig(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight",
                pad_inches=SAVE_PAD_INCHES, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")


def plot_val_curves(entries, timestamp):
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, e in enumerate(entries):
            if e["data"]["val_curve_mean"] is None:
                continue
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            mean = e["data"]["val_curve_mean"] * 100
            std = e["data"]["val_curve_std"] * 100
            xs = np.arange(1, len(mean) + 1)
            ax.plot(xs, mean, label=e["label"], color=color, linewidth=2)
            if e["data"]["n_seeds"] > 1:
                ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_title("Validation accuracy per epoch")
        ax.legend(fontsize=9)
        _apply_style(fig, ax)
        fig.tight_layout()
        _save_fig(fig, f"plot_folders_val_curve_{timestamp}.png")


def plot_metrics_bars(entries, timestamp):
    labels = [e["label"] for e in entries]
    train_means = [e["data"]["train_mean"] * 100 for e in entries]
    train_stds = [e["data"]["train_std"] * 100 for e in entries]
    val_means = [e["data"]["val_mean"] * 100 for e in entries]
    val_stds = [e["data"]["val_std"] * 100 for e in entries]
    f1_means = [e["data"]["f1_mean"] * 100 for e in entries]
    f1_stds = [e["data"]["f1_std"] * 100 for e in entries]

    x = np.arange(len(labels))
    width = 0.27

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 2.2), 5))
        ax.bar(x - width, train_means, width, yerr=train_stds, capsize=4,
               label="Train acc", color=COLOR_TRAIN, alpha=0.9)
        ax.bar(x,         val_means,   width, yerr=val_stds,   capsize=4,
               label="Val acc",   color=COLOR_VAL,   alpha=0.9)
        ax.bar(x + width, f1_means,    width, yerr=f1_stds,    capsize=4,
               label="Macro F1",  color=COLOR_F1,    alpha=0.95,
               edgecolor="#5fa85f", linewidth=0.6)

        for i in range(len(labels)):
            for offset, m in [(-width, train_means[i]),
                              (0,      val_means[i]),
                              (+width, f1_means[i])]:
                ax.text(i + offset, m + 0.6, f"{m:.1f}%", ha="center",
                        fontsize=8, color=STYLE["text_title"])

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Score (%)")
        ax.set_title("Train accuracy  |  Validation accuracy  |  Macro F1")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=9, loc="lower right")
        _apply_style(fig, ax)
        fig.tight_layout()
        _save_fig(fig, f"plot_folders_bars_{timestamp}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", help="Carpetas con .npz (orden preservado)")
    p.add_argument("--root", default=None,
                   help="Prefijo para nombres relativos (ej: results/models)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Etiquetas custom (mismo nro y orden que folders)")
    args = p.parse_args()

    folders = [os.path.join(args.root, f) if args.root else f for f in args.folders]
    if args.labels and len(args.labels) != len(folders):
        p.error("--labels debe tener la misma cantidad que folders")
    labels = args.labels or [os.path.basename(f.rstrip("/\\")) for f in folders]

    print("Loading data...")
    X_all, all_labels = load_digits(TRAIN_PATH)
    X_train, X_val, train_labels, val_labels = stratified_split(
        X_all, all_labels, val_size=0.2, random_state=SEED
    )
    print(f"  train: {len(X_train)} samples  |  val: {len(X_val)} samples")

    entries = []
    for folder, label in zip(folders, labels):
        print(f"\n--- {label} ({folder}) ---")
        data = collect_folder(folder, X_train, train_labels, X_val, val_labels)
        print(f"  seeds={data['n_seeds']}  "
              f"train={data['train_mean']*100:.2f}±{data['train_std']*100:.2f}  "
              f"val={data['val_mean']*100:.2f}±{data['val_std']*100:.2f}  "
              f"f1={data['f1_mean']*100:.2f}±{data['f1_std']*100:.2f}")
        entries.append({"label": label, "data": data})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_val_curves(entries, timestamp)
    plot_metrics_bars(entries, timestamp)


if __name__ == "__main__":
    main()
