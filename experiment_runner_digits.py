import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import datetime
from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron

RESULTS_DIR  = "results"
SUMMARY_CSV  = os.path.join(RESULTS_DIR, "summary.csv")
CURVES_CSV   = os.path.join(RESULTS_DIR, "curves.csv")

TRAIN_PATH = "datasets/digits.csv"
TEST_PATH  = "datasets/digits_test.csv"
SEED       = 1

# --- Step 1: LR sweep ---
# EXPERIMENTS = [
#     # {"name": "lr=0.001 [784,10,10]", "layers": [784, 10, 10], "lr": 0.001, "epochs": 500, "epsilon": 1e-6, "beta": 1.0},
#     # {"name": "lr=0.01  [784,10,10]", "layers": [784, 10, 10], "lr": 0.01,  "epochs": 500, "epsilon": 1e-6, "beta": 1.0},
#     # {"name": "lr=0.1 [784,10,10]", "layers": [784, 10, 10], "lr": 0.1,   "epochs": 500, "epsilon": 1e-6, "beta": 1.0},
#     {"name": "lr=0.15  [784,10,10]", "layers": [784, 10, 10], "lr": 0.15,   "epochs": 500, "epsilon": 1e-6, "beta": 1.0},
# ]
# Results: (500 epochs)
# Experiment                      Train acc   Test acc
# ====================================================
# lr=0.001 [784,10,10]               94.7%      77.9%
# lr=0.01  [784,10,10]               96.9%      79.9%
# lr=0.05  [784,10,10]               95.8%      77.9%
# lr=0.1   [784,10,10]               95.3%      80.4% <- 
# lr=0.15  [784,10,10]               93.2%      80.7%
# lr=0.2   [784,10,10]               88.4%      74.5%

# --- Step 2: 1 hidden layer size sweep (best LR from step 1 = 0.1) ---
# EXPERIMENTS = [
#     {"name": "[784,18,10]",  "layers": [784, 18,  10], "lr": 0.1, "epochs": 500, "epsilon": 1e-6, "beta": 1.0},
#     {"name": "[784,22,10]",  "layers": [784, 22,  10], "lr": 0.1, "epochs": 500, "epsilon": 1e-6, "beta": 1.0},
#     {"name": "[784,25,10]",  "layers": [784, 25,  10], "lr": 0.1, "epochs": 500, "epsilon": 1e-6, "beta": 1.0},
#     {"name": "[784,28,10]",  "layers": [784, 28,  10], "lr": 0.1, "epochs": 500, "epsilon": 1e-6, "beta": 1.0},
# ]
# Results (500 epochs)
# Experiment                      Train acc   Test acc
# ============================================================
# [784,10,10]                         95.3%      80.4%
# [784,15,10]                         94.9%      82.2%
# [784,18,10]                         94.5%      82.5%
# [784,20,10]                         95.2%      85.3% <-
# [784,22,10]                         93.9%      81.2%
# [784,25,10]                         93.5%      82.7%
# [784,28,10]                         94.4%      82.4%
# [784,30,10]                         92.3%      80.0%
# [784,50,10]                         71.0%      64.4%
# [784,100,10]                        67.2%      57.9%

# --- Step 3: Initialization sweep ---
EXPERIMENTS = [
    # {"name": "random [784,20,10]",  "layers": [784,  20, 10], "lr": 0.1, "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "initializer": "random"},
    # {"name": "xavier [784,20,10]",  "layers": [784,  20, 10], "lr": 0.1, "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
    # {"name": "random [784,100,10]", "layers": [784, 100, 10], "lr": 0.1, "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "initializer": "random"},
    # {"name": "xavier [784,100,10]", "layers": [784, 100, 10], "lr": 0.1, "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
    {"name": "xavier lr=0.01 [784,100,10]",  "layers": [784,  100, 10], "lr": 0.01, "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
    # {"name": "xavier lr=0.001 [784,10,10]",  "layers": [784,  20, 10], "lr": 0.01, "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
    # {"name": "xavier lr=0.1 [784,20,10]",  "layers": [784,  20, 10], "lr": 0.01, "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
]
# Results (500 epochs, LR=0.1)
# Experiment                         Train    Test    Best  BEpoch    Gap  E→80%  E→85%
# =====================================================================================
# random [784,100,10]                65.7%   54.8%   56.7%      63  10.9pp      -      -
# random [784,20,10]                 92.7%   82.2%   83.5%     394  10.6pp     48      -
# xavier [784,100,10]                46.5%   47.8%   50.5%     415  -1.3pp      -      -
# xavier [784,20,10]                 93.6%   82.6%   85.2%     384  11.0pp     42    384

# Results (500 epochs, different learning rates)
# xavier lr=0.001 [784,20,10]        98.9%   83.9%   84.1%      24  14.9pp      3      -
# xavier lr=0.01 [784,20,10]         99.0%   84.1%   84.8%      55  14.9pp      1      - <-
# xavier lr=0.1 [784,20,10]          93.6%   82.6%   85.2%     384  11.0pp     42    384


# xavier lr=0.01 [784,50,10]         99.4%   85.9%   86.3%      25  13.6pp      1      3


# --- Step 4: 2 hidden layers sweep (best LR=0.1, best single hidden size from step 2) ---
# EXPERIMENTS = [
#     {"name": "[784,100,10]",    "layers": [784, 100,     10], "lr": 0.1, "epochs": 500, "epsilon": 1e-6, "beta": 1.0},
#     {"name": "[784,100,50,10]", "layers": [784, 100, 50, 10], "lr": 0.1, "epochs": 500, "epsilon": 1e-6, "beta": 1.0},
#     {"name": "[784,100,25,10]", "layers": [784, 100, 25, 10], "lr": 0.1, "epochs": 500, "epsilon": 1e-6, "beta": 1.0},
# ]

# --- Step 4: Optimizer sweep (best arch=[784,20,10], Adam/RMSProp use smaller lr) ---
# EXPERIMENTS = [
#     {"name": "sgd     lr=0.1  [784,20,10]", "layers": [784, 20, 10], "lr": 0.1,   "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "optimizer": "sgd"},
#     {"name": "rmsprop lr=0.01 [784,20,10]", "layers": [784, 20, 10], "lr": 0.01,  "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "optimizer": "rmsprop"},
#     {"name": "adam    lr=0.01 [784,20,10]", "layers": [784, 20, 10], "lr": 0.01,  "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "optimizer": "adam"},
#     {"name": "adam    lr=0.001[784,20,10]", "layers": [784, 20, 10], "lr": 0.001, "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "optimizer": "adam"},
# ]


def load(path):
    df = pd.read_csv(path)
    images = df["image"].apply(lambda s: np.array(ast.literal_eval(s), dtype=np.float32))
    X = np.stack(images.to_numpy())
    y_labels = df["label"].to_numpy(dtype=np.int64)
    return X, y_labels


def encode(labels, n_outputs):
    targets = np.full((len(labels), n_outputs), -1.0)
    targets[np.arange(len(labels)), labels] = 1.0
    return targets


def _epochs_to_threshold(val_accuracies, threshold):
    for i, acc in enumerate(val_accuracies):
        if acc >= threshold:
            return i + 1
    return None


def run_experiment(config, X_train, y_train, train_labels, X_test, y_test, test_labels):
    print(f"\n--- {config['name']} ---")
    mlp = MultiLayerPerceptron(
        config["layers"], config["lr"], config["epochs"],
        config["epsilon"], SEED, config["beta"],
        initializer=config.get("initializer", "random"),
    )
    mlp.fit(X_train, y_train, X_val=X_test, y_val=y_test, val_labels=test_labels, name=config["name"])

    train_preds = np.argmax(mlp.predict(X_train), axis=1)
    test_preds  = np.argmax(mlp.predict(X_test),  axis=1)
    train_acc = np.mean(train_preds == train_labels)
    test_acc  = np.mean(test_preds  == test_labels)

    train_loss = [e / len(X_train) for e in mlp.errors_]
    test_loss  = [e / len(X_test)  for e in mlp.val_errors_]
    val_accs   = mlp.val_accuracies_

    best_epoch    = int(np.argmax(val_accs)) + 1 if val_accs else None
    best_test_acc = float(np.max(val_accs))      if val_accs else test_acc

    print(f"Train accuracy: {train_acc * 100:.1f}%  |  Test accuracy: {test_acc * 100:.1f}%"
          f"  |  Best: {best_test_acc * 100:.1f}% @ epoch {best_epoch}")

    return {
        "name":           config["name"],
        "config":         config,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "train_acc":      train_acc,
        "test_acc":       test_acc,
        "best_test_acc":  best_test_acc,
        "best_epoch":     best_epoch,
        "gap":            train_acc - test_acc,
        "epochs_to_80":   _epochs_to_threshold(val_accs, 0.80),
        "epochs_to_85":   _epochs_to_threshold(val_accs, 0.85),
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_test_loss":  test_loss[-1]  if test_loss  else None,
        "train_loss":     train_loss,
        "test_loss":      test_loss,
        "test_acc_per_epoch": val_accs,
    }


def plot_loss_curves(results):
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        epochs = range(1, len(r["train_loss"]) + 1)
        ax_train.plot(epochs, r["train_loss"], label=r["name"])
        ax_test.plot(epochs,  r["test_loss"],  label=r["name"])

    ax_train.set_title("Train loss")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Loss")
    ax_train.legend(fontsize=7)

    ax_test.set_title("Test loss")
    ax_test.set_xlabel("Epoch")
    ax_test.set_ylabel("Loss")
    ax_test.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("plots/experiment_loss_curves.png")
    print("Saved: plots/experiment_loss_curves.png")


def plot_accuracy_bars(results):
    names      = [r["name"]      for r in results]
    train_accs = [r["train_acc"] for r in results]
    test_accs  = [r["test_acc"]  for r in results]

    x = np.arange(len(names))
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

    plt.tight_layout()
    plt.savefig("plots/experiment_accuracy.png")
    print("Saved: plots/experiment_accuracy.png")


def plot_convergence(results):
    plt.figure(figsize=(10, 5))
    for r in results:
        accs = r["test_acc_per_epoch"]
        if accs:
            plt.plot(range(1, len(accs) + 1), [a * 100 for a in accs], label=r["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy (%)")
    plt.title("Test accuracy over epochs (convergence speed)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("plots/experiment_convergence.png")
    print("Saved: plots/experiment_convergence.png")


def print_summary(results):
    header = f"{'Experiment':<32} {'Train':>7} {'Test':>7} {'Best':>7} {'BEpoch':>7} {'Gap':>6} {'E→80%':>6} {'E→85%':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        e80 = str(r["epochs_to_80"]) if r["epochs_to_80"] else "-"
        e85 = str(r["epochs_to_85"]) if r["epochs_to_85"] else "-"
        print(
            f"{r['name']:<32}"
            f" {r['train_acc']*100:>6.1f}%"
            f" {r['test_acc']*100:>6.1f}%"
            f" {r['best_test_acc']*100:>6.1f}%"
            f" {str(r['best_epoch']):>7}"
            f" {r['gap']*100:>5.1f}pp"
            f" {e80:>6}"
            f" {e85:>6}"
        )
    print("=" * len(header))


def save_csvs(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_rows = []
    curve_rows   = []

    for r in results:
        cfg = r["config"]
        base = {
            "timestamp": r["timestamp"],
            "name":      r["name"],
            "layers":    str(cfg["layers"]),
            "lr":        cfg["lr"],
            "epochs_configured": cfg["epochs"],
            "optimizer": cfg.get("optimizer", "sgd"),
            "beta":      cfg.get("beta", 1.0),
        }
        summary_rows.append({
            **base,
            "train_acc":        round(r["train_acc"],      4),
            "test_acc":         round(r["test_acc"],       4),
            "best_test_acc":    round(r["best_test_acc"],  4),
            "best_epoch":       r["best_epoch"],
            "gap":              round(r["gap"],            4),
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

    summary_df = pd.DataFrame(summary_rows)
    curves_df  = pd.DataFrame(curve_rows)

    write_header = not os.path.exists(SUMMARY_CSV)
    summary_df.to_csv(SUMMARY_CSV, mode="a", header=write_header, index=False)

    write_header = not os.path.exists(CURVES_CSV)
    curves_df.to_csv(CURVES_CSV,  mode="a", header=write_header, index=False)

    print(f"Resultados guardados en {SUMMARY_CSV} y {CURVES_CSV}")


def _worker(args):
    config, X_train, y_train, train_labels, X_test, y_test, test_labels = args
    return run_experiment(config, X_train, y_train, train_labels, X_test, y_test, test_labels)


def main():
    print("Loading data...")
    X_train, train_labels = load(TRAIN_PATH)
    X_test,  test_labels  = load(TEST_PATH)
    y_train = encode(train_labels, 10)
    y_test  = encode(test_labels,  10)

    args = [(c, X_train, y_train, train_labels, X_test, y_test, test_labels) for c in EXPERIMENTS]

    n_workers = min(len(EXPERIMENTS), mp.cpu_count())
    print(f"Running {len(EXPERIMENTS)} experiments on {n_workers} CPU cores in parallel...")
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker, args)

    results.sort(key=lambda r: r["name"])
    print_summary(results)
    plot_loss_curves(results)
    plot_accuracy_bars(results)
    plot_convergence(results)


if __name__ == "__main__":
    main()
