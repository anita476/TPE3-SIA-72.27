import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from datasets.digit_dataset_loader import load_digits, encode_one_hot
from datetime import datetime
from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from utils.metrics import compute_metrics, epochs_to_threshold, save_results_csvs
from utils.visualization import (
    print_summary,
    plot_loss_curves, plot_accuracy_bars, plot_convergence,
    plot_confusion_matrices, plot_per_class_f1, plot_summary_table,
)

RESULTS_DIR   = "results"
SUMMARY_CSV   = os.path.join(RESULTS_DIR, "summary.csv")
CURVES_CSV    = os.path.join(RESULTS_DIR, "curves.csv")
PERCLASS_CSV  = os.path.join(RESULTS_DIR, "per_class.csv")

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
    # {"name": "xavier lr=0.01 [784,100,10]",  "layers": [784,  100, 10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
    # {"name": "xavier lr=0.01 [784,50,10]",  "layers": [784,  50, 10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
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

# Trying different number of layers
# xavier lr=0.01 [784,50,10]         99.4%   85.9%   86.3%      25  13.6pp      1      3
# xavier lr=0.01 [784,100,10]        99.7%   86.9%   87.0%     245  12.8pp      1      4

# -> CONCLUSION: xavier does best with learning rate = 0.01. At around 50 for the first layer, the performance is almost the same as with 100 so in terms of effiency it is kind of like a threshold. 

# --- Step 4: 2 hidden layers sweep (best LR=0.01, best single hidden size from step 2) ---
# EXPERIMENTS = [
#     {"name": "xavier lr=0. [784,50,25,10]",  "layers": [784, 50,  25, 10], "lr": 0.01, "initializer": "xavier", "epochs": 250, "epsilon": 1e-6, "beta": 1.0,},
#     {"name": "xavier lr=0.01 [784,50,10]",  "layers": [784,  50, 10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},

#     # {"name": "xavier [784,100,50,10]", "layers": [784, 100, 50, 10], "lr": 0.01, "initializer": "xavier", "epochs": 500, "epsilon": 1e-6, "beta": 1.0,},
#     # {"name": "xavier [784,100,25,10]", "layers": [784, 100, 25, 10], "lr": 0.01, "initializer": "xavier", "epochs": 500, "epsilon": 1e-6, "beta": 1.0,},
# ]

# Experiment                         Train    Test    Best  BEpoch    Gap  E→80%  E→85%
# =====================================================================================
# xavier [784,100,25,10]             99.8%   86.5%   86.7%      21  13.4pp      5      7
# xavier [784,100,50,10]             99.8%   86.5%   86.8%      44  13.4pp      1      5
# xavier [784,50,25,10]              99.7%   86.3%   86.5%      11  13.4pp      2      6 <-

# EXPERIMENTS = [
#     {"name": "xavier lr=0.01 [784,10,10]",  "layers": [784,  10, 10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
#     {"name": "xavier lr=0.01 [784,20,10]",  "layers": [784,  20, 10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
#     {"name": "xavier lr=0.01 [784,50,10]",  "layers": [784,  50, 10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
#     {"name": "xavier lr=0.001 [784,10,10]",  "layers": [784,  10, 10], "lr": 0.001, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
#     {"name": "xavier lr=0.001 [784,20,10]",  "layers": [784,  20, 10], "lr": 0.001, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
#     {"name": "xavier lr=0.001 [784,50,10]",  "layers": [784,  50, 10], "lr": 0.001, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
# ]

# xavier lr=0.001 [784,10,10]        97.9%   81.3%   82.5%      57  16.6pp    76.5%   0.0%     12      -
# xavier lr=0.001 [784,20,10]        98.8%   83.8%   84.1%      24  15.0pp    79.2%   0.0%      3      -
# xavier lr=0.001 [784,50,10]        99.4%   86.3%   86.3%     130  13.1pp    81.6%   0.0%      4     23
# xavier lr=0.01 [784,10,10]         97.5%   80.2%   81.8%      15  17.4pp    75.4%   0.0%      5      -
# xavier lr=0.01 [784,20,10]         99.0%   84.2%   84.8%      55  14.8pp    79.5%   0.0%      1      -
# xavier lr=0.01 [784,50,10]         99.4%   85.7%   86.3%      25  13.6pp    81.1%   0.0%      1      3

## Probando neuronas para learning rate = 0.01
# EXPERIMENTS = [
#     {"name": "xavier lr=0.01 [784,50,10]",  "layers": [784,  50, 10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
#     {"name": "xavier lr=0.01 [784,70,10]",  "layers": [784,  70, 10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
#     {"name": "xavier lr=0.01 [784,100,10]",  "layers": [784,  100, 10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
# ]


# EXPERIMENTS = [
#     # baseline (ya lo tenés, pero incluilo para comparación justa en el mismo run)
#     {"name": "1L [784,100,10]",       "layers": [784, 100,      10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
#     # 2 capas — funnel gradual
#     {"name": "2L [784,100,50,10]",    "layers": [784, 100, 50,  10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
#     # 2 capas — funnel más agresivo
#     {"name": "2L [784,100,30,10]",    "layers": [784, 100, 30,  10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
#     # 2 capas — idk
#     {"name": "2L [784,50,20,10]",    "layers": [784, 50, 20,  10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
#     # 3 capas — compresión progresiva
#     {"name": "3L [784,100,50,25,10]", "layers": [784, 100, 50, 25, 10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
# ]

# --- Step 4: Changing batch sizes ---
EXPERIMENTS = [
    {"name": "online b=1",    "layers": [784,100,10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier", "training_mode": "online",    "batch_size": 1},
    {"name": "mini-b b=16",   "layers": [784,100,10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier", "training_mode": "minibatch", "batch_size": 16},
    {"name": "mini-b b=64",   "layers": [784,100,10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier", "training_mode": "minibatch", "batch_size": 64},
    {"name": "mini-b b=256",  "layers": [784,100,10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier", "training_mode": "minibatch", "batch_size": 256},
    {"name": "mini-b b=1024", "layers": [784,100,10], "lr": 0.01, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier", "training_mode": "minibatch", "batch_size": 1024},
]



# --- Step 4: Optimizer sweep (best arch=[784,20,10], Adam/RMSProp use smaller lr) ---
# EXPERIMENTS = [
#     {"name": "sgd     lr=0.1  [784,20,10]", "layers": [784, 20, 10], "lr": 0.1,   "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "optimizer": "sgd"},
#     {"name": "rmsprop lr=0.01 [784,20,10]", "layers": [784, 20, 10], "lr": 0.01,  "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "optimizer": "rmsprop"},
#     {"name": "adam    lr=0.01 [784,20,10]", "layers": [784, 20, 10], "lr": 0.01,  "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "optimizer": "adam"},
#     {"name": "adam    lr=0.001[784,20,10]", "layers": [784, 20, 10], "lr": 0.001, "epochs": 500, "epsilon": 1e-6, "beta": 1.0, "optimizer": "adam"},
# ]


def run_experiment(config, X_train, y_train, train_labels, X_test, y_test, test_labels):
    print(f"\n--- {config['name']} ---")
    mlp = MultiLayerPerceptron(
        config["layers"], config["lr"], config["epochs"],
        config["epsilon"], SEED, config["beta"],
        initializer=config.get("initializer", "random"),
        training_mode=config.get("training_mode", "online"),
        batch_size=config.get("batch_size", 1),
        optimizer=config.get("optimizer", "sgd"),
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

    train_metrics = compute_metrics(train_labels, train_preds)
    test_metrics  = compute_metrics(test_labels,  test_preds)

    print(f"Train accuracy: {train_acc * 100:.1f}%  |  Test accuracy: {test_acc * 100:.1f}%"
          f"  |  Best: {best_test_acc * 100:.1f}% @ epoch {best_epoch}"
          f"  |  Macro F1: {test_metrics['macro_f1'] * 100:.1f}%")

    return {
        "name":             config["name"],
        "config":           config,
        "train_acc":        train_acc,
        "test_acc":         test_acc,
        "best_test_acc":    best_test_acc,
        "best_epoch":       best_epoch,
        "gap":              train_acc - test_acc,
        "epochs_to_80":     epochs_to_threshold(val_accs, 0.80),
        "epochs_to_85":     epochs_to_threshold(val_accs, 0.85),
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_test_loss":  test_loss[-1]  if test_loss  else None,
        "train_loss":       train_loss,
        "test_loss":        test_loss,
        "test_acc_per_epoch": val_accs,
        "train_metrics":    train_metrics,
        "test_metrics":     test_metrics,
    }

def _worker(args):
    config, X_train, y_train, train_labels, X_test, y_test, test_labels = args
    return run_experiment(config, X_train, y_train, train_labels, X_test, y_test, test_labels)


def main():
    print("Loading data...")
    X_train, train_labels = load_digits(TRAIN_PATH)
    X_test,  test_labels  = load_digits(TEST_PATH)
    y_train = encode_one_hot(train_labels, 10)
    y_test  = encode_one_hot(test_labels,  10)

    args = [(c, X_train, y_train, train_labels, X_test, y_test, test_labels) for c in EXPERIMENTS]

    n_workers = min(len(EXPERIMENTS), mp.cpu_count())
    print(f"Running {len(EXPERIMENTS)} experiments on {n_workers} CPU cores in parallel...")
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker, args)

    results.sort(key=lambda r: r["name"])
    print_summary(results)
    save_results_csvs(results, SUMMARY_CSV, CURVES_CSV, PERCLASS_CSV)
    plot_loss_curves(results)
    plot_accuracy_bars(results)
    plot_convergence(results)
    plot_confusion_matrices(results)
    plot_per_class_f1(results)
    plot_summary_table(results)


if __name__ == "__main__":
    main()
