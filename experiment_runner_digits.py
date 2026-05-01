import os
import json
import argparse
import numpy as np
import multiprocessing as mp
from digit_dataset_loader import load_dataset
from utils.test_data_split import stratified_split
from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from utils.metrics import compute_metrics
from utils.visualization import print_summary, plot_accuracy_bars, plot_val_accuracy, plot_overfitting_diagnosis

RESULTS_DIR = "results"
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")

TRAIN_PATH = "datasets/digits.csv"
SEED       = 1
VALID_OPTIMIZERS = {"gd", "sgd", "rmsprop", "adam"}
OFF_TARGET_BY_ACTIVATION = {"logistic": 0.0, "tanh": -1.0}


def load_digits(path):
    df = load_dataset(path)
    return np.stack(df["image"].to_numpy()), df["label"].to_numpy(dtype=np.int64)


def encode_one_hot(labels, n_outputs, activation="tanh"):
    if activation not in OFF_TARGET_BY_ACTIVATION:
        raise ValueError("activation must be 'tanh' or 'logistic'")

    labels = labels.astype(int)
    targets = np.full((len(labels), n_outputs), OFF_TARGET_BY_ACTIVATION[activation])
    targets[np.arange(len(labels)), labels] = 1.0
    return targets

EXPERIMENTS = [
   
    {"name": "b=64 mini", "layers": [784,100,10], "lr": 0.0001, "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier", "training_mode": "minibatch", "batch_size": 64,   "optimizer": "adam", "patience": 20, "min_delta": 1e-4},
]

def expand_seeds(configs):
    """Expand configs that have a 'seeds' list into one entry per seed.

    Example:
        {"name": "xavier", "seeds": [1, 2, 3], ...}
    becomes:
        {"name": "xavier", "seed": 1, ...}
        {"name": "xavier", "seed": 2, ...}
        {"name": "xavier", "seed": 3, ...}

    Configs without 'seeds' are passed through unchanged (uses SEED default).
    """
    out = []
    for c in configs:
        seeds = c.pop("seeds", None)
        if seeds:
            for s in seeds:
                out.append({**c, "seed": s})
        else:
            out.append(c)
    return out


def count_params(layers):
    """Total de parametros entrenables (pesos + biases) para una arquitectura."""
    return sum(layers[i-1] * layers[i] + layers[i] for i in range(1, len(layers)))

def run_experiment(config, X_train, train_labels, X_eval, eval_labels):
    print(f"\n--- {config['name']} ---")
    activation = config.get("activation", "tanh")
    y_train = encode_one_hot(train_labels, config["layers"][-1], activation)
    y_eval = encode_one_hot(eval_labels, config["layers"][-1], activation)

    mlp = MultiLayerPerceptron(
        config["layers"], config["lr"], config["epochs"],
        config["epsilon"], config.get("seed", SEED), config["beta"],
        activation=activation,
        initializer=config.get("initializer", "random"),
        training_mode=config.get("training_mode", "online"),
        batch_size=config.get("batch_size", 1),
        optimizer=config.get("optimizer", "sgd"),
        patience=config.get("patience", 0),
        min_delta=config.get("min_delta", 0.0),
    )
    mlp.fit(X_train, y_train, X_val=X_eval, y_val=y_eval,
            val_labels=eval_labels, train_labels=train_labels, name=config["name"])

    eval_preds = np.argmax(mlp.predict(X_eval), axis=1)
    train_preds = np.argmax(mlp.predict(X_train), axis=1)

    train_acc = float(np.mean(train_preds == train_labels))
    val_accs  = mlp.val_accuracies_
    best_val_acc = float(np.max(val_accs)) if val_accs else 0.0
    best_epoch   = int(np.argmax(val_accs)) + 1 if val_accs else None

    eval_metrics = compute_metrics(eval_labels, eval_preds)

    train_loss = [e / len(X_train) for e in mlp.errors_]
    val_loss   = [e / len(X_eval)  for e in mlp.val_errors_]

    stop_reason = getattr(mlp, "stop_reason_", "max_epochs")
    stop_epoch  = len(mlp.errors_)
    early_stopped = stop_reason != "max_epochs"

    print(f"Train acc: {train_acc*100:.1f}%  |  Best val: {best_val_acc*100:.1f}% @ ep {best_epoch}"
          f"  |  Macro F1: {eval_metrics['macro_f1']*100:.1f}%")
    print(f"Stop: {stop_reason} @ ep {stop_epoch}/{config['epochs']}"
          f"  (early_stopped={early_stopped})")

    safe_name  = config["name"].replace(" ", "_").replace("=", "_").replace("/", "-")
    config_dir = os.path.join(MODELS_DIR, safe_name)
    os.makedirs(config_dir, exist_ok=True)
    seed_val   = config.get("seed", SEED)
    mlp.save(os.path.join(config_dir, f"seed_{seed_val}"))

    return {
        "name":          config["name"],
        "params":        count_params(config["layers"]),
        "train_acc":     train_acc,
        "best_val_acc":  best_val_acc,
        "best_epoch":    best_epoch,
        "macro_f1":      eval_metrics["macro_f1"],
        "stop_reason":   stop_reason,
        "stop_epoch":    stop_epoch,
        "early_stopped": early_stopped,
        "train_loss":    train_loss,
        "val_loss":      val_loss,
        "val_acc":             val_accs,
        "train_acc_per_epoch": mlp.train_accuracies_,
        "config":    config,
    }

def _worker(args):
    return run_experiment(*args)


def _parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter sweep for digit MLP")
    p.add_argument("--config", type=str, default=None,
                   help="JSON file with experiment configs.")
    p.add_argument("--out", type=str, default=None,
                   help="Output dir. Default: results/sweep_<timestamp>/")
    return p.parse_args()


def main():
    cli = _parse_args()
    if cli.config:
        with open(cli.config) as f:
            experiments = json.load(f)
        print(f"Loaded {len(experiments)} experiments from {cli.config}")
    else:
        experiments = EXPERIMENTS

    if not experiments:
        print("No experiments to run.")
        return

    experiments = expand_seeds(experiments)

    print("Loading data...")
    X_all, all_labels = load_digits(TRAIN_PATH)
    X_train, X_val, train_labels, val_labels = stratified_split(
        X_all, all_labels, val_size=0.2, random_state=SEED
    )
    print(f"  train: {len(X_train)} samples  |  val: {len(X_val)} samples")

    args = [(c, X_train, train_labels, X_val, val_labels)
            for c in experiments]

    n_workers = min(len(experiments), mp.cpu_count())
    print(f"Running {len(experiments)} experiments on {n_workers} cores...")
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker, args)

    results.sort(key=lambda r: r["name"])
    print_summary(results)
    plot_accuracy_bars(results)
    plot_val_accuracy(results)
    plot_overfitting_diagnosis(results)

if __name__ == "__main__":
    main()
