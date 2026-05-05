# TPE3-SIA-72.27
Perceptrón Simple y Multicapa. Tercer trabajo práctico para Sistemas de Inteligencia Artificial

## Perceptron Types

| Type | Description | Implemented |
|------|-------------|-------------|
| `simple-step` | Step activation function, outputs -1 or 1 | yes |
| `linear` | Linear activation, raw weighted sum as output | yes |
| `non-linear` | Non-linear activation (`tanh` or `logistic`) | yes |
| `multilayer` | Multiple layers of neurons | yes |

## Entry points

### CLI / drivers

| Script | Purpose |
|--------|---------|
| `main.py` | Single model on any flat CSV dataset |
| `digits_main.py` | Single MLP run on the digit CSVs, metrics and plots |
| `experiment_runner_digits.py` | Batch MLP experiments (defined in `EXPERIMENTS` inside the file) |
| `remake_plots.py` | Regenerate digit plots from pickled models in `results/models/` (no retraining) |

### Fraud, linear vs non-linear, and generalization (`scripts/`)

| Script | Purpose |
|--------|---------|
| `scripts/experiment_runner_linear_nonlinear.py` | Batch **simple** linear and non-linear perceptron runs from a JSON config; stratified train/test split; fraud metrics and CSVs under `results/` |
| `scripts/plot_activation_tanh_vs_logistic.py` | **Tanh vs logistic** figures (LR sweep, generalization gap, last train MSE, convergence) → `plots/ej1/tanh_vs_logistica_*.png` |
| `scripts/generalization_study.py` | **Q2a / Q2b / Q2c** plots from runner output → `plots/ej1/gen_study_*.png` |
| `scripts/comparison_underfitting.py` | **Underfitting** study: trains on the **full** dataset (no test split when `test_per` is null); writes `*_linear_curves.csv` / `*_nonlinear_curves.csv` and recall CSVs |
| `scripts/comparison_underfitting_plot.py` | Plots MSE / BCE learning curves from the underfitting CSVs |

Run these from the **repository root** so paths such as `data/...` in the JSON configs resolve correctly.

---

## Requirements

To install dependencies run:

```bash
pip install -r requirements.txt
```

---

## Data format

### Tabular CSV (`data/`)

For `main.py`, one column is the target. By default the code expects a column named `label`; all other numeric columns are features.

```
x1,x2,label
-1,1,-1
1,1,1
```

### Digit images (`data/`)

`digits_main.py` loads `data/digits.csv` and `data/digits_test.csv` (one row per image). The `image` column is a string containing a Python list of 784 pixel values (28×28 flattened). `label` is the class 0–9.

`experiment_runner_digits.py` uses its own in-file path (`TRAIN_PATH`); if your files live only under `data/`, point that constant to `data/digits.csv` to match `digits_main.py`.

### Fraud example (`data/fraud_dataset.csv`)

Batch configs under `configs/` (e.g. `lr_exploration_tanh_logistic.json`) set the label column in JSON (`"label": "flagged_fraud"`) and may list extra feature columns to drop. Numeric features are used after drops.

---

## `main.py` usage

```bash
python main.py --data <path_to_csv> --type_p <perceptron_type> [options]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | required | Path to CSV file (target column: `label` unless you change the code) |
| `--type_p` | str | required | `simple-step`, `linear`, `non-linear`, `multilayer` |
| `--lr` | float | `0.01` | Learning rate |
| `--epochs` | int | `100` | Max training epochs |
| `--epsilon` | float | `0.001` | Early stopping threshold (stops if loss < epsilon) |
| `--tolerance` | float | `0.5` | Margin for counting a prediction as correct (linear / non-linear) |
| `--test_per` | float | `0.2` | Fraction of data held out for testing |
| `--seed` | int | `1` | Random seed |
| `--no_split` | flag | off | Skip train/test split, evaluate on full dataset |
| `--activation` | str | `tanh` | Activation for non-linear / multilayer: `tanh` or `logistic` |
| `--beta` | float | `1.0` | Beta scaling for activation function |
| `--layers` | int... | `2 2 1` | Layer sizes for multilayer (input → hidden... → output) |
| `--initializer` | str | `random` | Weight init for multilayer: `random`, `xavier`, or `xavier_n` |

### Examples

Train and evaluate with an 80/20 train/test split:

```bash
python main.py --data data/and_data.csv --type_p simple-step --lr 0.1 --epochs 100 --test_per 0.2 --seed 1
```

Evaluate on the full dataset (recommended for small toy sets):

```bash
python main.py --data data/and_data.csv --type_p simple-step --lr 0.1 --epochs 100 --no_split
```

Non-linear perceptron with logistic activation:

```bash
python main.py --data data/transactions.csv --type_p non-linear --activation logistic --beta 0.5 --lr 0.01 --epochs 500
```

XOR (multilayer):

```bash
python main.py --data data/xor_data.csv --type_p multilayer --layers 2 2 1 --lr 0.1 --epochs 500 --no_split --tolerance 0.5
```

### Plots saved to `plots/` (`main.py`)

- `loss_curve.png` — training loss over epochs (all types except simple-step)
- `confusion_matrix.png` — heatmap (multilayer with >1 output)
- `per_class_metrics.png` — precision, recall, F1 per class (multilayer with >1 output)

---

## `digits_main.py`

```bash
python digits_main.py [options]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--layers` | int... | `784 100 10` | Layer sizes (input → hidden... → output) |
| `--lr` | float | `0.01` | Learning rate |
| `--epochs` | int | `250` | Max training epochs |
| `--epsilon` | float | `1e-6` | Early stopping threshold |
| `--seed` | int | `1` | Random seed |
| `--activation` | str | `tanh` | `tanh` or `logistic` |
| `--beta` | float | `1.0` | Beta scaling for activation |
| `--initializer` | str | `xavier` | `random`, `xavier`, or `xavier_n` |

Prints train/test accuracy, macro F1, and min class F1. Saves `loss_curve.png` (train + test), `confusion_matrix.png`, and `per_class_metrics.png` to `plots/`.

---

## `experiment_runner_digits.py`

Runs the configurations from the `EXPERIMENTS` list at the top of the file, or from a JSON file passed with `--config`, using a process pool. Training data path is `TRAIN_PATH` in that file (`datasets/digits.csv` by default—point it to `data/digits.csv` if you keep digits there).

```bash
python experiment_runner_digits.py
```

When experiments finish, it prints an aggregated summary (`print_summary`) and calls the plotting helpers from `utils/visualization.py` (`plot_accuracy_bars`, `plot_val_accuracy`, `plot_overfitting_diagnosis`). Whether figures are saved to disk depends on those functions; this runner does **not** mirror the single-run PNG set from `digits_main.py`. Use `remake_plots.py` if you rely on saved models under `results/models/`.

---

## Fraud workflow (configs + `scripts/`)

### 1. Train experiments → CSVs

The default config (if you omit `--config`) is `configs/lr_exploration_tanh_logistic.json`.

```bash
python scripts/experiment_runner_linear_nonlinear.py
# same as:
python scripts/experiment_runner_linear_nonlinear.py --config configs/lr_exploration_tanh_logistic.json
```

Useful flags: `--workers N`, `--dry-run`, `--drop COL ...`, `--no-linear`.

Typical outputs in `results/` (fixed filenames):

| File | Contents |
|------|---------|
| `linear_vs_nonlinear_summary.csv` | One row per run: metrics, ROC-AUC, best threshold / F1, etc. |
| `linear_vs_nonlinear_curves.csv` | Per-epoch train (and test) MSE over seeds |
| `linear_vs_nonlinear_roc.csv` | ROC/PR-style curve samples |
| `linear_vs_nonlinear_confusion_runs.csv` | Confusion-related summaries |

Other JSON grids (e.g. `configs/experiments_generalization.json`) use the same runner; align `grid` and `base` with the analysis scripts you run next.

### 2. Tanh vs logistic plots

Requires the summary + curves + ROC files from step 1 (defaults assume them under `results/`).

```bash
python scripts/plot_activation_tanh_vs_logistic.py
```

Optional: `--config`, `--summary`, `--curves`, `--roc`. Figures go to `plots/ej1/` (`tanh_vs_logistica_lr_sweep.png`, `tanh_vs_logistica_brecha.png`, `tanh_vs_logistica_last_mse.png`, `tanh_vs_logistica_convergencia.png`, `tanh_vs_logistica_convergencia_todos_lrs.png`).

### 3. Generalization study (Q2a–Q2c)

After producing results with a config consistent with `configs/experiments_generalization.json`:

```bash
python scripts/generalization_study.py --config configs/experiments_generalization.json
```

Figures: `plots/ej1/gen_study_*.png` (distribution, metrics, PR, F2 vs test size, final model, threshold, confusion summary, optional big-model comparison when data is available).

### 4. Underfitting (optional, full-dataset training)

```bash
python scripts/comparison_underfitting.py --config configs/underfitting_compare.json --outpath results/underfitting_runs
```

Then plot (filenames depend on `base.name` in the JSON; example below uses `fraud`):

```bash
python scripts/comparison_underfitting_plot.py --linear results/underfitting_runs/fraud_linear_curves.csv --nonlinear results/underfitting_runs/fraud_nonlinear_curves.csv --log --out results/plots
```

---

## `remake_plots.py`

Reloads models from `results/models/<config_name>/` and rebuilds digit plots without training. See the docstring at the top of the file for flags.

```bash
python remake_plots.py
```
