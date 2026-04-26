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

| Script | Purpose |
|--------|---------|
| `main.py` | Single model: train once, evaluate, save plots |
| `experiment_runner_digits.py` | Batch experiments in parallel on the digit dataset |
 
---
 
## Requirements

To install dependencies run:
 
```bash
pip install -r requirements.txt
```
 
---
 
## Data Format

### Standard (`data/`)

Every column except `label` is a feature. `label` is the target.

```
x1,x2,label
-1,1,-1
1,1,1
```

### Digit dataset (`datasets/`)

One row per image. The `image` column is a Python list literal of 784 pixel values (28×28 flattened). `label` is the digit class 0–9.

```
image,label
"[0.0, 0.0, ..., 0.0]",3
```
 
---
 
## Usage
 
```bash
python main.py --data <path_to_csv> --type_p <perceptron_type> [options]
```
 
### Arguments
 

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | required | Path to CSV file (must have a `label` column) |
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
 
Evaluate on the full dataset (recommended for small datasets):
 
```bash
python main.py --data data/and_data.csv --type_p simple-step --lr 0.1 --epochs 100 --no_split
```

Non-linear perceptron with logistic activation:

```bash
python main.py --data data/transactions.csv --type_p non-linear --activation logistic --beta 0.5 --lr 0.01 --epochs 500
```

XOR (multilayer)

```bash
python main.py --data data/xor_data.csv --type_p multilayer --layers 2 2 1 --lr 0.1 --epochs 500 --no_split --tolerance 0.5
```

Multi-class classification

```bash
python main.py --data datasets/digits.csv --type_p multilayer \
  --layers 784 10 10 --lr 0.01 --epochs 100 --initializer xavier
```
 
### Plots saved to `plots/`

- `loss_curve.png` — training loss over epochs (all types except simple-step)
- `confusion_matrix.png` — predicted vs true class heatmap (multilayer with >1 output)
- `per_class_metrics.png` — precision, recall, F1 per class (multilayer with >1 output)

### Observations: Cutoff Criteria

If data converges (no errors in a certain epoch), the loop stops even if the number of epochs has not been reached.


---

## `experiment_runner_digits.py`

Runs multiple configurations in parallel on the 28×28 digit dataset (`datasets/digits.csv` / `datasets/digits_test.csv`).

```bash
python experiment_runner_digits.py
```

Edit the `EXPERIMENTS` list at the top of the file to define what to test:

```python
EXPERIMENTS = [
    {"name": "1L [784,100,10]", "layers": [784, 100, 10], "lr": 0.01,
     "epochs": 250, "epsilon": 1e-6, "beta": 1.0, "initializer": "xavier"},
]
```

### Results saved to `results/` (append mode)

| File | Contents |
|------|---------|
| `summary.csv` | One row per experiment: accuracy, F1, gap, convergence epoch |
| `curves.csv` | One row per epoch per experiment: train loss, test loss, test accuracy |
| `per_class.csv` | One row per class per split per experiment: precision, recall, F1, support |

### Plots saved to `plots/`

| File | What it shows |
|------|--------------|
| `loss_curves.png` | Train and test loss over epochs, one line per experiment |
| `accuracy_bars.png` | Train vs test accuracy per experiment |
| `convergence.png` | Test accuracy % over epochs |
| `confusion_matrices.png` | Confusion matrix per experiment |
| `per_class_f1.png` | Heatmap: digit classes × experiments, color = F1 |
| `summary_table.png` | Color-coded table of all key metrics |

---