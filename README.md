# TPE3-SIA-72.27
Perceptrón Simple y Multicapa. Tercer trabajo práctico para Sistemas de Inteligencia Artificial

## Perceptron Types
 
| Type | Description | Implemented |
|------|-------------|-------------|
| `simple-step` | Step activation function, outputs -1 or 1 | yes |
| `linear` | Linear activation, raw weighted sum as output | yes |
| `non-linear` | Non-linear activation (`tanh` or `logistic`) | yes |
| `multilayer` | Multiple layers of neurons | no |
 
---
 
## Requirements

To install dependencies run:
 
```bash
pip install -r requirements.txt
```
 
---
 
## Sample Data Format
 
Input data must be a CSV file where every column except `label` is treated as a feature (label is expected output):
 
```
x1,x2,label
-1,1,-1
1,-1,-1
-1,-1,-1
1,1,1
```
 
---
 
## Usage
 
```bash
python main.py --data <path_to_csv> --type_p <perceptron_type> [options]
```
 
### Arguments
 
| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--data` | str | ✓ | — | Path to CSV file |
| `--type_p` | str | ✓ | — | Perceptron type: `simple-step`, `linear`, `non-linear` |
| `--lr` | float | | `0.01` | Learning rate |
| `--epochs` | int | | `100` | Max number of training epochs |
| `--epsilon` | float | | `0.001` | Convergence threshold |
| `--tolerance` | float | | `0.5` | Tolerance for counting a prediction as correct (linear/non-linear) |
| `--test_per` | float | | `0.2` | Fraction of data for test split (e.g. `0.2` = 20%) |
| `--seed` | int | | `1` | Random seed for reproducibility |
| `--no_split` | flag | | `False` | Skip train/test split and evaluate on full dataset |
| `--activation` | str | | `tanh` | Activation for non-linear perceptron: `tanh` or `logistic` |
| `--beta` | float | | `1.0` | Beta scaling parameter for non-linear perceptron |
 
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
 

### Observations: Cutoff Criteria

If data converges (no errors in a certain epoch), the loop stops even if the number of epochs has not been reached.