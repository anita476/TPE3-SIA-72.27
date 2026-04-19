import numpy as np
import pandas as pd

np.random.seed(42)
x = np.linspace(-10, 10, 400)
y = 2 * x + 1 + np.random.normal(0, 2.0, size=x.shape)  # noise σ=2

df = pd.DataFrame({"x1": x, "label": y})
df.to_csv("data/linear_function_data_noisy.csv", index=False)