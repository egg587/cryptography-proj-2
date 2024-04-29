import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from concrete.ml.quantization.quantized_module import QuantizedModule
from concrete.ml.torch.compile import compile_brevitas_qat_model

# Data set generation
# Generate a simple checkerboard pattern, which is a common benchmark for non-linear classification problems.

IN_FEAT = 2
OUT_FEAT = 1
N_SIDE = 100
N_EXAMPLE_TOTAL = N_SIDE ** 2
N_TEST = 500
CLUSTERS = 3

print("Generating dataset...")

# Generate the grid points and put them in a 2 column list of X,Y coordinates
xx, yy = np.meshgrid(np.linspace(0, 1, N_SIDE), np.linspace(0, 1, N_SIDE))
X = np.c_[np.ravel(xx), np.ravel(yy)]

# Genberate the labels, using the XOR function to produce the checkerboard pattern
y = (np.rint(xx * CLUSTERS).astype(np.int64) % 2) ^ ((np.rint(yy * CLUSTERS).astype(np.int64) % 2))
y = np.ravel(y)

# Add some noise to the data
X += np.random.randn(X.shape[0], X.shape[1]) * 0.01

# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
plt.title("Original Dataset")
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=N_TEST / N_EXAMPLE_TOTAL, random_state=42)


