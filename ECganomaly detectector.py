#pip install torch numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

torch.manual_seed(42)
np.random.seed(42)


# 3. LOAD ECG5000 DATASET


train_data = pd.read_csv(
    "ECG5000_TRAIN.txt",
    sep=r"\s+",
    header=None
)

test_data = pd.read_csv(
    "ECG5000_TEST.txt",
    sep=r"\s+",
    header=None
)

# First column = class label
# Remaining 140 columns = ECG signal

X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ECG5000 class 1 = normal

normal_data = X_train[y_train == 1]

print("Normal samples:", len(normal_data))

# Train only on normal ECG signals

X_train_normal, X_val_normal = train_test_split(
    normal_data,
    test_size=0.15,
    random_state=42
)

# Fit normalization only on training data

mean = X_train_normal.mean(axis=0)
std = X_train_normal.std(axis=0)

std[std == 0] = 1

X_train_normal = (X_train_normal - mean) / std
X_val_normal = (X_val_normal - mean) / std
X_test = (X_test - mean) / std

