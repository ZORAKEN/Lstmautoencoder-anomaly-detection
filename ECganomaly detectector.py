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

#pytorch tensors

X_train_tensor = torch.tensor(
    X_train_normal,
    dtype=torch.float32
).unsqueeze(-1)

X_val_tensor = torch.tensor(
    X_val_normal,
    dtype=torch.float32
).unsqueeze(-1)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).unsqueeze(-1)

y_test_tensor = torch.tensor(
    (y_test != 1).astype(np.int64)
)

print("Input shape:", X_train_tensor.shape)

train_loader = DataLoader(
    TensorDataset(X_train_tensor),
    batch_size=64,
    shuffle=True
)
class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        latent_dim=16,
        seq_len=140
    ):

        super().__init__()

        self.seq_len = seq_len

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.hidden_to_latent = nn.Linear(
            hidden_dim,
            latent_dim
        )

        # Latent to decoder input
        self.latent_to_hidden = nn.Linear(
            latent_dim,
            hidden_dim
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            input_dim
        )

    def forward(self, x):

        batch_size = x.size(0)

        # Encoder
        _, (hidden, cell) = self.encoder(x)

        # Last hidden state
        hidden = hidden[-1]

        # Latent representation
        latent = self.hidden_to_latent(hidden)

        # Expand latent representation
        decoder_input = self.latent_to_hidden(latent)

        decoder_input = decoder_input.unsqueeze(1).repeat(
            1, self.seq_len, 1
        )

        # Decoder
        decoded, _ = self.decoder(decoder_input)

        # Reconstruct original ECG
        reconstructed = self.output_layer(decoded)

        return reconstructed


model = LSTMAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print(model)

#Training

num_epochs = 30

train_losses = []
val_losses = []

for epoch in range(num_epochs):

    model.train()

    total_loss = 0

    for (batch,) in train_loader:

        batch = batch.to(device)

        # Forward pass
        reconstructed = model(batch)

        # Reconstruction loss
        loss = criterion(reconstructed, batch)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

 # Validation
    model.eval()

    with torch.no_grad():

        val_reconstructed = model(
            X_val_tensor.to(device)
        )

        val_loss = criterion(
            val_reconstructed,
            X_val_tensor.to(device)
        ).item()

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {avg_train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM Autoencoder Training")

plt.legend()
plt.grid()
plt.show()

def reconstruction_error(data):

    model.eval()

    with torch.no_grad():

        data = data.to(device)

        reconstructed = model(data)

        # Error for each ECG sample
        error = torch.mean(
            (data - reconstructed) ** 2,
            dim=(1, 2)
        )

    return error.cpu().numpy()


# Normal validation errors
val_errors = reconstruction_error(X_val_tensor)

# Test errors
test_errors = reconstruction_error(X_test_tensor)


# ANOMALY THRESHOLD
# 95th percentile of normal validation errors

threshold = np.percentile(val_errors, 95)

print("Anomaly Threshold:", threshold)

# 0 = Normal
# 1 = Anomaly

predictions = (test_errors > threshold).astype(int)

actual = (y_test != 1).astype(int)

print("\nClassification Report:")

print(
    classification_report(
        actual,
        predictions,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)


#confusion
cm = confusion_matrix(actual, predictions)

print("Confusion Matrix:")
print(cm)



# ROC-AUC SCORE
auc = roc_auc_score(actual, test_errors)
print("ROC-AUC:", auc)


#  VISUALIZE ECG SIGNALS
normal_indices = np.where(actual == 0)[0]
anomaly_indices = np.where(actual == 1)[0]

plt.figure(figsize=(12, 5))

if len(normal_indices) > 0:
    idx = normal_indices[0]

    plt.plot(
        X_test[idx],
        label="Normal ECG",
        linewidth=2
    )

if len(anomaly_indices) > 0:
    idx = anomaly_indices[0]

    plt.plot(
        X_test[idx],
        label="Anomalous ECG",
        linewidth=2
    )

plt.xlabel("Time Steps")
plt.ylabel("Normalized ECG Amplitude")

plt.title("Normal vs Anomalous ECG")

plt.legend()
plt.grid()
plt.show()


#  RECONSTRUCTION 


model.eval()

with torch.no_grad():

    sample = X_test_tensor[:1].to(device)

    reconstructed = model(sample)

    original = sample.cpu().numpy()[0, :, 0]

    reconstructed = (
        reconstructed.cpu().numpy()[0, :, 0]
    )

plt.figure(figsize=(12, 5))

plt.plot(original, label="Original ECG")
plt.plot(reconstructed, label="Reconstructed ECG")

plt.xlabel("Time Steps")
plt.ylabel("Normalized Amplitude")

plt.title("ECG Reconstruction")

plt.legend()
plt.grid()
plt.show()


#model

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "threshold": threshold,
        "mean": mean,
        "std": std
    },
    "ecg_lstm_autoencoder.pth"
)

print("Model saved successfully!")




