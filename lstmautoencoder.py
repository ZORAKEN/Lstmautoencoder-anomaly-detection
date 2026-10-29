import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Generate synthetic sine-wave sequences
torch.manual_seed(42)

num_samples = 2000
seq_len = 30
input_size = 1

t = torch.linspace(0, 4 * torch.pi, seq_len)

data = []

for _ in range(num_samples):
    frequency = torch.empty(1).uniform_(0.5, 2.0).item()
    phase = torch.empty(1).uniform_(0, 2 * torch.pi).item()

    signal = torch.sin(frequency * t + phase)
    signal += 0.05 * torch.randn(seq_len)

    data.append(signal.unsqueeze(-1))

data = torch.stack(data)  # (2000, 30, 1)

dataset = TensorDataset(data)
train_loader = DataLoader(
    dataset, batch_size=64, shuffle=True
)

print(data.shape)  # (2000, 30, 1)

class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        latent_size=16
    ):
        super().__init__()

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Compress the encoder's final hidden state
        self.to_latent = nn.Linear(
            hidden_size, latent_size
        )

        # Expand latent representation for the decoder
        self.from_latent = nn.Linear(
            latent_size, hidden_size
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Predict the original feature at each time step
        self.output_layer = nn.Linear(
            hidden_size, input_size
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        # 1. Encode the sequence
        _, (hidden, cell) = self.encoder(x)

        # hidden shape: (1, batch, hidden_size)
        hidden = hidden[-1]

        # 2. Create compressed latent representation
        latent = self.to_latent(hidden)

        # 3. Expand latent representation
        decoder_input = self.from_latent(latent)

        # Repeat the same context vector at every time step
        decoder_input = decoder_input.unsqueeze(1).repeat(
            1, x.size(1), 1
        )

        # 4. Decode the sequence
        decoded, _ = self.decoder(decoder_input)

        # 5. Reconstruct the original features
        reconstructed = self.output_layer(decoded)

        return reconstructed, latent


model = LSTMAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for (sequences,) in train_loader:
        sequences = sequences.to(device)

        # Forward pass
        reconstructed, latent = model(sequences)

        # Original and reconstructed sequences have
        # the same shape: (batch, 30, 1)
        loss = criterion(reconstructed, sequences)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)

    avg_loss = total_loss / len(dataset)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {avg_loss:.6f}"
    )
