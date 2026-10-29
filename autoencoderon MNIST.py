
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Load MNIST dataset
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False
)


# 3. Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: 784 -> 256 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 32)
        )

        # Decoder: 32 -> 64 -> 256 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 256),
            nn.ReLU(),

            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Flatten image: (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)

        # Compress input
        latent = self.encoder(x)

        # Reconstruct input
        reconstructed = self.decoder(latent)

        return reconstructed


# 4. Initialize model, loss, and optimizer
model = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 5. Train the autoencoder
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, _ in train_loader:
        images = images.to(device)

        # Forward pass
        reconstructed = model(images)

        # Compare reconstruction with original pixels
        original = images.view(images.size(0), -1)
        loss = criterion(reconstructed, original)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_dataset)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {avg_loss:.6f}"
    )


# 6. Evaluate and visualize reconstructed images
model.eval()

with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)

    reconstructed = model(images)

    original_images = images[:10].cpu()
    reconstructed_images = reconstructed[:10].cpu().view(-1, 1, 28, 28)

# Plot original vs reconstructed images
fig, axes = plt.subplots(2, 10, figsize=(15, 3))

for i in range(10):
    axes[0, i].imshow(original_images[i].squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title("Original")

    axes[1, i].imshow(
        reconstructed_images[i].squeeze(), cmap="gray"
    )
    axes[1, i].axis("off")
    axes[1, i].set_title("Reconstructed")

plt.tight_layout()
plt.show()

# 7. Save the trained model
torch.save(model.state_dict(), "autoencoder.pth")
print("Model saved successfully!")
