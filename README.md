#Lstmautoencoder-anomaly-detection
# LSTM Autoencoder & Anomaly Detection

**Exploring Deep Learning, Sequence Reconstruction, and Anomaly Detection using PyTorch.**

This repository contains practical implementations of Autoencoders and LSTM-based encoder-decoder architectures, with experiments on ECG signals, synthetic time-series data, and handwritten digit images.

The main focus is understanding how neural networks learn data representations, reconstruct inputs, and identify unusual patterns through reconstruction errors.

---

## Projects

### 1. ECG Anomaly Detection using LSTM Autoencoder

**File:** `ECganomaly detectector.py`

An LSTM Autoencoder designed to identify abnormal patterns in ECG signals.

**How it works:**

- Trains on normal ECG signals to learn their temporal patterns.
- Uses an LSTM Encoder to compress sequential information into a latent representation.
- Uses an LSTM Decoder to reconstruct the input signal.
- Calculates reconstruction error using Mean Squared Error (MSE).
- Identifies anomalies when reconstruction error exceeds a threshold derived from normal validation samples.

**Evaluation and Visualization:**
- Confusion Matrix
- Classification Report
- ROC-AUC Score
- Training and Validation Loss
- Original vs. Reconstructed ECG Signals

### 2. LSTM Autoencoder for Time-Series Reconstruction

**File:** `lstmautoencoder.py`

An implementation exploring how LSTM networks learn and reconstruct sequential data.

**Highlights:**
- Generates 2,000 synthetic sine-wave sequences.
- Introduces variations in frequency, phase, and Gaussian noise.
- Builds an LSTM Encoder-Decoder architecture.
- Learns compressed representations using a 16-dimensional latent space.
- Reconstructs input sequences using the trained model.

This experiment explores temporal dependencies, sequence learning, and reconstruction using recurrent neural networks.

### 3. Image Reconstruction using Autoencoder (MNIST)

**File:** `autoencoderon MNIST.py`

A fully connected Autoencoder trained on the MNIST handwritten digit dataset.

**Highlights:**
- Processes 28 × 28 grayscale handwritten digit images.
- Compresses 784-dimensional inputs into a 32-dimensional latent representation.
- Reconstructs images through a decoder.
- Uses MSE loss to measure reconstruction quality.
- Visualizes original and reconstructed images.

**Architecture:**

`784 → 256 → 64 → 32 → 64 → 256 → 784`

---

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Image Processing | Torchvision |

---

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/ZORAKEN/Lstmautoencoder-anomaly-detection.git
```

**2. Navigate to the project**

```bash
cd Lstmautoencoder-anomaly-detection
```

**3. Install dependencies**

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

**4. Run an implementation**

```bash
python lstmautoencoder.py
```

```bash
python "ECganomaly detectector.py"
```

```bash
python "autoencoderon MNIST.py"
```

*Note: The ECG implementation requires the ECG5000 training and testing dataset files.*

---

## Key Concepts

- Autoencoders and Encoder-Decoder Architectures
- LSTM Networks and Sequential Learning
- Latent Space Representations
- Time-Series Reconstruction
- Unsupervised Anomaly Detection
- Reconstruction Error and Thresholding
- Neural Network Training and Evaluation

---

## What I Learned

Working on these implementations helped me understand how Autoencoders learn compressed representations and reconstruct different types of data.

I explored how LSTM networks capture temporal dependencies in sequential signals and how reconstruction errors can be used to detect abnormal patterns without training directly on anomalous examples.

I also gained practical experience working with PyTorch, time-series preprocessing, model training, evaluation, and visualization.

---

## Future Improvements

- Experiment with different LSTM architectures.
- Explore GRU and Variational Autoencoders.
- Investigate adaptive anomaly detection thresholds.
- Improve model performance through hyperparameter tuning.
- Extend anomaly detection experiments to additional real-world signals.

---

## Author

**Olivia Gijo**

[GitHub Profile](https://github.com/ZORAKEN)

*Exploring Machine Learning, Deep Learning, and real-world applications of AI.*
