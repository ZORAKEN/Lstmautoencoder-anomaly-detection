# Lstmautoencoder-anomaly-detection
# LSTM Autoencoder & Anomaly Detection

A Deep Learning project exploring Autoencoders, LSTM networks, sequence reconstruction, and anomaly detection using PyTorch. This repository contains implementations on synthetic time-series data, ECG signals, and the MNIST handwritten digit dataset.

## Overview

The goal of this project is to understand how Autoencoders learn compressed representations of data and how reconstruction errors can be used to identify unusual patterns.

The repository explores three implementations:

1. **ECG Anomaly Detection using LSTM Autoencoder** – Detecting abnormal ECG patterns by learning from normal heart signal sequences.
2. **LSTM Autoencoder on Synthetic Time-Series Data** – Building an encoder-decoder architecture to learn and reconstruct sequential sine-wave data.
3. **Autoencoder on MNIST** – Learning compressed representations of handwritten digit images and reconstructing them.

## Projects

### 1. ECG Anomaly Detection using LSTM Autoencoder

**File:** `ECganomaly detectector.py`

This implementation uses the ECG5000 dataset to identify anomalous ECG signals.

**Approach:**
- Load and preprocess the ECG5000 dataset.
- Separate normal ECG signals from anomalous samples.
- Train the LSTM Autoencoder exclusively on normal ECG signals.
- Learn temporal patterns using an LSTM encoder.
- Compress the learned representation into a latent vector.
- Reconstruct the original ECG sequence using an LSTM decoder.
- Calculate reconstruction errors using Mean Squared Error (MSE).
- Set the anomaly detection threshold using the 95th percentile of normal validation reconstruction errors.
- Classify samples with reconstruction errors above the threshold as anomalies.

**Evaluation:**
- Classification Report
- Confusion Matrix
- ROC-AUC Score
- Training and validation loss visualization
- Original vs reconstructed ECG signal comparison

The implementation follows an unsupervised anomaly detection approach, where the model learns normal signal behaviour and identifies deviations during evaluation.

### 2. LSTM Autoencoder on Synthetic Time-Series Data

**File:** `lstmautoencoder.py`

This implementation explores how LSTM-based Autoencoders learn sequential patterns.

**Approach:**
- Generate 2,000 synthetic sine-wave sequences.
- Introduce variations in frequency and phase.
- Add Gaussian noise to simulate variations in signal patterns.
- Build an LSTM encoder-decoder architecture.
- Compress sequential information into a latent representation.
- Reconstruct the original sequences.
- Train the network using MSE loss and the Adam optimizer.

**Model Architecture:**
- LSTM Encoder
- Latent Representation (16 dimensions)
- LSTM Decoder
- Reconstruction Layer

This experiment provides a foundation for understanding sequence modelling, temporal dependencies, latent representations, and reconstruction-based learning.

### 3. Autoencoder on MNIST

**File:** `autoencoderon MNIST.py`

This implementation explores image reconstruction using a fully connected Autoencoder.

**Approach:**
- Load the MNIST handwritten digit dataset using torchvision.
- Convert images into tensors.
- Flatten 28 × 28 images into 784-dimensional vectors.
- Compress the input using an encoder.
- Learn a 32-dimensional latent representation.
- Reconstruct the original image using a decoder.
- Compare reconstructed images with their original versions using MSE loss.

**Model Architecture:**

Encoder:
`784 → 256 → 64 → 32`

Decoder:
`32 → 64 → 256 → 784`

The implementation also visualizes original and reconstructed handwritten digit images.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Torchvision

## Installation

Clone the repository:

```bash
git clone https://github.com/ZORAKEN/Lstmautoencoder-anomaly-detection.git
```

Navigate into the project directory:

```bash
cd Lstmautoencoder-anomaly-detection
```

Install the required dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

## Running the Project

Run the individual implementations:

```bash
python "ECganomaly detectector.py"
```

```bash
python lstmautoencoder.py
```

```bash
python "autoencoderon MNIST.py"
```

**Note:** The ECG anomaly detection implementation requires the `ECG5000_TRAIN.txt` and `ECG5000_TEST.txt` dataset files in the working directory.

## Key Concepts Explored

- Deep Learning and Neural Networks
- Autoencoders and Encoder-Decoder Architectures
- Long Short-Term Memory (LSTM)
- Latent Space Representations
- Time-Series Modelling
- Signal Reconstruction
- Reconstruction-Based Anomaly Detection
- Unsupervised Learning
- Model Evaluation and Visualization

## Learning Outcomes

Through this project, I explored how different types of Autoencoders can be applied to image data and sequential signals.

I gained practical experience in building LSTM encoder-decoder architectures, working with time-series data, learning compressed representations, and applying reconstruction errors as a mechanism for anomaly detection.

One of the main areas of interest in this project was understanding how a model trained on normal ECG signals can identify deviations without requiring anomalous examples during training.

## Future Improvements

- Experiment with different LSTM architectures and latent dimensions.
- Explore GRU-based and Variational Autoencoders.
- Investigate different anomaly threshold selection techniques.
- Improve anomaly detection performance through hyperparameter tuning.
- Explore real-world applications of reconstruction-based anomaly detection.

## Author

**Olivia Gijo**

GitHub: [ZORAKEN](https://github.com/ZORAKEN)

---

*This project is part of my hands-on exploration of Deep Learning, sequence modelling, and anomaly detection using PyTorch.*
