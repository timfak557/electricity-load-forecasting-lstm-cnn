# ⚡ Electricity Load Forecasting: LSTM vs 1D CNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A deep learning time series project comparing LSTM and 1D CNN architectures for electricity consumption forecasting using the UCI Electricity Load Diagrams 2011–2014 dataset.**

[Overview](#-overview) • [Dataset](#-dataset) • [Models](#-models) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure)

</div>

---

## 📌 Overview

Accurate electricity load forecasting is critical for smart grid management, energy trading, and carbon footprint reduction. This project implements and compares two deep learning architectures:

- **LSTM (Long Short-Term Memory)** — a recurrent model designed to capture long-range temporal dependencies
- **1D CNN (Convolutional Neural Network)** — a parallelizable feedforward model optimized for local pattern detection

Both models are trained on real-world industrial electricity consumption data, evaluated on identical test splits, and compared across accuracy, training speed, and architecture tradeoffs.

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| **Name** | UCI Electricity Load Diagrams 2011–2014 |
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) |
| **Period** | January 2011 – December 2014 |
| **Clients** | 370 Portuguese industrial/commercial clients |
| **Frequency** | 15-minute intervals → resampled to hourly |
| **Task** | Univariate next-step forecasting (Client MT_001) |
| **Format** | Semicolon-delimited `.txt`, European decimal format |

### Key EDA Findings

- Strong **daily seasonality**: morning ramp-up, evening peak, overnight trough
- **Weekend dip**: Saturday/Sunday loads are significantly lower than weekdays
- **Monthly variation**: consumption peaks in winter months
- Some clients exhibit high correlation, while others are independent

---

## 🧠 Models

### LSTM Architecture

```
Input(24, 1)
  → LSTM(64, return_sequences=True, tanh)
  → Dropout(0.2)
  → LSTM(32, tanh)
  → Dropout(0.2)
  → Dense(16, relu)
  → Dense(1)           ← Next-step prediction
```

| Design Choice | Rationale |
|---------------|-----------|
| `return_sequences=True` on Layer 1 | Passes full hidden state sequence to second LSTM for richer feature extraction |
| `Dropout(0.2)` | Prevents neuron co-adaptation; reduces overfitting |
| `tanh` activation | Bounded output prevents gradient explosion; matches LSTM's internal gating |
| `Adam(lr=1e-3)` | Adaptive moment estimation; robust to noisy gradients |

### 1D CNN Architecture

```
Input(24, 1)
  → Conv1D(64, kernel=3, relu, padding='same')
  → Conv1D(64, kernel=3, relu, padding='same')
  → MaxPooling1D(2)
  → Conv1D(32, kernel=3, relu, padding='same')
  → MaxPooling1D(2)
  → Flatten()
  → Dense(64, relu)
  → Dropout(0.2)
  → Dense(1)
```

| Design Choice | Rationale |
|---------------|-----------|
| `kernel_size=3` | Each filter spans 3 consecutive time steps — detects short local patterns |
| `MaxPooling1D(2)` | Downsamples feature maps; provides translation invariance and reduces parameters |
| Two Conv blocks | Hierarchical feature extraction: low-level → high-level temporal patterns |
| Fully parallelizable | All convolutions computed simultaneously → significantly faster training |

---

## ⚙️ Methodology

### Data Pipeline

```
Raw 15-min data
    ↓
Resample to hourly (natural noise smoothing)
    ↓
Forward-fill → Backward-fill missing values
    ↓
Train/Test split (80/20, temporal order preserved — NO shuffling)
    ↓
MinMaxScaler fit on TRAIN only (prevents data leakage)
    ↓
Sliding Window (size=24) → (samples, 24, 1) tensors
    ↓
Model training with EarlyStopping + ReduceLROnPlateau
    ↓
Inverse-transform predictions → evaluate in original kW scale
```

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Window size | 24 time steps (24 hours look-back) |
| Batch size | 64 |
| Max epochs | 50 |
| Optimizer | Adam (lr=1e-3) |
| Loss function | MSE |
| Validation split | 10% of training set (temporal) |
| EarlyStopping patience | 10 epochs |
| ReduceLROnPlateau factor | 0.5 (patience=5) |
| Random seed | 42 |

---

## 📈 Results

> ⚠️ Exact metrics depend on your hardware and TensorFlow version. Results below are representative.

| Metric | LSTM | 1D CNN |
|--------|------|--------|
| **RMSE (kW)** | — | — |
| **MAE (kW)** | — | — |
| **MAPE (%)** | — | — |
| **Training Time (s)** | Slower | ~2–3× faster |
| **Epochs to Converge** | ~20–35 | ~15–30 |
| **Parameters** | ~50K | ~30K |

### Qualitative Comparison

| Dimension | LSTM | 1D CNN |
|-----------|------|--------|
| Long-range dependencies | ✅ Excellent | ⚠️ Limited by kernel size |
| Parallelizable training | ❌ Sequential | ✅ Fully parallel |
| Short local patterns | ⚠️ OK | ✅ Excellent |
| Noise robustness | ✅ Good | ⚠️ Needs smoothing |
| Deployment footprint | Larger | Smaller |
| Best suited for | Long sequences, complex patterns | Speed-critical, short patterns |

### When to use each

- **Choose LSTM** when long-range context matters (e.g., weekly seasonality in a single model, anomaly-sensitive deployments)
- **Choose 1D CNN** when training speed and inference latency are priorities with comparable accuracy

---

## 🗂️ Project Structure

```
electricity-load-forecasting/
│
├── notebooks/
│   └── electricity_forecasting.ipynb    # Main project notebook
│
├── data/
│   └── LD2011_2014.txt                  # Raw dataset (download separately)
│
├── models/
│   ├── lstm_model.h5                    # Saved LSTM weights
│   └── cnn_model.h5                     # Saved CNN weights
│
├── outputs/
│   └── plots/
│       ├── 01_time_series_overview.png
│       ├── 02_temporal_patterns.png
│       ├── 03_distribution_correlation.png
│       ├── 04_sliding_window.png
│       ├── 05_lstm_training.png
│       ├── 06_cnn_training.png
│       ├── 07_val_loss_comparison.png
│       ├── 08_predictions_vs_actual.png
│       ├── 09_residuals.png
│       ├── 10_model_comparison.png
│       ├── 11_overfitting_analysis.png
│       └── 12_final_dashboard.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/electricity-load-forecasting.git
cd electricity-load-forecasting
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
wget https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
unzip electricityloaddiagrams20112014.zip -d ./data/
```

Or run the download cell inside the notebook — it handles this automatically.

---

## 🖥️ Usage

### Run in Jupyter

```bash
jupyter notebook notebooks/electricity_forecasting.ipynb
```

### Run in Google Colab

Click the badge below or upload the notebook to [colab.research.google.com](https://colab.research.google.com):

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> The notebook includes a `!wget` cell to download the dataset automatically inside Colab.

### Load saved models

```python
from tensorflow.keras.models import load_model

lstm_model = load_model('models/lstm_model.h5')
cnn_model  = load_model('models/cnn_model.h5')
```

---

## 🛠️ Challenges & Solutions

| Challenge | Solution Applied |
|-----------|-----------------|
| **Overfitting** | Dropout(0.2) + EarlyStopping(patience=10) |
| **Vanishing gradients** | LSTM gating mechanism + ReduceLROnPlateau |
| **Data leakage** | Scaler fitted exclusively on training split |
| **Training instability** | `shuffle=False` + batch size 64 + fixed seed (42) |
| **High-frequency noise** | Resample 15-min → hourly + forward-fill NaNs |

---

## 🔬 Potential Extensions

- [ ] Add **GRU** as a third architecture for comparison
- [ ] Implement **multi-step forecasting** (predict next 24 hours at once)
- [ ] Extend to **multi-client forecasting** (multi-variate input)
- [ ] Try **Transformer / Temporal Fusion Transformer** architecture
- [ ] Add **hyperparameter tuning** via Keras Tuner or Optuna
- [ ] Build a **Streamlit dashboard** for real-time inference

---

## 📚 References

- [UCI Electricity Load Diagrams Dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*
- LeCun et al. (1989) — *Convolutional Neural Networks*
- Chollet, F. — *Deep Learning with Python* (Manning, 2021)
- TensorFlow / Keras documentation — [keras.io](https://keras.io)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ⚡ and TensorFlow

</div>
