<div align="center">

# 🛡️ Multimodal Phishing Detection System

**A Deep Learning system that detects phishing websites by simultaneously analyzing both the URL text and the visual appearance of the webpage.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/svspraveen/multimodal-phishing-detection-dataset)
[![Kaggle Model](https://img.shields.io/badge/Model%20Weights-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/models/svspraveen/multimodal-phishing-detection)

<br/>

*Developed by [SVS Praveen](https://www.linkedin.com/in/svs-praveen-s) · Academic Project*

</div>

---

## 📌 Overview

Traditional phishing detection systems rely on static URL blacklists or simple heuristic rules that fail against modern, sophisticated phishing attacks. This project takes a fundamentally different approach — it **mimics how a human expert detects phishing**: by reading the URL *and* visually inspecting the page at the same time.

The system fuses two deep learning branches — a **Bi-LSTM** for URL text analysis and a **VGG16 CNN** for visual screenshot analysis — into a single **Fusion Neural Network** that outputs a phishing probability score in real time.

---

## 🏗️ Architecture

```
        ┌────────────────────────┐     ┌────────────────────────┐
        │     URL (Text Input)   │     │  Screenshot (Visual)   │
        └────────────┬───────────┘     └────────────┬───────────┘
                     │                               │
             ┌───────▼───────┐               ┌───────▼───────┐
             │  Text Branch  │               │ Visual Branch │
             │  (Bi-LSTM)    │               │   (VGG16)     │
             │               │               │               │
             │ • Tokenization│               │ • Headless    │
             │ • Word2Vec    │               │   Screenshot  │
             │ • 128 units   │               │ • Transfer    │
             │ • Bidirection.│               │   Learning    │
             └───────┬───────┘               └───────┬───────┘
                     │                               │
                     └──────────────┬────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │   Fusion Layer    │
                          │  (Dense + Dropout)│
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  Output: Phishing │
                          │  Probability 0–1  │
                          └───────────────────┘
```

### Text Branch — Bi-LSTM
- Tokenizes the URL into structural units (subdomains, TLDs, keywords, special characters)
- Trains **Word2Vec embeddings** (100-dimensional) on the URL corpus
- Passes through a **Bidirectional LSTM** (128 units) to capture sequential patterns in both directions

### Visual Branch — VGG16 (Transfer Learning)
- Drives a **headless Chrome browser** via Selenium to capture a live screenshot of the URL
- Preprocesses the screenshot to `224×224` pixels
- Uses a **VGG16 CNN** pre-trained on ImageNet (top 15 layers frozen) to extract rich visual features

### Fusion Layer
- Concatenates both feature vectors into a unified representation
- Passes through **Dense (64) → Dropout (0.5) → Sigmoid** for final classification

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔀 **Multimodal Fusion** | Combines URL text + visual rendering for higher accuracy |
| 🌐 **Live Screenshot Capture** | Headless Selenium renders actual pages in real time |
| 📊 **Automated Data Pipeline** | Fetches from PhishTank, URLhaus, and Tranco Top-1M |
| 🛡️ **Trusted Domain Whitelist** | Prevents false positives on popular legitimate domains |
| 🚀 **REST API + Web UI** | Flask-based API with a responsive, dark-themed interface |
| 🔁 **Colab Training Ready** | Full training pipeline designed for Google Colab (GPU) |

---

## 📁 Project Structure

```
Multimodel-Phishing-Detection/
│
├── configs/
│   └── config.py               # All hyperparameters & path configurations
│
├── preprocessing/
│   ├── text_processor.py       # URL tokenization & Word2Vec embedding
│   └── visual_processor.py    # Headless screenshot capture via Selenium
│
├── models/
│   ├── text_branch.py          # Bi-LSTM model definition
│   ├── visual_branch.py        # VGG16 transfer learning model definition
│   └── fusion_model.py         # Combined Fusion Neural Network
│
├── scripts/
│   ├── collect_data.py         # Automated data collection from APIs
│   ├── train_model.py          # Local training script
│   └── predict_phishing.py     # Standalone inference / prediction script
│
├── deployment/
│   ├── web_api.py              # Flask REST API + Web Interface
│   └── run_deployment.py       # One-click server launcher
│
├── notebooks/
│   ├── data_collection.ipynb   # Data exploration notebook
│   └── training_complete.ipynb # Full training walkthrough notebook
│
├── colab_train.py              # 🚀 Google Colab end-to-end training script
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/SVSPraveen/Multimodel-Phishing-Detection.git
cd Multimodel-Phishing-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download the dataset from Kaggle and place it in the `data/` folder:

> 📦 **[Multimodal Phishing Detection Dataset on Kaggle](https://www.kaggle.com/datasets/svspraveen/multimodal-phishing-detection-dataset)**

```
data/
└── combined_urls.csv
```

### 4. Download Pre-trained Model Weights
The trained model weights are hosted on Kaggle Models (~200MB) and are **not stored in this repo**.

> ⬇️ **[Download `fusion_model.keras` + `tokenizer.pkl` on Kaggle Models](https://www.kaggle.com/models/svspraveen/multimodal-phishing-detection)**

Place the downloaded files in the `models/` directory:
```
models/
├── fusion_model.keras
└── tokenizer.pkl
```

### 5. Run the Web Interface
```bash
python deployment/run_deployment.py
```
Open your browser at **`http://localhost:5000`** and start detecting phishing URLs!

---

## 🔁 Training From Scratch (Google Colab)

For best performance, training is designed to run on Google Colab with GPU acceleration.

1. Upload `colab_train.py` to your Google Drive
2. Open [Google Colab](https://colab.research.google.com/) and enable **GPU runtime**
3. Run:
```python
!python colab_train.py
```
This will automatically:
- Download and balance the dataset
- Train both text and visual branches
- Fuse them and save the final `fusion_model.keras`

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Total Samples** | 1,697 URLs |
| **Legitimate URLs** | 1,147 (68%) |
| **Phishing URLs** | 550 (32%) |
| **Sources** | PhishTank, URLhaus, Tranco Top-1M |
| **Format** | CSV with URL and label columns |

> 📦 **[Download Dataset on Kaggle](https://www.kaggle.com/datasets/svspraveen/multimodal-phishing-detection-dataset)**

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Text Analysis:** Bi-LSTM, Word2Vec (Gensim)
- **Visual Analysis:** VGG16 CNN (Transfer Learning), Selenium, Pillow
- **Data:** PhishTank API, URLhaus, Tranco Top-1M List
- **Backend:** Flask, Flask-CORS
- **Utilities:** scikit-learn, NumPy, Pandas, Matplotlib

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | ~92%+ |
| **Precision (Phishing)** | High |
| **Recall (Phishing)** | High |
| **Architecture** | Bi-LSTM + VGG16 Fusion |

> Performance may vary based on dataset version and training configuration.

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**SVS Praveen**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/svs-praveen-s)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/SVSPraveen)

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

</div>
