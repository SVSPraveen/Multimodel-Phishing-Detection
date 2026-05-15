"""
Configuration for Multimodal Phishing Detection
Based on research paper parameters
"""

import os

# Data Configuration
PHISHTANK_API_URL = "https://data.phishtank.com/data/online-valid.json"
URLHAUS_API_URL = "https://urlhaus.abuse.ch/downloads/csv_recent/"
TRANCO_URL = "https://tranco-list.eu/1000000.csv"

# Dataset Configuration
DATASET_SIZE = 1697  # Total samples from research
LEGITIMATE_SAMPLES = 1147  # 68% of dataset
PHISHING_SAMPLES = 550  # 32% of dataset (natural imbalance)

# Text Processing Configuration
MAX_SEQUENCE_LENGTH = 200  # Max URL length
WORD_EMBEDDING_DIM = 100  # Word2Vec dimension from research
BILSTM_UNITS = 128  # Bi-LSTM hidden units
VOCAB_SIZE = 10000  # Maximum vocabulary size

# Visual Processing Configuration
IMAGE_SIZE = (224, 224)  # VGG16 input size
IMAGE_CHANNELS = 3
VGG16_FROZEN_LAYERS = 15  # Frozen layers for transfer learning
VISUAL_FEATURE_DIM = 256  # Dense layer output dimension

# Fusion Configuration
FUSION_UNITS = 64  # FC layer units after concatenation
DROPOUT_RATE = 0.5  # Dropout rate from research

# Training Configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Paths Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Model Paths
WORD2VEC_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec.model")
BILSTM_MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_model.keras")
VGG16_MODEL_PATH = os.path.join(MODELS_DIR, "vgg16_model.keras")
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "fusion_model.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")

# Dataset Paths
PHISHING_DATA_PATH = os.path.join(DATASETS_DIR, "phishing_urls.csv")
LEGITIMATE_DATA_PATH = os.path.join(DATASETS_DIR, "legitimate_urls.csv")
COMBINED_DATA_PATH = os.path.join(DATA_DIR, "combined_urls.csv")
TEXT_FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "text_features.npy")
VISUAL_FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "visual_features.npy")

# Screenshot Configuration
SCREENSHOT_TIMEOUT = 10  # Seconds
SCREENSHOT_WIDTH = 1280
SCREENSHOT_HEIGHT = 720
HEADLESS = True  # Headless browser mode

# Preprocessing Configuration
HTML_TAGS_TO_REMOVE = ['script', 'style', 'meta', 'link']
OVERLAY_SELECTORS = [
    '.cookie-banner', '.cookie-notice', '.gdpr-banner',
    '.popup', '.modal', '.overlay', '.consent-banner'
]

# Random State for Reproducibility
RANDOM_STATE = 42
