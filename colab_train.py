##############################################################################
# MULTIMODAL PHISHING DETECTION - COMPLETE COLAB TRAINING
# ========================================================
# This notebook fetches REAL data from PhishTank + Tranco,
# captures REAL screenshots, and trains the full Bi-LSTM + VGG16 model.
#
# INSTRUCTIONS:
# 1. Open Google Colab -> New Notebook
# 2. Runtime -> Change runtime type -> GPU (T4)
# 3. Copy this ENTIRE script into ONE cell and run
# 4. Download the 2 output files and put in your models/ folder
# 5. Run locally: python deployment/run_deployment.py
##############################################################################

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("selenium")
install("webdriver-manager")
install("Pillow")
install("gensim")

# Install Chrome in Colab
import os
if os.path.exists("/content"):  # We're in Colab
    os.system("apt-get update -qq")
    os.system("wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb")
    os.system("apt-get install -y -qq ./google-chrome-stable_current_amd64.deb > /dev/null 2>&1")
    os.system("rm -f google-chrome-stable_current_amd64.deb")

print("Dependencies installed!")

# ============================================================================
# CELL 2: Fetch data from PhishTank + Tranco
# ============================================================================
import requests
import json
import io
import csv
import random
import time
import numpy as np
import pandas as pd
import re
import pickle

print("=" * 60)
print("STEP 1: Fetching real data from PhishTank + Tranco")
print("=" * 60)

# --- Fetch Phishing URLs from PhishTank ---
print("\nFetching phishing URLs from PhishTank...")
phishing_urls = []

try:
    # PhishTank verified online phishing database
    resp = requests.get(
        "https://data.phishtank.com/data/online-valid.json",
        headers={"User-Agent": "phishtank/research"},
        timeout=30
    )
    if resp.status_code == 200:
        data = resp.json()
        phishing_urls = [entry["url"] for entry in data if "url" in entry]
        print(f"  PhishTank: got {len(phishing_urls)} phishing URLs")
    else:
        print(f"  PhishTank returned status {resp.status_code}")
except Exception as e:
    print(f"  PhishTank error: {e}")

# Fallback: also try URLhaus
if len(phishing_urls) < 200:
    print("  Trying URLhaus as backup...")
    try:
        resp = requests.get(
            "https://urlhaus.abuse.ch/downloads/csv_recent/",
            timeout=30
        )
        if resp.status_code == 200:
            lines = resp.text.split("\n")
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split('","')
                if len(parts) >= 2:
                    url = parts[2].strip('"') if len(parts) > 2 else parts[1].strip('"')
                    if url.startswith("http"):
                        phishing_urls.append(url)
            print(f"  URLhaus: total phishing URLs now {len(phishing_urls)}")
    except Exception as e:
        print(f"  URLhaus error: {e}")

# Fallback: upload existing CSV if APIs fail
if len(phishing_urls) < 100:
    print("\n  APIs returned few results. Uploading your combined_urls.csv...")
    from google.colab import files
    uploaded = files.upload()
    fname = list(uploaded.keys())[0]
    df_existing = pd.read_csv(fname)
    phishing_urls = df_existing[df_existing['label'] == 1]['url'].tolist()
    print(f"  Loaded {len(phishing_urls)} phishing URLs from CSV")

# Limit to 850 phishing URLs (manageable for training)
random.seed(42)
if len(phishing_urls) > 850:
    phishing_urls = random.sample(phishing_urls, 850)
print(f"Using {len(phishing_urls)} phishing URLs")

# --- Fetch Legitimate URLs from Tranco ---
print("\nFetching legitimate URLs from Tranco top-1M list...")
legit_urls = []

try:
    resp = requests.get("https://tranco-list.eu/download/X4JNQ/1000000", timeout=30)
    if resp.status_code != 200:
        resp = requests.get("https://tranco-list.eu/top-1m.csv.zip", timeout=30)

    if resp.status_code == 200:
        content = resp.text
        for line in content.strip().split("\n")[:5000]:
            parts = line.strip().split(",")
            if len(parts) == 2:
                domain = parts[1].strip()
                if domain and "." in domain:
                    legit_urls.append(f"https://www.{domain}")
        print(f"  Tranco: got {len(legit_urls)} legitimate domains")
except Exception as e:
    print(f"  Tranco error: {e}")

# Fallback: hardcoded top domains
if len(legit_urls) < 200:
    print("  Using hardcoded legitimate domains as fallback...")
    top_domains = [
        "google.com","youtube.com","facebook.com","amazon.com","wikipedia.org",
        "twitter.com","instagram.com","linkedin.com","reddit.com","netflix.com",
        "microsoft.com","apple.com","github.com","stackoverflow.com","medium.com",
        "nytimes.com","bbc.com","cnn.com","walmart.com","ebay.com","paypal.com",
        "spotify.com","zoom.us","adobe.com","dropbox.com","slack.com","oracle.com",
        "ibm.com","intel.com","samsung.com","nike.com","adidas.com","uber.com",
        "airbnb.com","booking.com","expedia.com","zillow.com","indeed.com",
        "coursera.org","udemy.com","khanacademy.org","pinterest.com","tumblr.com",
        "twitch.tv","discord.com","whatsapp.com","telegram.org","mozilla.org",
        "wordpress.com","shopify.com","etsy.com","target.com","bestbuy.com",
        "chase.com","bankofamerica.com","wellsfargo.com","visa.com","mastercard.com",
        "cloudflare.com","docker.com","gitlab.com","notion.so","figma.com",
        "canva.com","trello.com","hubspot.com","mailchimp.com","zendesk.com",
        "python.org","nodejs.org","tensorflow.org","pytorch.org","numpy.org",
        "reuters.com","bloomberg.com","forbes.com","wsj.com","economist.com",
        "harvard.edu","mit.edu","stanford.edu","yale.edu","oxford.ac.uk",
        "who.int","un.org","nasa.gov","nih.gov","cdc.gov","usa.gov",
        "tesla.com","ford.com","toyota.com","honda.com","bmw.com",
        "delta.com","united.com","marriott.com","hilton.com","ikea.com",
        "att.com","verizon.com","t-mobile.com","allstate.com","geico.com",
        "tiktok.com","snapchat.com","soundcloud.com","bandcamp.com","chess.com",
        "imdb.com","rottentomatoes.com","goodreads.com","hulu.com","crunchyroll.com",
        "epicgames.com","xbox.com","playstation.com","nintendo.com","roblox.com",
        "fedex.com","ups.com","usps.com","dhl.com","costco.com",
        "sephora.com","nordstrom.com","macys.com","rei.com","patagonia.com",
        "norton.com","mcafee.com","avast.com","kaspersky.com","bitdefender.com",
        "fiverr.com","upwork.com","behance.net","dribbble.com","kaggle.com",
        "hackerrank.com","leetcode.com","geeksforgeeks.org","w3schools.com",
        "webmd.com","mayoclinic.org","healthline.com","nature.com","arxiv.org",
        "grammarly.com","evernote.com","todoist.com","1password.com","nordvpn.com",
        "expressVPN.com","surveymonkey.com","typeform.com","calendly.com",
        "kickstarter.com","gofundme.com","patreon.com","stripe.com","twilio.com",
        "heroku.com","netlify.com","vercel.com","digitalocean.com",
    ]
    legit_urls = [f"https://www.{d}" for d in top_domains]

# Match phishing count for balance
if len(legit_urls) > len(phishing_urls):
    legit_urls = random.sample(legit_urls, len(phishing_urls))
elif len(legit_urls) < len(phishing_urls):
    phishing_urls = random.sample(phishing_urls, len(legit_urls))

print(f"\nBalanced dataset: {len(phishing_urls)} phishing + {len(legit_urls)} legitimate")

all_urls = phishing_urls + legit_urls
all_labels = [1] * len(phishing_urls) + [0] * len(legit_urls)

# Shuffle
combined = list(zip(all_urls, all_labels))
random.shuffle(combined)
all_urls, all_labels = zip(*combined)
all_urls = list(all_urls)
all_labels = np.array(all_labels)

print(f"Total dataset: {len(all_urls)} URLs")

# ============================================================================
# CELL 3: Capture screenshots with Selenium
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Capturing screenshots (VGG16 visual branch)")
print("=" * 60)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from PIL import Image
import io as iomod

IMAGE_SIZE = (224, 224)
SCREENSHOT_DIR = "/content/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def get_driver():
    from webdriver_manager.chrome import ChromeDriverManager
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,720")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")  # faster capture

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(8)
    return driver

def capture_screenshot(driver, url):
    """Capture and return a (224, 224, 3) numpy array."""
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        driver.get(url)
        time.sleep(1)
        png_data = driver.get_screenshot_as_png()
        img = Image.open(iomod.BytesIO(png_data)).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        # VGG16 preprocessing: ImageNet mean subtraction, RGB->BGR
        arr[:, :, 0] -= 123.68
        arr[:, :, 1] -= 116.779
        arr[:, :, 2] -= 103.939
        arr = arr[:, :, ::-1]  # RGB to BGR
        return arr
    except Exception:
        return None

# Capture screenshots
print(f"Capturing screenshots for {len(all_urls)} URLs...")
print("(Timeout URLs get a blank image - this is normal for dead phishing sites)")

visual_data = []
success_count = 0
driver = get_driver()

for i, url in enumerate(all_urls):
    if (i + 1) % 50 == 0 or i == 0:
        print(f"  [{i+1}/{len(all_urls)}] Capturing... ({success_count} successful so far)")

    # Restart driver every 100 URLs to prevent memory leaks
    if i > 0 and i % 100 == 0:
        try:
            driver.quit()
        except:
            pass
        driver = get_driver()

    img = capture_screenshot(driver, url)
    if img is not None:
        visual_data.append(img)
        success_count += 1
    else:
        # Blank image fallback for dead/timeout URLs
        blank = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
        visual_data.append(blank)

try:
    driver.quit()
except:
    pass

X_visual = np.array(visual_data, dtype=np.float32)
print(f"\nScreenshots captured: {success_count}/{len(all_urls)} successful")
print(f"Visual input shape: {X_visual.shape}")

# ============================================================================
# CELL 4: Text tokenization (Bi-LSTM branch)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: URL tokenization (Bi-LSTM text branch)")
print("=" * 60)

MAX_SEQUENCE_LENGTH = 200

# Build vocabulary from all URLs
word_freq = {}
for url in all_urls:
    tokens = [t for t in re.split(r'[/\.\-_?=&:+@]', url.lower()) if t]
    for t in tokens:
        word_freq[t] = word_freq.get(t, 0) + 1

vocab = [w for w, freq in sorted(word_freq.items(), key=lambda x: -x[1])[:10000]]
word_to_index = {word: i + 1 for i, word in enumerate(vocab)}

tokenizer_data = {
    'vocab': vocab,
    'word_to_index': word_to_index,
    'max_seq_length': MAX_SEQUENCE_LENGTH
}

with open('/content/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer_data, f)

# Tokenize
sequences = []
for url in all_urls:
    tokens = [t for t in re.split(r'[/\.\-_?=&:+@]', url.lower()) if t]
    seq = [word_to_index.get(tok, 0) for tok in tokens]
    if len(seq) > MAX_SEQUENCE_LENGTH:
        seq = seq[:MAX_SEQUENCE_LENGTH]
    else:
        seq = seq + [0] * (MAX_SEQUENCE_LENGTH - len(seq))
    sequences.append(seq)

X_text = np.array(sequences, dtype=np.int32)

print(f"Vocabulary size: {len(vocab)}")
print(f"Text input shape: {X_text.shape}")

# ============================================================================
# CELL 5: Build and train the multimodal model
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Building Bi-LSTM + VGG16 Multimodal Model")
print("=" * 60)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense, Dropout,
    GlobalMaxPooling1D, Concatenate, BatchNormalization, Flatten
)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

vocab_size = len(vocab) + 1

# --- Text Branch (Bi-LSTM) ---
text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')
embedding = Embedding(
    vocab_size, 100, input_length=MAX_SEQUENCE_LENGTH,
    trainable=True, name='word_embedding'
)(text_input)
bilstm = Bidirectional(
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    name='bilstm'
)(embedding)
text_pool = GlobalMaxPooling1D(name='global_max_pool')(bilstm)
text_feat = Dense(128, activation='relu', name='text_features')(text_pool)

# --- Visual Branch (VGG16) ---
visual_input = Input(shape=(224, 224, 3), name='visual_input')
base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze first 15 layers (transfer learning)
for layer in base_vgg.layers[:15]:
    layer.trainable = False
vgg_out = base_vgg(visual_input)
flat = Flatten(name='flatten')(vgg_out)
vis_feat = Dense(256, activation='relu', name='visual_features')(flat)

# --- Fusion ---
concat = Concatenate(name='concatenate')([text_feat, vis_feat])
bn = BatchNormalization(name='batch_norm')(concat)
fc = Dense(64, activation='relu', name='fusion_fc')(bn)
drop = Dropout(0.5, name='dropout')(fc)
output = Dense(1, activation='sigmoid', name='output')(drop)

model = Model(
    inputs=[text_input, visual_input],
    outputs=output,
    name='multimodal_fusion'
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Train ---
print("\nTraining multimodal model with GPU...")
history = model.fit(
    [X_text, X_visual], all_labels,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        )
    ],
    verbose=1
)

# ============================================================================
# CELL 6: Evaluate the model
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Evaluation Results")
print("=" * 60)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Use last 20% as test set
split = int(0.8 * len(all_labels))
X_text_test, X_vis_test, y_test = X_text[split:], X_visual[split:], all_labels[split:]

preds = model.predict([X_text_test, X_vis_test])
binary_preds = (preds > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_test, binary_preds, target_names=['Legitimate', 'Phishing']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, binary_preds))

try:
    auc = roc_auc_score(y_test, preds)
    print(f"\nROC-AUC Score: {auc:.4f}")
except:
    pass

# Quick demo test
print("\n" + "=" * 60)
print("Quick Test on Sample URLs:")
print("=" * 60)

test_urls = [
    ("https://www.google.com", "LEGITIMATE"),
    ("https://www.github.com", "LEGITIMATE"),
    ("https://www.amazon.com", "LEGITIMATE"),
    ("http://verify-your-paypal.weebly.com", "PHISHING"),
    ("https://secure-login-account.blogspot.com", "PHISHING"),
    ("http://allegro.7238239034g34-212.shop", "PHISHING"),
]

for url, expected in test_urls:
    tokens = [t for t in re.split(r'[/\.\-_?=&:+@]', url.lower()) if t]
    seq = [word_to_index.get(tok, 0) for tok in tokens]
    if len(seq) > MAX_SEQUENCE_LENGTH:
        seq = seq[:MAX_SEQUENCE_LENGTH]
    else:
        seq = seq + [0] * (MAX_SEQUENCE_LENGTH - len(seq))
    t_arr = np.array([seq], dtype=np.int32)
    v_arr = np.zeros((1, 224, 224, 3), dtype=np.float32)

    score = float(model.predict([t_arr, v_arr], verbose=0)[0][0])
    pred = "PHISHING" if score > 0.5 else "LEGITIMATE"
    conf = score if score > 0.5 else (1 - score)
    mark = "OK" if pred == expected else "XX"
    print(f"  [{mark}] {url}")
    print(f"       -> {pred} ({conf*100:.1f}%) | Expected: {expected}")

# ============================================================================
# CELL 7: Save and download
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: Saving and downloading model files")
print("=" * 60)

model.save('/content/fusion_model.keras')
print("Model saved!")

# Also save training history
with open('/content/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Download
from google.colab import files
print("\nDownloading fusion_model.keras...")
files.download('/content/fusion_model.keras')
print("Downloading tokenizer.pkl...")
files.download('/content/tokenizer.pkl')

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
print("1. Put fusion_model.keras in your models/ folder")
print("2. Put tokenizer.pkl in your models/ folder")
print("3. Run: python deployment/run_deployment.py")
print("4. Open http://localhost:5000")
print("=" * 60)
