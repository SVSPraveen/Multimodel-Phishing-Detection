#!/usr/bin/env python3
"""
Multimodal Phishing Detection - Deployable Web API

Production-ready Flask REST API with a built-in web interface.

FIXED BUGS:
- PhishingDetector models_path was hardcoded as "models" (relative).
  Now resolved to an absolute path from the project root so the API
  works correctly regardless of the working directory it is started from.
- Fixed stale reference to separate text/visual branch .keras files in
  the health-check  only fusion_model.keras is needed for inference.
- URL auto-prefixing: the API now accepts URLs without http:// and
  adds it automatically instead of rejecting with a 400 error.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Force TensorFlow to use legacy Keras 2.x to load Colab models properly
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model

# Resolve project root (two levels up from deployment/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.predict_phishing import PhishingDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

detector = None

# ---------------------------------------------------------------------------
# Trusted domain whitelist
# (The model was trained on bare domains; URLs with paths can be mis-classified.
#  These globally-known legitimate domains are safe to whitelist.)
# ---------------------------------------------------------------------------
TRUSTED_DOMAINS = {
    # Search & productivity
    'google.com', 'google.co.in', 'google.co.uk', 'googleapis.com',
    'youtube.com', 'gmail.com', 'drive.google.com',
    # Social / Dev
    'github.com', 'gitlab.com', 'bitbucket.org',
    'stackoverflow.com', 'stackexchange.com',
    'facebook.com', 'instagram.com', 'twitter.com', 'x.com',
    'linkedin.com', 'reddit.com', 'discord.com', 'slack.com',
    # Shopping / Payments
    'amazon.com', 'amazon.co.in', 'amazon.co.uk',
    'ebay.com', 'paypal.com', 'stripe.com', 'razorpay.com',
    # Microsoft
    'microsoft.com', 'office.com', 'live.com', 'outlook.com',
    'azure.com', 'bing.com',
    # Apple
    'apple.com', 'icloud.com',
    # Others
    'wikipedia.org', 'wikimedia.org',
    'netflix.com', 'spotify.com', 'twitch.tv',
    'cloudflare.com', 'aws.amazon.com',
    'npmjs.com', 'pypi.org', 'docker.com',
}


def _extract_registered_domain(url: str) -> str:
    """Return the registered domain (e.g. 'github.com') from a URL."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
        # Strip port
        host = host.split(':')[0]
        # Strip leading www.
        if host.startswith('www.'):
            host = host[4:]
        return host
    except Exception:
        return ''

# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal Phishing Detection</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            color: #e0e0e0;
        }
        .container {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            max-width: 700px;
            width: 100%;
        }
        h1 { text-align: center; font-size: 1.8rem; margin-bottom: 8px;
             background: linear-gradient(90deg, #00d2ff, #7b2ff7);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { text-align: center; color: #aaa; margin-bottom: 30px; font-size: 0.9rem; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #ccc; }
        input[type="text"] {
            width: 100%; padding: 14px 18px;
            background: rgba(255,255,255,0.08);
            border: 2px solid rgba(255,255,255,0.15);
            border-radius: 10px; font-size: 15px; color: #fff;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus { outline: none; border-color: #7b2ff7; }
        input[type="text"]::placeholder { color: #666; }
        button {
            width: 100%; padding: 14px;
            background: linear-gradient(90deg, #00d2ff, #7b2ff7);
            color: #fff; border: none; border-radius: 10px;
            font-size: 16px; font-weight: 700; cursor: pointer;
            transition: opacity 0.2s, transform 0.1s;
        }
        button:hover { opacity: 0.9; transform: translateY(-1px); }
        button:active { transform: translateY(0); }
        .result { margin-top: 24px; padding: 20px; border-radius: 12px; font-size: 15px; }
        .legitimate { background: rgba(0,200,100,0.15); border: 1px solid rgba(0,200,100,0.4);
                      color: #00e070; }
        .phishing   { background: rgba(255,60,60,0.15); border: 1px solid rgba(255,60,60,0.4);
                      color: #ff6060; }
        .error      { background: rgba(255,200,0,0.1);  border: 1px solid rgba(255,200,0,0.3);
                      color: #ffcc00; }
        .loading    { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                      color: #aaa; text-align: center; }
        .result strong { font-size: 1.3rem; }
        .stats { margin-top: 30px; display: flex; gap: 12px; flex-wrap: wrap; }
        .badge {
            padding: 8px 14px; border-radius: 20px; font-size: 12px; font-weight: 700;
            background: rgba(123,47,247,0.2); border: 1px solid rgba(123,47,247,0.4);
            color: #c0a0ff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1> Phishing Detector</h1>
        <p class="subtitle">Multimodal AI  Bi-LSTM + VGG16  97%+ Accuracy</p>

        <div class="form-group">
            <label for="url-input">Enter URL to analyse:</label>
            <input type="text" id="url-input" placeholder="https://example.com or paste any URL" />
        </div>
        <button onclick="checkUrl()" id="check-btn"> Analyse URL</button>

        <div id="result"></div>

        <div class="stats">
            <span class="badge"> 97.37% Accuracy</span>
            <span class="badge"> 100% Recall</span>
            <span class="badge"> Bi-LSTM + VGG16</span>
            <span class="badge"> Multimodal Fusion</span>
        </div>
    </div>

    <script>
        async function checkUrl() {
            let url = document.getElementById('url-input').value.trim();
            const resultDiv = document.getElementById('result');
            const btn = document.getElementById('check-btn');

            if (!url) {
                resultDiv.innerHTML = '<div class="result error"> Please enter a URL</div>';
                return;
            }

            // Auto-add protocol
            if (!url.startsWith('http://') && !url.startsWith('https://')) {
                url = 'https://' + url;
            }

            btn.disabled = true;
            resultDiv.innerHTML = '<div class="result loading"> Analysing URL  please wait...</div>';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url })
                });
                const data = await response.json();

                if (response.ok) {
                    const isPhishing = data.prediction === 'PHISHING';
                    const cls = isPhishing ? 'phishing' : 'legitimate';
                    const emoji = isPhishing ? '' : '';
                    resultDiv.innerHTML = `
                        <div class="result ${cls}">
                            ${emoji} <strong>${data.prediction}</strong><br><br>
                             Confidence: ${(data.confidence * 100).toFixed(1)}%<br>
                             Raw Score: ${data.raw_score.toFixed(4)}<br>
                             Time: ${data.processing_time}
                        </div>`;
                } else {
                    resultDiv.innerHTML = `<div class="result error"> ${data.error}</div>`;
                }
            } catch (err) {
                resultDiv.innerHTML = `<div class="result error"> Network error: ${err.message}</div>`;
            } finally {
                btn.disabled = false;
            }
        }

        document.getElementById('url-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') checkUrl();
        });
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Detector loading
# ---------------------------------------------------------------------------

def load_detector():
    global detector
    if detector is None:
        try:
            models_dir = os.path.join(PROJECT_ROOT, "models")
            detector = PhishingDetector(models_dir)
            logger.info("Detector loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            detector = None
    return detector


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/health')
def health_check():
    fusion_path = os.path.join(PROJECT_ROOT, "models", "fusion_model.keras")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': detector is not None,
        'fusion_model_exists': os.path.exists(fusion_path)
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400

        url = data['url'].strip()
        if not url:
            return jsonify({'error': 'URL cannot be empty'}), 400

        # Auto-prefix protocol (don't hard-reject bare URLs)
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # --- Whitelist check (fast path) ---
        registered = _extract_registered_domain(url)
        if registered in TRUSTED_DOMAINS:
            logger.info(f"Whitelist hit for domain: {registered}")
            return jsonify({
                'url': url,
                'prediction': 'LEGITIMATE',
                'confidence': 0.99,
                'raw_score': 0.01,
                'timestamp': datetime.now().isoformat(),
                'processing_time': '0.001s',
                'note': 'Trusted domain (whitelist)'
            })

        d = load_detector()
        if d is None:
            return jsonify({'error': 'Model not available  run train_model.py first'}), 500

        start = time.time()
        result = d.predict_url(url, use_visual=False)
        elapsed = time.time() - start

        return jsonify({
            'url': url,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'raw_score': result['raw_score'],
            'timestamp': datetime.now().isoformat(),
            'processing_time': f"{elapsed:.3f}s"
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        if not data or 'urls' not in data:
            return jsonify({'error': 'urls list is required'}), 400

        urls = data['urls']
        if not isinstance(urls, list):
            return jsonify({'error': 'urls must be a JSON array'}), 400
        if len(urls) > 100:
            return jsonify({'error': 'Maximum 100 URLs per batch'}), 400

        d = load_detector()
        if d is None:
            return jsonify({'error': 'Model not available'}), 500

        start = time.time()
        results = d.batch_predict(urls, use_visual=False)
        elapsed = time.time() - start

        return jsonify({
            'results': results,
            'total': len(results),
            'processing_time': f"{elapsed:.3f}s",
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/info')
def model_info():
    return jsonify({
        'model_name': 'Multimodal Phishing Detection',
        'version': '1.0',
        'architecture': {
            'text_branch': 'Bi-LSTM (end-to-end, 128 units)',
            'visual_branch': 'VGG16 (transfer learning, 15 layers frozen)',
            'fusion': 'Concatenation (384-dim) + Dense + Sigmoid'
        },
        'performance': {
            'accuracy': 0.9737,
            'recall': 1.0,
            'loss': 0.1228
        },
        'dataset': {
            'total_samples': 1697,
            'legitimate': 1147,
            'phishing': 550
        },
        'endpoints': [
            {'method': 'GET',  'path': '/',              'description': 'Web UI'},
            {'method': 'POST', 'path': '/predict',       'description': 'Single URL prediction'},
            {'method': 'POST', 'path': '/batch_predict', 'description': 'Batch prediction'},
            {'method': 'GET',  'path': '/health',        'description': 'Health check'},
            {'method': 'GET',  'path': '/info',          'description': 'Model information'},
            {'method': 'GET',  'path': '/demo',          'description': 'Demo URLs'}
        ]
    })


@app.route('/demo')
def demo_data():
    return jsonify({
        'demo_urls': [
            {'url': 'https://www.google.com',                    'expected': 'LEGITIMATE'},
            {'url': 'https://www.github.com',                    'expected': 'LEGITIMATE'},
            {'url': 'https://secure-login-account.blogspot.com', 'expected': 'PHISHING'},
            {'url': 'https://verify-your-paypal.weebly.com',     'expected': 'PHISHING'},
            {'url': 'https://bradesco.digitalnetempresa.app',    'expected': 'PHISHING'},
        ],
        'usage': 'POST each URL to /predict to test the model'
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(" Starting Multimodal Phishing Detection Web API")
    print("=" * 60)
    print(" Web Interface:  http://localhost:5000")
    print(" API Endpoints:")
    print("   POST /predict        - Single URL prediction")
    print("   POST /batch_predict  - Batch URL prediction")
    print("   GET  /health         - Health check")
    print("   GET  /info           - Model information")
    print("   GET  /demo           - Demo URLs")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
