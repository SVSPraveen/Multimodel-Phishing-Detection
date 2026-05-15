#!/usr/bin/env python3
"""
Multimodal Phishing Detection - Inference Script

Loads the trained end-to-end fusion model and predicts whether a URL
is phishing or legitimate.

Usage:
    python predict_phishing.py --url "https://example.com"
    python predict_phishing.py --file urls.txt --output results.csv

FIXED BUGS:
- URL preprocessing now uses a URL-aware tokeniser (splits on /, ., -, etc.)
  instead of NLTK sentence tokenisation which stripped critical URL characters.
- Added protocol (http://) prefix guard before passing URL to Selenium.
- Inference now uses the single end-to-end fusion model directly, eliminating
  the need to run each branch independently.
- Removed incorrect text_features_pred[0][0] fallback that assumed a 1-dim
  output from the text branch (it was 128-dim).
- Fixed the screenshot window size: VGG16 needs 224x224 after resize, so
  the browser window is now set to a reasonable 1280x720 and resized in PIL.
"""

import argparse
import os
import re
import pickle
import numpy as np

# Force TensorFlow to use legacy Keras 2.x to load Colab models properly
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import time

# -----------------------------------------------------------------------
# Optional Selenium import (visual mode only)
# -----------------------------------------------------------------------
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    _SELENIUM_AVAILABLE = True
except ImportError:
    _SELENIUM_AVAILABLE = False

# -----------------------------------------------------------------------
# Config-independent defaults (used when running standalone)
# -----------------------------------------------------------------------
_MAX_SEQ_LENGTH_DEFAULT = 200
_IMAGE_SIZE = (224, 224)


class PhishingDetector:
    """Load the trained end-to-end fusion model and run predictions."""

    def __init__(self, models_path="models"):
        self.models_path = models_path
        self.fusion_model = None
        self.tokenizer_data = None
        self.word_to_index = None
        self.max_seq_length = None
        self.load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self):
        """Load the fusion model and tokenizer."""
        print("Loading trained models...")

        # Load tokenizer
        tokenizer_path = os.path.join(self.models_path, "tokenizer.pkl")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Run preprocessing/text_processor.py first."
            )
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer_data = pickle.load(f)

        self.word_to_index = self.tokenizer_data['word_to_index']
        self.max_seq_length = self.tokenizer_data.get(
            'max_seq_length', _MAX_SEQ_LENGTH_DEFAULT
        )

        # Load end-to-end fusion model
        fusion_path = os.path.join(self.models_path, "fusion_model.keras")
        if not os.path.exists(fusion_path):
            raise FileNotFoundError(
                f"Fusion model not found at {fusion_path}. "
                "Run scripts/train_model.py first."
            )
        self.fusion_model = load_model(fusion_path)
        print("Models loaded successfully!")

    # ------------------------------------------------------------------
    # Text preprocessing (URL-aware)
    # ------------------------------------------------------------------

    def preprocess_text(self, url):
        """
        URL-aware tokenisation.

        Splits on common URL delimiters (/ . - _ ? = & : + @) rather than
        treating the URL as a natural-language sentence.  This preserves
        structural phishing signals such as hyphens, subdomains, and
        suspicious path patterns.
        """
        url_lower = url.lower()
        tokens = re.split(r'[/\.\-_?=&:+@]', url_lower)
        tokens = [t for t in tokens if t]  # drop empty strings

        sequence = [self.word_to_index.get(tok, 0) for tok in tokens]

        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        else:
            sequence = sequence + [0] * (self.max_seq_length - len(sequence))

        return np.array(sequence, dtype=np.int32)

    # ------------------------------------------------------------------
    # Screenshot capture
    # ------------------------------------------------------------------

    def capture_screenshot(self, url, output_path="temp_screenshot.png"):
        """Capture a screenshot of the given URL using headless Chrome."""
        if not _SELENIUM_AVAILABLE:
            print("Selenium not installed; skipping visual analysis.")
            return None

        # FIX: ensure URL has a protocol before passing to driver
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1280,720")
        chrome_options.add_argument("--disable-gpu")

        try:
            # Try to use webdriver_manager if installed, otherwise default Chrome
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                from selenium.webdriver.chrome.service import Service
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
            except Exception:
                driver = webdriver.Chrome(options=chrome_options)
                
            driver.set_page_load_timeout(15)
            driver.get(url)
            time.sleep(3)  # let the page render

            # Remove noisy UI elements
            driver.execute_script("""
                var elements = document.querySelectorAll('script, style, noscript');
                elements.forEach(function(el) { el.remove(); });
            """)

            driver.save_screenshot(output_path)
            driver.quit()
            return output_path

        except Exception as e:
            try:
                driver.quit()
            except:
                pass
            print(f"Screenshot failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def preprocess_image(self, image_path):
        """
        Load, resize and normalise an image for VGG16.

        Uses ImageNet mean-subtraction (BGR order) to match the
        preprocessing used during training.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(_IMAGE_SIZE, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32)

            # ImageNet mean subtraction (RGB values)
            img_array[:, :, 0] -= 123.68   # R
            img_array[:, :, 1] -= 116.779  # G
            img_array[:, :, 2] -= 103.939  # B

            # Convert RGB → BGR (VGG16 convention)
            img_array = img_array[:, :, ::-1]

            return img_array.reshape(1, _IMAGE_SIZE[0], _IMAGE_SIZE[1], 3)
        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            return None

    def _dummy_image(self):
        """Return a zero (blank) image when screenshot is unavailable."""
        return np.zeros((1, _IMAGE_SIZE[0], _IMAGE_SIZE[1], 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # Prediction (single URL)
    # ------------------------------------------------------------------

    def predict_url(self, url, use_visual=True):
        """
        Run the end-to-end model on a single URL.

        The fusion model takes (text_sequence, image) → sigmoid probability.
        If visual capture fails, a blank image is used as fallback instead
        of trying to use an intermediate branch output.
        """
        print(f"Analysing URL: {url}")

        # --- Text ---
        text_seq = self.preprocess_text(url).reshape(1, -1)

        # --- Visual ---
        if use_visual:
            screenshot_path = self.capture_screenshot(url)
            if screenshot_path:
                visual = self.preprocess_image(screenshot_path)
                if visual is None:
                    visual = self._dummy_image()
                # Clean up temp file
                if os.path.exists(screenshot_path):
                    os.remove(screenshot_path)
            else:
                print("Screenshot unavailable — using blank image for visual branch.")
                visual = self._dummy_image()
        else:
            visual = self._dummy_image()

        # --- Inference via end-to-end model ---
        raw_score = float(self.fusion_model.predict(
            [text_seq, visual], verbose=0
        )[0][0])

        is_phishing = raw_score > 0.5
        confidence = raw_score if is_phishing else (1.0 - raw_score)

        return {
            'url': url,
            'is_phishing': bool(is_phishing),
            'confidence': confidence,
            'raw_score': raw_score,
            'prediction': 'PHISHING' if is_phishing else 'LEGITIMATE'
        }

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def batch_predict(self, urls, use_visual=False):
        """Run predictions on a list of URLs."""
        results = []
        for url in urls:
            try:
                result = self.predict_url(url, use_visual)
                results.append(result)
            except Exception as e:
                print(f"Error processing {url}: {e}")
                results.append({
                    'url': url,
                    'is_phishing': None,
                    'confidence': 0.0,
                    'raw_score': 0.0,
                    'prediction': 'ERROR',
                    'error': str(e)
                })
        return results


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Multimodal Phishing Detection')
    parser.add_argument('--url',          type=str, help='URL to check')
    parser.add_argument('--file',         type=str, help='File with URLs (one per line)')
    parser.add_argument('--models-path',  type=str, default='models',
                        help='Path to trained models directory')
    parser.add_argument('--no-visual',    action='store_true',
                        help='Disable visual analysis (faster)')
    parser.add_argument('--output',       type=str, help='Output CSV file for results')

    args = parser.parse_args()

    if not args.url and not args.file:
        print("Error: provide --url or --file")
        return

    detector = PhishingDetector(args.models_path)
    use_visual = not args.no_visual

    if args.url:
        result = detector.predict_url(args.url, use_visual)
        print("\n" + "=" * 60)
        print("PHISHING DETECTION RESULTS")
        print("=" * 60)
        print(f"URL:        {result['url']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Raw Score:  {result['raw_score']:.4f}")
        print("=" * 60)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(f"URL,Prediction,Confidence\n")
                f.write(f"{result['url']},{result['prediction']},{result['confidence']:.4f}\n")
            print(f"Results saved to {args.output}")

    elif args.file:
        with open(args.file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]

        print(f"Processing {len(urls)} URLs...")
        results = detector.batch_predict(urls, use_visual)

        print("\n" + "=" * 80)
        print("BATCH PHISHING DETECTION RESULTS")
        print("=" * 80)
        for r in results:
            if r['prediction'] != 'ERROR':
                print(f"{r['url']:<50} -> {r['prediction']:<12} ({r['confidence']:.4f})")
            else:
                print(f"{r['url']:<50} -> ERROR: {r.get('error', 'Unknown')}")
        print("=" * 80)

        if args.output:
            with open(args.output, 'w') as f:
                f.write("URL,Prediction,Confidence,Raw Score\n")
                for r in results:
                    f.write(
                        f"{r['url']},{r['prediction']},"
                        f"{r['confidence']:.4f},{r['raw_score']:.4f}\n"
                    )
            print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
