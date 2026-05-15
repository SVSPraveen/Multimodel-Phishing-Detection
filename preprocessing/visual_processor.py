"""
visual_processor.py
-------------------
Visual preprocessing for multimodal phishing detection.

FIXED BUGS:
- --disable-images Chrome flag was set during dataset preprocessing,
  which meant screenshots were mostly blank (no images loaded). This flag
  has been removed so pages render normally.
- Added explicit page-load timeout via driver.set_page_load_timeout() to
  prevent infinite hangs on slow/broken pages.
- The VGG16 feature output is now a raw numpy array of images (shape
  N x 224 x 224 x 3) saved directly as visual_features.npy, matching
  what FusionModel.train() expects. The old approach saved flattened
  block5_pool features which were incompatible with the VGG16 sub-model
  inside FusionModel.
- ImageNet normalisation (mean subtraction, RGB→BGR) is now applied
  consistently inside preprocess_image(), matching inference behaviour.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import time
import io
import pickle
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    COMBINED_DATA_PATH, VISUAL_FEATURES_PATH, VGG16_MODEL_PATH,
    IMAGE_SIZE, IMAGE_CHANNELS, SCREENSHOT_TIMEOUT, SCREENSHOT_WIDTH,
    SCREENSHOT_HEIGHT, HEADLESS, OVERLAY_SELECTORS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualProcessor:
    """Visual processor for phishing detection."""

    def __init__(self):
        self.driver = None
        self.image_size = IMAGE_SIZE
        self.channels = IMAGE_CHANNELS

    # ------------------------------------------------------------------
    # WebDriver
    # ------------------------------------------------------------------

    def setup_driver(self):
        """Setup headless Chrome WebDriver."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager

            options = Options()
            if HEADLESS:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument(
                "--window-size={},{}".format(SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT)
            )
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-plugins")
            # FIX: removed --disable-images so pages render with actual content

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.set_page_load_timeout(SCREENSHOT_TIMEOUT)  # FIX: explicit timeout

            logger.info("Chrome WebDriver setup complete")

        except Exception as e:
            logger.error(f"Error setting up WebDriver: {e}")
            raise

    def remove_overlays(self):
        """Remove cookie banners, popups, and other overlays."""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            WebDriverWait(self.driver, 3).until(
                EC.presence_of_element_located(('tag name', 'body'))
            )
            for selector in OVERLAY_SELECTORS:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        self.driver.execute_script("arguments[0].remove();", element)
                except Exception:
                    continue
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Error removing overlays: {e}")

    # ------------------------------------------------------------------
    # Screenshot capture
    # ------------------------------------------------------------------

    def capture_screenshot(self, url):
        """Capture a screenshot of a URL and return a PIL Image."""
        from PIL import Image

        try:
            # FIX: ensure protocol is present before passing to driver
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url

            self.driver.get(url)
            self.remove_overlays()

            screenshot_bytes = self.driver.get_screenshot_as_png()
            image = Image.open(io.BytesIO(screenshot_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image

        except Exception as e:
            logger.warning(f"Error capturing screenshot for {url}: {e}")
            return None

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def preprocess_image(self, image):
        """
        Resize and normalise a PIL Image for VGG16.

        Applies ImageNet mean subtraction and RGB→BGR conversion
        to match the VGG16 preprocessing used at inference time.
        """
        from PIL import Image

        try:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            img_array = np.array(image, dtype=np.float32)

            # ImageNet mean subtraction
            img_array[:, :, 0] -= 123.68   # R
            img_array[:, :, 1] -= 116.779  # G
            img_array[:, :, 2] -= 103.939  # B

            # RGB → BGR (VGG16 convention)
            img_array = img_array[:, :, ::-1]

            return img_array  # shape: (224, 224, 3)

        except Exception as e:
            logger.warning(f"Error preprocessing image: {e}")
            return None

    def create_dummy_image(self):
        """Return a zero image when screenshot fails."""
        return np.zeros(
            (self.image_size[0], self.image_size[1], self.channels),
            dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Dataset processing
    # ------------------------------------------------------------------

    def process_dataset(self, df):
        """
        Capture screenshots for all URLs and return a normalised image array.

        Returns
        -------
        np.ndarray of shape (N, 224, 224, 3) — ready for FusionModel.train()
        """
        logger.info("Processing visual features (screenshot capture)...")
        self.setup_driver()

        visual_features = []
        urls = df['url'].tolist()

        for url in tqdm(urls, desc="Capturing screenshots"):
            image = self.capture_screenshot(url)

            if image is not None:
                processed = self.preprocess_image(image)
                visual_features.append(
                    processed if processed is not None else self.create_dummy_image()
                )
            else:
                visual_features.append(self.create_dummy_image())

            time.sleep(0.5)   # be polite to servers

        if self.driver:
            self.driver.quit()

        visual_array = np.array(visual_features, dtype=np.float32)
        logger.info(f"Visual features shape: {visual_array.shape}")

        # FIX: save the raw preprocessed images (N x 224 x 224 x 3) so
        # FusionModel.train() can feed them directly into its VGG16 sub-layer.
        # The old code saved flattened block5_pool features, which were
        # incompatible with the end-to-end architecture.
        return visual_array

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------

    def save_config(self):
        """Save processor configuration."""
        os.makedirs(os.path.dirname(VGG16_MODEL_PATH), exist_ok=True)
        config = {
            'image_size': self.image_size,
            'channels': self.channels,
            'frozen_layers': 15
        }
        config_path = VGG16_MODEL_PATH.replace('.keras', '_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        logger.info("Visual processor config saved")


def main():
    logger.info("=" * 60)
    logger.info("VISUAL PROCESSING FOR MULTIMODAL PHISHING DETECTION")
    logger.info("=" * 60)

    if not os.path.exists(COMBINED_DATA_PATH):
        logger.error(f"Dataset not found: {COMBINED_DATA_PATH}")
        logger.error("Please run scripts/collect_data.py first")
        return

    df = pd.read_csv(COMBINED_DATA_PATH)
    logger.info(f"Loaded dataset: {len(df)} URLs")

    processor = VisualProcessor()
    visual_features = processor.process_dataset(df)

    os.makedirs(os.path.dirname(VISUAL_FEATURES_PATH), exist_ok=True)
    np.save(VISUAL_FEATURES_PATH, visual_features)
    logger.info(f"Visual features saved to {VISUAL_FEATURES_PATH}")

    processor.save_config()

    logger.info("=" * 60)
    logger.info("VISUAL PROCESSING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Visual features shape: {visual_features.shape}")
    logger.info("Next step: Run scripts/train_model.py")


if __name__ == "__main__":
    main()
