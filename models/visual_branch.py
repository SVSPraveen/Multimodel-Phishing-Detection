"""
visual_branch.py
---------------
VGG16 visual branch for multimodal phishing detection.

NOTE: This module is kept for reference and standalone testing only.
      In the end-to-end architecture the VGG16 layers are embedded
      directly inside FusionModel, so you do not need to call this
      class during training or inference.

FIXED BUGS:
- The standalone model now has a proper Dense(1, sigmoid) classification
  head so it can be compiled and tested with binary_crossentropy correctly.
- Fixed NameError in main(): `self.image_size` and `self.channels` were
  referenced outside the class body. Now uses local variables.
"""

import os
import sys
import numpy as np
import logging
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    VGG16_MODEL_PATH, IMAGE_SIZE, IMAGE_CHANNELS,
    VGG16_FROZEN_LAYERS, VISUAL_FEATURE_DIM, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualBranch:
    """
    Standalone VGG16 visual branch.

    Use this for independent testing.
    For production, the same architecture is embedded in FusionModel.
    """

    def __init__(self):
        self.model = None
        self.base_model = None
        self.image_size = IMAGE_SIZE
        self.channels = IMAGE_CHANNELS

    def build_model(self, include_classifier=True):
        """
        Build VGG16 model with transfer learning.

        Parameters
        ----------
        include_classifier : bool
            If True, appends Dense(1, sigmoid) for standalone training.
            Set False to return 256-dim feature vectors only.
        """
        logger.info("Building VGG16 visual branch...")

        self.base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(self.image_size[0], self.image_size[1], self.channels)
        )

        # Freeze first VGG16_FROZEN_LAYERS layers
        for layer in self.base_model.layers[:VGG16_FROZEN_LAYERS]:
            layer.trainable = False

        logger.info(f"Frozen first {VGG16_FROZEN_LAYERS} layers of VGG16")

        input_layer = Input(
            shape=(self.image_size[0], self.image_size[1], self.channels),
            name='visual_input'
        )
        base_output = self.base_model(input_layer)
        flattened = Flatten(name='flatten')(base_output)
        visual_features = Dense(
            VISUAL_FEATURE_DIM, activation='relu', name='visual_features'
        )(flattened)

        if include_classifier:
            # FIX: proper classification head so binary_crossentropy works
            dropout = Dropout(0.3, name='dropout')(visual_features)
            output = Dense(1, activation='sigmoid', name='output')(dropout)
            self.model = Model(
                inputs=input_layer, outputs=output, name='visual_branch'
            )
        else:
            self.model = Model(
                inputs=input_layer, outputs=visual_features,
                name='visual_branch_features'
            )

        logger.info("VGG16 visual branch built successfully")
        self.model.summary()
        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compile for standalone training."""
        if self.model is None:
            self.build_model(include_classifier=True)

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Visual branch model compiled")

    def predict(self, images):
        """Generate predictions for images."""
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model.predict(images)

    def save_model(self, path):
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(path)
        logger.info(f"Visual branch model saved to {path}")

        config = {
            'image_size': self.image_size,
            'channels': self.channels,
            'frozen_layers': VGG16_FROZEN_LAYERS,
            'feature_dim': VISUAL_FEATURE_DIM
        }
        config_path = path.replace('.keras', '_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        logger.info(f"Visual branch config saved to {config_path}")

    def load_model(self, path):
        """Load the model."""
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        logger.info(f"Visual branch model loaded from {path}")

        config_path = path.replace('.keras', '_config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            self.image_size = config['image_size']
            self.channels = config['channels']
            logger.info(f"Visual branch config loaded from {config_path}")

    def get_feature_extractor(self):
        """Return the model with visual_features as output (no classifier)."""
        if self.model is None:
            self.build_model(include_classifier=False)
        # If built with classifier, cut at visual_features layer
        return Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('visual_features').output
        )


def main():
    logger.info("=" * 60)
    logger.info("TESTING VISUAL BRANCH (standalone)")
    logger.info("=" * 60)

    visual_branch = VisualBranch()

    try:
        model = visual_branch.build_model(include_classifier=True)
        visual_branch.compile_model()

        # FIX: use the instance attributes, not bare `self` outside the class
        h, w, c = visual_branch.image_size[0], visual_branch.image_size[1], visual_branch.channels
        dummy_input = np.random.rand(4, h, w, c).astype(np.float32)

        predictions = visual_branch.predict(dummy_input)
        logger.info(f"Test predictions shape: {predictions.shape}")
        logger.info(f"Test predictions: {predictions.flatten()}")

        logger.info("=" * 60)
        logger.info("VISUAL BRANCH TEST COMPLETE!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error testing visual branch: {e}")
        logger.error("Make sure TensorFlow is properly installed")


if __name__ == "__main__":
    main()
