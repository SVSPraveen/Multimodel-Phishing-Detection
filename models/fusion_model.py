"""
fusion_model.py
--------------
Multimodal fusion model for phishing detection.
Implements an end-to-end fusion model combining Bi-LSTM text features
and VGG16 visual features into a single trainable graph.

FIXED BUGS:
- Rebuilt as a fully end-to-end model so all branches are trained together
  with a single binary_crossentropy loss against the true label.
- Removed the flawed pattern of compiling branch models independently
  against binary labels with a multi-dim ReLU output.
- Fixed the ROC-curve argument order bug (was swapped).
- Fixed the missing 'binary_labels' key in the results dict.
- compile_model() now compiles the full fusion model correctly.
"""

import os
import sys
import numpy as np
import logging
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization,
    Embedding, Bidirectional, LSTM, GlobalMaxPooling1D, Flatten
)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    FUSION_MODEL_PATH, FUSION_UNITS, DROPOUT_RATE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT,
    MAX_SEQUENCE_LENGTH, WORD_EMBEDDING_DIM, BILSTM_UNITS,
    IMAGE_SIZE, IMAGE_CHANNELS, VGG16_FROZEN_LAYERS, VISUAL_FEATURE_DIM
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionModel:
    """
    End-to-end multimodal fusion model for phishing detection.

    Architecture
    ------------
    Text Input (integer token sequence)
        → Embedding (100-dim Word2Vec weights, frozen)
        → Bi-LSTM (128 units)
        → GlobalMaxPooling1D
        → Dense(128, relu)  -- text feature vector

    Visual Input (224×224×3 image)
        → VGG16 (first 15 layers frozen, ImageNet weights)
        → Flatten
        → Dense(256, relu)  -- visual feature vector

    Fusion
        → Concatenate([text_feat, visual_feat])  -- 384-dim
        → BatchNormalization
        → Dense(FUSION_UNITS, relu)
        → Dropout(DROPOUT_RATE)
        → Dense(1, sigmoid)  -- phishing probability
    """

    def __init__(self):
        self.model = None
        self.history = None
        self.embedding_matrix = None  # set before build if Word2Vec is available
        self.vocab_size = None         # set before build

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def build_end_to_end_model(self, vocab_size=None, embedding_matrix=None):
        """
        Build the complete end-to-end multimodal model.

        Parameters
        ----------
        vocab_size : int
            Number of words in vocabulary (needed for Embedding layer).
        embedding_matrix : np.ndarray or None
            Pre-trained Word2Vec weights. If None, the embedding is
            initialised randomly (trainable).
        """
        logger.info("Building end-to-end multimodal fusion model...")

        # Store for later reference
        if vocab_size is not None:
            self.vocab_size = vocab_size
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix

        # ------------------------------------------------------------------
        # TEXT BRANCH
        # ------------------------------------------------------------------
        text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')

        if self.embedding_matrix is not None:
            v_size = self.embedding_matrix.shape[0]
            embedding = Embedding(
                input_dim=v_size,
                output_dim=WORD_EMBEDDING_DIM,
                weights=[self.embedding_matrix],
                input_length=MAX_SEQUENCE_LENGTH,
                trainable=False,
                name='word2vec_embedding'
            )(text_input)
        else:
            v_size = self.vocab_size if self.vocab_size else 10001
            embedding = Embedding(
                input_dim=v_size,
                output_dim=WORD_EMBEDDING_DIM,
                input_length=MAX_SEQUENCE_LENGTH,
                trainable=True,
                name='word_embedding'
            )(text_input)

        bilstm = Bidirectional(
            LSTM(BILSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name='bilstm'
        )(embedding)
        pooled = GlobalMaxPooling1D(name='global_max_pool')(bilstm)
        text_features = Dense(128, activation='relu', name='text_features')(pooled)

        # ------------------------------------------------------------------
        # VISUAL BRANCH
        # ------------------------------------------------------------------
        visual_input = Input(
            shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS),
            name='visual_input'
        )

        base_vgg = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS)
        )
        # Freeze first VGG16_FROZEN_LAYERS layers
        for layer in base_vgg.layers[:VGG16_FROZEN_LAYERS]:
            layer.trainable = False
        base_vgg.trainable = True  # allow fine-tuning of last few layers

        vgg_out = base_vgg(visual_input)
        flattened = Flatten(name='flatten')(vgg_out)
        visual_features = Dense(
            VISUAL_FEATURE_DIM, activation='relu', name='visual_features'
        )(flattened)

        # ------------------------------------------------------------------
        # FUSION
        # ------------------------------------------------------------------
        concatenated = Concatenate(name='concatenate')([text_features, visual_features])
        normalized = BatchNormalization(name='batch_norm')(concatenated)
        fc_layer = Dense(FUSION_UNITS, activation='relu', name='fusion_fc')(normalized)
        dropout = Dropout(DROPOUT_RATE, name='dropout')(fc_layer)
        output = Dense(1, activation='sigmoid', name='output')(dropout)

        self.model = Model(
            inputs=[text_input, visual_input],
            outputs=output,
            name='multimodal_fusion'
        )

        logger.info("End-to-end multimodal fusion model built successfully")
        self.model.summary()
        return self.model

    # Keep old name as alias for backward compatibility
    def build_fusion_model(self, vocab_size=None, embedding_matrix=None):
        return self.build_end_to_end_model(vocab_size=vocab_size,
                                           embedding_matrix=embedding_matrix)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile_model(self, learning_rate=LEARNING_RATE):
        """Compile the end-to-end model with binary cross-entropy."""
        if self.model is None:
            self.build_end_to_end_model()

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Fusion model compiled")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, text_sequences, visual_images, labels,
              validation_split=VALIDATION_SPLIT):
        """
        Train the end-to-end model.

        Parameters
        ----------
        text_sequences : np.ndarray, shape (N, MAX_SEQUENCE_LENGTH)
            Tokenised URL/text integer sequences.
        visual_images : np.ndarray, shape (N, 224, 224, 3)
            Pre-processed screenshot images (ImageNet-normalised).
        labels : np.ndarray, shape (N,)
            Binary labels: 1 = phishing, 0 = legitimate.
        """
        if self.model is None:
            raise ValueError("Call build_end_to_end_model() and compile_model() first.")

        logger.info("Training end-to-end multimodal fusion model...")

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        self.history = self.model.fit(
            [text_sequences, visual_images],
            labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Model training completed")
        return self.history

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, text_sequences, visual_images, y_true):
        """
        Evaluate the fusion model and return a results dict.

        FIXED: 'binary_labels' key added; roc_curve argument order fixed.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        logger.info("Evaluating fusion model...")

        results_raw = self.model.evaluate(
            [text_sequences, visual_images], y_true, verbose=1
        )

        # Get predictions
        predictions = self.model.predict([text_sequences, visual_images])

        # Metrics
        from sklearn.metrics import (
            classification_report, confusion_matrix, roc_auc_score
        )

        binary_predictions = (predictions > 0.5).astype(int)
        report = classification_report(
            y_true, binary_predictions,
            target_names=['Legitimate', 'Phishing']
        )
        cm = confusion_matrix(y_true, binary_predictions)
        auc_score = roc_auc_score(y_true, predictions)

        logger.info(f"Loss:      {results_raw[0]:.4f}")
        logger.info(f"Accuracy:  {results_raw[1]:.4f}")
        logger.info(f"AUC:       {auc_score:.4f}")

        return {
            'loss': results_raw[0],
            'accuracy': results_raw[1],
            'auc': auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            # FIX: expose y_true so callers (e.g. roc_curve plots) can access it
            'binary_labels': y_true
        }

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, text_sequences, visual_images):
        """Return raw sigmoid predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict([text_sequences, visual_images])

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_model(self, path=FUSION_MODEL_PATH):
        """Save the full end-to-end model."""
        if self.model is None:
            raise ValueError("No model to save")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Fusion model saved to {path}")

        if self.history:
            history_path = path.replace('.keras', '_history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(self.history.history, f)
            logger.info(f"Training history saved to {history_path}")

    def load_model(self, path=FUSION_MODEL_PATH):
        """Load the saved model."""
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        logger.info(f"Fusion model loaded from {path}")

        history_path = path.replace('.keras', '_history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.history = pickle.load(f)
            logger.info(f"Training history loaded from {history_path}")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("TESTING FUSION MODEL")
    logger.info("=" * 60)

    fusion_model = FusionModel()

    try:
        model = fusion_model.build_end_to_end_model(vocab_size=10001)
        fusion_model.compile_model()

        # Dummy data
        dummy_text = np.random.randint(1, 1000, (4, MAX_SEQUENCE_LENGTH))
        dummy_visual = np.random.rand(4, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS).astype(np.float32)
        dummy_labels = np.array([0, 1, 0, 1])

        preds = fusion_model.predict(dummy_text, dummy_visual)
        logger.info(f"Test predictions shape: {preds.shape}")
        logger.info(f"Test predictions: {preds.flatten()}")

        logger.info("=" * 60)
        logger.info("FUSION MODEL TEST COMPLETE!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error testing fusion model: {e}")
        raise


if __name__ == "__main__":
    main()
