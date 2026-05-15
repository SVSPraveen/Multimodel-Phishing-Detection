"""
text_branch.py
-------------
Bi-LSTM text branch for multimodal phishing detection.

NOTE: This module is now kept for reference and standalone testing only.
      In the end-to-end architecture the Bi-LSTM layers are embedded
      directly inside FusionModel, so you do not need to call this class
      during training or inference.

FIXED BUGS:
- The model used binary_crossentropy with a 128-dim Dense(relu) output which
  caused Keras to broadcast labels across all 128 dimensions — meaningless.
  The standalone model now has a proper Dense(1, sigmoid) classification head
  so it can be compiled and tested independently if desired.
- Removed incorrect reference to VOCAB_SIZE constant (config uses it as an
  upper cap, but the actual vocab depends on the tokenizer).
"""

import os
import sys
import numpy as np
import logging
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense,
    GlobalMaxPooling1D, Dropout
)
from tensorflow.keras.optimizers import Adam

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    WORD2VEC_MODEL_PATH, TOKENIZER_PATH, MAX_SEQUENCE_LENGTH,
    WORD_EMBEDDING_DIM, BILSTM_UNITS, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextBranch:
    """
    Standalone Bi-LSTM text branch.

    Use this class for independent testing or pre-training experiments.
    For production, the same architecture is embedded in FusionModel.
    """

    def __init__(self):
        self.word2vec_model = None
        self.tokenizer_data = None
        self.model = None
        self.embedding_matrix = None

    def load_models(self):
        """Load Word2Vec model and tokenizer."""
        # Load Word2Vec
        if os.path.exists(WORD2VEC_MODEL_PATH):
            try:
                from gensim.models import Word2Vec
                self.word2vec_model = Word2Vec.load(WORD2VEC_MODEL_PATH)
                logger.info(f"Word2Vec model loaded from {WORD2VEC_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Could not load Word2Vec: {e}")
        else:
            logger.warning(
                f"Word2Vec model not found at {WORD2VEC_MODEL_PATH}. "
                "Embedding layer will be trained from scratch."
            )

        # Load tokenizer
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as f:
                self.tokenizer_data = pickle.load(f)
            logger.info(f"Tokenizer loaded from {TOKENIZER_PATH}")
        else:
            raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    def create_embedding_matrix(self):
        """Create embedding matrix from Word2Vec model."""
        if self.word2vec_model is None or self.tokenizer_data is None:
            return None

        word_to_index = self.tokenizer_data['word_to_index']
        vocab_size = len(word_to_index) + 1
        embedding_matrix = np.zeros((vocab_size, WORD_EMBEDDING_DIM))

        for word, index in word_to_index.items():
            if word in self.word2vec_model.wv:
                embedding_matrix[index] = self.word2vec_model.wv[word]
            else:
                embedding_matrix[index] = np.random.normal(0, 0.1, WORD_EMBEDDING_DIM)

        self.embedding_matrix = embedding_matrix
        logger.info(f"Embedding matrix created: {embedding_matrix.shape}")
        return embedding_matrix

    def build_model(self, include_classifier=True):
        """
        Build Bi-LSTM model.

        Parameters
        ----------
        include_classifier : bool
            If True, appends Dense(1, sigmoid) so the model can be compiled
            and trained standalone with binary_crossentropy.
            Set False to return feature vectors only (used in FusionModel).
        """
        vocab_size = (
            len(self.tokenizer_data['word_to_index']) + 1
            if self.tokenizer_data else 10001
        )

        if self.embedding_matrix is None and self.word2vec_model is not None:
            self.create_embedding_matrix()

        input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')

        if self.embedding_matrix is not None:
            embedding_layer = Embedding(
                input_dim=self.embedding_matrix.shape[0],
                output_dim=WORD_EMBEDDING_DIM,
                weights=[self.embedding_matrix],
                input_length=MAX_SEQUENCE_LENGTH,
                trainable=False,
                name='word2vec_embedding'
            )(input_layer)
        else:
            embedding_layer = Embedding(
                input_dim=vocab_size,
                output_dim=WORD_EMBEDDING_DIM,
                input_length=MAX_SEQUENCE_LENGTH,
                trainable=True,
                name='word_embedding'
            )(input_layer)

        bilstm_layer = Bidirectional(
            LSTM(BILSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name='bilstm'
        )(embedding_layer)

        pooled_layer = GlobalMaxPooling1D(name='global_max_pool')(bilstm_layer)

        # 128-dim feature vector
        feature_layer = Dense(128, activation='relu', name='text_features')(pooled_layer)

        if include_classifier:
            # FIX: add a proper classification head so the model can be
            # trained with binary_crossentropy without dimensionality mismatch
            dropout = Dropout(0.3, name='dropout')(feature_layer)
            output_layer = Dense(1, activation='sigmoid', name='output')(dropout)
            self.model = Model(
                inputs=input_layer, outputs=output_layer, name='text_branch'
            )
        else:
            self.model = Model(
                inputs=input_layer, outputs=feature_layer, name='text_branch_features'
            )

        logger.info("Bi-LSTM text branch built successfully")
        self.model.summary()
        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compile the model for standalone training."""
        if self.model is None:
            self.build_model(include_classifier=True)

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Text branch model compiled")

    def predict(self, text_sequences):
        """Generate predictions for text sequences."""
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model.predict(text_sequences)

    def save_model(self, path):
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        logger.info(f"Text branch model saved to {path}")

    def load_model(self, path):
        """Load the model."""
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        logger.info(f"Text branch model loaded from {path}")


def main():
    logger.info("=" * 60)
    logger.info("TESTING TEXT BRANCH (standalone)")
    logger.info("=" * 60)

    text_branch = TextBranch()

    try:
        text_branch.load_models()
        model = text_branch.build_model(include_classifier=True)
        text_branch.compile_model()

        dummy_input = np.random.randint(1, 1000, (10, MAX_SEQUENCE_LENGTH))
        predictions = text_branch.predict(dummy_input)
        logger.info(f"Test predictions shape: {predictions.shape}")
        logger.info(f"Test predictions sample: {predictions[:3].flatten()}")

        logger.info("=" * 60)
        logger.info("TEXT BRANCH TEST COMPLETE!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error testing text branch: {e}")
        logger.error("Make sure to run preprocessing/text_processor.py first")


if __name__ == "__main__":
    main()
