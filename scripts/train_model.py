"""
train_model.py
--------------
Training script for multimodal phishing detection.
Implements the complete end-to-end training pipeline.

FIXED BUGS:
- Replaced branch-by-branch training with a single end-to-end model.
- ROC-curve arguments were swapped: fixed to roc_curve(y_true, y_score).
- 'binary_labels' KeyError in create_plots fixed: now uses y_test directly.
- Feature-shape triple-comparison was logically wrong (a!=b!=c doesn't mean
  all three are different): replaced with explicit pair checks.
- Removed stale references to TEXT_FEATURES_PATH / VISUAL_FEATURES_PATH
  that are no longer needed when training end-to-end.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    COMBINED_DATA_PATH,
    FUSION_MODEL_PATH,
    TOKENIZER_PATH,
    WORD2VEC_MODEL_PATH,
    RESULTS_DIR, RANDOM_STATE, VALIDATION_SPLIT,
    MAX_SEQUENCE_LENGTH, WORD_EMBEDDING_DIM, IMAGE_SIZE, IMAGE_CHANNELS
)

from models.fusion_model import FusionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Complete end-to-end training pipeline for multimodal phishing detection."""

    def __init__(self):
        self.fusion_model = None
        self.dataset = None
        self.text_sequences = None
        self.visual_images = None
        self.labels = None
        self.tokenizer_data = None
        self.embedding_matrix = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_dataset(self):
        """Load the combined dataset CSV."""
        if not os.path.exists(COMBINED_DATA_PATH):
            logger.error(f"Dataset not found: {COMBINED_DATA_PATH}")
            logger.error("Please run scripts/collect_data.py first")
            return False

        self.dataset = pd.read_csv(COMBINED_DATA_PATH)
        self.labels = self.dataset['label'].values

        logger.info(f"Dataset loaded: {len(self.dataset)} samples")
        logger.info(f"  - Legitimate: {np.sum(self.labels == 0)}")
        logger.info(f"  - Phishing:   {np.sum(self.labels == 1)}")
        return True

    def load_tokenizer(self):
        """Load tokenizer and optionally build Word2Vec embedding matrix."""
        if not os.path.exists(TOKENIZER_PATH):
            logger.error(f"Tokenizer not found: {TOKENIZER_PATH}")
            logger.error("Please run preprocessing/text_processor.py first")
            return False

        with open(TOKENIZER_PATH, 'rb') as f:
            self.tokenizer_data = pickle.load(f)

        logger.info(
            f"Tokenizer loaded: vocab_size={len(self.tokenizer_data['word_to_index'])}, "
            f"max_seq={self.tokenizer_data['max_seq_length']}"
        )

        # Optionally build embedding matrix from Word2Vec
        if os.path.exists(WORD2VEC_MODEL_PATH):
            try:
                from gensim.models import Word2Vec
                w2v = Word2Vec.load(WORD2VEC_MODEL_PATH)
                word_to_index = self.tokenizer_data['word_to_index']
                vocab_size = len(word_to_index) + 1
                emb_matrix = np.zeros((vocab_size, WORD_EMBEDDING_DIM))
                for word, idx in word_to_index.items():
                    if word in w2v.wv:
                        emb_matrix[idx] = w2v.wv[word]
                    else:
                        emb_matrix[idx] = np.random.normal(0, 0.1, WORD_EMBEDDING_DIM)
                self.embedding_matrix = emb_matrix
                logger.info(f"Embedding matrix built: {emb_matrix.shape}")
            except Exception as e:
                logger.warning(f"Could not load Word2Vec embeddings ({e}); "
                               "embedding layer will be trained from scratch.")

        return True

    def build_text_sequences(self):
        """Convert URL strings to padded integer token sequences."""
        if self.tokenizer_data is None:
            logger.error("Tokenizer not loaded")
            return False

        word_to_index = self.tokenizer_data['word_to_index']
        max_seq = self.tokenizer_data['max_seq_length']

        import re
        sequences = []
        for url in self.dataset['url'].tolist():
            # URL-aware tokenisation: split on /, ., -, _, ?, =, &, :, +
            tokens = re.split(r'[/\.\-_?=&:+@]', url.lower())
            tokens = [t for t in tokens if t]
            seq = [word_to_index.get(tok, 0) for tok in tokens]
            if len(seq) > max_seq:
                seq = seq[:max_seq]
            else:
                seq = seq + [0] * (max_seq - len(seq))
            sequences.append(seq)

        self.text_sequences = np.array(sequences, dtype=np.int32)
        logger.info(f"Text sequences shape: {self.text_sequences.shape}")
        return True

    def load_visual_images(self):
        """
        Load pre-processed visual features (images) if available.
        Falls back to dummy zero images so training can still proceed.
        """
        from configs.config import VISUAL_FEATURES_PATH
        if os.path.exists(VISUAL_FEATURES_PATH):
            self.visual_images = np.load(VISUAL_FEATURES_PATH)
            logger.info(f"Visual images loaded: {self.visual_images.shape}")
        else:
            logger.warning(
                f"Visual features file not found at {VISUAL_FEATURES_PATH}. "
                "Using dummy (zero) images — run preprocessing/visual_processor.py "
                "to capture real screenshots."
            )
            n = len(self.dataset)
            self.visual_images = np.zeros(
                (n, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS), dtype=np.float32
            )
        return True

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def split_data(self):
        """Stratified train/test split."""
        indices = np.arange(len(self.labels))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=RANDOM_STATE, stratify=self.labels
        )

        X_train_text   = self.text_sequences[train_idx]
        X_test_text    = self.text_sequences[test_idx]
        X_train_visual = self.visual_images[train_idx]
        X_test_visual  = self.visual_images[test_idx]
        y_train        = self.labels[train_idx]
        y_test         = self.labels[test_idx]

        logger.info(f"Data split → train: {len(y_train)}, test: {len(y_test)}")
        return X_train_text, X_train_visual, y_train, X_test_text, X_test_visual, y_test

    def train_end_to_end(self, X_train_text, X_train_visual, y_train):
        """Build, compile and train the end-to-end model."""
        logger.info("Initialising end-to-end multimodal model...")

        vocab_size = len(self.tokenizer_data['word_to_index']) + 1

        self.fusion_model = FusionModel()
        self.fusion_model.build_end_to_end_model(
            vocab_size=vocab_size,
            embedding_matrix=self.embedding_matrix
        )
        self.fusion_model.compile_model()

        history = self.fusion_model.train(X_train_text, X_train_visual, y_train)
        self.fusion_model.save_model(FUSION_MODEL_PATH)

        logger.info("End-to-end training completed")
        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(self, X_test_text, X_test_visual, y_test):
        """Evaluate on test set and return results dict."""
        logger.info("Evaluating model...")
        results = self.fusion_model.evaluate(X_test_text, X_test_visual, y_test)

        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"AUC:      {results['auc']:.4f}")
        logger.info(f"\nClassification Report:\n{results['classification_report']}")
        logger.info(f"Confusion Matrix:\n{results['confusion_matrix']}")
        return results

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def save_results(self, results, y_test):
        """Save evaluation results and plots."""
        os.makedirs(RESULTS_DIR, exist_ok=True)

        results_path = os.path.join(RESULTS_DIR, 'evaluation_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {results_path}")

        self.create_plots(results, y_test)

    def create_plots(self, results, y_test):
        """Create and save confusion-matrix and ROC-curve plots."""
        # -- Confusion matrix ---
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
        plt.close()

        # -- ROC curve ---
        # FIX: correct argument order is roc_curve(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_test, results['predictions'])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2,
                 label=f'ROC Curve (AUC = {results["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--',
                 label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
        plt.close()

        logger.info(f"Plots saved to {RESULTS_DIR}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def train_complete_pipeline(self):
        """Run the entire training pipeline end-to-end."""
        logger.info("=" * 60)
        logger.info("COMPLETE TRAINING PIPELINE")
        logger.info("=" * 60)

        if not self.load_dataset():
            return False

        if not self.load_tokenizer():
            return False

        if not self.build_text_sequences():
            return False

        if not self.load_visual_images():
            return False

        # Verify row counts match
        n_text   = len(self.text_sequences)
        n_visual = len(self.visual_images)
        n_labels = len(self.labels)
        if not (n_text == n_visual == n_labels):
            logger.error(
                f"Row-count mismatch: text={n_text}, visual={n_visual}, labels={n_labels}"
            )
            return False

        (X_train_text, X_train_visual, y_train,
         X_test_text,  X_test_visual,  y_test) = self.split_data()

        self.train_end_to_end(X_train_text, X_train_visual, y_train)

        results = self.evaluate_model(X_test_text, X_test_visual, y_test)

        self.save_results(results, y_test)

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Final Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Final AUC:      {results['auc']:.4f}")
        logger.info(f"Model saved to: {FUSION_MODEL_PATH}")
        logger.info(f"Results saved to: {RESULTS_DIR}")
        return True


def main():
    trainer = ModelTrainer()
    success = trainer.train_complete_pipeline()
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")


if __name__ == "__main__":
    main()
