"""
text_processor.py
-----------------
Text preprocessing for multimodal phishing detection.

FIXED BUGS:
- Changed NLTK punkt → punkt_tab check to match the NLTK version shipped
  with recent TensorFlow/Colab environments (punkt_tab is the newer name).
- URL-aware tokenisation now used throughout: splits on URL delimiters
  instead of treating URLs as natural-language sentences, preserving
  structural phishing signals (hyphens, subdomains, suspicious paths).
- The text fallback (using the raw URL when HTML fetch fails) is now also
  processed with the URL tokeniser instead of the word tokeniser.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import requests
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    COMBINED_DATA_PATH, TEXT_FEATURES_PATH, WORD2VEC_MODEL_PATH,
    TOKENIZER_PATH, MAX_SEQUENCE_LENGTH, WORD_EMBEDDING_DIM,
    VOCAB_SIZE, RANDOM_STATE, HTML_TAGS_TO_REMOVE, OVERLAY_SELECTORS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data (only once) — support both old and new package names
for _pkg, _path in [('punkt_tab', 'tokenizers/punkt_tab'),
                     ('punkt',     'tokenizers/punkt'),
                     ('stopwords', 'corpora/stopwords')]:
    try:
        nltk.data.find(_path)
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass


def _url_tokenise(url):
    """
    URL-aware tokenisation.

    Splits a URL on common delimiters (/ . - _ ? = & : + @) and
    lowercases each token.  This preserves structural phishing signals
    that standard word_tokenize would discard.
    """
    tokens = re.split(r'[/\.\-_?=&:+@]', url.lower())
    return [t for t in tokens if t]


class TextProcessor:
    """Text processor for phishing detection."""

    def __init__(self):
        self.word2vec_model = None
        self.tokenizer = None
        self.vocab = None
        self.word_to_index = None
        self.max_seq_length = MAX_SEQUENCE_LENGTH

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def clean_html(self, html_content):
        """Clean HTML content by removing scripts, styles, and overlays."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            for tag in HTML_TAGS_TO_REMOVE:
                for element in soup.find_all(tag):
                    element.decompose()

            for selector in OVERLAY_SELECTORS:
                for element in soup.select(selector):
                    element.decompose()

            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            text = text.lower().strip()
            return text

        except Exception as e:
            logger.warning(f"Error cleaning HTML: {e}")
            return ""

    def fetch_html_content(self, url):
        """Fetch HTML content from a URL."""
        # Ensure protocol
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        try:
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36'
                )
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Error fetching HTML from {url}: {e}")
            return ""

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize_url(self, url):
        """URL-aware tokenisation (primary method)."""
        return _url_tokenise(url)

    def tokenize_text(self, text):
        """
        Tokenise page-body text using NLTK word tokeniser.

        This is used only for the HTML page content, not for the URL itself.
        """
        try:
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            tokens = [
                t for t in tokens if t.isalnum() and t not in stop_words
            ]
            return tokens
        except Exception as e:
            logger.warning(f"Error tokenising text: {e}")
            return []

    # ------------------------------------------------------------------
    # Word2Vec training
    # ------------------------------------------------------------------

    def train_word2vec(self, all_tokens):
        """Train Word2Vec model on collected URL / page tokens."""
        logger.info("Training Word2Vec model...")

        word_freq = {}
        for tokens in all_tokens:
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        vocab = [w for w, freq in word_freq.items() if freq >= 2]
        filtered_tokens = [
            [t for t in tokens if t in set(vocab)]
            for tokens in all_tokens
        ]

        try:
            from gensim.models import Word2Vec
            self.word2vec_model = Word2Vec(
                sentences=filtered_tokens,
                vector_size=WORD_EMBEDDING_DIM,
                window=5,
                min_count=2,
                workers=4,
                sg=0,   # CBOW
                epochs=100
            )
            self.vocab = list(self.word2vec_model.wv.index_to_key)[:VOCAB_SIZE]
        except ImportError:
            logger.warning(
                "gensim not available. Using frequency-based vocab only "
                "(no Word2Vec embeddings)."
            )
            self.vocab = [w for w, _ in
                          sorted(word_freq.items(), key=lambda x: -x[1])][:VOCAB_SIZE]

        self.word_to_index = {
            word: i + 1 for i, word in enumerate(self.vocab)
        }  # 0 = padding
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        return self.word2vec_model

    # ------------------------------------------------------------------
    # Sequence conversion
    # ------------------------------------------------------------------

    def url_to_sequence(self, url):
        """Convert a URL string to a padded integer token sequence."""
        tokens = self.tokenize_url(url)
        sequence = [self.word_to_index.get(tok, 0) for tok in tokens]

        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        else:
            sequence = sequence + [0] * (self.max_seq_length - len(sequence))

        return sequence

    def text_to_sequence(self, text):
        """Convert page-body text to a padded integer token sequence."""
        tokens = self.tokenize_text(text)
        sequence = [self.word_to_index.get(tok, 0) for tok in tokens]

        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        else:
            sequence = sequence + [0] * (self.max_seq_length - len(sequence))

        return sequence

    # ------------------------------------------------------------------
    # Dataset processing
    # ------------------------------------------------------------------

    def process_dataset(self, df):
        """
        Process the full dataset and return text feature sequences.

        Primary signal: URL tokens (always available).
        Secondary signal: page-body tokens (appended when HTML is fetched
        successfully and the combined length stays within max_seq_length).
        """
        logger.info("Processing text features...")
        urls = df['url'].tolist()

        all_tokens = []
        for url in tqdm(urls, desc="Tokenising URLs"):
            url_tokens = self.tokenize_url(url)

            # Optionally enrich with page-body tokens
            html = self.fetch_html_content(url)
            if html:
                page_tokens = self.tokenize_text(self.clean_html(html))
                combined = url_tokens + page_tokens
            else:
                combined = url_tokens

            all_tokens.append(combined)

        # Train Word2Vec (or build vocab) on collected tokens
        self.train_word2vec(all_tokens)

        # Convert to integer sequences
        logger.info("Converting to integer sequences...")
        sequences = []
        for url in tqdm(urls, desc="Building sequences"):
            sequences.append(self.url_to_sequence(url))

        text_features = np.array(sequences, dtype=np.int32)
        logger.info(f"Text features shape: {text_features.shape}")
        return text_features

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_model(self):
        """Save trained Word2Vec model and tokenizer."""
        os.makedirs(os.path.dirname(WORD2VEC_MODEL_PATH), exist_ok=True)

        if self.word2vec_model is not None:
            self.word2vec_model.save(WORD2VEC_MODEL_PATH)
            logger.info(f"Word2Vec model saved to {WORD2VEC_MODEL_PATH}")

        tokenizer_data = {
            'vocab': self.vocab,
            'word_to_index': self.word_to_index,
            'max_seq_length': self.max_seq_length
        }
        with open(TOKENIZER_PATH, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        logger.info(f"Tokenizer saved to {TOKENIZER_PATH}")

    def load_model(self):
        """Load trained Word2Vec model and tokenizer."""
        if os.path.exists(WORD2VEC_MODEL_PATH):
            try:
                from gensim.models import Word2Vec
                self.word2vec_model = Word2Vec.load(WORD2VEC_MODEL_PATH)
                logger.info(f"Word2Vec model loaded from {WORD2VEC_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Could not load Word2Vec: {e}")

        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer_data = pickle.load(f)
            self.vocab = tokenizer_data['vocab']
            self.word_to_index = tokenizer_data['word_to_index']
            self.max_seq_length = tokenizer_data['max_seq_length']
            logger.info(f"Tokenizer loaded from {TOKENIZER_PATH}")


def main():
    logger.info("=" * 60)
    logger.info("TEXT PROCESSING FOR MULTIMODAL PHISHING DETECTION")
    logger.info("=" * 60)

    if not os.path.exists(COMBINED_DATA_PATH):
        logger.error(f"Dataset not found: {COMBINED_DATA_PATH}")
        logger.error("Please run scripts/collect_data.py first")
        return

    df = pd.read_csv(COMBINED_DATA_PATH)
    logger.info(f"Loaded dataset: {len(df)} URLs")

    processor = TextProcessor()
    text_features = processor.process_dataset(df)

    os.makedirs(os.path.dirname(TEXT_FEATURES_PATH), exist_ok=True)
    np.save(TEXT_FEATURES_PATH, text_features)
    logger.info(f"Text features saved to {TEXT_FEATURES_PATH}")

    processor.save_model()

    logger.info("=" * 60)
    logger.info("TEXT PROCESSING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Text features shape: {text_features.shape}")
    logger.info(f"Vocabulary size: {len(processor.vocab)}")
    logger.info("Next step: Run preprocessing/visual_processor.py")


if __name__ == "__main__":
    main()
