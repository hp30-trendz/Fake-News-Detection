"""
TF-IDF feature extraction module.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.logger import setup_logger

logger = setup_logger(__name__)


class TFIDFFeatureExtractor:
    """
    Convert text into TF-IDF feature vectors.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ) -> None:
        """
        Initialize the TF-IDF vectorizer.
        """

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )

    def fit_transform(
        self,
        text: pd.Series,
    ) -> csr_matrix:
        """
        Learn the vocabulary and transform the training text.
        """

        logger.info("Fitting TF-IDF vectorizer...")

        features = self.vectorizer.fit_transform(text)

        logger.info(
            "TF-IDF feature matrix shape: %s",
            features.shape,
        )

        return features

    def transform(
        self,
        text: pd.Series,
    ) -> csr_matrix:
        """
        Transform new text using the fitted vectorizer.
        """

        return self.vectorizer.transform(text)

    def save(
        self,
        output_path: Path,
    ) -> None:
        """
        Save the fitted vectorizer.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.vectorizer,
            output_path,
        )

        logger.info(
            "TF-IDF vectorizer saved to %s",
            output_path,
        )

    @staticmethod
    def load(
        model_path: Path,
    ) -> TfidfVectorizer:
        """
        Load a saved TF-IDF vectorizer.
        """

        logger.info(
            "Loading TF-IDF vectorizer from %s",
            model_path,
        )

        return joblib.load(model_path)