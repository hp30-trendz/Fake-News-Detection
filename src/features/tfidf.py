"""
TF-IDF feature extraction module.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import Config
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

    def save(self) -> Path:
        """
        Save the fitted TF-IDF vectorizer.

        Returns
        -------
        Path
            Path to the saved vectorizer.
        """

        output_directory = Config.MODELS_DIR / "vectorizers"

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_directory / "tfidf_vectorizer.joblib"
        )

        joblib.dump(
            self.vectorizer,
            output_path,
        )

        logger.info(
            "TF-IDF vectorizer saved to %s",
            output_path,
        )

        return output_path

    @staticmethod
    def load(
        model_path: Path,
    ) -> TfidfVectorizer:
        """
        Load a saved TF-IDF vectorizer.

        Parameters
        ----------
        model_path : Path
            Path to the saved TF-IDF vectorizer.

        Returns
        -------
        TfidfVectorizer
            Loaded TF-IDF vectorizer.
        """

        logger.info(
            "Loading TF-IDF vectorizer from %s",
            model_path,
        )

        return joblib.load(model_path)