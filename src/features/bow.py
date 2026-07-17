"""
Bag of Words feature extraction module.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class BagOfWordsVectorizer:
    """
    Convert text into Bag of Words feature vectors.
    """

    def __init__(
        self,
        max_features: int = 5000,
    ) -> None:
        """
        Initialize the CountVectorizer.

        Parameters
        ----------
        max_features : int, default=5000
            Maximum number of words to keep in the vocabulary.
        """

        self.vectorizer = CountVectorizer(
            max_features=max_features,
        )

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        text_column: str = "clean_text",
    ):
        """
        Learn the vocabulary and transform the dataset.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataset.

        text_column : str
            Name of the text column.

        Returns
        -------
        scipy.sparse.csr_matrix
            Bag of Words feature matrix.
        """

        logger.info("Generating Bag of Words features...")

        features = self.vectorizer.fit_transform(
            dataframe[text_column]
        )

        logger.info(
            "Bag of Words matrix shape: %s",
            features.shape,
        )

        return features

    def transform(
        self,
        dataframe: pd.DataFrame,
        text_column: str = "clean_text",
    ):
        """
        Transform new text using the learned vocabulary.

        Parameters
        ----------
        dataframe : pd.DataFrame

        text_column : str

        Returns
        -------
        scipy.sparse.csr_matrix
        """

        return self.vectorizer.transform(
            dataframe[text_column]
        )

    def save(self) -> None:
        """
        Save the fitted vectorizer.
        """

        output_path = Config.MODELS_DIR / "bow_vectorizer.joblib"

        joblib.dump(
            self.vectorizer,
            output_path,
        )

        logger.info(
            "Bag of Words vectorizer saved to %s",
            output_path,
        )

    def get_feature_names(
        self,
    ) -> list[str]:
        """
        Return the vocabulary.

        Returns
        -------
        list[str]
        """

        return list(
            self.vectorizer.get_feature_names_out()
        )   