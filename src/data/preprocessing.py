"""
Text preprocessing module.
"""

from __future__ import annotations

import re
import string

import emoji
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class TextPreprocessor:
    """
    Clean raw text before feature engineering.

    This class preprocesses a single piece of text.
    """

    def __init__(
        self,
        use_stemming: bool = True,
        use_lemmatization: bool = False,
    ) -> None:
        """
        Initialize preprocessing resources.

        Parameters
        ----------
        use_stemming : bool
            Whether to apply stemming.

        use_lemmatization : bool
            Whether to apply lemmatization.
        """

        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization

        self.stop_words = set(stopwords.words("english"))

        self.stemmer = PorterStemmer()

        self.lemmatizer = WordNetLemmatizer()

        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")

        self.email_pattern = re.compile(r"\S+@\S+\.\S+")

        self.number_pattern = re.compile(r"\d+")

        self.whitespace_pattern = re.compile(r"\s+")

        self.punctuation_table = str.maketrans(
            "",
            "",
            string.punctuation,
        )

    @staticmethod
    def remove_html(text: str) -> str:
        """Remove HTML tags."""
        return BeautifulSoup(
            text,
            "html.parser",
        ).get_text(separator=" ")

    @staticmethod
    def lowercase(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    def remove_urls(self, text: str) -> str:
        """Remove URLs."""
        return self.url_pattern.sub("", text)

    def remove_emails(self, text: str) -> str:
        """Remove email addresses."""
        return self.email_pattern.sub("", text)

    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove emojis."""
        return emoji.replace_emoji(
            text,
            replace="",
        )

    def remove_numbers(self, text: str) -> str:
        """Remove numbers."""
        return self.number_pattern.sub("", text)

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation."""
        return text.translate(self.punctuation_table)

    def normalize_whitespace(self, text: str) -> str:
        """Remove extra spaces."""
        return self.whitespace_pattern.sub(
            " ",
            text,
        ).strip()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split text into tokens."""
        return text.split()

    def remove_stopwords(
        self,
        tokens: list[str],
    ) -> list[str]:
        """Remove English stopwords."""
        return [
            token
            for token in tokens
            if token not in self.stop_words
        ]

    def stem(
        self,
        tokens: list[str],
    ) -> list[str]:
        """Apply stemming."""
        return [
            self.stemmer.stem(token)
            for token in tokens
        ]

    def lemmatize(
        self,
        tokens: list[str],
    ) -> list[str]:
        """Apply lemmatization."""
        return [
            self.lemmatizer.lemmatize(token)
            for token in tokens
        ]

    def preprocess(
        self,
        text: str,
    ) -> str:
        """
        Execute the preprocessing pipeline.

        Parameters
        ----------
        text : str
            Raw text.

        Returns
        -------
        str
            Cleaned text.
        """

        if not isinstance(text, str):
            return ""

        text = self.lowercase(text)
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_emojis(text)
        text = self.remove_numbers(text)
        text = self.remove_punctuation(text)
        text = self.normalize_whitespace(text)

        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)

        if self.use_stemming:
            tokens = self.stem(tokens)

        if self.use_lemmatization:
            tokens = self.lemmatize(tokens)

        return " ".join(tokens)


class DatasetPreprocessor:
    """
    Apply text preprocessing to an entire dataset.
    """

    def __init__(
        self,
        use_stemming: bool = True,
        use_lemmatization: bool = False,
    ) -> None:
        """
        Initialize the dataset preprocessor.

        Parameters
        ----------
        use_stemming : bool
            Whether to apply stemming.

        use_lemmatization : bool
            Whether to apply lemmatization.
        """

        tqdm.pandas()

        self.text_preprocessor = TextPreprocessor(
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization,
        )

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Preprocess every article in the dataset.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Dataset containing the new 'clean_text' column.
        """

        logger.info("=" * 60)
        logger.info("Starting dataset preprocessing...")
        logger.info("=" * 60)

        dataframe = dataframe.copy()

        dataframe["clean_text"] = (
            dataframe["text"]
            .fillna("")
            .progress_apply(
                self.text_preprocessor.preprocess
            )
        )

        logger.info("Dataset preprocessing completed.")

        return dataframe

    def save(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Save the processed dataset.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Processed dataset.
        """

        dataframe.to_csv(
            Config.OUTPUT_DATASET,
            index=False,
        )

        logger.info(
            "Processed dataset saved successfully to %s",
            Config.OUTPUT_DATASET,
        )