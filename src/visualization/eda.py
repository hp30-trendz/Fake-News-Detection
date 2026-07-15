"""
Exploratory Data Analysis (EDA) module.

This module provides reusable methods for generating
basic dataset statistics.
"""

from __future__ import annotations

import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__)


class EDAAnalyzer:
    """
    Perform Exploratory Data Analysis.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the analyzer.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataset to analyze.
        """
        self.df = dataframe

    def dataset_shape(self) -> None:
        """Print dataset shape."""
        logger.info("Dataset Shape")
        print(f"Rows    : {self.df.shape[0]}")
        print(f"Columns : {self.df.shape[1]}")

    def dataset_info(self) -> None:
        """Print dataframe information."""
        print("\n")
        self.df.info()

    def missing_values(self) -> pd.DataFrame:
        """
        Return missing value statistics.
        """
        missing = (
            self.df.isnull()
            .sum()
            .to_frame("Missing Values")
        )

        print("\nMissing Values")
        print(missing)

        return missing

    def duplicate_count(self) -> int:
        """
        Count duplicate rows.
        """
        duplicates = self.df.duplicated().sum()

        print(f"\nDuplicate Rows : {duplicates}")

        return duplicates

    def class_distribution(self) -> pd.Series:
        """
        Display label distribution.
        """
        distribution = self.df["label"].value_counts()

        print("\nClass Distribution")
        print(distribution)

        return distribution

    def summary(self) -> None:
        """
        Execute all summary methods.
        """
        print("=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        self.dataset_shape()

        self.dataset_info()

        self.missing_values()

        self.duplicate_count()

        self.class_distribution()

        print("=" * 60)