"""
Data loading module.

This module provides functionality for loading raw datasets used by the
Fake News Detection project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Load raw datasets from disk."""

    def __init__(self) -> None:
        """Initialize dataset paths."""

        self.datasets: dict[str, Path] = {
            "fake": Config.FAKE_DATASET,
            "real": Config.REAL_DATASET,
        }

    @staticmethod
    def load_csv(file_path: Path) -> pd.DataFrame:
        """
        Load a CSV file.

        Parameters
        ----------
        file_path : Path
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded dataframe.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """

        if not file_path.exists():
            logger.error("Dataset not found: %s", file_path)
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        try:
            dataframe = pd.read_csv(file_path)

            dataframe.columns = dataframe.columns.str.strip()

            logger.info(
                "Loaded %s | Rows=%d | Columns=%d",
                file_path.name,
                dataframe.shape[0],
                dataframe.shape[1],
            )

            return dataframe

        except Exception as exc:
            logger.exception("Failed to load %s", file_path.name)
            raise exc

    def load_all_datasets(self) -> dict[str, pd.DataFrame]:
        """
        Load all configured datasets.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary containing loaded datasets.
        """

        logger.info("=" * 60)
        logger.info("Loading datasets...")
        logger.info("=" * 60)

        loaded_datasets: dict[str, pd.DataFrame] = {}

        for name, path in self.datasets.items():
            loaded_datasets[name] = self.load_csv(path)

        logger.info(
            "Successfully loaded %d datasets.",
            len(loaded_datasets),
        )

        return loaded_datasets

    @staticmethod
    def summarize(
        dataframe: pd.DataFrame,
        name: str,
    ) -> None:
        """
        Display dataset summary.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataset.

        name : str
            Dataset name.
        """

        print("\n" + "=" * 60)
        print(f"Dataset : {name}")
        print("=" * 60)

        print(f"Shape : {dataframe.shape}")

        print("\nColumns")
        print(dataframe.columns.tolist())

        print("\nMissing Values")
        print(dataframe.isnull().sum())

        print("\nDuplicate Rows")
        print(dataframe.duplicated().sum())

        print("\nPreview")
        print(dataframe.head())

        print("=" * 60)