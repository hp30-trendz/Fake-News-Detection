"""
Data loading module.

This module provides functionality for loading all raw datasets
used in the Fake News Detection project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """
    Initialize dataset paths.
    """

    def __init__(self) -> None:
        """Initialize the data loader."""

        self.datasets = {
            "fake": Config.FAKE_DATASET,
            "real": Config.REAL_DATASET,
        }

    def load_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load a CSV file safely.

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
            If the CSV file does not exist.
        """

        if not file_path.exists():
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"{file_path} not found.")

        try:
            dataframe = pd.read_csv(file_path)

            # Remove leading and trailing whitespace from column names
            dataframe.columns = dataframe.columns.str.strip()

            logger.info(
                "Loaded %s | Rows=%d Columns=%d",
                file_path.name,
                dataframe.shape[0],
                dataframe.shape[1],
            )

            return dataframe

        except Exception as exc:
            logger.exception("Unable to load %s", file_path.name)
            raise exc

    def load_all_datasets(self) -> dict[str, pd.DataFrame]:
        """
        Load every configured dataset.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary containing all loaded datasets.
        """

        loaded_data = {}

        logger.info("=" * 60)
        logger.info("Loading raw datasets...")
        logger.info("=" * 60)

        for dataset_name, dataset_path in self.datasets.items():
            loaded_data[dataset_name] = self.load_csv(dataset_path)

        logger.info("Successfully loaded %d datasets.", len(loaded_data))

        return loaded_data

    @staticmethod
    def dataset_summary(dataframe: pd.DataFrame, name: str) -> None:
        """
        Print a short summary of a dataframe.

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

        print(f"Shape      : {dataframe.shape}")

        print("\nColumns")
        print(dataframe.columns.tolist())

        print("\nMissing Values")
        print(dataframe.isnull().sum())

        print("\nDuplicates :", dataframe.duplicated().sum())

        print("\nFirst Five Rows")
        print(dataframe.head())

        print("=" * 60)