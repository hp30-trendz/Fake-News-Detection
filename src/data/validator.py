"""
Dataset validation and preparation module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class DatasetValidator:
    """
    Validate, merge, and prepare datasets for downstream tasks.
    """

    REQUIRED_COLUMNS = [
        "title",
        "text",
        "subject",
        "date",
    ]

    def validate_columns(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> None:
        """
        Validate required columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataset.

        dataset_name : str
            Dataset name.

        Raises
        ------
        ValueError
            If required columns are missing.
        """

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{dataset_name} missing columns: {missing_columns}"
            )

        logger.info(
            "%s schema validation successful.",
            dataset_name,
        )

    def prepare_dataset(
        self,
        fake_df: pd.DataFrame,
        real_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge and clean datasets.

        Parameters
        ----------
        fake_df : pd.DataFrame
            Fake news dataset.

        real_df : pd.DataFrame
            Real news dataset.

        Returns
        -------
        pd.DataFrame
            Clean merged dataset.
        """

        logger.info("=" * 60)
        logger.info("Preparing datasets...")
        logger.info("=" * 60)

        self.validate_columns(fake_df, "Fake Dataset")
        self.validate_columns(real_df, "Real Dataset")

        fake_df = fake_df.copy()
        real_df = real_df.copy()

        fake_df["label"] = "FAKE"
        real_df["label"] = "REAL"

        merged_df = pd.concat(
            [fake_df, real_df],
            ignore_index=True,
        )

        logger.info(
            "Merged dataset shape: %s",
            merged_df.shape,
        )

        duplicate_count = merged_df.duplicated().sum()

        logger.info(
            "Duplicate rows detected: %d",
            duplicate_count,
        )

        merged_df.drop_duplicates(inplace=True)

        merged_df.dropna(
            subset=["title", "text"],
            inplace=True,
        )

        merged_df["title"] = merged_df["title"].astype(str).str.strip()

        merged_df["text"] = merged_df["text"].astype(str).str.strip()

        merged_df = merged_df[
            (merged_df["title"] != "")
            & (merged_df["text"] != "")
        ]

        merged_df.reset_index(
            drop=True,
            inplace=True,
        )

        logger.info(
            "Final dataset shape: %s",
            merged_df.shape,
        )

        return merged_df

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> Path:
        """
        Save processed dataset.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Clean dataset.

        Returns
        -------
        Path
            Saved file path.
        """

        Config.PROCESSED_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            Config.OUTPUT_DATASET,
            index=False,
        )

        logger.info(
            "Processed dataset saved to %s",
            Config.OUTPUT_DATASET,
        )

        return Config.OUTPUT_DATASET

    @staticmethod
    def dataset_statistics(
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Print dataset statistics.
        """

        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)

        print(f"Shape: {dataframe.shape}")

        print("\nMissing Values")
        print(dataframe.isnull().sum())

        print("\nDuplicate Rows")
        print(dataframe.duplicated().sum())

        print("\nClass Distribution")
        print(dataframe["label"].value_counts())

        print("=" * 60)