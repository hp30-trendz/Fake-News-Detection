"""
Dataset validation module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class DatasetValidator:
    """
    Validate and merge fake news datasets.
    """

    REQUIRED_COLUMNS = ["title", "text", "subject", "date"]

    def validate_columns(self, dataframe: pd.DataFrame) -> None:
        """
        Validate that all required columns exist.
        """
        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def prepare_dataset(
        self,
        fake_df: pd.DataFrame,
        real_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge, clean and validate datasets.
        """

        logger.info("Preparing datasets...")

        self.validate_columns(fake_df)
        self.validate_columns(real_df)

        fake_df = fake_df.copy()
        real_df = real_df.copy()

        fake_df["label"] = "FAKE"
        real_df["label"] = "REAL"

        dataframe = pd.concat(
            [fake_df, real_df],
            ignore_index=True,
        )

        before_rows = len(dataframe)

        dataframe.drop_duplicates(inplace=True)

        duplicates_removed = before_rows - len(dataframe)

        dataframe.dropna(
            subset=["title", "text"],
            inplace=True,
        )

        dataframe.reset_index(
            drop=True,
            inplace=True,
        )

        logger.info(
            "Duplicates removed: %d",
            duplicates_removed,
        )

        logger.info(
            "Final dataset shape: %s",
            dataframe.shape,
        )

        return dataframe

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> Path:
        """
        Save cleaned dataset.
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