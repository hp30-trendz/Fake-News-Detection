"""
Visualization utilities for Exploratory Data Analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class PlotGenerator:
    """
    Generate and save EDA plots.
    """

    def __init__(self) -> None:
        """Initialize plot generator."""

        Config.IMAGES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _save_plot(filename: str) -> None:
        """
        Save current figure.

        Parameters
        ----------
        filename : str
            Name of image file.
        """

        output_path = Config.IMAGES_DIR / filename

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        logger.info("Saved plot -> %s", output_path)

        plt.show()

        plt.close()

    def plot_class_distribution(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Plot label distribution.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataset.
        """

        counts = dataframe["label"].value_counts()

        plt.figure(figsize=(8, 6))

        plt.bar(
            counts.index,
            counts.values,
        )

        plt.title("Class Distribution")

        plt.xlabel("Class")

        plt.ylabel("Number of Articles")

        self._save_plot("class_distribution.png")

    def plot_text_length_distribution(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Plot article length distribution.
        """

        lengths = dataframe["text"].str.len()

        plt.figure(figsize=(10, 6))

        plt.hist(
            lengths,
            bins=40,
        )

        plt.title("Character Length Distribution")

        plt.xlabel("Characters")

        plt.ylabel("Frequency")

        self._save_plot(
            "character_length_distribution.png"
        )

    def plot_title_length_distribution(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Plot title length distribution.
        """

        lengths = dataframe["title"].str.len()

        plt.figure(figsize=(10, 6))

        plt.hist(
            lengths,
            bins=30,
        )

        plt.title("Title Length Distribution")

        plt.xlabel("Characters")

        plt.ylabel("Frequency")

        self._save_plot(
            "title_length_distribution.png"
        )