"""
Word cloud and word frequency visualization module.
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class WordCloudGenerator:
    """
    Generate word clouds and word frequency plots.
    """

    def __init__(self) -> None:
        """Create output directory."""

        Config.IMAGES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _save_plot(filename: str) -> None:
        """
        Save current matplotlib figure.
        """

        output_path = Config.IMAGES_DIR / filename

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        logger.info("Saved %s", output_path)

        plt.show()

        plt.close()

    def generate_wordcloud(
        self,
        dataframe: pd.DataFrame,
        label: str,
    ) -> None:
        """
        Generate a word cloud.

        Parameters
        ----------
        dataframe : pd.DataFrame

        label : str
            FAKE or REAL.
        """

        text = " ".join(
            dataframe.loc[
                dataframe["label"] == label,
                "text",
            ].astype(str)
        )

        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            max_words=300,
        ).generate(text)

        plt.figure(figsize=(14, 7))

        plt.imshow(wordcloud)

        plt.axis("off")

        plt.title(f"{label} News Word Cloud")

        self._save_plot(
            f"{label.lower()}_wordcloud.png"
        )

    def top_words(
        self,
        dataframe: pd.DataFrame,
        label: str,
        top_n: int = 20,
    ) -> None:
        """
        Plot most frequent words.

        Parameters
        ----------
        dataframe : pd.DataFrame

        label : str

        top_n : int
        """

        words = (
            " ".join(
                dataframe.loc[
                    dataframe["label"] == label,
                    "text",
                ].astype(str)
            )
            .lower()
            .split()
        )

        counter = Counter(words)

        common = counter.most_common(top_n)

        labels = [item[0] for item in common]

        counts = [item[1] for item in common]

        plt.figure(figsize=(12, 7))

        plt.barh(
            labels[::-1],
            counts[::-1],
        )

        plt.title(
            f"Top {top_n} Words ({label})"
        )

        plt.xlabel("Frequency")

        self._save_plot(
            f"{label.lower()}_top_words.png"
        )