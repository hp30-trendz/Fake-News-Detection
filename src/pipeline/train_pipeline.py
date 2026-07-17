"""
Training pipeline for the Fake News Detection project.
"""

from __future__ import annotations

from src.data.loader import DataLoader
from src.data.preprocessing import DatasetPreprocessor
from src.data.validator import DatasetValidator
from src.visualization.eda import EDAAnalyzer
from src.visualization.plots import PlotGenerator
from src.visualization.wordclouds import WordCloudGenerator


class TrainingPipeline:
    """
    Execute the complete data preparation pipeline.
    """

    def run(self) -> None:
        """
        Run the complete pipeline.
        """

        # Load datasets
        loader = DataLoader()
        datasets = loader.load_all_datasets()

        # Validate and merge datasets
        validator = DatasetValidator()

        merged_dataset = validator.prepare_dataset(
            fake_df=datasets["fake"],
            real_df=datasets["real"],
        )

        validator.save_dataset(merged_dataset)

        validator.dataset_statistics(
            merged_dataset
        )

        # Preprocess dataset
        preprocessor = DatasetPreprocessor()

        merged_dataset = preprocessor.process(
            merged_dataset
        )

        preprocessor.save(
            merged_dataset
        )

        # Exploratory Data Analysis
        eda = EDAAnalyzer(
            merged_dataset
        )

        eda.summary()

        # Plots
        plotter = PlotGenerator()

        plotter.plot_class_distribution(
            merged_dataset
        )

        plotter.plot_text_length_distribution(
            merged_dataset
        )

        plotter.plot_title_length_distribution(
            merged_dataset
        )

        # Word Clouds
        wordcloud = WordCloudGenerator()

        wordcloud.generate_wordcloud(
            merged_dataset,
            "FAKE",
        )

        wordcloud.generate_wordcloud(
            merged_dataset,
            "REAL",
        )

        wordcloud.top_words(
            merged_dataset,
            "FAKE",
        )

        wordcloud.top_words(
            merged_dataset,
            "REAL",
        )

        print("\n" + "=" * 60)
        print("Pipeline completed successfully.")
        print("=" * 60)