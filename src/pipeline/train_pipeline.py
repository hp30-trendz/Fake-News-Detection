"""
Training pipeline for the Fake News Detection project.
"""

from __future__ import annotations

from pathlib import Path

from src.data.loader import DataLoader
from src.data.preprocessing import DatasetPreprocessor
from src.data.validator import DatasetValidator
from src.features.tfidf import TFIDFFeatureExtractor
from src.models.dataset_splitter import DatasetSplitter
from src.models.model_manager import ModelManager
from src.models.trainer import ModelTrainer
from src.visualization.eda import EDAAnalyzer
from src.visualization.plots import PlotGenerator
from src.visualization.wordclouds import WordCloudGenerator


class TrainingPipeline:
    """
    Execute the complete Fake News Detection pipeline.
    """

    def run(self) -> None:
        """
        Run the complete machine learning pipeline.
        """

        # ==========================================================
        # Load datasets
        # ==========================================================
        loader = DataLoader()

        datasets = loader.load_all_datasets()

        # ==========================================================
        # Validate datasets
        # ==========================================================
        validator = DatasetValidator()

        merged_dataset = validator.prepare_dataset(
            fake_df=datasets["fake"],
            real_df=datasets["real"],
        )

        validator.dataset_statistics(
            merged_dataset
        )

        # ==========================================================
        # Preprocess dataset
        # ==========================================================
        preprocessor = DatasetPreprocessor()

        merged_dataset = preprocessor.process(
            merged_dataset
        )

        preprocessor.save(
            merged_dataset
        )

        # ==========================================================
        # Split dataset
        # ==========================================================
        splitter = DatasetSplitter()

        X_train, X_test, y_train, y_test = splitter.split(
            merged_dataset
        )

        # ==========================================================
        # TF-IDF Feature Extraction
        # ==========================================================
        tfidf = TFIDFFeatureExtractor()

        X_train_features = tfidf.fit_transform(
            X_train
        )

        X_test_features = tfidf.transform(
            X_test
        )

        # ==========================================================
        # Train Models
        # ==========================================================
        trainer = ModelTrainer()

        trained_models, results = trainer.train(
            X_train_features,
            X_test_features,
            y_train,
            y_test,
        )

        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)
        print(results)

        # ==========================================================
        # Save evaluation results
        # ==========================================================
        metrics_dir = Path("models") / "metrics"

        metrics_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results.to_csv(
            metrics_dir / "model_results.csv",
            index=False,
        )

        # ==========================================================
        # Save best model
        # ==========================================================
        best_model_name = results.iloc[0]["Model"]

        model_manager = ModelManager()

        model_manager.save_model(
            trained_models[best_model_name],
            best_model_name.lower().replace(" ", "_"),
        )

        # ==========================================================
        # Save TF-IDF Vectorizer
        # ==========================================================
        tfidf.save()

        # ==========================================================
        # Exploratory Data Analysis
        # ==========================================================
        eda = EDAAnalyzer(
            merged_dataset
        )

        eda.summary()

        # ==========================================================
        # Visualizations
        # ==========================================================
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

        # ==========================================================
        # Word Clouds
        # ==========================================================
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
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)