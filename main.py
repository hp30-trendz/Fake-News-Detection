"""
Project entry point.
"""

from src.data.loader import DataLoader
from src.data.validator import DatasetValidator
from src.visualization.eda import EDAAnalyzer
from src.visualization.plots import PlotGenerator


def main() -> None:
    """Run the data ingestion pipeline."""

    loader = DataLoader()

    datasets = loader.load_all_datasets()

    validator = DatasetValidator()

    merged_dataset = validator.prepare_dataset(
        fake_df=datasets["fake"],
        real_df=datasets["real"],
    )

    validator.save_dataset(merged_dataset)

    validator.dataset_statistics(merged_dataset)

    eda = EDAAnalyzer(merged_dataset)

    eda.summary()

    plotter = PlotGenerator()
    plotter.plot_class_distribution(merged_dataset)

    plotter.plot_text_length_distribution(
        merged_dataset
    )

    plotter.plot_title_length_distribution(
        merged_dataset

    )

    print("\n")
    print("=" * 60)
    print("Dataset successfully prepared.")
    print("=" * 60)
    print(f"Shape : {merged_dataset.shape}")
    print("\nClass Distribution")
    print(merged_dataset["label"].value_counts())
    print("=" * 60)


if __name__ == "__main__":
    main()