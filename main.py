"""
Project entry point.
"""

from src.pipeline.train_pipeline import TrainingPipeline


def main() -> None:
    """
    Execute the training pipeline.
    """

    pipeline = TrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()