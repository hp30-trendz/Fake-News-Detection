"""
Project configuration module.
"""

from pathlib import Path


class Config:
    """
    Central configuration for project paths.
    """

    # Project root directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    # Project directories
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    IMAGES_DIR = PROJECT_ROOT / "images"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # Dataset files
    FAKE_DATASET = RAW_DATA_DIR / "Fake.csv"
    REAL_DATASET = RAW_DATA_DIR / "Real.csv"

    # Processed data
    OUTPUT_DATASET = (
        PROCESSED_DATA_DIR / "fake_news_dataset.csv"
    )

    VALIDATION_REPORT = (
        REPORTS_DIR / "data_validation_report.csv"
    )


# Create required directories automatically
REQUIRED_DIRECTORIES = [
    Config.DATA_DIR,
    Config.RAW_DATA_DIR,
    Config.PROCESSED_DATA_DIR,
    Config.MODELS_DIR,
    Config.REPORTS_DIR,
    Config.IMAGES_DIR,
    Config.LOGS_DIR,
]

for directory in REQUIRED_DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )