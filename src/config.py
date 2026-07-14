"""
Project configuration module.
"""

from pathlib import Path


class Config:
    """Central configuration for project paths."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DATA_DIR = PROJECT_ROOT / "data"

    RAW_DATA_DIR = DATA_DIR / "raw"

    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    MODELS_DIR = PROJECT_ROOT / "models"

    REPORTS_DIR = PROJECT_ROOT / "reports"

    IMAGES_DIR = PROJECT_ROOT / "images"

    LOGS_DIR = PROJECT_ROOT / "logs"

    FAKE_DATASET = RAW_DATA_DIR / "Fake.csv"

    REAL_DATASET = RAW_DATA_DIR / "Real.csv"

    OUTPUT_DATASET = PROCESSED_DATA_DIR / "fake_news_dataset.csv"

    VALIDATION_REPORT = REPORTS_DIR / "data_validation_report.csv"