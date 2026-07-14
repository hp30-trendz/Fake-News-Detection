"""
Logging configuration.
"""

from pathlib import Path
import logging


def setup_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger.
    """

    log_directory = Path("logs")

    log_directory.mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_directory / "project.log",
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger