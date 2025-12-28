"""Logging configuration using loguru.

This module sets up structured, colorized logging for the entire application.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    colorize: bool = True,
) -> None:
    """Configure loguru logger for the application.

    Sets up console logging with colors and optional file logging with rotation.

    Args:
        log_level: Logging level. Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        log_file: Path to log file. If None, only console logging is enabled.
        rotation: When to rotate log file (e.g., "10 MB", "1 day", "00:00").
        retention: How long to keep old log files (e.g., "7 days", "1 week").
        colorize: If True, use colored output in console.

    Example:
        >>> setup_logger(log_level="DEBUG", log_file="logs/app.log")
        >>> logger.info("Application started")
        >>> logger.debug("Debug message")
    """
    # Remove default logger
    logger.remove()

    # Console logger with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=colorize,
    )

    # File logger (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",  # Compress rotated files
        )

    logger.info(f"Logger configured - Level: {log_level}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")


def get_logger(name: str):
    """Get a logger instance with a specific name.

    Args:
        name: Name for the logger (typically __name__).

    Returns:
        Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module started")
    """
    return logger.bind(name=name)
