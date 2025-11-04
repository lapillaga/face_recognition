"""Logging configuration for face recognition pipeline.

This module provides structured logging with timestamps, module names,
and configurable log levels.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels (terminal only)."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors if terminal supports it."""
        # Only use colors if output is a terminal
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
                )
                record.name = f"{self.BOLD}{record.name}{self.RESET}"

        return super().format(record)


def setup_logging(
    name: str = "face_recognition",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup and configure logger with consistent formatting.

    Args:
        name: Logger name (usually module name or 'face_recognition' for root)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, reads from environment via Config.
        log_file: Optional file path to also log to a file.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logging(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("Failed to detect face", exc_info=True)
    """
    # Get or create logger
    logger = logging.getLogger(name)

    # If logger already has handlers, return it (avoid duplicate handlers)
    if logger.handlers:
        return logger

    # Determine log level
    if level is None:
        try:
            from app.config import get_config

            level = get_config().log_level
        except Exception:
            level = "INFO"

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)

    # Format: 2025-11-04 15:30:45 | INFO | module.name | Message
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    colored_formatter = ColoredFormatter(fmt, datefmt=date_fmt)
    console_handler.setFormatter(colored_formatter)

    logger.addHandler(console_handler)

    # Optional file handler (without colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logger.level)
        plain_formatter = logging.Formatter(fmt, datefmt=date_fmt)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoid duplicate messages)
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance.

    Example:
        >>> from app.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return setup_logging(name)