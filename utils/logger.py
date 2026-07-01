"""
utils/logger.py

Configures the application's root logger.

A single call to ``setup_logger()`` at startup ensures that every module
using ``logging.getLogger(__name__)`` automatically writes to both the
console and a rotating log file.
"""

import logging
import logging.handlers
from pathlib import Path


def setup_logger(
    name: str = "predictive_monitor",
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a named logger with console + file handlers.

    Args:
        name:    Logger name (appears in every log line).
        log_dir: Directory where ``<name>.log`` will be written.
        level:   Minimum log level (default INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Avoid adding duplicate handlers on repeated calls
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler (max 5 MB, keep 3 backups)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
