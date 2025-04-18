from typing import Optional, Dict, Any
import logging
import os
import sys


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    module_levels: Optional[Dict[str, str]] = None,
) -> None:
    """
    Configure the logging system for the entire application.

    Args:
        log_level: Default logging level for all modules (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to (in addition to console)
        log_format: Optional custom log format string
        module_levels: Optional dict mapping module names to their specific log levels
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Default format includes timestamp, level, and message with contextual information
    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

    # Basic configuration for root logger
    handlers = [logging.StreamHandler(sys.stdout)]

    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    # Configure root logger
    logging.basicConfig(level=numeric_level, format=log_format, handlers=handlers)

    # Set specific levels for modules if specified
    if module_levels:
        for module, level in module_levels.items():
            module_logger = logging.getLogger(module)
            module_logger.setLevel(getattr(logging, level.upper(), numeric_level))

    # Suppress overly verbose loggers from libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Name of the module, typically __name__ from the calling module

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
