"""
Logging utilities for the Polymarket analysis project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "polymarket_analysis",
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_default_logger() -> logging.Logger:
    """Get the default logger for the project."""
    try:
        from .config import config
        log_file = config.logs_dir / f"polymarket_analysis_{datetime.now().strftime('%Y%m%d')}.log"
        return setup_logger(log_file=log_file)
    except ImportError:
        # Fallback if config is not available (during initialization)
        return setup_logger(console_output=True)
