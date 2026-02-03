"""
Logging utilities for Smart Campus Security & Attendance 2.0
"""

import sys
from loguru import logger
from pathlib import Path


def setup_logger(log_file: str = "data/logs/app.log", level: str = "INFO"):
    """
    Setup logger configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    # Create logs directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    logger.info("Logger initialized")


if __name__ == "__main__":
    setup_logger()
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
