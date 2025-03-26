"""
Logging utilities for Finite Monkey Engine
"""

import os
import logging
import sys
from typing import Optional

# Create the logger
logger = logging.getLogger("finite-monkey")

def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Set up logger configuration
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Optional log format string
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format if none provided
    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    # Set level for our package logger
    logger.setLevel(numeric_level)
    
    # Add colored output for console if rich is available
    try:
        from rich.logging import RichHandler
        # Remove default handler and add rich handler
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RichHandler):
                logger.removeHandler(handler)
        
        rich_handler = RichHandler(rich_tracebacks=True, markup=True)
        logger.addHandler(rich_handler)
        logger.info("Rich logging enabled")
    except ImportError:
        pass