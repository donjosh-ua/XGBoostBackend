"""
Logging configuration for the application.
Provides standardized logging setup and helper functions.
"""
import logging
import os
import sys
from typing import Optional

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_DIR = os.path.join("app", "logs")


def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Set up a logger with the given name and level.
    
    Args:
        name (str): Name for the logger
        level (Optional[int], optional): Logging level. Defaults to None (uses LOG_LEVEL).
    
    Returns:
        logging.Logger: Configured logger
    """
    if level is None:
        level = LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger if they don't exist already
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


def setup_file_logger(name: str, filename: str, level: Optional[int] = None) -> logging.Logger:
    """
    Set up a logger with a file handler.
    
    Args:
        name (str): Name for the logger
        filename (str): Path to the log file
        level (Optional[int], optional): Logging level. Defaults to None (uses LOG_LEVEL).
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Get logger
    logger = setup_logger(name, level)
    
    # Create file handler
    file_path = os.path.join(LOG_DIR, filename)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level if level is not None else LOG_LEVEL)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    
    # Add file handler to logger if not already present
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)
    
    return logger


# Application loggers
app_logger = setup_logger("app")
xgboost_logger = setup_logger("xgboost")
nn_logger = setup_logger("neural_network")
data_logger = setup_logger("data_management") 