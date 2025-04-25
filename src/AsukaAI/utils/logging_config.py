import os
import logging
from logging.handlers import RotatingFileHandler
import datetime as dt
from typing import Optional

# --- Custom Formatter ---

class CustomFormatter(logging.Formatter):
    """Adds color to console log messages based on level."""
    COLORS = {
        logging.DEBUG: '\033[0;36m',  # Cyan
        logging.INFO: '\033[0;32m',   # Green
        logging.WARNING: '\033[0;33m',  # Yellow
        logging.ERROR: '\033[0;31m',  # Red
        logging.CRITICAL: '\033[0;35m'  # Magenta
    }
    RESET = '\033[0m'

    def __init__(self, fmt: str, datefmt: Optional[str] = None):
        # Store the original format string and date format
        self._fmt = fmt
        self._datefmt = datefmt
        # Initialize formatters for each level, including the default
        self._formatters = {}
        for levelno, color in self.COLORS.items():
            log_fmt = color + self._fmt + self.RESET
            self._formatters[levelno] = logging.Formatter(log_fmt, datefmt=self._datefmt)
        # Default formatter for levels without specific colors
        self._default_formatter = logging.Formatter(self._fmt, datefmt=self._datefmt)


    def format(self, record: logging.LogRecord) -> str:
        # Choose the formatter based on the record's level
        formatter = self._formatters.get(record.levelno, self._default_formatter)
        return formatter.format(record)

# --- Helper Functions for Handler Setup ---

def _setup_file_handler(logger: logging.Logger, log_dir: str, max_file_size: int, backup_count: int) -> None:
    """Sets up the rotating file handler."""
    os.makedirs(log_dir, exist_ok=True)

    current_time = dt.datetime.now().strftime('%Y-%m-%d')
    log_filename = os.path.join(log_dir, f'log_{current_time}.log') # Use os.path.join
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=max_file_size, backupCount=backup_count, encoding='utf-8' # Add encoding
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s', # Adjusted padding
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

def _setup_console_handler(logger: logging.Logger) -> None:
    """Sets up the console handler with custom formatting."""
    console_handler = logging.StreamHandler()
    # Use the CustomFormatter directly
    console_formatter = CustomFormatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', # Adjusted padding
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# --- Main Configuration Function ---

def configure_logger(name: str, log_dir: str = 'logs', log_level: int = logging.DEBUG,
                     max_file_size: int = 5 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    """
    Configures and returns a logger instance with file and console handlers.

    Args:
        name (str): The name of the logger (usually __name__).
        log_dir (str): Directory to store log files.
        log_level (int): The minimum logging level (e.g., logging.DEBUG, logging.INFO).
        max_file_size (int): Maximum size of a log file before rotation (in bytes).
        backup_count (int): Number of backup log files to keep.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # Prevent adding handlers multiple times if called again for the same logger name
    if not logger.handlers:
        _setup_file_handler(logger, log_dir, max_file_size, backup_count)
        _setup_console_handler(logger)

    # Prevent propagation to root logger if it has handlers (like basicConfig)
    logger.propagate = False

    return logger