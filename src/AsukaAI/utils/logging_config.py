import os
import logging
from logging.handlers import RotatingFileHandler
import datetime as dt
from typing import Optional
from utils import constants

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
        self._fmt = fmt
        self._datefmt = datefmt
        self._formatters = {}
        for levelno, color in self.COLORS.items():
            log_fmt = color + self._fmt + self.RESET
            self._formatters[levelno] = logging.Formatter(log_fmt, datefmt=self._datefmt)
        self._default_formatter = logging.Formatter(self._fmt, datefmt=self._datefmt)

    def format(self, record: logging.LogRecord) -> str:
        formatter = self._formatters.get(record.levelno, self._default_formatter)
        return formatter.format(record)

# --- Helper Functions for Handler Setup ---
def _setup_file_handler(logger: logging.Logger, log_dir: str, max_file_size: int, backup_count: int) -> None:
    """Sets up the rotating file handler."""
    os.makedirs(log_dir, exist_ok=True)
    current_time = dt.datetime.now().strftime('%Y-%m-%d')
    log_filename = os.path.join(log_dir, f'log_{current_time}.log')
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=max_file_size, backupCount=backup_count, encoding='utf-8'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

def _setup_console_handler(logger: logging.Logger) -> None:
    """Sets up the console handler with custom formatting."""
    console_handler = logging.StreamHandler()
    console_formatter = CustomFormatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# --- Main Configuration Function ---
def configure_logger(name: str, log_dir: str = 'logs',
                     log_level: Optional[int] = None, # Allows explicit override
                     max_file_size: int = 5 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    """
    Configures and returns a logger instance.
    Log level defaults to DEBUG if constants.DEBUG_MODE is True, otherwise INFO.
    This can be overridden by passing the log_level argument.
    """
    logger = logging.getLogger(name)

    # Determine effective log level
    if log_level is None: # If no explicit level is passed, use DEBUG_MODE
        effective_log_level = logging.DEBUG if constants.DEBUG_MODE else logging.INFO
    else: # If an explicit level is passed, use it
        effective_log_level = log_level
    
    logger.setLevel(effective_log_level)

    if not logger.handlers: # Avoid adding handlers multiple times
        _setup_file_handler(logger, log_dir, max_file_size, backup_count)
        _setup_console_handler(logger)
    
    logger.propagate = False # Prevent propagation to root logger

    # Log the effective log level, useful for debugging the logger itself
    # This message will use the configured handlers and level.
    init_log_message = f"Logger '{name}' configured with level: {logging.getLevelName(logger.level)}."
    if log_level is None: # If level was determined by DEBUG_MODE
        init_log_message += f" (DEBUG_MODE is {'ON' if constants.DEBUG_MODE else 'OFF'})"
    else: # If level was explicitly passed
        init_log_message += " (Explicitly set)"
    
    # Use a temporary, basic config for this one message if logger not fully set up
    # or rely on it being logged correctly after setup. For simplicity:
    logger.log(logging.DEBUG if constants.DEBUG_MODE else logging.INFO, init_log_message)


    return logger