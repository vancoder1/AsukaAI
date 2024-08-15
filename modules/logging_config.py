import os
import logging
from logging.handlers import RotatingFileHandler
import datetime as dt
from typing import Optional

class CustomFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: '\033[0;36m',  # Cyan
        logging.INFO: '\033[0;32m',   # Green
        logging.WARNING: '\033[0;33m',  # Yellow
        logging.ERROR: '\033[0;31m',  # Red
        logging.CRITICAL: '\033[0;35m'  # Magenta
    }
    RESET = '\033[0m'

    def __init__(self, fmt: str):
        super().__init__()
        self.fmt = fmt

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.COLORS.get(record.levelno, self.RESET) + self.fmt + self.RESET
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

class LogConfig:
    def __init__(self, log_dir: str = 'logs', log_level: int = logging.DEBUG,
                 max_file_size: int = 5 * 1024 * 1024, backup_count: int = 5):
        self.log_dir = log_dir
        self.log_level = log_level
        self.max_file_size = max_file_size
        self.backup_count = backup_count

    def configure_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)

        if not logger.handlers:
            self._setup_file_handler(logger)
            self._setup_console_handler(logger)

        return logger

    def _setup_file_handler(self, logger: logging.Logger) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

        current_time = dt.datetime.now().strftime('%Y-%m-%d')
        log_filename = f'{self.log_dir}/log_{current_time}.log'
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=self.max_file_size, backupCount=self.backup_count
        )
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    def _setup_console_handler(self, logger: logging.Logger) -> None:
        console_handler = logging.StreamHandler()
        console_formatter = CustomFormatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

def configure_logger(name: str, log_dir: str = 'logs', log_level: int = logging.DEBUG,
                     max_file_size: int = 5 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    log_config = LogConfig(log_dir, log_level, max_file_size, backup_count)
    return log_config.configure_logger(name)