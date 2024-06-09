import os
import logging
import datetime as dt

class CustomFormatter(logging.Formatter):
    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def configure_logger(name, log_dir = 'logs') -> logging.Logger:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    current_time = dt.datetime.now().strftime('%Y-%m-%d')
    log_filename = f'{log_dir}/log_{current_time}.log'
    file_fmt = '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d %(message)s'
    logging.basicConfig(filename=log_filename, filemode='a', format=file_fmt)

    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(CustomFormatter(fmt))
    logger.addHandler(handler)
    return logger