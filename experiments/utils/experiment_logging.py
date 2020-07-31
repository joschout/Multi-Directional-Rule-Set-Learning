import logging
import sys

TargetAttr = str


def create_logger(logger_name: str, log_file_name: str) -> logging.Logger:
    # create formatters
    file_log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # create handlers
    file_handler = logging.FileHandler(log_file_name, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_log_formatter)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_log_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def create_stdout_logger(logger_name: str = None) -> logging.Logger:

    if logger_name is None:
        logger_name = "stdout_logger"

    # create formatters
    console_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # create handlers
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_log_formatter)

    # add the handlers to the logger
    logger.addHandler(console_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
