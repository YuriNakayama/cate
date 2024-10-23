import sys
from logging import Formatter, Logger, StreamHandler, getLogger, INFO


def get_logger(name: str, level: int = INFO) -> Logger:
    base_logger = getLogger()
    for handler in base_logger.handlers:
        base_logger.removeHandler(handler)

    logger = getLogger(name)
    logger.setLevel(level)
    formatter = Formatter(
        "%(levelname)s  %(asctime)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    std_handler = StreamHandler(sys.stdout)
    std_handler.setFormatter(formatter)
    logger.addHandler(std_handler)
    return logger