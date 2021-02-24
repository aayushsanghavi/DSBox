from enum import Enum
from logging import Logger, StreamHandler, Formatter, getLogger


class LoggingLevel(Enum):
    """
    Defines the enum for the logging levels
    """
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0


def get_logger(logger_name: str, level: LoggingLevel = LoggingLevel.INFO) -> Logger:
    logger = getLogger(logger_name)
    logger.setLevel(level.name)

    # create console handler and set level to debug
    ch = StreamHandler()
    ch.setLevel(level.name)

    # create formatter
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger
