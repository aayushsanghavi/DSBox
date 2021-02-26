from enum import Enum
from logging import Logger, StreamHandler, Formatter, getLogger


class LoggingLevel(Enum):
    """
    Defines the enum for the logging levels
    """
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10


def get_logger(logger_name: str, level: LoggingLevel = LoggingLevel.INFO) -> Logger:
    """
    Creates a configured instance of a logger object

    General view of different log levels and when to use them:
    DEBUG -> purely for debugging purposes. Usually set for verbosity
    INFO -> indicates normal processing information
    WARNING -> indicates that something failed but normal flow of execution can continue
    ERROR -> indicates that something failed and there is no way to continue the normal execution flow
    CRITICAL -> indicates a severe failure in the system and that the application will crash

    :param logger_name: name of the class / module using the logger object
    :param level: logging level used to display logs.
    Setting the value as INFO would mean that all log statements at and above level INFO will be logged.
    :return: configured logger object
    """

    logger = getLogger(logger_name)
    logger.setLevel(level.name)

    # create console handler and set level to debug
    handler = StreamHandler()
    handler.setLevel(level.name)

    # create formatter
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to handler
    handler.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(handler)
    return logger
