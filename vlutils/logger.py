"""Module of logging"""
import os
import sys
import warnings
import logging
import logging.config
from logging import LogRecord
import datetime
import multiprocessing
import time

from vlutils.base.decoratorContextManager import DecoratorContextManager
from .io import rotateItems


__all__ = [
    "WaitingBar",
    "LoggingDisabler",
    "configLogging"
]


class WaitingBar(DecoratorContextManager):
    """A CLI tool for printing waiting bar.

    Example:
    ```python
        @WaitingBar("msg")
        def longTime():
            # Long time operation
            ...

        with WaitingBar("msg"):
            # Long time operation
            ...
    ```

    Args:
        msg (str): Addtional message shows after bar.
        ncols (int): Total columns of bar.
    """
    def __init__(self, msg: str, ncols: int = 10):
        if ncols <= 8:
            raise ValueError("ncols must greater than 8, got %d", ncols)
        self._msg = msg
        self._ticker = None
        self._stillRunning = None
        self._ncols = ncols
        self.animation = list()
        # "       =       "
        template = (" " * (ncols + 1) + "=" * (ncols - 8) + " " * (ncols + 1))
        for i in range(2 * (ncols - 2)):
            start = 2 * (ncols - 2) - i
            end = 3 * (ncols - 2) - i
            self.animation.append("[" + template[start:end] + "]" + r" %s")

    def __enter__(self):
        self._stillRunning = multiprocessing.Value("b", True)
        self._ticker = multiprocessing.Process(name="waitingBarTicker", target=self._print, args=[self._stillRunning])
        self._ticker.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stillRunning.value = False
        self._ticker.join()
        print(" " * (len(self._msg) + self._ncols + 1), end="\r", file=sys.stderr)

    def _print(self, stillRunning: multiprocessing.Value):
        i = 0
        while bool(stillRunning.value):
            print(self.animation[i % len(self.animation)] % self._msg, end='\r', file=sys.stderr)
            time.sleep(.06)
            i += 1


class LoggingDisabler:
    """Disable or enable logging temporarily.

    Example:
    ```python
        # True -> disable logging, False -> enable logging
        with LoggingDisabler(logger, True):
            # Some operations
            ...
    ```

    Args:
        logger (logging.Logger): The target logger to interpolate.
        disable (bool): Whether to disable logging.
    """
    def __init__(self, logger: logging.Logger, disable: bool):
        self._logger = logger
        self._disable = disable
        self._previous_status = False

    def __enter__(self):
        if self._disable:
            self._previous_status = self._logger.disabled
            self._logger.disabled = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._disable:
            self._logger.disabled = self._previous_status


class _DeprecationFilter:
    def filter(self, record: LogRecord):
        if "depreca" in record.msg:
            return 0
        return 1


def configLogging(logDir: str, rootName: str = "", level: str = logging.INFO, logName: str = None, rotateLogs: int = 10, ignoreWarnings: list = None) -> logging.Logger:
    """Logger configuration.

    Args:
        logDir (str): Log files placed in this folder.
        rootName (str): Logger's root name.
        level (str): Minimal level that will be logged.
        useTqdm (bool, optional): Not used. Defaults to False.
        logName (str, optional): Log file name, if None, use the formatted current time. Defaults to None.
        rotateLogs (int, optional): Whether to perform rotate in this folder, -1 is don't rotate. Defaults to 10.
        ignoreWarnings (list, optional): Which warnings don't want to be logged. Defaults to None.

    Returns:
        logging.Logger: The configed logger.
    """
    os.makedirs(logDir, exist_ok=True)
    if rotateLogs > 0:
        rotateItems(logDir, rotateLogs)
    if logName is None:
        fPrefix = os.path.join(logDir, "{0}".format(datetime.datetime.now().strftime(r"%y%m%d-%H%M%S")))
    else:
        fPrefix = os.path.join(logDir, logName)
    logging_config = {
        "version": 1,
        "formatters": {
            "full": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "simple": {
                "format": "%(asctime)s - %(message)s",
                "datefmt": "%m/%d %H:%M:%S"
            }
        },
        "filters": {
            "deprecation": {
                "()": _DeprecationFilter
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "info_file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "full",
                "filename": f"{fPrefix}.log",
                "mode": "w"
            }
        },
        "loggers": {
            rootName: {
                "propagate": False,
                "level": level,
                "handlers": [
                    "console",
                    "info_file"
                ]
            }
        }
    }
    logging.config.dictConfig(logging_config)

    def handleException(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger = logging.getLogger(rootName)
        logger.exception(repr(exc_value), exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handleException

    def handleWarning(message, category, filename, lineno, file=None, line=None):
        _ = file
        logger = logging.getLogger(rootName)
        if ignoreWarnings is not None and category in ignoreWarnings:
            return
        logger.warning(warnings.formatwarning(message, category, filename, lineno, line))
    warnings.showwarning = handleWarning
    return logging.getLogger(rootName)
