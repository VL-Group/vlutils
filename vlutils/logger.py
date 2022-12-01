"""Module of logging"""
from typing import ClassVar, List, Optional, TypeVar, Union
import abc
import functools
import os
import sys
import warnings
import logging
import logging.config
from logging import LogRecord
import datetime
import multiprocessing
import time
import rich.logging
from rich.console import ConsoleRenderable
from rich.text import Text

from vlutils.base.decoratorContextManager import DecoratorContextManager
from .io import rotateItems
from .runtime import functionFullName


__all__ = [
    "WaitingBar",
    "LoggingDisabler",
    "configLogging",
    "readableSize"
]

T = TypeVar("T")

class LoggerBase(abc.ABC):
    @abc.abstractmethod
    def setLevel(self, *_, **__):
        raise NotImplementedError

    @abc.abstractmethod
    def debug(self, *_, **__):
        raise NotImplementedError

    @abc.abstractmethod
    def info(self, *_, **__):
        raise NotImplementedError

    @abc.abstractmethod
    def warning(self, *_, **__):
        raise NotImplementedError

    @abc.abstractmethod
    def warn(self, *_, **__):
        raise NotImplementedError

    @abc.abstractmethod
    def error(self, *_, **__):
        raise NotImplementedError

    @abc.abstractmethod
    def exception(self, *_, **__):
        raise NotImplementedError

    @abc.abstractmethod
    def critical(self, *_, **__):
        raise NotImplementedError

    @abc.abstractmethod
    def log(self, *_, **__):
        raise NotImplementedError


def trackingFunctionCalls(function: T, logger: Union[logging.Logger, LoggerBase] = logging.root) -> T:
    fullName = functionFullName(function)
    if isinstance(function, functools.partial):
        funcArgs = function.args
        funcKwArgs = function.keywords
    else:
        funcArgs = ()
        funcKwArgs = dict()
    def wrapper(*args, **kwArgs):
        allArgs = ", ".join(str(arg) for arg in (args + funcArgs))
        allkwArgs = ", ".join(f"{key}={value}" for key, value in {**kwArgs, **funcKwArgs}.items())
        if len(allArgs) > 0:
            logger.debug("Call %s(%s, %s)", fullName, allArgs, allkwArgs)
        else:
            logger.debug("Call %s(%s)", fullName, allkwArgs)
        return function(*args, **kwArgs)
    return wrapper


def readableSize(byteSize: int, floating: int = 2, binary: bool = True) -> str:
    """Convert bytes to human-readable string (like `-h` option in POSIX).

    Args:
        size (int): Total bytes.
        floating (int, optional): Floating point length. Defaults to 2.
        binary (bool, optional): Format as XB or XiB. Defaults to True.

    Returns:
        str: Human-readable string of size.
    """
    size = float(byteSize)
    unit = "B"
    if binary:
        for unit in ["B", "kiB", "MiB", "GiB", "TiB", "PiB"]:
            if size < 1024.0 or unit == "PiB":
                break
            size /= 1024.0
        return f"{size:.{floating}f}{unit}"
    for unit in ["B", "kB", "MB", "GB", "TB", "PB"]:
        if size < 1000.0 or unit == "PB":
            break
        size /= 1000.0
    return f"{size:.{floating}f}{unit}"


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


class KeywordRichHandler(rich.logging.RichHandler):
    KEYWORDS: ClassVar[Optional[List[str]]] = [
        r"(?P<green>\b([gG]ood|[bB]etter|[bB]est|[sS]uccess(|ful|fully))\b)",
        r"(?P<magenta>\b([bB]ase|[cC]all(|s|ed|ing)|[Mm]ount(|s|ed|ing))\b)",
        r"(?P<cyan>\b([mM]aster|nccl|NCCL|[mM]ain|···|[tT]otal|[tT]rain(|s|ed|ing)|[vV]alidate(|s|d)|[vV]alidat(|ing|ion)|[tT]est(|s|ed|ing))\b)",
        r"(?P<yellow>\b([lL]atest|[lL]ast|[sS]tart(|s|ed|ing)|[bB]egin(|s|ning)|[bB]egun|[cC]reate(|s|d|ing)|[gG]et(|s|ting)|[gG]ot|)\b)",
        r"(?P<red>\b([eE]nd(|s|ed|ing)|[fF]inish(|es|ed|ing)|[kK]ill(|s|ed|ing)|[iI]terrupt(|s|ed|ting)|[qQ]uit|QUIT|[eE]xit|EXIT|[bB]ad|[wW]orse|[sS]low(|er))\b)",
        r"(?P<italic>\b([aA]ll|[aA]ny|[nN]one)\b)"
    ]

    def render_message(self, record: LogRecord, message: str) -> ConsoleRenderable:
        use_markup = getattr(record, "markup", self.markup)
        message_text = Text.from_markup(message) if use_markup else Text(message)

        highlighter = getattr(record, "highlighter", self.highlighter)

        if self.KEYWORDS:
            for keyword in self.KEYWORDS:
                message_text.highlight_regex(keyword)
                # message_text.highlight_words(value, key, case_sensitive=False)

        if highlighter:
            message_text = highlighter(message_text)

        return message_text


def configLogging(logDir: Optional[str] = None, rootName: str = "", level: Union[str, int] = logging.INFO, logName: Optional[str] = None, rotateLogs: int = 10, ignoreWarnings: Optional[list] = None) -> logging.Logger:
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
    if logDir is not None:
        os.makedirs(logDir, exist_ok=True)
        if rotateLogs > 0:
            rotateItems(logDir, rotateLogs)
        if logName is None:
            fPrefix = os.path.join(logDir, "{0}".format(datetime.datetime.now().strftime(r"%y%m%d-%H%M%S")))
        else:
            fPrefix = os.path.join(logDir, logName)
        logFile = f"{fPrefix}.log"
    else:
        logFile = os.devnull
    logging_config = {
        "version": 1,
        "formatters": {
            "full": {
                "format": r"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "simple": {
                "format": r"%(asctime)s - %(message)s",
                "datefmt": r"%m/%d %H:%M:%S"
            }
        },
        "filters": {
            "deprecation": {
                "()": _DeprecationFilter
            }
        },
        "handlers": {
            "console": {
                "class": "vlutils.logger.KeywordRichHandler",
                "level": level,
                "rich_tracebacks": True,
                "tracebacks_show_locals": False,
                "log_time_format": r"%m/%d %H:%M",
                "markup": False,
                "enable_link_path": False
            },
            "info_file": {
                "class": "logging.FileHandler",
                "level": level,
                "formatter": "full",
                "filename": logFile,
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
