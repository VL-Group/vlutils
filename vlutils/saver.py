"""Module of Saver"""
from typing import Any, Dict
import os
import logging
import shutil
import datetime
from logging import Logger
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from .io import rotateItems
from .config import serialize


__all__ = [
    "Saver"
]


class Saver(SummaryWriter):
    """A class for load and save model

    Example:
    ```python
        #  Save path: `saved/myModel`.
        #     config: Any object that can dump to yaml.
        # autoManage: If True, rename `latest` folder to current time and recreate `latest` folder.
        #   maxItems: Keep items (files/folders) in the save path no more than it. If <= 0, ignore.
        #    reserve: When autoManage is on and `latest` folder exists, continue to save file in current `latest` folder other than rotate it.
        #   dumpFile: Dump source code and config to the save path.
        saver = Saver("saved/myModel", config, autoManage=True, maxItems=10, reserve=False, dumpFile="mySoureCode/path")
    ```

    Args:
        saveDir (str): Direcotry to save model
        config (Any): The config that can dump to yaml
        autoManage (bool, optional): Auto create `latest` folder and rotate folders by time to save the new model. Defaults to True.
        maxItems (int, optional): Max old checkpoints to preserve. Defaults to 25.
        reserve (bool, optional): When autoManage is on and `latest` folder exists, continue to save file in current `latest` folder other than rotate it. Defaults to False.
        dumpFile (str, optional): The path of any folder want to save.
    """
    NewestDir = "latest"

    @staticmethod
    def composePath(saveDir: str, saveName: str = "saved.ckpt", autoManage: bool = True):
        if saveDir.endswith(Saver.NewestDir):
            autoManage = False
        if autoManage:
            _saveDir = os.path.join(saveDir, Saver.NewestDir)
        else:
            _saveDir = saveDir
        return os.path.join(_saveDir, saveName)

    def __init__(self, saveDir: str, saveName: str = "saved.ckpt", logger: Logger = None, config: Any = None, autoManage: bool = True, maxItems: int = 25, reserve: bool = False, dumpFile: str = None):
        self.setLogger(logger)
        if saveDir.endswith(self.NewestDir):
            autoManage = False

        if autoManage:
            if os.path.exists(os.path.join(saveDir, self.NewestDir)) and not reserve:
                newDir = os.path.join(saveDir, datetime.datetime.now().strftime(r"%y%m%d-%H%M%S"))
                shutil.move(os.path.join(saveDir, self.NewestDir), newDir)
                self.debug("Auto rename %s to %s", os.path.join(saveDir, self.NewestDir), newDir)
            os.makedirs(os.path.join(saveDir, self.NewestDir), exist_ok=True)
            if maxItems > 0:
                rotateItems(saveDir, maxItems)
            self._saveDir = os.path.join(saveDir, self.NewestDir)
        else:
            self._saveDir = saveDir
        super().__init__(self._saveDir)
        self._savePath = os.path.join(self._saveDir, saveName)
        self.debug("Saver located at %s", self._saveDir)
        if config is not None:
            with open(os.path.join(self._saveDir, "config.yaml"), "w") as fp:
                yaml.dump(serialize(config), fp)
        if dumpFile is not None and not str.isspace(dumpFile) and os.path.exists(dumpFile):
            self._dumpFile(dumpFile)

    @property
    def Logger(self) -> logging.Logger:
        return self._logger

    @Saver.Logger
    def setLogger(self, logger: logging.Logger):
        logger = logger or logging
        self._logger = logger

    def setLevel(self, level):
        """
        Set the logging level of this logger.  level must be an int or a str.
        """
        self._logger.setLevel(level)

    def debug(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'ERROR'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.error("Houston, we have a %s", "major problem", exc_info=1)
        """
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """
        Convenience method for logging an ERROR with exception information.
        """
        self._logger.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'CRITICAL'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.critical("Houston, we have a %s", "major disaster", exc_info=1)
        """
        self._logger.critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """
        Log 'msg % args' with the integer severity 'level'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.log(level, "We have a %s", "mysterious problem", exc_info=1)
        """
        self._logger.log(level, msg, *args, **kwargs)


    def _dumpFile(self, path: str):
        shutil.copytree(path, os.path.join(self._saveDir, "dump"), symlinks=True, ignore=lambda src, path: [x for x in path if x == "__pycache__"], ignore_dangling_symlinks=True)

    @property
    def SaveDir(self) -> Path:
        """Return the current saving directory path"""
        return Path(self._saveDir)

    @property
    def SavePath(self) -> Path:
        """Return the current saving ckpt absolute path"""
        return Path(self._savePath)

    def moveTo(self, dest: str):
        """Move current saving dir to dest

        Args:
            dest (str): The path of destination dir, execute by shutil.move().
        """
        self.flush()
        self.close()
        shutil.move(self._saveDir, dest)
        self._saveDir = dest
        self._savePath = os.path.join(dest, self.SavePath.name)
        self.log_dir = self._saveDir

    def save(self, path: str = None, **objs: Any):
        """Save anything

        Args:
            **objs (Any): The saved items ordered by names.
        """
        saveDict = dict()
        for key, value in objs.items():
            if isinstance(value, torch.nn.Module):
                saveDict[key] = value.state_dict()
            elif hasattr(value, "state_dict"):
                saveDict[key] = value.state_dict()
            else:
                saveDict[key] = value
        torch.save(saveDict, path or self._savePath)
        self.debug("Successfully saved checkpoint with keys: %s", list(saveDict.keys()))

    @staticmethod
    def load(filePath: str, mapLocation: Dict[str, str] = None, strict: bool = True, logger: Logger = None, **objs: Any) -> Dict[str, Any]:
        """Load from ckpt.

        Args:
            filePath (str): The destination path to load ckpt.
            mapLocation (Dict[str, str], optional): See torch.load(mapLocation). Defaults to None.
            strict (bool, optional): See torch.load_state_dict(strict). Defaults to True.
            logger (Logger, optional): For logging. Defaults to None.
            **objs (Any): Anything to load by name. If it has `.load_state_dict()`, use this method. Else the loaded item will be placed in resulting dict.

        Returns:
            Dict[str, Any]: The loaded dict.
        """
        savedDict = torch.load(filePath, map_location=mapLocation)
        logger = logger or logging
        logger.debug("Load state_dict with keys:\r\n%s", savedDict.keys())
        for key, value in objs.items():
            stateDict = savedDict[key]
            if isinstance(value, torch.nn.Module):
                value.load_state_dict(stateDict, strict=strict)
            elif callable(getattr(value, "load_state_dict", None)):
                value.load_state_dict(stateDict)
            else:
                if isinstance(value, torch.Tensor):
                    value.data = stateDict
                else:
                    objs[key] = stateDict
        return objs
