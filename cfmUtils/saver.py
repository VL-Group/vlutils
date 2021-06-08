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

    def __init__(self, saveDir: str, saveName: str = "saved.ckpt", config: Any = None, autoManage: bool = True, maxItems: int = 25, reserve: bool = False, dumpFile: str = None, logger: Logger = None):
        logger = logger or logging
        if saveDir.endswith(self.NewestDir):
            autoManage = False

        if autoManage:
            if os.path.exists(os.path.join(saveDir, self.NewestDir)) and not reserve:
                newDir = os.path.join(saveDir, datetime.datetime.now().strftime(r"%y%m%d-%H%M%S"))
                shutil.move(os.path.join(saveDir, self.NewestDir), newDir)
                logger.debug("Auto rename %s to %s", os.path.join(saveDir, self.NewestDir), newDir)
            os.makedirs(os.path.join(saveDir, self.NewestDir), exist_ok=True)
            if maxItems > 0:
                rotateItems(saveDir, maxItems)
            self._saveDir = os.path.join(saveDir, self.NewestDir)
        else:
            self._saveDir = saveDir
        super().__init__(self._saveDir)
        self._savePath = os.path.join(self._saveDir, saveName)
        logger.debug("Saver located at %s", self._saveDir)
        if config is not None:
            with open(os.path.join(self._saveDir, "config.yaml"), "w") as fp:
                yaml.dump(serialize(config), fp)
        if dumpFile is not None and not str.isspace(dumpFile) and os.path.exists(dumpFile):
            self._dumpFile(dumpFile)

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

    def save(self, logger: Logger = None, **objs: Any):
        """Save anything

        Args:
            logger (Logger, optional): For logging. Defaults to None.
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
        torch.save(saveDict, self._savePath)
        (logger or logging).debug("Successfully saved checkpoint with keys: %s", list(saveDict.keys()))

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
