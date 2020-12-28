"""Module of Saver"""
from typing import Any, Dict
import os
import logging
import shutil
import datetime
from logging import Logger

import torch
from torch.utils.tensorboard import SummaryWriter

from .io import rotateItems
from .config import Config

class Saver(SummaryWriter):
    """A class for load and save model"""
    NewestDir = "latest"
    def __init__(self, config: Config, saveDir: str, autoManage: bool = True, maxItems: int = 25, reserve: bool = False):
        """init

        Args:
            config (Config): The running config
            saveDir (str): Direcotry to save model
            autoManage (bool, optional): Auto create `latest` folder and rotate folders by time to save the new model. Defaults to True.
            maxItems (int, optional): Max old checkpoints to preserve. Defaults to 25.
            reserve (bool, optional): When autoManage is on and `latest` folder exists, continue to save file in current `latest` folder other than rotate it. Defaults to False.
        """
        if saveDir.endswith(self.NewestDir):
            autoManage = False

        if autoManage:
            if os.path.exists(os.path.join(saveDir, self.NewestDir)) and not reserve:
                shutil.move(os.path.join(saveDir, self.NewestDir), os.path.join(saveDir, datetime.datetime.now().strftime(r"%y%m%d-%H%M%S")))
            os.makedirs(os.path.join(saveDir, self.NewestDir), exist_ok=True)
            rotateItems(saveDir, maxItems)
            self._saveDir = os.path.join(saveDir, self.NewestDir)
        else:
            self._saveDir = saveDir
        super().__init__(self._saveDir)
        self._savePath = os.path.join(self._saveDir, self.NewestDir)
        if not reserve:
            self._dumpFile(saveDir, config)

    def _dumpFile(self, path: str, config):
        shutil.copytree(path, os.path.join(self._saveDir, "dump"), symlinks=True, ignore=lambda src, path: [x for x in path if x == "__pycache__"], ignore_dangling_symlinks=True)
        with open(os.path.join(self._saveDir, "config.json"), "w") as fp:
            fp.write(str(config))

    @property
    def SaveDir(self) -> str:
        """Return the current saving directory path"""
        return self._saveDir

    @property
    def SavePath(self) -> str:
        """Return the current saving ckpt absolute path"""
        return self._savePath

    def save(self, logger: Logger = None, **objs: Any):
        """Save anything

        Args:
            logger (Logger, optional): For logging. Defaults to None.
            **objs (Any): The saved items ordered by names.
        """
        saveDict = dict()
        for key, value in objs.items():
            if isinstance(value, torch.nn.DataParallel):
                saveDict[key] = value.module.state_dict()
            elif hasattr(value, "state_dict"):
                saveDict[key] = value.state_dict()
            else:
                saveDict[key] = value
        torch.save(saveDict, self._savePath)
        (logger or logging).debug("Successfully saved checkpoint with keys: %s", list(saveDict.keys()))

    @staticmethod
    def load(filePath, logger: Logger = None, **objs: Any) -> Dict[str, Any]:
        """Load from ckpt.

        Args:
            logger (Logger, optional): For logging. Defaults to None.
            **objs (Any): Anything to load by name. If it has `.load_state_dict()`, use this method. Else the loaded item will be placed in resulting dict.

        Returns:
            Dict[str, Any]: The loaded dict.
        """
        savedDict = torch.load(filePath)
        logger = logger or logging
        logger.debug("Load state_dict with keys:\r\n%s", savedDict.keys())
        for key, value in objs.items():
            stateDict = savedDict[key]
            if isinstance(value, torch.nn.DataParallel):
                value.module.load_state_dict(stateDict)
            elif callable(getattr(value, "load_state_dict", None)):
                value.load_state_dict(stateDict)
            else:
                value.data = stateDict
        return objs
