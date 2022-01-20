"""Module of runtime utils"""
import functools
from typing import Callable, Union
from types import FunctionType, MethodType
import os
import logging
from pprint import pformat
import time
from typing import List, Dict, Tuple

from .types import StrPath

import torch
import pynvml


__all__ = [
    "preAllocateMem",
    "queryGPU",
    "gpuInfo",
    "Timer"
]

def inspectFunction(function: Callable) -> Callable:
    if isinstance(function, functools.partial):
        return function.func
    return function

def _fullName(function):
    if hasattr(function, "__qualname__"):
        return f"{function.__module__}.{function.__qualname__}"
    if hasattr(function, "__class__"):
        return f"{function.__module__}.{function.__class__.__name__}"
    return repr(function)

def functionFullName(function: Callable) -> str:
    if isinstance(function, functools.partial):
        return functionFullName(function.func)
    if isinstance(function, (FunctionType, MethodType)):
        return _fullName(function)
    if (function.__str__ is not object.__str__) or (type(function).__str__ is not object.__str__):
        return function.__str__()
    return _fullName(function)

def relativePath(path: StrPath, start: Union[StrPath, None] = None):
    result = os.path.relpath(path, start)
    if not result.startswith("../"):
        result = "./" + result
    return result

def preAllocateMem(memSize: int):
    """Pre-allocate VRAM in GPUs.

    Args:
        memSize (int): Preserved VRAM amount (MiB).
    """
    devices = torch.cuda.device_count()
    for d in range(devices):
        x = torch.rand((256, 1024, memSize), device=f"cuda:{d}")
        del x
    return


def gpuInfo() -> List[Dict[str, int]]:
    """Helper for list all gpus.

    Returns:
        List[Dict[str, int]]: A list of dicts { "memory.used": int, "memory.total": int }, sorted by `CUDA_DEVICE_ORDER`
    """
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    gpus = list()
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpus.append({"memory.used": info.used / 1048576, "memory.total": info.total / 1048576})
    return gpus


def queryGPU(wantsMore: bool = False, givenList: list = None, needGPUs: int = -1, needVRamEachGPU: int = -1, writeOSEnv: bool = True, logger: logging.Logger = None) -> List[Tuple[int, int]]:
    """Query GPUs that meet requirements.

    Example:
    ```python
        # Find a GPU has free VRAM >= 2000MiB
        queryGPU(needGPUs=1, needVRamEachGPU=2000)
        # Only when all GPUs are free, otherwise raise EnvironmentError
        queryGPU(needGPUs=-1, needVRamEachGPU=-1)
        # Don't write CUDA_VISIBLE_DEVICE, return real GPU ID
        gpuList = queryGPU(writeOSEnv=False)
        # A typical usage
        while True:
            try:
                queryGPU(needGPUs=3, needVRamEachGPU=8000)
                # Go to run
                break
            except EnvironmentError:
                # Wait 15s
                time.sleep(1500)
            # Begin GPU tasks
            run()
    ```

    Args:
        wantsMore (bool, optional): Wants at least `needGPUs` GPUs, if there are more GPUs, use them all. Defaults to False.
        givenList (list, optional): If given, only GPUs id in this list will be queried. Defaults to None.
        needGPUs (int, optional): How many GPUs take in demand. Defaults to -1.
        needVRamEachGPU (int, optional): It is OK to use a GPU if free VRAM (MiB) is larger than this threshold. Defaults to -1.
        writeOSEnv (bool, optional): Overwrite env parameter 'CUDA_VISIBLE_DEVICES'. Defaults to True.
        logger (logging.Logger, optional): For logging. Defaults to None.

    Raises:
        EnvironmentError: No GPU satisfied requirements.

    Returns:
        List[Tuple[int, int]]: List of available gpu id and free VRAM.
    """
    logger = logger or logging

    # keep the devices order same as in nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    gpus = gpuInfo()
    if needGPUs < 0:
        needGPUs = len(gpus)

    # logger.debug("\n" + str(gpus))
    if isinstance(givenList, list):
        it = givenList
        needGPUs = min(needGPUs, len(givenList))
    else:
        it = range(len(gpus))

    gpus = [(i, gpus[i]) for i in it]
    if wantsMore:
        gpus = sorted(gpus, key=lambda item: item[1]['memory.used'])

    gpuList = []
    for i, g in gpus:
        if needVRamEachGPU < 0:
            if g['memory.used'] < 64:
                # give space for basic vram
                gpuList.append((i, (g['memory.total'] - g['memory.used'] - 64)))
                logger.debug("adding gpu[%d] with %.2fMB free.", i, g['memory.total'] - g['memory.used'])
        elif g['memory.total'] - g['memory.used'] > needVRamEachGPU + 64:
            gpuList.append((i, (g['memory.total'] - g['memory.used'] - 64)))
            logger.debug("adding gpu[%d] with %.2fMB free.", i, g['memory.total'] - g['memory.used'])
        if len(gpuList) >= needGPUs and not wantsMore:
            break

    if len(gpuList) >= needGPUs:
        # keep order
        gpuList = sorted(gpuList, key=lambda item: item[0])
        logger.debug("Found %d %s satisfied with args %s.", len(gpuList), "gpu" if len(gpuList) == 1 else "gpus", pformat(
            {
                "wantsMore": wantsMore,
                "givenList": givenList,
                "needGPUs": needGPUs,
                "needVRamEachGPU": needVRamEachGPU
            }, indent=4
        ))
        if writeOSEnv:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, [item[0] for item in gpuList]))
            newGPUList = []
            j = 0
            for i, mem in gpuList:
                newGPUList.append((j, mem))
                j += 1
            gpuList = newGPUList
        else:
            try:
                os.environ.pop("CUDA_VISIBLE_DEVICES")
            except KeyError:
                pass
        return gpuList
    else:
        raise EnvironmentError("Current system status is not satisfied")


class Timer:
    """A simple timer

    Example:
    ```python
        # Start the timer
        timer = Timer()
        ...
        # Last interval and total spent from start
        # 3.0,    3.0
        interval, total = timer.Tick()
        ...
        # 4.4,    7.4
        interval, total = timer.Tick()
    ```
    """
    def __init__(self):
        self._initial = self._tick = time.time()

    def tick(self):
        tock = time.time()
        interval = tock - self._tick
        self._tick = tock
        return interval, self._tick - self._initial
