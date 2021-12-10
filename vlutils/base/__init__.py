"""Package of Base classes.

Exports:
    DecoratorContextManager: Allow a context manager to be used as a decorator.
    Dictable: An interface with `state_dict()` and `load_state_dict()` method.
    Restorable: A class implements PyTorch-like state_dict.
    DataParallel: Modified `nn.DataParallel` runs class-method decorated by `parallelFunction` parallelly.
    parallelFunction: A decorator for class-methods that want to parallel-apply.
    Factory: Factory design mode.
    Module: Custom `torch.nn.Module` with function mapping.
"""
from .dataParallel import DataParallel, parallelFunction
from .decoratorContextManager import DecoratorContextManager
from .registry import Registry
from .restorable import Restorable, Dictable
from .module import Module
from .freqHook import FrequecyHook


__all__ = ["DataParallel",
           "parallelFunction",
           "Registry",
           "Restorable",
           "Dictable",
           "Module",
           "DecoratorContextManager",
           "FrequecyHook"]
