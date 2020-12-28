"""Package of Base classes.

Exports:
    DecoratorContextManager: Allow a context manager to be used as a decorator.
    Restorable: A class implements PyTorch-like state_dict.
    DataParallel: Modified `nn.DataParallel` runs class-method decorated by `parallelFunction` parallelly.
    parallelFunction: A decorator for class-methods that want to parallel-apply.
    Factory: Factory design mode.
"""
from .dataParallel import DataParallel, parallelFunction
from .decoratorContextManager import DecoratorContextManager
from .factory import Factory
from .restorable import Restorable
from .module import Module
