"""Module for custom nn.Module"""
from typing import Any, Union
from abc import abstractmethod
import types

import torch
from torch import nn


__all__ = [
    "Module"
]


class Module(nn.Module):
    """Custom nn.Module

    Example:
    ```python
        # Custom network
        class Net(Module):
            ...
            # Function used for normal forward
            @Module.register("forward")
            def _ff(self, ...):
                ...
            # Function used for computing loss
            @Module.register("loss")
            def _loss(self, ...):
                ...

        net = Net()
        # Call net._ff
        y = net("forward", x)
        # Call net._loss
        loss = net("loss", y, label)
    ```
    """
    @staticmethod
    def register(key):
        """Decorator for register forward function into module.

        Args:
            key (str): The key for registering a forward function.
        """
        def _wrapped(fn: Any):
            if isinstance(fn, property):
                fn.fget._vlutilsModuleMappedFunction = key
            else:
                fn._vlutilsModuleMappedFunction = key
            return fn
        return _wrapped

    def __init__(self):
        super().__init__()
        self._functions = dict()
        for methodname in dir(self):
            method = getattr(self, methodname)
            if isinstance(method, property):
                method = method.fget
            if hasattr(method, "_vlutilsModuleMappedFunction"):
                self._functions[method._vlutilsModuleMappedFunction] = method

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        # Redirect mapped function to new replica.
        replica._functions = {k: types.MethodType(v.__func__, replica) for k, v in replica._functions.items()}
        return replica

    @abstractmethod
    def _forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward(self, key: Union[torch.Tensor, str], *args: Any, **kwargs: Any) -> Any:
        """Custom forward function

        Args:
            key (str): Key of the target fuction.

        Returns:
            Any: Result.
        """
        if not isinstance(key, str):
            args = (key, ) + args
            return self._forward(*args, **kwargs)
        return self._functions[key](*args, **kwargs)
