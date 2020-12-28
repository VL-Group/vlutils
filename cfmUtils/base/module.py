"""Module for custom nn.Module"""
from typing import Any

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
                fn.fget._cfmUtilsModuleMappedFunction = key
            else:
                fn._cfmUtilsModuleMappedFunction = key
            return fn
        return _wrapped

    def __init__(self):
        super().__init__()
        self._functions = dict()
        for methodname in dir(self):
            method = getattr(self, methodname)
            if isinstance(method, property):
                method = method.fget
            if hasattr(method, "_cfmUtilsModuleMappedFunction"):
                self._functions[method._cfmUtilsModuleMappedFunction] = method

    def forward(self, key: str, *args, **kwargs) -> Any:
        """Custom forward function

        Args:
            key (str): Key of the target fuction.

        Returns:
            Any: Result.
        """
        return self._functions[key](*args, **kwargs)
