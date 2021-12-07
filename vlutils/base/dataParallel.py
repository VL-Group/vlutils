"""Module of modified `nn.DataParallel`."""
from itertools import chain
from typing import Callable, Optional, Union, List
import functools
import threading

import torch
from torch import nn
from torch.cuda._utils import _get_device_index
from torch._utils import ExceptionWrapper


__all__ = [
    "parallelFunction",
    "DataParallel"
]


def _get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(_get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(_get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def _parallel_apply(modules, funcName: str, inputs, kwargs_tup=None, devices=None):
    """Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):  # pylint: disable = redefined-builtin
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = _get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = getattr(module, funcName)(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:  # pylint: disable = broad-except
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


def parallelFunction(function: Callable):
    """A decorator for class-methods that want to parallel-apply.

    Example:
    ```python
        class Foo(nn.Module):
            ...

            # This function will be parallel-applied, just like forward(...)
            @parallelFunction
            def add(self, x, y):
                return x + y
    ```
    See also `DataParallel`.

    Args:
        function (Callable): The class-method to be decorated.

    Returns:
        Callable: The decorated class-method.
    """
    @functools.wraps(function)
    def wrappedFn(*args, **kwargs):
        return function(*args, **kwargs)
    return wrappedFn


class DataParallel(nn.DataParallel):
    """Customized `nn.DataParallel`.

    Example:
    ```python
        class Foo(nn.Module):
            ...

            # This function will be parallel-applied, just like forward(...)
            @parallelFunction
            def add(self, x, y):
                return x + y

        foo = Foo()
        foo = DataParallel(foo)
        # parallel-apply
        foo.add(3, 4)
    ```

    Args:
        module (nn.Module): Module to be parallelized.
        device_ids (Optional[List[Union[int, torch.device]]], optional): Devices to be placed on. Defaults to None.
        output_device (Union[int, torch.device], optional): Devices where results placed. Defaults to None.
        dim (int, optional): Which dimension of input to be splitted for parallel. Defaults to 0.
    """
    def __init__(self, module: nn.Module, device_ids: Optional[List[Union[int, torch.device]]] = None, output_device: Union[int, torch.device] = None, dim: int = 0):
        super().__init__(module, device_ids, output_device, dim)
        fns = [getattr(module, fn) for fn in dir(module) if callable(getattr(module, fn))]
        for fn in fns:
            if getattr(fn, "__wrapped__", None) is not None:
                setattr(self, fn.__name__, self._parallelFnWrapper(fn))

    def _parallelFnWrapper(self, func):
        @functools.wraps(func)
        def parallelApply(*inputs, **kwargs):
            if not self.device_ids:
                return func(*inputs, **kwargs)

            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError("module must have its parameters and buffers "
                                       "on device {} (device_ids[0]) but found one of "
                                       "them on device: {}".format(self.src_device_obj, t.device))

            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                return func(*inputs[0], **kwargs[0])
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = _parallel_apply(replicas, func.__name__, inputs, kwargs, self.device_ids[:len(replicas)])
            return self.gather(outputs, self.output_device)
        return parallelApply
