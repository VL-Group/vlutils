"""Module of DecoratorContextManager.

Originally from https://github.com/pytorch/pytorch/blob/671ee71ad4b6f507218d1cad278a8e743780b716/torch/autograd/grad_mode.py#L16
"""
import inspect
import functools
import abc


__all__ = [
    "DecoratorContextManager"
]


class DecoratorContextManager(abc.ABC):
    """Allow a context manager to be used as a decorator

    Example:
    ```python
        class Foo(DecoratorContextManager):
            ...

            def __enter__(self):
                ...

            def __exit__(self, exc_type, exc_val, exc_tb):
                ...

        def add(x, y):
            return a + b
        @Foo()
        def addD(x, y):
            return a + b

        # Equivalent
        with Foo():
            add(3, 4)
        # Equivalent
        addD(3, 4)
    ```
    """

    def __call__(self, func):
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_context

    def _wrap_generator(self, func):
        """Wrap each generator invocation with the context manager"""
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)
            while True:
                try:
                    with self:
                        x = next(gen)
                    yield x
                except StopIteration:
                    break
        return generator_context

    @abc.abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError
