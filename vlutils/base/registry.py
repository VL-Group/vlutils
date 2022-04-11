"""Module of Registry."""
import functools
import os
import logging
from typing import Dict, Union, Generic, TypeVar

from vlutils.utils import pPrint

T = TypeVar("T")

__all__ = [
    "Registry"
]


class Registry(Generic[T]):
    """A registry. Inherit from it to create a lots of factories.

    Example:
    ```python
        # Inherit to make a factory.
        class Geometry(Registry):
            ...

        # Register with auto-key "Foo"
        @Geometry.register
        class Foo:
            ...

        # Register with manual-key "Bar"
        @Geometry.register("Bar")
        class Bar:
            ...

        instance = Geometry.get("Foo")()
        assert isinstance(instance, Foo)

        instance = Geometry["Bar"]()
        assert isinstance(instance, Bar)
    ```
    """
    _map: Dict[str, T]
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._map: Dict[str, T] = dict()

    @classmethod
    def register(cls, key):
        """Decorator for register anything into registry.

        Args:
            key (str): The key for registering an object.
        """
        if isinstance(key, str):
            def insert(value):
                cls._map[key] = value
                return value
            return insert
        else:
            cls._map[key.__name__] = key
            return key

    @classmethod
    def get(cls, key: str, logger: Union[logging.Logger, "vlutils.logger.LoggerBase"] = logging.root) -> T:
        """Get an object from registry.

        Args:
            key (str): The key for the registered object.
        """
        result = cls._map.get(key)
        if result is None:
            raise KeyError(f"No entry for {cls.__name__}. Avaliable entries are: {os.linesep + cls.summary()}.")
        elif isinstance(result, functools.partial):
            logger.debug("Get <%s.%s> from \"%s\".", result.func.__module__, result.func.__qualname__, cls.__name__)
        else:
            logger.debug("Get <%s.%s> from \"%s\".", result.__module__, result.__qualname__, cls.__name__)
        return result

    @classmethod
    def values(cls):
        """Get all registered objects."""
        return cls._map.values()

    @classmethod
    def keys(cls):
        """Get all registered keys."""
        return cls._map.keys()

    @classmethod
    def items(cls):
        """Get all registered key-value pairs."""
        return cls._map.items()

    @classmethod
    def summary(cls) -> str:
        """Get registry summary.
        """
        return pPrint({
            k: v.__module__ + "." + v.__name__ for k, v in cls._map.items()
        })
