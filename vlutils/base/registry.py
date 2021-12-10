"""Module of Registry."""
from typing import Any, Union
from vlutils.logger import pPrint


__all__ = [
    "Registry"
]


class _registryMeta(type):
    def __getitem__(self, idx):
        return self._map[idx]


class Registry(metaclass=_registryMeta):
    """A registry. Inherit from it to create a lots of factories.

    Example:
    ```python
        # Inherit to make a factory.
        class Geometry(Registry):
            ...

        @Geometry.register("Foo")
        class Foo:
            ...

        instance = Geometry.get("Foo")()
        assert isinstance(instance, Foo)
    ```
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._map = dict()

    @classmethod
    def register(cls, key: Union[str, Any]):
        """Decorator for register anything into registry.

        Args:
            key (str): The key for registering an object.
        """
        if isinstance(key, str):
            def insert(value: Any):
                cls._map[key] = value
                return value
            return insert
        else:
            cls._map[key.__name__] = key
            return key

    @classmethod
    def get(cls, key: str):
        """Get an object from registry.

        Args:
            key (str): The key for the registered object.
        """
        return cls._map[key]

    @classmethod
    def summary(cls) -> str:
        return pPrint({
            k: v.__module__ + v.__name__ for k, v in cls._map.items()
        })
