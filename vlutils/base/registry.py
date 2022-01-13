"""Module of Registry."""
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
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._map: Dict[str, T] = dict()

    @classmethod
    def register(cls, key: Union[str, T]) -> T:
        """Decorator for register anything into registry.

        Args:
            key (str): The key for registering an object.
        """
        if isinstance(key, str):
            def insert(value: T):
                cls._map[key] = value
                return value
            return insert
        else:
            cls._map[key.__name__] = key
            return key

    @classmethod
    def get(cls, key: str) -> T:
        """Get an object from registry.

        Args:
            key (str): The key for the registered object.
        """
        return cls._map[key]

    @classmethod
    def summary(cls) -> str:
        """Get registry summary.
        """
        return pPrint({
            k: v.__module__ + v.__name__ for k, v in cls._map.items()
        })
