"""Module of Factory."""
from typing import Any

__all__ = [
    "Factory"
]

class Factory:
    """A factory. Inherit from it to create a lots of factories.

    Example:
    ```python
        @Factory.register("Foo")
        class Foo:
            ...

        instance = Factory.get("Foo")()
        assert isinstance(instance, Foo)

        # Inherit to make other factories, or all value will be shared in the same factory.
        class Geometry(Factory):
            ...
    ```
    """
    _map = dict()

    @classmethod
    def register(cls, key: str):
        """Decorator for register anything into factory.

        Args:
            key (str): The key for registering an object.
        """
        def insert(value: Any):
            cls._map[key] = value
            return value
        return insert

    @classmethod
    def get(cls, key: str):
        """Get an object from factory.

        Args:
            key (str): The key for the registered object.
        """
        return cls._map[key]
