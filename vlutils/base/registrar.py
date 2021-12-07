"""Module of Registrar."""
from typing import Any, Union


__all__ = [
    "Registrar"
]


class _RegistrarMeta(type):
    def __getitem__(self, idx):
        return self._map[idx]


class Registrar(metaclass=_RegistrarMeta):
    """A registrar. Inherit from it to create a lots of factories.

    Example:
    ```python
        # Inherit to make a factory.
        class Geometry(Registrar):
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
        """Decorator for register anything into registrar.

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
        """Get an object from registrar.

        Args:
            key (str): The key for the registered object.
        """
        return cls._map[key]
