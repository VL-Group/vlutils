"""Module of Restorable class"""
from typing import Any, Dict
import abc


__all__ = [
    "Dictable",
    "Restorable"
]


class Dictable(abc.ABC):
    """An abstract class implements PyTorch-like state-dict."""

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Export current state-dict

        Returns:
            Dict[str, Any]: The state-dict.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(self, stateDict: Dict[str, Any]):
        """Load from state-dict

        Args:
            stateDict (Dict[str, Any]): State-dict to be loaded.
        """
        raise NotImplementedError


class Restorable(Dictable):
    """A class implements PyTorch-like state-dict.

    Example:
    ```python
        class Foo(Restorable):
            super().__init__()
            # all fields start with '_' will be stored in the state-dict.
            self._x = 3

        foo = Foo()
        # { "_x": 3 }
        foo.state_dict()
        foo.load_state_dict({ "_x": 1 })
    ```

    Attributes:
        valuesToSave (set): names of all attributes to be saved.
    """
    def __init__(self):
        self.valuesToSave = set()

    def state_dict(self):
        return {key: self.__dict__[key].state_dict() if callable(getattr(self.__dict__[key], "state_dict", None)) else self.__dict__[key] for key in self.valuesToSave}

    def load_state_dict(self, stateDict):
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                continue
            if callable(getattr(value, "load_state_dict", None)):
                value.load_state_dict(stateDict[key])  # pylint: disable = no-member
            else:
                self.__dict__[key] = stateDict[key]

    def __setattr__(self, name: str, value: Any):
        self.__dict__[name] = value
        if name.startswith("_"):
            self.valuesToSave.add(name)
