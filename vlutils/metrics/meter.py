from typing import Any, Iterable, List
import abc


class Handler(abc.ABC):
    accumulated: float = 0.0
    length: int = 0

    def __init__(self, format: str = r"%.2f"):
        super().__init__()
        self._format = format

    def to(self, device: Any) -> "Handler":
        return self

    @property
    def ShowInSummary(self) -> bool:
        """True only if results can be represented in summary string.
        """
        return True

    @property
    def Length(self) -> int:
        return self.length

    @property
    def Accumulated(self) -> float:
        return self.accumulated

    @property
    def Result(self) -> float:
        return self.accumulated / self.length

    def __call__(self, *args: Any, **kwds: Any):
        results = self.handle(*args, **kwds)
        self.length += len(results)
        self.accumulated += sum(results)

    def __str__(self) -> str:
        return self._format % (self.accumulated / self.length)

    def __repr__(self) -> str:
        return self.__class__.__name__ + ": " + str(self)

    def reset(self):
        self.accumulated = 0.0
        self.length = 0

    @abc.abstractmethod
    def handle(self, *args: Any, **kwds: Any) -> List[float]:
        raise NotImplementedError

class Meters:
    def __init__(self, handlers: Iterable[Handler]):
        self._handlers = list(h for h in handlers)

    def __call__(self, *args, **kwds):
        for handler in self._handlers:
            handler(*args, **kwds)

    def reset(self):
        for handler in self._handlers:
            handler.reset()

    def summary(self, reset: bool = False):
        result = ", ".join(repr(handler) for handler in self._handlers if handler.ShowInSummary)
        if reset:
            self.reset()
        return result

    def results(self, reset: bool = False):
        result = {
            handler.__class__.__name__: handler.Result
                for handler
                    in self._handlers
        }
        if reset:
            self.reset()
        return result
