from typing import Any, Iterable, List
import abc


class Handler(abc.ABC):
    accumulated: float = 0.0
    length: int = 0

    def __init__(self, format: str = r"%.2f"):
        super().__init__()
        self._format = format

    def to(self, device):
        pass

    @property
    def Length(self) -> int:
        return self.length

    @property
    def Accumulated(self) -> float:
        return self.accumulated

    @Length.setter
    def Length(self, length: int):
        self.length = length

    @Accumulated.setter
    def Accumulated(self, accumulated: float):
        self.accumulated = accumulated

    def __call__(self, *args: Any, **kwds: Any):
        results = self.handle(*args, **kwds)
        self.length += len(results)
        self.accumulated += sum(results)

    def __str__(self) -> str:
        return self._format % (self.accumulated / self.length)

    @abc.abstractmethod
    def handle(self, *args: Any, **kwds: Any) -> List[float]:
        raise NotImplementedError

class Meters:
    def __init__(self, handlers: Iterable[Handler]):
        self._handlers = list(h for h in handlers)

    def __call__(self, *args, **kwds):
        for handler in self._handlers:
            handler(*args, **kwds)

    def summary(self, reset: bool = False):
        return {
            handler.__class__.__name__: str(handler)
                for handler
                    in self._handlers
        }
