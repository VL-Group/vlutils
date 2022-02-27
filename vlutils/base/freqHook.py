import logging
from typing import Dict, Callable, Any, List, Tuple, Union

from vlutils.runtime import functionFullName

__all__ = [
    "FrequecyHook"
]


class FrequecyHook:
    """A hook to call function by frequency

    Example:
    ```python
        def foo(*, bar, **_):
            print(bar)

        def lorem(*, ipsum, **_):
            print(ipsum)

        hook = FrequecyHook({
            2: foo,
            6: lorem
        })

        for i in range(1, 10):
            hook(i, bar=f"Hello bar{i}.", ipsum=f"Hello ipsum{i}.")

    >>> Hello bar2.
    Hello bar4.
    Hello bar6.
    Hello ipsum6.
    Hello bar8.
    ```

    Args:
        freqAndHooks (Dict[int, Callable]): The function (`value`) to call every `key` steps.
    """
    def __init__(self, *freqAndHooks: Tuple[int, Callable], logger: Union[logging.Logger, "vlutils.logger.LoggerBase"]=logging):
        self._hooks: Dict[int, List[Callable]] = dict()
        self._logger = logger
        for key, value in freqAndHooks:
            if key not in self._hooks:
                self._hooks[key] = list()
            self._hooks[key].append(value)

    def extend(self, *freqAndHooks: Tuple[int, Callable]):
        for key, value in freqAndHooks:
            if key not in self._hooks:
                self._hooks[key] = list()
            self._hooks[key].append(value)

    def append(self, freq: int, hook: Callable):
        if freq not in self._hooks:
            self._hooks[freq] = list()
        self._hooks[freq].append(hook)

    def remove(self, freq: int):
        self._hooks.pop(freq)

    def __call__(self, step: int, *args: Any, **kwArgs: Any) -> Dict[int, Any]:
        """Check whether the step % key == 0, if True, call value by args and kwArgs.

        Args:
            step (int): Current step.

        Returns:
            Dict[int, Any]: if the function is called, add its key and return value into this dict.
        """
        results = dict()
        for key, value in self._hooks.items():
            if step % key == 0:
                results[key] = list()
                for fn in value:
                    fullName = functionFullName(fn)
                    self._logger.debug("Call %s(...) by FrequecyHook@%d", fullName, key)
                    results[key].append(fn(step, *args, **kwArgs))
        return results

    def __str__(self) -> str:
        pretty = { f"{key}": [f"<{fullName}>" for fullName in (functionFullName(func) for func in value)] for key, value in self._hooks.items() }
        result = ""
        for key, value in pretty.items():
            value = ", ".join(value)
            result += f"    {key}:\r\n        [{value}]\r\n"
        return f"FrequencyHook(\r\n{result})"


class ChainHook:
    def __init__(self, *hooks: Union[Callable, None]) -> None:
        self._hooks: List[Callable] = [h for h in hooks if h is not None]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for hook in self._hooks:
            hook(*args, **kwds)

    def __str__(self) -> str:
        hookNames = [functionFullName(h) for h in self._hooks]
        hookNames = ",\r\n    ".join(hookNames)
        return f"ChainHook(\r\n    {hookNames})"
