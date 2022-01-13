from typing import Dict, Callable, Any, List, Tuple, Union


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
    def __init__(self, *freqAndHooks: Tuple[int, Callable]):
        self._hooks = dict()
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
                    results[key].append(fn(step, *args, **kwArgs))
        return results


class ChainHook:
    def __init__(self, *hooks: Union[Callable, None]) -> None:
        self._hooks: List[Callable] = [h for h in hooks if h is not None]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for hook in self._hooks:
            hook(*args, **kwds)
