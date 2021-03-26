from collections.abc import MutableMapping
import math


class TransformedDict(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, freq, inDict):
        self.store = inDict.copy()
        self._freq = freq

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        result = math.log(key) / math.log(self._freq)
        if result.is_integer():
            result = int(result)
        return result


class FrequecyHook():
    def __init__(self, base: int, freqAndHooks: dict):
        self._hooks = TransformedDict(base, freqAndHooks)

    def __call__(self, step, *args, **kwArgs):
        hook = self._hooks.get(step, None)
        if hook is not None:
            return hook(*args, **kwArgs)
