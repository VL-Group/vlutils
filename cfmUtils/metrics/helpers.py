from typing import Any

import torch
import numpy as np


__all__ = [
    "recursiveCompare"
]


def recursiveCompare(a: Any, b: Any) -> bool:
    if isinstance(a, (list, set, tuple)):
        return isinstance(b, (list, set, tuple)) and all(recursiveCompare(x, y) for x, y in zip(a, b))
    elif isinstance(a, dict):
        if not isinstance(b, dict):
            return False
        for key, value in a.items():
            if key not in b:
                return False
            if not recursiveCompare(value, b[key]):
                return False
        return True
    else:
        if type(a) != type(b):
            return False
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        return a == b
