"""Package of datasets.

Exports:
    Zip: Wrapper dataset behaves like `zip()`.
    Enumerate: Wrapper dataset behaves like `enumerate()` with element-wise count.
"""
from .wrappers import Zip, Enumerate


__all__ = [
    "Zip",
    "Enumerate"
]
