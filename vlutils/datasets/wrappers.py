"""Module of wrapper datasets."""
from torch.utils.data import Dataset


__all__ = [
    "Zip",
    "Enumerate"
]


class Zip(Dataset):
    """Wrapper dataset behaves like `zip()`.

    Args:
        *data (List of Dataset): A series of datasets.
    """
    def __init__(self, *datas: Dataset):
        super().__init__()
        self._data = datas

    def __len__(self):
        return len(self._data[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self._data)


class Enumerate(Dataset):
    """Wrapper dataset behaves like `enumerate()` with element-wise count.

    Args:
        data (Dataset): Any datasets.
        mode (str): The representation of the return indices.
                    "asis": -1 -> -1
                    "absolute": -1 -> len - 1
    """
    def __init__(self, data: Dataset, mode: str = "asis"):
        super().__init__()
        self._data = data
        if mode not in ["asis", "absolute"]:
            raise ValueError(f"Given mode not in ['asis', 'absolute'], got {mode}.")
        self._map = {
            "asis": self._asIs,
            "absolute": self._absolute
        }
        self._idxMapping = self._map[mode]

    def mode(self, mode: str = "asis"):
        if mode not in ["asis", "absolute"]:
            raise ValueError(f"Given mode not in ['asis', 'absolute'], got {mode}.")
        self._idxMapping = self._map[mode]

    def _asIs(self, idx):
        return idx

    def _absolute(self, idx):
        return (idx + len(self)) % len(self)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._idxMapping(idx), self._data[idx]
