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
        assert len(datas) > 0
        self.datas = datas

    def __len__(self):
        return len(self.datas[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datas)


class Enumerate(Dataset):
    """Wrapper dataset behaves like `enumerate()` with element-wise count.

    Args:
        data (Dataset): Any datasets.
    """
    def __init__(self, data: Dataset):
        assert len(data) > 0
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx]
