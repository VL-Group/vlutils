import numpy as np
import pytest

from vlutils.datasets import Zip, Enumerate


class TestDataset:
    @pytest.fixture
    def getData(self):
        return np.arange(0, 5), np.arange(1, 6), np.arange(0, 10).reshape((-1, 2))

    def testZip(self, getData):
        a, b, c = getData
        zipDataset = Zip(a, b, c)
        assert len(zipDataset[0]) == 3
        aa, bb, cc = zipDataset[-1]
        assert aa == 4 and bb == 5 and np.array_equal(np.array([8, 9]), cc)
        assert len(zipDataset) == 5

    def testEnumerate(self, getData):
        a, b, c = getData
        zipDataset = Zip(a, b, c)
        enumDataset = Enumerate(zipDataset, "asis")
        assert len(enumDataset) == 5
        i, (aa, bb, cc) = enumDataset[-1]
        assert aa == 4 and bb == 5 and np.array_equal(np.array([8, 9]), cc) and i == -1

        enumDataset = Enumerate(zipDataset, "absolute")
        assert len(enumDataset) == 5
        i, (aa, bb, cc) = enumDataset[-1]
        assert aa == 4 and bb == 5 and np.array_equal(np.array([8, 9]), cc) and i == 4
        i, (aa, bb, cc) = enumDataset[2]
        assert aa == 2 and bb == 3 and np.array_equal(np.array([4, 5]), cc) and i == 2
