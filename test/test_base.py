import pytest

from torch import nn
import torch

from vlutils.base import DataParallel, Module, Registrar, Restorable
from vlutils.metrics.helpers import recursiveCompare


def _initToOnes(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.ones_(m.bias)


class Factory(Registrar):
    ...


@Factory.register
@Factory.register("AnotherKey")
class ForFactory:
    ...


class NewFactory(Factory):
    ...


@NewFactory.register("ForNewFactory")
@NewFactory.register("ForNewFactory1")
class ForNewFactory:
    ...


class NestedRestorable(Restorable):
    def __init__(self):
        super().__init__()
        self._t = torch.ones([3])


class SimpleRestorable(Restorable):
    def __init__(self):
        super().__init__()
        self._a = 3
        self._b = "str"
        self._c = NestedRestorable()


class TestBase:
    @pytest.fixture
    def getModule(self):
        class linear(Module):
            def __init__(self, d1in: int, d1out: int, d2in: int, d2out: int, bias: bool = True):
                super().__init__()
                self._linear1 = nn.Linear(d1in, d1out, bias)
                self._linear2 = nn.Linear(d2in, d2out, bias)
                self.apply(_initToOnes)

            def _forward(self, x1, x2):
                return self._linear1(x1), self._linear2(x2)

            @Module.register("_ff1")
            def _ff1(self, x):
                return self._linear1(x)

            @Module.register("_ff2")
            def _ff2(self, x):
                return self._linear2(x)

        return linear(3, 6, 7, 4)

    @pytest.fixture
    def getLegacyModule(self):
        class linear(nn.Module):
            def __init__(self, din: int, dout: int, bias: bool = True):
                super().__init__()
                self._linear = nn.Linear(din, dout, bias)
                self.apply(_initToOnes)

            def forward(self, x):
                return self._linear(x)
        return linear(3, 6), linear(7, 4)

    @pytest.fixture
    def getDataParallel(self, getLegacyModule):
        return DataParallel(getLegacyModule[0].cuda())

    @pytest.fixture
    def getLegacyDataParallel(self, getLegacyModule):
        return nn.DataParallel(getLegacyModule[0].cuda())

    @pytest.fixture
    def getInput(self):
        return torch.randn(100, 3), torch.randn(45, 7)

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="System not CUDA-enabled.")
    def testDataParallel(self, getInput, getDataParallel, getLegacyDataParallel):
        x = getInput[0].cuda()
        # Basic usage
        y1 = getDataParallel(x)
        y2 = getDataParallel(x)
        assert y1.shape == y2.shape
        assert nn.functional.mse_loss(y1, y2) < 1e-12

    @pytest.mark.slow
    def testModule(self, getInput, getModule, getLegacyModule):
        x1, x2 = getInput
        legacyModule1, legacyModule2 = getLegacyModule
        y1 = legacyModule1(x1)
        y2 = legacyModule2(x2)
        # Basic forward
        y11, y12 = getModule(x1, x2)
        # Mapped forward
        y21, y22 = getModule("_ff1", x1), getModule("_ff2", x2)
        assert nn.functional.mse_loss(y1, y11) < 1e-12
        assert nn.functional.mse_loss(y1, y21) < 1e-12
        assert nn.functional.mse_loss(y2, y12) < 1e-12
        assert nn.functional.mse_loss(y2, y22) < 1e-12

    def testFactory(self):
        # Mapping independency
        assert "ForFactory" not in NewFactory._map
        assert "ForNewFactory" not in Factory._map
        # Basic function
        assert ForFactory == Factory.get(ForFactory.__name__) == Factory.get("AnotherKey") == Factory["ForFactory"]
        assert ForNewFactory == NewFactory.get("ForNewFactory") == NewFactory.get("ForNewFactory1") == NewFactory["ForNewFactory"]

    def testRestorable(self):
        instance = SimpleRestorable()
        assert recursiveCompare(instance.state_dict(), {
            "_a": 3,
            "_b": "str",
            "_c": {
                "_t": torch.ones([3])
            }
        })
        instance.load_state_dict({
            "_a": "mm",
            "_b": 6,
            "_c": {
                "_t": torch.zeros([4])
            }
        })
        assert recursiveCompare(instance.state_dict(), {
            "_a": "mm",
            "_b": 6,
            "_c": {
                "_t": torch.zeros([4])
            }
        })
