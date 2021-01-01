"""test/conftest.py

DO NOT CHANGE FILE NAME"""
import pytest
import torch
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def testInit():
    torch.autograd.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
