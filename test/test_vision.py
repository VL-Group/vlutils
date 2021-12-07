import numpy as np
import torch
# import pytest
from skimage.color import rgb2hsv as sk_rgb2hsv, hsv2rgb as sk_hsv2rgb

from vlutils.vision.colorSpace import rgb2hsv, hsv2rgb


class TestDataset:
    def testRgb2Hsv(self):
        x = np.random.uniform(low=0.0, high=1.0, size=(1, 3, 255, 255))
        xT = torch.from_numpy(x)
        y = sk_rgb2hsv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        yT = rgb2hsv(xT)
        assert ((y - yT.numpy()) ** 2).sum() < 1e-12

    def testHsv2Rgb(self):
        x = np.random.uniform(low=0.0, high=1.0, size=(1, 3, 255, 255))
        xT = torch.from_numpy(x)
        y = sk_hsv2rgb(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        yT = hsv2rgb(xT)
        assert ((y - yT.numpy()) ** 2).sum() < 1e-12
