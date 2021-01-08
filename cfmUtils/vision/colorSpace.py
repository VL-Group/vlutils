import torch


__all__ = [
    "rgb2hsv",
    "hsv2rgb"
]


def rgb2hsv(x: torch.Tensor, mode: str = "normal", eps: float = 1e-8) -> torch.Tensor:
    """Map rgb color space to HSV color space

    before call this function, rgb should first reshape to [N, C, H, W] and normalize to [0, 1]

    Args:
        x (torch.Tensor): [N, C, H, W] Image tensor.
        mode (str): Routine to comute hsv, one of "normal", "advance".
        eps (float, optional): Epsilon for small values. Defaults to 1e-8.

    returns:
        torch.Tensor: [N, C, H, W] Converted tensor.
    """
    return {
        "normal": _hsvNormal,
        "advance": _hsvAdvance
    }[mode](x, eps)


def _hsvNormal(x: torch.Tensor, eps: float) -> torch.Tensor:
    cMax, maxIdx = x.max(1)
    cMin, minIdx = x.min(1)

    sub = cMax - cMin

    # don't need requires_grad
    hsv = torch.empty_like(x)

    idx = maxIdx == 0
    hsv[:, 0][idx] = (x[:, 1][idx] - x[:, 2][idx]) / sub[idx]

    idx = maxIdx == 1
    hsv[:, 0][idx] = 2. + (x[:, 2][idx] - x[:, 0][idx]) / sub[idx]

    idx = maxIdx == 2
    hsv[:, 0][idx] = 4. + (x[:, 0][idx] - x[:, 1][idx]) / sub[idx]

    hsv[:, 0] = hsv[:, 0] / 6. % 1.
    hsv[:, 0][maxIdx == minIdx] = 0.0

    s = 1 - cMin / cMax
    s[cMax.abs() < eps] = 0.0

    hsv[:, 1] = s
    hsv[:, 2] = cMax

    return hsv


def _hsvAdvance(x: torch.Tensor, eps: float) -> torch.Tensor:
    raise NotImplementedError


def hsv2rgb(x: torch.Tensor, mode: str = "normal", eps: float = 1e-8) -> torch.Tensor:
    """Map rgb color space to HSV color space

    before call this function, rgb should first reshape to [N, C, H, W] and normalize to [0, 1]

    Args:
        x (torch.Tensor): [N, C, H, W] Image tensor.
        mode (str): Routine to comute hsv, one of "normal", "advance".
        eps (float, optional): Epsilon for small values. Defaults to 1e-8.

    returns:
        torch.Tensor: [N, C, H, W] Converted tensor.
    """
    return {
        "normal": _rgbNormal,
        "advance": _rgbAdvance
    }[mode](x, eps)


def _rgbNormal(x: torch.Tensor, eps: float) -> torch.Tensor:
    hi = (x[:, 0] * 6).floor()
    f = x[:, 0] * 6 - hi
    p = x[:, 2] * (1 - x[:, 1])
    q = x[:, 2] * (1 - f * x[:, 1])
    t = x[:, 2] * (1 - (1 - f) * x[:, 1])
    v = x[:, 2]

    hi = torch.stack([hi, hi, hi], axis=1).byte() % 6
    return ((hi == 0) * torch.stack((v, t, p), axis=1)) + ((hi == 1) * torch.stack((q, v, p), axis=1)) + ((hi == 2) * torch.stack((p, v, t), axis=1)) + ((hi == 3) * torch.stack((p, q, v), axis=1)) + ((hi == 4) * torch.stack((t, p, v), axis=1)) + ((hi == 5) * torch.stack((v, p, q), axis=1))


def _rgbAdvance(x: torch.Tensor, eps: float) -> torch.Tensor:
    raise NotImplementedError
