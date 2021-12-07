import torch


__all__ = [
    "l2DistanceWithNorm",
    "l2Distance"
]


def l2DistanceWithNorm(A: torch.Tensor, B: torch.Tensor):
    diff = ((A.unsqueeze(1) - B) ** 2).sum(2)
    maxi, _ = diff.max(1, keepdim=True)
    norm = diff / maxi
    return norm


def l2Distance(A: torch.Tensor, B: torch.Tensor):
    return ((A.unsqueeze(1) - B) ** 2).sum(2)
