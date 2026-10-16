"""Shared tensor checks and the residual block used by the camera-1 codec."""

import torch
from torch import nn


def validate_tensor(value, sample_shape, name):
    if (
        value.ndim != len(sample_shape) + 1
        or tuple(value.shape[1:]) != tuple(sample_shape)
        or not len(value)
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
    ):
        raise ValueError(
            f"{name} must be finite float32 with shape (N, {sample_shape}), N > 0."
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.correction = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.correction(x)
