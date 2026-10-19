"""Image-to-vector blocks. Each loads only its own encoder and normalization."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ._images import ResidualBlock, validate_tensor


class _ImageNetwork(nn.Module):
    def __init__(self, shape, latent, residual):
        super().__init__()
        self.shape = tuple(shape)
        self.padded = tuple(math.ceil(v / 16) * 16 for v in shape)
        widths = (1, 8, 16, 24, 32)
        self.encoder = nn.Sequential(
            *[
                layer
                for a, b in zip(widths, widths[1:])
                for layer in (nn.Conv2d(a, b, 3, stride=2, padding=1), nn.SiLU())
            ]
        )
        self.encoder_residuals = nn.ModuleList(
            ResidualBlock(c) if residual else nn.Identity() for c in widths[1:]
        )
        n = 32 * math.prod(v // 16 for v in self.padded)
        self.to_latent = nn.Sequential(
            nn.Flatten(), nn.Linear(n, 128), nn.SiLU(), nn.Linear(128, latent)
        )

    def forward(self, x):
        h, w = self.shape
        x = F.pad(x, (0, self.padded[1] - w, 0, self.padded[0] - h))
        for i, residual in enumerate(self.encoder_residuals):
            x = residual(self.encoder[2 * i + 1](self.encoder[2 * i](x)))
        return self.to_latent(x)


class ImageEncoder(nn.Module):
    """Encode prepared nonnegative images (N, 1, H, W) to (N, latent).

    Supports the TopView baseline and camera-1 residual encoder. Normalization
    is restored from the checkpoint and never refitted on incoming images.
    Inputs must be on the same device as the encoder.
    """

    def __init__(self, weights, device="cpu"):
        super().__init__()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.network = _ImageNetwork(
            checkpoint["shape"], checkpoint["latent"], checkpoint["residual"]
        )
        self.network.load_state_dict(checkpoint["model"], strict=True)
        for name in ("unit", "reference", "scale"):
            self.register_buffer(name, checkpoint["norm"][name])
        self.to(device=device).eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images):
        validate_tensor(images, (1, *self.network.shape), "images")
        if (images < 0).any():
            raise ValueError("Prepared image intensities must be nonnegative.")
        normalized = (torch.log1p(images / self.unit) - self.reference) / self.scale
        return self.network(normalized)


class _GaiaNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (1, 8, 16, 32, 32)
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Conv2d(a, b, 3, 2, 1), nn.SiLU())
                for a, b in zip(channels, channels[1:])
            ]
        )
        self.encoder_dense = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 19 * 19, 128), nn.SiLU()
        )
        self.mu = nn.Linear(128, 14)

    def forward(self, images):
        return self.mu(self.encoder_dense(self.encoder(images)))


class GaiaEncoder(nn.Module):
    """Deterministic 14-coordinate posterior mean, without VAE sampling.

    Inputs must be on the same device as the encoder.
    """

    def __init__(self, weights, device="cpu"):
        super().__init__()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.network = _GaiaNetwork()
        self.network.load_state_dict(checkpoint["model"], strict=True)
        for name in ("reference", "scale"):
            self.register_buffer(name, checkpoint["norm"][name])
        self.to(device=device).eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images):
        validate_tensor(images, (1, 304, 304), "Gaia images")
        if (images < 0).any():
            raise ValueError("Prepared image intensities must be nonnegative.")
        return self.network((images - self.reference) / self.scale)


class PCAEncoder:
    """Encode prepared, cropped images (N, H, W) with the frozen PCA fit."""

    def __init__(self, weights):
        with np.load(weights, allow_pickle=False) as data:
            self.mean = data["mean"].astype(np.float64)
            self.components = data["components"].astype(np.float64)
            self.shape = tuple(data["shape"])
            self.crop = tuple(data["crop"])
        if (
            self.mean.shape != (math.prod(self.shape),)
            or self.components.shape != (8, self.mean.size)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.components).all()
        ):
            raise ValueError("Invalid eight-component PCA checkpoint.")

    def __call__(self, images):
        images = np.asarray(images, dtype=np.float32)
        if (
            images.ndim != 3
            or images.shape[1:] != self.shape
            or not np.isfinite(images).all()
        ):
            raise ValueError(
                f"Expected finite prepared images with shape (N, {self.shape})."
            )
        # The frozen context was exported using float64, one image at a time.
        return np.asarray(
            [
                self.components @ (image.ravel().astype(np.float64) - self.mean)
                for image in images
            ],
            dtype=np.float32,
        ).reshape(len(images), 8)
