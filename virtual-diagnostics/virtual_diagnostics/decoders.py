"""Camera-1 image decoder only: no image encoder layers or weights."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ._images import ResidualBlock, validate_tensor


class _DecoderNetwork(nn.Module):
    def __init__(self, shape: tuple[int, int], latent: int):
        super().__init__()
        self.shape = tuple(shape)
        self.small = tuple(math.ceil(v / 16) for v in shape)
        n = 32 * math.prod(self.small)
        self.from_latent = nn.Sequential(
            nn.Linear(latent, 128),
            nn.SiLU(),
            nn.Linear(128, n),
            nn.SiLU(),
        )
        channels = (32, 24, 16, 8, 1)
        self.decoder = nn.ModuleList(
            nn.Conv2d(a, b, 3, padding=1) for a, b in zip(channels, channels[1:])
        )
        self.decoder_residuals = nn.ModuleList(ResidualBlock(c) for c in channels[:-1])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z).reshape(-1, 32, *self.small)
        for i, (residual, conv) in enumerate(zip(self.decoder_residuals, self.decoder)):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = conv(residual(x))
            if i < len(self.decoder) - 1:
                x = F.silu(x)
        return x[:, :, : self.shape[0], : self.shape[1]]


class ImageDecoder(nn.Module):
    """Decode float32 tensors of shape (batch, 20) to (batch, 1, 189, 285).

    Returns images in the original reconstructed intensity scale. Use device='cpu'
    for decoder-only checks; Kumo itself requires CUDA. Inputs must be on the
    same device as the decoder. The decoder does not change global TF32 settings.
    """

    def __init__(self, weights, device: str | torch.device = "cpu"):
        super().__init__()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.network = _DecoderNetwork(checkpoint["shape"], checkpoint["latent"])
        self.network.load_state_dict(checkpoint["model"], strict=True)
        for name in ("unit", "reference", "scale"):
            self.register_buffer(name, checkpoint["norm"][name])
        self.max_log = float(checkpoint["norm"]["max_log"])
        self.latent = checkpoint["latent"]
        self.to(device=device).eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        validate_tensor(z, (self.latent,), "latents")
        output = self.network(z)
        return (
            torch.expm1((output * self.scale + self.reference).clamp(0, self.max_log))
            * self.unit
        )
