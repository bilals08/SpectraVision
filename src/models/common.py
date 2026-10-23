# Shared projection and peak m/z features.

import torch
from torch import nn


class FeatureToEncoding(nn.Module):
    # 256-dimensional projection.

    def __init__(self, input_dim: int = 512, embedding_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.proj(x)


class _MzSinusoidalEncoding(nn.Module):
    # Fixed, logarithmically spaced wavelengths for scalar m/z values.

    def __init__(self, d_model, min_wavelength=0.2, max_wavelength=1000):
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even")
        if min_wavelength <= 0 or max_wavelength < min_wavelength:
            raise ValueError("m/z wavelengths must be positive and ordered")
        wavelengths = torch.logspace(
            torch.log10(torch.tensor(float(min_wavelength))),
            torch.log10(torch.tensor(float(max_wavelength))),
            d_model // 2,
            dtype=torch.float32,
        )
        self.register_buffer("angular_frequencies", 2 * torch.pi / wavelengths)

    def forward(self, values):
        angles = values.unsqueeze(-1) * self.angular_frequencies
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
