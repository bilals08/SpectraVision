# Peak-aware SpectraVision Vision Transformer.

import torch
from torch import nn
from torch.nn import functional as F
from .common import FeatureToEncoding, _MzSinusoidalEncoding


class SpectraVisionPeakViT(nn.Module):
    # Encode a one-channel spectrum image with a peak-aware ViT.

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        in_channels=1,
        d_model=512,
        embedding_dim=256,
        num_layers=2,
        nhead=8,
        position_encoding="learned",
        mask_empty_patches=True,
        bin_multiply_factor=10,
        max_mz=5000,
        intensity_channel=0,
        min_mz_wavelength=0.2,
        max_mz_wavelength=1000,
    ):
        super().__init__()
        if position_encoding != "learned":
            raise ValueError("PeakViT requires learned patch positions")
        if in_channels != 1 or intensity_channel != 0:
            raise ValueError("PeakViT requires a one-channel intensity image")
        if not mask_empty_patches:
            raise ValueError("PeakViT requires empty-patch masking")
        if bin_multiply_factor <= 0 or max_mz <= 0:
            raise ValueError("bin_multiply_factor and max_mz must be positive")
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if len(img_size) != 2 or len(patch_size) != 2 or any(size <= 0 for size in (*img_size, *patch_size)):
            raise ValueError("image and patch sizes must be positive pairs")
        if img_size[0] % patch_size[0] or img_size[1] % patch_size[1]:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")

        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.mask_empty_patches = mask_empty_patches
        self.bin_multiply_factor = bin_multiply_factor
        self.max_mz = max_mz
        self.intensity_channel = intensity_channel

        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.position_embedding = nn.Parameter(torch.empty(1, num_patches, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.encoder = FeatureToEncoding(d_model, embedding_dim)
        self.patch_projection = nn.Linear(patch_size[0] * patch_size[1], d_model)
        self.mz_encoder = _MzSinusoidalEncoding(
            d_model=d_model,
            min_wavelength=min_mz_wavelength,
            max_wavelength=max_mz_wavelength,
        )

        total_pixels = img_size[0] * img_size[1]
        pixel_mz = torch.arange(total_pixels, dtype=torch.float32).reshape(1, 1, *img_size)
        pixel_mz = pixel_mz * (float(max_mz) / total_pixels)
        patch_mz = F.unfold(pixel_mz, kernel_size=patch_size, stride=patch_size)
        self.register_buffer("patch_mz", patch_mz.clamp_max(float(max_mz)))

    def _dominant_patch_mz(self, intensity_channel):
        patch_intensities = F.unfold(
            intensity_channel,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        dominant_pixel = patch_intensities.argmax(dim=1, keepdim=True)
        patch_mz = self.patch_mz.expand(patch_intensities.shape[0], -1, -1)
        dominant_mz = patch_mz.gather(1, dominant_pixel).squeeze(1)
        padding_mask = patch_intensities.amax(dim=1).eq(0)
        return dominant_mz.masked_fill(padding_mask, 0.0), padding_mask

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or tuple(x.shape[-2:]) != self.img_size:
            raise ValueError(f"Expected [batch, 1, {self.img_size[0]}, {self.img_size[1]}] input")
        x = x.float() / 255.0
        dominant_mz, padding_mask = self._dominant_patch_mz(x)
        mz_embedding = self.mz_encoder(dominant_mz).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        tokens = self.patch_projection(patches) + self.position_embedding + mz_embedding

        all_empty = padding_mask.all(dim=1)
        if all_empty.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_empty, 0] = False
        features = self.transformer(tokens, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.encoder(pooled)
