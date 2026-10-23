# One-channel SpectraVision CNN.

from torch import nn
from .common import FeatureToEncoding


class SpectraVision1CCNN(nn.Module):
    def __init__(self, img_size=(224, 224)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(4, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.encoder = FeatureToEncoding(8 * (img_size[0] // 4) * (img_size[1] // 4), 256)

    def forward(self, x):
        features = self.conv(x).reshape(x.size(0), -1)
        return self.encoder(features)
