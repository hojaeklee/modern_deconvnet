import pytest
import torch
from torch import nn


@pytest.fixture
def tiny_vgg() -> nn.Sequential:
    """A small random VGG-style net: 3x3 convs, ReLUs, non-overlapping 2x2 pools."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
    ).eval()
