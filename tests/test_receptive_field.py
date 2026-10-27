import pytest
import torch
from torch import nn

from modern_deconvnet.receptive_field import receptive_field


def gradient_support(net, layer_idx, h, w, size=64):
    """Bounding box of input pixels with nonzero gradient for one unit (the empirical RF)."""
    x = torch.randn(1, 3, size, size, requires_grad=True)
    net[: layer_idx + 1](x)[0, :, h, w].sum().backward()
    rows, cols = torch.nonzero(x.grad.abs().sum(1)[0], as_tuple=True)
    return rows.min().item(), cols.min().item(), rows.max().item() + 1, cols.max().item() + 1


@pytest.mark.parametrize("layer_idx", [1, 4, 7])
def test_rf_matches_gradient_support(tiny_vgg, layer_idx):
    rf = receptive_field(tiny_vgg, layer_idx)
    h, w = 3, 4  # an interior unit, so the box isn't clipped by the image border
    assert rf.box(h, w) == gradient_support(tiny_vgg, layer_idx, h, w)


def test_rf_strided_conv_and_overlapping_pool():
    # AlexNet's first two stages: 11x11/4 conv, 3x3/2 pool, 5x5 conv.
    net = nn.Sequential(nn.Conv2d(3, 4, 11, stride=4, padding=2), nn.ReLU(),
                        nn.MaxPool2d(3, 2), nn.Conv2d(4, 4, 5, padding=2), nn.ReLU())
    rf = receptive_field(net, 4)
    assert (rf.size, rf.jump) == (51, 8)
    assert rf.box(4, 4) == gradient_support(net, 4, 4, 4, size=160)
