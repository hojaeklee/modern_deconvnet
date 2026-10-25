import pytest
import torchvision
from torch import nn

from modern_deconvnet.models import (_ALEXNET_LAYERS, _VGG16_LAYERS, IMAGENET_MEAN, IMAGENET_STD,
                                     Preprocess, _assemble)


@pytest.mark.parametrize("build, layers", [(torchvision.models.alexnet, _ALEXNET_LAYERS),
                                           (torchvision.models.vgg16, _VGG16_LAYERS)])
def test_layers_point_at_non_inplace_relus(build, layers):
    # Random weights: this checks structure only, without downloading anything.
    net = _assemble("x", Preprocess(IMAGENET_MEAN, IMAGENET_STD), build(weights=None).features, layers)
    assert isinstance(net.features[0], Preprocess)
    for idx in net.layers.values():
        assert isinstance(net.features[idx], nn.ReLU)
        assert isinstance(net.features[idx - 1], nn.Conv2d)
    assert not any(m.inplace for m in net.features if isinstance(m, nn.ReLU))
