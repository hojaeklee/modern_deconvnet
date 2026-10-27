"""Load a CNN as one flat ``nn.Sequential`` that takes [0, 1] RGB images.

Every model here is returned as ``Sequential(Preprocess, conv, relu, pool, ...)``:

* The input normalization is the first module, so a deconvnet projection
  lands in plain [0, 1] pixel space whatever the weights expect.
* All ReLUs are separate, non-inplace modules. Captum's Deconvolution works
  by hooking ``nn.ReLU`` modules; an inplace ReLU breaks those hooks.
* ``layers`` maps the paper's "layer 1..5" to the index of the ReLU whose
  output we visualize (Zeiler & Fergus show rectified feature maps).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn

KERAS_VGG16_URL = (
    "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/"
    "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
)


class Preprocess(nn.Module):
    """Per-channel affine map from [0, 1] RGB to whatever the weights expect.

    ``y[c] = (x[perm[c]] * scale - mean[c]) / std[c]``. It is linear, so its
    backward pass just rescales the projection per channel.
    """

    def __init__(self, mean, std, scale=1.0, bgr=False):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.scale = scale
        self.bgr = bgr

    def forward(self, x):
        if self.bgr:
            x = x.flip(1)
        return (x * self.scale - self.mean) / self.std


@dataclass
class ConvNet:
    name: str
    features: nn.Sequential  # Preprocess + conv trunk, in eval mode
    layers: dict[str, int]  # "layer1".."layer5" -> index into `features`


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Index of the ReLU after each "layer", counted *before* Preprocess is prepended.
_ALEXNET_LAYERS = {"layer1": 1, "layer2": 4, "layer3": 7, "layer4": 9, "layer5": 11}
# For VGG16 we take the last ReLU of each of the 5 conv blocks.
_VGG16_LAYERS = {"layer1": 3, "layer2": 8, "layer3": 15, "layer4": 22, "layer5": 29}


def _assemble(name, preprocess, trunk, layers) -> ConvNet:
    modules = [preprocess]
    for m in trunk:
        modules.append(nn.ReLU(inplace=False) if isinstance(m, nn.ReLU) else m)
    features = nn.Sequential(*modules).eval()
    for p in features.parameters():
        p.requires_grad_(False)
    return ConvNet(name, features, {k: v + 1 for k, v in layers.items()})


def load_model(name: str, weights_path: Path | None = None) -> ConvNet:
    """``name`` is one of ``alexnet``, ``vgg16`` (torchvision weights) or
    ``vgg16-keras`` (the original Caffe VGG16 weights, via the Keras mirror)."""
    if name == "alexnet":
        net = torchvision.models.alexnet(weights="IMAGENET1K_V1")
        pre = Preprocess(IMAGENET_MEAN, IMAGENET_STD)
        return _assemble(name, pre, net.features, _ALEXNET_LAYERS)
    if name == "vgg16":
        net = torchvision.models.vgg16(weights="IMAGENET1K_V1")
        pre = Preprocess(IMAGENET_MEAN, IMAGENET_STD)
        return _assemble(name, pre, net.features, _VGG16_LAYERS)
    if name == "vgg16-keras":
        trunk = torchvision.models.vgg16(weights=None).features
        _load_keras_vgg16(trunk, weights_path or _download(KERAS_VGG16_URL))
        # Caffe convention: BGR, 0-255, mean-subtracted, no std division.
        pre = Preprocess((103.939, 116.779, 123.68), (1.0, 1.0, 1.0), scale=255.0, bgr=True)
        return _assemble(name, pre, trunk, _VGG16_LAYERS)
    raise ValueError(f"unknown model {name!r}")


def _download(url: str) -> Path:
    dest = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(dest))
    return dest


def _load_keras_vgg16(trunk: nn.Sequential, path: Path) -> None:
    """Copy Keras ``blockB_convC`` kernels (H, W, in, out) into torch convs (out, in, H, W).

    This exists because some sandboxes can't reach download.pytorch.org; on a
    normal machine just use ``vgg16``.
    """
    import h5py  # optional dependency: `uv sync --extra keras`

    convs = [m for m in trunk if isinstance(m, nn.Conv2d)]
    names = [f"block{b}_conv{c}" for b, n in [(1, 2), (2, 2), (3, 3), (4, 3), (5, 3)]
             for c in range(1, n + 1)]
    with h5py.File(path, "r") as f:
        for conv, name in zip(convs, names, strict=True):
            group = f[name]
            w = next(np.asarray(group[k]) for k in group if "_W" in k)
            b = next(np.asarray(group[k]) for k in group if "_b" in k)
            conv.weight.data.copy_(torch.from_numpy(w.transpose(3, 2, 0, 1)))
            conv.bias.data.copy_(torch.from_numpy(b))
