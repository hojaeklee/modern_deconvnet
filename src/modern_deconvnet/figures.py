"""Stage 3: Fig. 2-style panels.

For each chosen channel we draw two 3x3 grids side by side: the deconvnet
projections of its top-9 activations (left) and the matching image patches
(right). Both are cropped to the unit's receptive field, as in the paper.
"""

import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from .project import project_unit
from .receptive_field import receptive_field
from .scan import TopK

TILE = 64  # every crop is resized to TILE x TILE pixels for display


def crop(images: torch.Tensor, boxes: list[tuple[int, int, int, int]], fill: float) -> torch.Tensor:
    """Crop one square box per image, padding with ``fill`` where it leaves the image."""
    pad = max(max(-t, -l, b - images.shape[-2], r - images.shape[-1], 0) for t, l, b, r in boxes)
    padded = F.pad(images, (pad, pad, pad, pad), value=fill)
    crops = [padded[i, :, t + pad : b + pad, l + pad : r + pad] for i, (t, l, b, r) in enumerate(boxes)]
    # Shrink with area averaging (nearest would alias fine detail into noise);
    # enlarge small layer-1 crops with nearest so pixels stay crisp.
    mode = "area" if crops[0].shape[-1] > TILE else "nearest-exact"
    return F.interpolate(torch.stack(crops), size=(TILE, TILE), mode=mode)


def to_display(projection: torch.Tensor) -> torch.Tensor:
    """Map signed projections to [0, 1] around mid-gray, one scale per crop.

    Scaling by the 99th percentile of |p| rather than the max keeps a few hot
    pixels from washing out the rest.
    """
    flat = projection.abs().flatten(1)
    scale = flat.quantile(0.99, dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
    return (0.5 + projection / (2 * scale)).clamp(0, 1)


def channel_panels(features: nn.Sequential, layer_idx: int, top: TopK, channel: int,
                   dataset: Dataset, method: str = "deconv") -> tuple[torch.Tensor, torch.Tensor]:
    """(projections, patches) for one channel, each (k, 3, TILE, TILE)."""
    keep = top.image_idx[channel] >= 0  # fewer than k images seen leaves -1 slots
    idx, h, w = top.image_idx[channel][keep], top.h[channel][keep], top.w[channel][keep]
    images = torch.stack([dataset[i][0] for i in idx.tolist()])

    projections = project_unit(features, layer_idx, images, channel, h, w, method)
    rf = receptive_field(features, layer_idx)
    boxes = [rf.box(hh, ww) for hh, ww in zip(h.tolist(), w.tolist())]
    return to_display(crop(projections, boxes, fill=0.0)), crop(images, boxes, fill=0.5)


def _grid(tiles: torch.Tensor, cols: int = 3, gap: int = 2) -> torch.Tensor:
    """Tile (n, 3, T, T) into one (3, H, W) image with white gaps."""
    n, c, t, _ = tiles.shape
    rows = math.ceil(n / cols)
    out = torch.ones(c, rows * (t + gap) - gap, cols * (t + gap) - gap)
    for i in range(n):
        r, q = divmod(i, cols)
        out[:, r * (t + gap) : r * (t + gap) + t, q * (t + gap) : q * (t + gap) + t] = tiles[i]
    return out


def layer_figure(features: nn.Sequential, layer_idx: int, top: TopK, channels: list[int],
                 dataset: Dataset, title: str, method: str = "deconv", cols: int = 4) -> plt.Figure:
    rows = math.ceil(len(channels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, channel in zip(axes.flat, channels):
        projections, patches = channel_panels(features, layer_idx, top, channel, dataset, method)
        left, right = _grid(projections), _grid(patches)
        side_by_side = torch.cat([left, torch.ones(3, left.shape[1], 6), right], dim=2)
        ax.imshow(side_by_side.permute(1, 2, 0).numpy())
        ax.set_title(f"channel {channel}", fontsize=9)
    rf = receptive_field(features, layer_idx)
    fig.suptitle(f"{title}  (receptive field {rf.size}px; left: {method} projection, right: patch)")
    fig.tight_layout()
    return fig
