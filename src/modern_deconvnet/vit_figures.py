"""Figures for ViT units.

* ``top9_figure``: the Fig. 2 layout for one site. Per unit, a 3x3 grid of its
  top-9 images with the unit's per-token activation laid over each one. Every
  heatmap uses the same scale (0 to the unit's top-1 value), so the 9th image
  looks weaker than the 1st when it is.
* ``projection_figure``: one unit, 9 columns (its top-9 images), one row per
  method: the activation heatmap, then each projection from ``vit_project``.

There is no receptive-field crop: attention lets every token see the whole
image from block 0, so we always show the full image.
"""

import math

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .figures import _grid, to_display
from .sae import TopKSAE
from .vit import ViT, forward, patch_map
from .vit_project import project_vit
from .vit_scan import Site, SiteScan, site_acts

TILE = 112  # images are drawn at half resolution
METHOD_LABELS = {"gradient": "gradient", "deconv": "GELU deconv", "chefer": "Chefer", "attnlrp": "AttnLRP"}


def colorize(maps: torch.Tensor, cmap: str, vmin: float, vmax: float) -> torch.Tensor:
    """(B, h, w) scalars -> (B, 3, h, w) RGB."""
    normed = ((maps - vmin) / (vmax - vmin)).clamp(0, 1)
    rgb = matplotlib.colormaps[cmap](normed.cpu().numpy())[..., :3]
    return torch.from_numpy(rgb).float().permute(0, 3, 1, 2)


def resize(x: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    kwargs = {} if mode == "nearest" else dict(align_corners=False)
    return F.interpolate(x.float().cpu(), size=(TILE, TILE), mode=mode, **kwargs)


def activation_tiles(images: torch.Tensor, token_maps: torch.Tensor, vmax: float, alpha: float = 0.6):
    """Overlay (B, 16, 16) activations on (B, 3, H, W) images, blocky so you see the tokens."""
    heat = colorize(resize(token_maps[:, None], "nearest")[:, 0], "inferno", 0.0, vmax)
    return (1 - alpha) * resize(images) + alpha * heat


def signed_heatmap(maps: torch.Tensor) -> torch.Tensor:
    """(B, 1, h, w) signed relevance -> RGB, red positive, blue negative, one scale per map."""
    scale = maps.abs().flatten(1).quantile(0.99, dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
    return colorize(resize(maps / scale)[:, 0], "bwr", -1.0, 1.0)


def method_tiles(method: str, maps: torch.Tensor) -> torch.Tensor:
    if method in ("gradient", "deconv"):  # RGB projections, as in the CNN figures
        return to_display(resize(maps))
    if method == "attnlrp":
        return signed_heatmap(maps)
    if method == "chefer":  # non-negative, one value per patch
        scale = maps.flatten(1).quantile(0.99, dim=1).clamp_min(1e-12).view(-1, 1, 1)
        return colorize(resize(maps, "nearest")[:, 0] / scale, "inferno", 0.0, 1.0)
    raise ValueError(method)


def unit_images(top, unit: int, dataset: Dataset):
    keep = top.image_idx[unit] >= 0  # fewer than k images seen leaves -1 slots
    idx = top.image_idx[unit][keep]
    return torch.stack([dataset[i][0] for i in idx.tolist()]), top.h[unit][keep], top.w[unit][keep]


@torch.no_grad()
def token_maps(vit: ViT, saes, site: Site, images: torch.Tensor, unit: int) -> torch.Tensor:
    trace = forward(vit, images, last_block=site.block)
    return patch_map(site_acts(trace, site, saes), vit)[:, unit].float().cpu()  # (B, 16, 16)


def top9_figure(vit: ViT, saes: dict[int, TopKSAE], site: Site, scan: SiteScan, units: list[int],
                dataset: Dataset, title: str, device: str = "cpu", cols: int = 4) -> plt.Figure:
    rows = math.ceil(len(units) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, unit in zip(axes.flat, units):
        images, _, _ = unit_images(scan.top, unit, dataset)
        maps = token_maps(vit, saes, site, images.to(device), unit)
        ax.imshow(_grid(activation_tiles(images, maps, float(scan.top.values[unit, 0]))).permute(1, 2, 0).numpy())
        ax.set_title(f"{'neuron' if site.kind == 'mlp' else 'feature'} {unit}", fontsize=9)
    fig.suptitle(f"{title}: top-9 images, per-token activation (0 to top-1 value)")
    fig.tight_layout()
    return fig


def projection_figure(vit: ViT, saes: dict[int, TopKSAE], site: Site, scan: SiteScan, unit: int,
                      dataset: Dataset, methods: list[str], title: str, device: str = "cpu") -> plt.Figure:
    images, h, w = unit_images(scan.top, unit, dataset)
    images = images.to(device)
    token = (vit.num_prefix + h * vit.grid[1] + w).to(device)
    maps = token_maps(vit, saes, site, images, unit)
    rows = [("activation", activation_tiles(images, maps, float(scan.top.values[unit, 0])))]
    for m in methods:
        rows.append((METHOD_LABELS[m], method_tiles(m, project_vit(vit, saes, images, site, unit, token, m))))

    n = images.shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.25, len(rows) * 1.35), squeeze=False)
    for r, (label, tiles) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(tiles[c].permute(1, 2, 0).numpy())
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            if r == 0:  # outline the token the projections start from
                s = TILE / vit.grid[0]
                ax.add_patch(plt.Rectangle((w[c].item() * s - 0.5, h[c].item() * s - 0.5), s, s,
                                           fill=False, edgecolor="cyan", lw=1))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
