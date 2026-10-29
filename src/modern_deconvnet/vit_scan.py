"""Find the top-k images for every unit of a ViT: MLP neurons and SAE features.

The protocol is the same as ``scan.py`` for the CNN: per unit, keep the k
images whose strongest activation over the patch grid is largest, and which
patch it was. The 16x16 patch grid plays the role of a conv feature map, so
``scan.TopK`` and ``scan.batch_candidates`` are reused unchanged.

Only patch tokens compete. CLS and register tokens have no place in the image,
so they can't be drawn as a heatmap; ``prefix_max`` records how strongly each
unit fires on them instead.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .sae import TopKSAE
from .scan import TopK, batch_candidates
from .vit import ViT, forward, patch_map


@dataclass(frozen=True)
class Site:
    """A family of units: the MLP neurons of one block, or one block's SAE features."""

    kind: str  # "mlp" or "sae"
    block: int  # 0-based, as in timm

    @property
    def name(self) -> str:
        return f"block{self.block}_{self.kind}"


def site_acts(trace, site: Site, saes: dict[int, TopKSAE], sparse: bool = True) -> torch.Tensor:
    """(B, N, C) activations of every unit of ``site``, for every token.

    For SAE features ``sparse=True`` is the TopK code (what the scan ranks by);
    ``sparse=False`` is the dense ReLU score, which is what we backpropagate from.
    """
    if site.kind == "mlp":
        return trace.mlp[site.block]
    if site.kind == "sae":
        sae = saes[site.block]
        x = trace.resid[site.block]
        return sae.encode(x) if sparse else sae.pre_acts(x)
    raise ValueError(site.kind)


@dataclass
class SiteScan:
    top: TopK  # (C, k), with (h, w) the patch row and column
    prefix_max: torch.Tensor  # (C,) largest activation on any CLS/register token, over the dataset


@torch.no_grad()
def scan_vit(vit: ViT, sites: list[Site], saes: dict[int, TopKSAE], dataset: Dataset, k: int = 9,
             batch_size: int = 32, device: str = "cpu", num_workers: int = 2) -> dict[Site, SiteScan]:
    last = max(s.block for s in sites)
    top: dict[Site, TopK] = {}
    prefix: dict[Site, torch.Tensor] = {}
    for images, idx in tqdm(DataLoader(dataset, batch_size=batch_size, num_workers=num_workers), desc="scan"):
        trace = forward(vit, images.to(device), last_block=last)
        for s in sites:
            acts = site_acts(trace, s, saes).float()
            cand = batch_candidates(patch_map(acts, vit).cpu(), idx)
            top[s] = top[s].merge(cand) if s in top else TopK.empty(acts.shape[2], k).merge(cand)
            p = acts[:, : vit.num_prefix].amax(dim=(0, 1)).cpu()
            prefix[s] = torch.maximum(prefix[s], p) if s in prefix else p
    return {s: SiteScan(top[s], prefix[s]) for s in sites}


@torch.no_grad()
def collect_resid(vit: ViT, block: int, dataset: Dataset, batch_size: int = 32, device: str = "cpu",
                  num_workers: int = 2) -> torch.Tensor:
    """Patch-token residual stream after ``block`` for every image: (N_images * 256, D)."""
    out = []
    for images, _ in tqdm(DataLoader(dataset, batch_size=batch_size, num_workers=num_workers), desc="collect"):
        trace = forward(vit, images.to(device), last_block=block)
        out.append(trace.resid[block][:, vit.num_prefix :].flatten(0, 1))
    return torch.cat(out)
