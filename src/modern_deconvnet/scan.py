"""Stage 1: find the top-k images for every channel of every chosen layer.

This is the dataset sweep behind Fig. 2. For each (layer, channel) we keep the
k images whose *strongest* activation anywhere in that channel's feature map
is largest, and where in the map it happened. Taking one location per image
(the spatial argmax) means the k results are k different images, as in the paper.

One forward pass per batch serves all layers: forward hooks grab each layer's
output, so the cost is one pass through the network up to the deepest layer.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class TopK:
    """Top-k records for every channel of one layer; all tensors are (C, k)."""

    values: torch.Tensor  # activation value at the argmax location
    image_idx: torch.Tensor  # dataset index of the image
    h: torch.Tensor  # argmax row in the feature map
    w: torch.Tensor  # argmax column in the feature map

    @classmethod
    def empty(cls, channels: int, k: int) -> "TopK":
        long = lambda: torch.full((channels, k), -1, dtype=torch.long)
        return cls(torch.full((channels, k), -float("inf")), long(), long(), long())

    def merge(self, other: "TopK") -> "TopK":
        """Keep the k best of ``self`` and ``other``, per channel, sorted descending."""
        k = self.values.shape[1]
        cat = lambda a, b: torch.cat([a, b], dim=1)
        values = cat(self.values, other.values)
        best = values.topk(k, dim=1).indices
        pick = lambda a, b: cat(a, b).gather(1, best)
        return TopK(pick(self.values, other.values), pick(self.image_idx, other.image_idx),
                    pick(self.h, other.h), pick(self.w, other.w))


def batch_candidates(acts: torch.Tensor, image_idx: torch.Tensor) -> TopK:
    """Turn a (B, C, H, W) activation batch into (C, B) candidates: one per image."""
    b, c, _, width = acts.shape
    values, flat = acts.flatten(2).max(dim=2)  # both (B, C)
    return TopK(values.T, image_idx.view(b, 1).expand(b, c).T, (flat // width).T, (flat % width).T)


@torch.no_grad()
def scan(features: nn.Sequential, layers: dict[str, int], dataset: Dataset,
         k: int = 9, batch_size: int = 32, num_workers: int = 2) -> dict[str, TopK]:
    """Return ``{layer_name: TopK}`` over the whole dataset."""
    deepest = max(layers.values())
    trunk = features[: deepest + 1]  # nothing past the deepest layer is needed

    captured: dict[str, torch.Tensor] = {}
    hooks = [trunk[i].register_forward_hook(lambda _m, _in, out, name=name: captured.__setitem__(name, out))
             for name, i in layers.items()]

    results: dict[str, TopK] = {}
    try:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        for images, idx in tqdm(loader, desc="scan"):
            trunk(images)
            for name, acts in captured.items():
                if name not in results:
                    results[name] = TopK.empty(acts.shape[1], k)
                results[name] = results[name].merge(batch_candidates(acts, idx))
    finally:
        for h in hooks:
            h.remove()
    return results
