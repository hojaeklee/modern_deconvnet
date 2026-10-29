"""Milestone 2's model-randomization test, on the ViT projections.

Same question as ``sanity.py``: if we re-initialize the network's weights and
project the same unit again, does the map change? Here we randomize whole
transformer blocks, from the unit's block downward (*cascade*) or one at a
time (*independent*), and compare with the trained map by Spearman rank
correlation over the full image (``sanity.similarity``).

What "re-initialize" means for a block: every Linear and LayerNorm gets
PyTorch's default init. LayerScale keeps its trained values. Its init is 1e-5,
so resetting it too would make each randomized block an identity, which tests
"delete the block" rather than "scramble its weights".

The SAE (for SAE units) stays as trained: it defines which direction the unit
is.
"""

import copy

import torch
from torch import nn

from .sae import TopKSAE
from .sanity import RandomizationResult, similarity
from .vit import ViT
from .vit_project import project_vit
from .vit_scan import Site


def randomize_blocks(vit: ViT, blocks: list[int], seed: int = 0) -> ViT:
    model = copy.deepcopy(vit.model)
    with torch.random.fork_rng(devices=[]):
        for b in blocks:
            torch.manual_seed(seed + b)  # the same block gets the same weights in every step
            for m in model.blocks[b].modules():
                if isinstance(m, (nn.Linear, nn.LayerNorm)):
                    m.reset_parameters()
    return ViT(vit.name, model, vit.preprocess)


def vit_randomization(vit: ViT, saes: dict[int, TopKSAE], site: Site, images: torch.Tensor,
                      unit: torch.Tensor, token: torch.Tensor, methods, mode: str = "cascade",
                      seed: int = 0) -> RandomizationResult:
    """One unit per image: ``unit[i]`` at ``token[i]`` on ``images[i]``."""
    blocks = list(range(site.block + 1))
    steps = [("trained", [])]
    if mode == "cascade":
        steps += [(f"block{b}+", blocks[b:]) for b in reversed(blocks)]
    elif mode == "independent":
        steps += [(f"block{b}", [b]) for b in reversed(blocks)]
    else:
        raise ValueError(mode)

    n = images.shape[0]
    signed = {m: torch.full((len(steps), n), float("nan")) for m in methods}
    absolute = {m: torch.full((len(steps), n), float("nan")) for m in methods}
    example = {m: [] for m in methods}
    reference = {}
    for s, (_label, reset) in enumerate(steps):
        net = randomize_blocks(vit, reset, seed) if reset else vit
        for m in methods:
            maps = project_vit(net, saes, images, site, unit, token, m).detach().cpu()
            if s == 0:
                reference[m] = maps
            for i in range(n):
                signed[m][s, i], absolute[m][s, i] = similarity(reference[m][i], maps[i])
            example[m].append(maps[0])
    return RandomizationResult([label for label, _ in steps], signed, absolute, example)
