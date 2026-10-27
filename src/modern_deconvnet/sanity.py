"""Milestone 2: do the projections depend on the network, or just on the image?

Two papers say deconvnet and guided backprop mostly *reconstruct the input*:
Adebayo et al., "Sanity Checks for Saliency Maps" (arXiv 1810.03292), and the
theory in Nie, Zhang & Patel (arXiv 1805.07039). This module runs their tests
on the same units the Fig. 2 harness draws, with the plain gradient as a
control:

* **Model randomization** (Adebayo §3.1). Re-initialize conv weights and
  project the same unit (channel, h, w) of the same image again. *Cascading*
  randomizes from the top conv down, one more layer per step; *independent*
  randomizes one conv at a time. A method that explains the model should
  change; a method that recovers the image won't.
* **Unit sensitivity** (Nie et al.). Project several *different* channels at
  the same location of the same image. If the maps all look alike, the
  projection says little about which unit it came from.
* **Edge baseline** (Adebayo Fig. 1). Correlate each map with a Sobel edge
  map of the image patch, which uses no model at all.

Similarity is Spearman rank correlation over the unit's receptive field,
both on the signed map (channel sum) and on its magnitude (sum of |.| over
channels). Adebayo recommends reading the signed one: the absolute value
hides sign flips, so it overstates similarity.

We project from the conv output just *before* the chosen ReLU. For the
trained net the top-k units are positive there, so the maps are identical to
the harness's; after randomization the unit can go negative, and projecting
from the ReLU output would make guided backprop and the gradient zero out.
"""

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .project import project_unit
from .receptive_field import receptive_field

ALL_METHODS = ("deconv", "guided", "gradient")


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation of two flat tensors; NaN if either is constant."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def similarity(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """(signed, abs) Spearman between two (3, h, w) maps."""
    return (spearman(x.sum(0).flatten(), y.sum(0).flatten()),
            spearman(x.abs().sum(0).flatten(), y.abs().sum(0).flatten()))


def sobel_edges(patch: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude of a (3, h, w) image patch, as a (1, h, w) map."""
    gray = patch.mean(0, keepdim=True).unsqueeze(0)
    kx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), kx.transpose(2, 3))
    return (gx**2 + gy**2).sqrt()[0]


@dataclass
class Units:
    """One unit per image: project ``channel[i], h[i], w[i]`` on ``images[i]``."""

    images: torch.Tensor  # (N, 3, H, W)
    channel: torch.Tensor  # (N,)
    h: torch.Tensor
    w: torch.Tensor
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # RF, clipped to the image

    def crops(self, maps: torch.Tensor) -> list[torch.Tensor]:
        return [maps[i, :, t:b, l:r] for i, (t, l, b, r) in enumerate(self.boxes)]


def make_units(features: nn.Sequential, layer_idx: int, images, channel, h, w) -> Units:
    rf = receptive_field(features, layer_idx)
    size_h, size_w = images.shape[-2:]
    boxes = []
    for hh, ww in zip(h.tolist(), w.tolist()):
        t, l, b, r = rf.box(hh, ww)
        boxes.append((max(t, 0), max(l, 0), min(b, size_h), min(r, size_w)))
    return Units(images, torch.as_tensor(channel), h, w, boxes)


def convs_below(features: nn.Sequential, layer_idx: int) -> list[int]:
    """Indices of the convs feeding ``features[layer_idx]``, bottom to top."""
    return [i for i, m in enumerate(features[: layer_idx + 1]) if isinstance(m, nn.Conv2d)]


def randomize(features: nn.Sequential, conv_indices: list[int], seed: int = 0) -> nn.Sequential:
    """A copy of ``features`` with the given convs re-initialized (PyTorch's default init)."""
    net = copy.deepcopy(features)
    with torch.random.fork_rng():
        for i in conv_indices:
            torch.manual_seed(seed + i)  # the same conv gets the same weights in every step
            net[i].reset_parameters()
    return net


def project_crops(features, layer_idx, units: Units, method: str, channel=None) -> list[torch.Tensor]:
    channel = units.channel if channel is None else channel
    # layer_idx - 1 is the conv (pre-ReLU) output; see the module docstring.
    maps = project_unit(features, layer_idx - 1, units.images, channel, units.h, units.w, method)
    return units.crops(maps.detach())


@dataclass
class RandomizationResult:
    labels: list[str]  # one per step; step 0 is the trained net
    signed: dict[str, torch.Tensor]  # method -> (steps, N) Spearman vs the trained map
    absolute: dict[str, torch.Tensor]
    example: dict[str, list[torch.Tensor]]  # method -> one crop per step, for image 0


def model_randomization(features: nn.Sequential, layer_idx: int, units: Units,
                        methods=ALL_METHODS, mode: str = "cascade", seed: int = 0) -> RandomizationResult:
    """Adebayo et al.'s model randomization test, ``mode`` = "cascade" or "independent"."""
    convs = convs_below(features, layer_idx)
    names = {i: f"conv{n}" for n, i in enumerate(convs, start=1)}
    steps = [("trained", [])]
    if mode == "cascade":
        steps += [(f"{names[i]}+", convs[k:]) for k, i in reversed(list(enumerate(convs)))]
    elif mode == "independent":
        steps += [(names[i], [i]) for i in reversed(convs)]
    else:
        raise ValueError(mode)

    n = len(units.boxes)
    signed = {m: torch.full((len(steps), n), float("nan")) for m in methods}
    absolute = {m: torch.full((len(steps), n), float("nan")) for m in methods}
    example = {m: [] for m in methods}
    reference = {}
    for s, (_label, reset) in enumerate(steps):
        net = randomize(features, reset, seed) if reset else features
        for m in methods:
            crops = project_crops(net, layer_idx, units, m)
            if s == 0:
                reference[m] = crops
            for i, (x, y) in enumerate(zip(reference[m], crops)):
                signed[m][s, i], absolute[m][s, i] = similarity(x, y)
            example[m].append(crops[0])
    return RandomizationResult([label for label, _ in steps], signed, absolute, example)


def unit_sensitivity(features: nn.Sequential, layer_idx: int, units: Units, methods=ALL_METHODS,
                     n_other: int = 4, seed: int = 0) -> dict[str, dict[str, float]]:
    """Mean Spearman between the chosen unit's map and maps of other channels at
    the same location of the same image (higher = less unit-specific)."""
    channels = features[layer_idx - 1].out_channels
    g = torch.Generator().manual_seed(seed)
    others = []
    for c in units.channel.tolist():
        pool = torch.tensor([o for o in range(channels) if o != c])
        others.append(pool[torch.randperm(len(pool), generator=g)[:n_other]])
    others = torch.stack(others)  # (N, n_other)

    out = {}
    for m in methods:
        mine = project_crops(features, layer_idx, units, m)
        scores = []
        for j in range(n_other):
            theirs = project_crops(features, layer_idx, units, m, channel=others[:, j])
            scores += [similarity(x, y) for x, y in zip(mine, theirs)]
        s = torch.tensor(scores)
        out[m] = {"signed": float(s[:, 0].nanmean()), "abs": float(s[:, 1].nanmean())}
    return out


def edge_similarity(features: nn.Sequential, layer_idx: int, units: Units,
                    methods=ALL_METHODS) -> dict[str, float]:
    """Mean Spearman between |map| and the Sobel edges of the image patch."""
    patches = units.crops(units.images)
    edges = [sobel_edges(p) for p in patches]
    out = {}
    for m in methods:
        crops = project_crops(features, layer_idx, units, m)
        scores = [spearman(c.abs().sum(0).flatten(), e.flatten()) for c, e in zip(crops, edges)]
        out[m] = float(torch.tensor(scores).nanmean())
    return out
