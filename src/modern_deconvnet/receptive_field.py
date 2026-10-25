"""Receptive fields of units in a Sequential of convs and pools.

Zeiler & Fergus show each projection next to "the corresponding image patch",
i.e. the region of the input that the unit can see. For a plain stack of
convs and pools that region is a square we can compute in closed form by
walking the layers and tracking three numbers (Dumoulin & Visin's recurrence):

* ``size``:  side length of the region, in input pixels
* ``jump``:  distance in input pixels between neighbouring units
* ``start``: input coordinate of the centre of unit 0 (can be negative with padding)
"""

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ReceptiveField:
    size: int
    jump: int
    start: float

    def box(self, h: int, w: int) -> tuple[int, int, int, int]:
        """(top, left, bottom, right) of unit (h, w), half-open, may lie outside the image."""
        half = (self.size - 1) / 2
        top = round(self.start + h * self.jump - half)
        left = round(self.start + w * self.jump - half)
        return top, left, top + self.size, left + self.size


def _pair_first(v) -> int:
    return v[0] if isinstance(v, tuple) else v


def receptive_field(features: nn.Sequential, layer_idx: int) -> ReceptiveField:
    """Receptive field of the output of ``features[layer_idx]`` (square kernels assumed)."""
    size, jump, start = 1, 1, 0.0  # each input pixel is its own unit
    for m in features[: layer_idx + 1]:
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
            k = _pair_first(m.kernel_size)
            s = _pair_first(m.stride)
            p = _pair_first(m.padding)
            d = _pair_first(m.dilation)
            k_eff = d * (k - 1) + 1
            start += ((k_eff - 1) / 2 - p) * jump
            size += (k_eff - 1) * jump
            jump *= s
    return ReceptiveField(size, jump, start)
