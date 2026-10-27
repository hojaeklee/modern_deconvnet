"""A by-hand deconvnet, written the way Zeiler & Fergus describe it (Sec. 2.1).

Not used by the pipeline. It is here so you can read the algorithm in ~40
lines and so ``tests/test_project.py`` can check Captum against it.

Forward: run the net, and at every max-pool record the argmax "switches".
Backward, starting from a map that is zero except for the chosen unit:

* ReLU      -> rectify the signal: ``relu(signal)``
* MaxPool   -> unpool: put each value back where the switch says, zeros elsewhere
* Conv2d    -> transposed conv with the *same* filters

Supports Conv2d, ReLU and *non-overlapping* MaxPool2d (as in VGG). With
overlapping pools (AlexNet's 3x3 stride 2), two windows can share an argmax:
autograd adds both contributions while ``max_unpool2d`` keeps one, so the two
would differ slightly at those pixels.
"""

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def reference_deconv(net: nn.Sequential, layer_idx: int, image: torch.Tensor,
                     channel: int, h: int, w: int) -> torch.Tensor:
    """Project unit (channel, h, w) of ``net[layer_idx]`` for a single (1, 3, H, W) image."""
    layers = list(net[: layer_idx + 1])

    # Forward pass, remembering each layer's input size and the pool switches.
    x, sizes, switches = image, [], {}
    for i, m in enumerate(layers):
        sizes.append(x.shape[-2:])
        if isinstance(m, nn.MaxPool2d):
            x, switches[i] = F.max_pool2d(x, m.kernel_size, m.stride, m.padding,
                                          return_indices=True)
        else:
            x = m(x)

    # Seed: 1 at the chosen unit, 0 everywhere else (Captum seeds with the gradient 1 too).
    signal = torch.zeros_like(x)
    signal[0, channel, h, w] = 1.0

    for i in reversed(range(len(layers))):
        m = layers[i]
        if isinstance(m, nn.ReLU):
            signal = F.relu(signal)
        elif isinstance(m, nn.MaxPool2d):
            signal = F.max_unpool2d(signal, switches[i], m.kernel_size, m.stride,
                                    m.padding, output_size=sizes[i])
        elif isinstance(m, nn.Conv2d):
            out = F.conv_transpose2d(signal, m.weight, stride=m.stride, padding=m.padding)
            signal = F.pad(out, (0, sizes[i][1] - out.shape[-1], 0, sizes[i][0] - out.shape[-2]))
        else:
            raise TypeError(f"reference_deconv doesn't handle {type(m).__name__}")
    return signal
