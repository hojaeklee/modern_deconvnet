"""Stage 2: project single units back to pixel space with Captum's deconvnet.

Captum's ``NeuronDeconvolution`` runs ordinary backprop from one unit with one
change: every ``nn.ReLU`` passes ``relu(grad_out)`` backward instead of
``grad_out * (input > 0)``. The other two pieces of Zeiler & Fergus's
deconvnet fall out of autograd for free:

* max-pool backward routes the signal to the argmax of each window, which is
  exactly the paper's "switches";
* conv backward is a transposed conv with the same filters.

``reference_deconv.py`` spells the same thing out by hand, and
``tests/test_project.py`` checks the two agree.

``method="guided"`` swaps in guided backprop (Springenberg et al. 2014), which
also applies the forward ReLU mask. It is here for comparison: on deep VGG
layers the pure deconvnet output gets noisy, guided backprop stays sharp.
"""

import torch
from captum.attr import NeuronDeconvolution, NeuronGuidedBackprop
from torch import nn

METHODS = {"deconv": NeuronDeconvolution, "guided": NeuronGuidedBackprop}


def project_unit(features: nn.Sequential, layer_idx: int, images: torch.Tensor,
                   channel: torch.Tensor | int, h: torch.Tensor, w: torch.Tensor,
                   method: str = "deconv") -> torch.Tensor:
    """Project unit ``(channel[i], h[i], w[i])`` of ``features[layer_idx]`` for each image i.

    Returns a (B, 3, H, W) tensor in input-pixel space. Only the chosen unit
    seeds the backward pass, which is the paper's "set all other activations
    in the layer to zero".
    """
    trunk = features[: layer_idx + 1]
    batch = torch.arange(images.shape[0])
    channel = torch.as_tensor(channel).expand(images.shape[0])

    # Captum wants one scalar per example. A callable selector lets every
    # image in the batch pick a different (c, h, w).
    def selector(layer_out: torch.Tensor) -> torch.Tensor:
        return layer_out[batch, channel, h, w]

    attribution = METHODS[method](trunk, trunk[layer_idx])
    return attribution.attribute(images.clone().requires_grad_(), neuron_selector=selector)
