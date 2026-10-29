"""The "projection" panel for ViT units: four ways to map one unit back to the image.

A ViT has no ReLUs and no max-pool switches, so Zeiler & Fergus's deconvnet
doesn't carry over directly. We compare four stand-ins, all seeded from a
single unit at a single token (the image's strongest patch for that unit, as
in the CNN harness):

* ``gradient``: d(unit) / d(pixels). The plain control, as in milestone 2.
* ``deconv``: the same backward pass, except each GELU passes
  ``relu(grad_out)``, ignoring its input. This is the deconvnet's ReLU rule
  bolted onto GELU; LayerNorm and softmax keep their true derivatives. It's a
  hack, and the sanity checks are there to say how much it means.
* ``chefer``: Chefer, Gur & Wolf, "Generic Attention-model Explainability"
  (arXiv 2103.15679). Attention rollout weighted by gradients, at the token
  level: start from R = I, and for each block up to the unit's,
  ``R += mean_heads(relu(dA * A)) @ R``. Row ``token`` of R scores every token
  by how much the unit's token reads from it. One value per patch (16x16).
* ``attnlrp``: Achtibat et al., "AttnLRP: Attention-Aware Layer-wise Relevance
  Propagation for Transformers" (arXiv 2402.05602). Layer-wise relevance
  propagation, written as gradient x input with three rules swapped in
  (see ``vit.Rules``): GELU passes relevance through unchanged, LayerNorm's 1/std
  is held constant, and the two attention matmuls split relevance evenly between
  their factors. Softmax keeps its true gradient, which is what AttnLRP's softmax
  rule reduces to in this form. Relevance = normalized pixels x that gradient.

``gradient`` and ``deconv`` come back as RGB (B, 3, H, W), like the CNN
projections; ``attnlrp`` as one signed channel (B, 1, H, W) and ``chefer`` as
(B, 1, 16, 16) non-negative patch scores.
"""

import torch

from .sae import TopKSAE
from .vit import ATTN_LRP, DECONV, GRADIENT, ViT, forward
from .vit_scan import Site, site_acts

METHODS = ("gradient", "deconv", "chefer", "attnlrp")


def project_vit(vit: ViT, saes: dict[int, TopKSAE], images: torch.Tensor, site: Site,
                unit: torch.Tensor | int, token: torch.Tensor, method: str) -> torch.Tensor:
    """Project unit ``unit[i]`` at token ``token[i]`` (index into the full sequence,
    prefix included) of ``site``, for each image i."""
    batch = torch.arange(images.shape[0], device=images.device)
    unit = torch.as_tensor(unit, device=images.device).expand(images.shape[0])
    rules = {"gradient": GRADIENT, "deconv": DECONV, "chefer": GRADIENT, "attnlrp": ATTN_LRP}[method]

    x = images.clone().requires_grad_()
    with torch.enable_grad():
        trace = forward(vit, x, rules, last_block=site.block, retain_attn_grad=method == "chefer")
        acts = site_acts(trace, site, saes, sparse=False)
        acts[batch, token, unit].sum().backward()  # images are independent, so one backward serves all

    if method in ("gradient", "deconv"):
        return x.grad
    if method == "attnlrp":
        normalized_minus_mean = x.detach() - vit.preprocess.mean  # = std * normalized pixels
        return (normalized_minus_mean * x.grad).sum(1, keepdim=True)
    if method == "chefer":
        return _chefer(trace.attn, batch, token, vit)
    raise ValueError(method)


def _chefer(attn: list[torch.Tensor], batch, token, vit: ViT) -> torch.Tensor:
    b, _, n, _ = attn[0].shape
    rel = torch.eye(n, device=attn[0].device).expand(b, n, n)
    for a in attn:
        cam = (a.grad * a).clamp_min(0).mean(1)  # (B, N, N), averaged over heads
        rel = rel + cam @ rel
    row = rel[batch, token, vit.num_prefix :]  # (B, 256): drop CLS and register columns
    return row.view(b, 1, *vit.grid).detach()
