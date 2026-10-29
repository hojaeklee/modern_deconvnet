"""DINOv2 (with registers), with its forward pass written out by hand.

The weights come from timm, but the forward pass is spelled out here, block by
block. That has two uses: you can read what DINOv2 actually computes, and
every explanation method in ``vit_project.py`` is just this forward pass with
a different backward rule for three operations (GELU, LayerNorm, matmul).
``tests/test_vit.py`` checks it gives the same output as timm's.

What one forward pass does, for a 224x224 image and ViT-S/14:

1. **Patchify.** A 14x14 conv with stride 14 turns the image into a 16x16 grid
   of 384-dim patch tokens (256 tokens).
2. **Position.** Add a learned position embedding to each patch token.
3. **Prefix tokens.** Prepend 1 CLS token and 4 *register* tokens. Registers
   (Darcet et al., "Vision Transformers Need Registers", arXiv 2309.16588) are
   extra learned tokens that no loss looks at. Without them, large ViTs
   repurpose a few background patches as scratch space, which shows up as
   high-norm artifact tokens in attention maps. Registers give the model
   somewhere else to put that, so patch-token maps get cleaner.
   Token order: ``[CLS, reg1..reg4, patch_0 .. patch_255]``.
4. **12 blocks**, each pre-norm with LayerScale (a learned per-channel scale
   on each branch, initialized near zero, from CaiT/DINOv2)::

       x = x + ls1 * Attention(LayerNorm1(x))
       x = x + ls2 * fc2(GELU(fc1(LayerNorm2(x))))

   The 1536-dim GELU output is what "MLP neuron" means here, and ``x`` after
   each block is "the residual stream" the SAE reads.

We stop after the blocks: the final LayerNorm and heads aren't needed.
"""

from dataclasses import dataclass, field
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from torch import nn

from .models import IMAGENET_MEAN, IMAGENET_STD, Preprocess

MODELS = {
    "dinov2-s-reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2-b-reg": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov2-s": "vit_small_patch14_dinov2.lvd142m",  # no registers, for comparison
}


@dataclass
class ViT:
    name: str
    model: timm.models.VisionTransformer  # eval mode, requires_grad False
    preprocess: Preprocess  # [0, 1] RGB -> ImageNet-normalized

    @property
    def num_prefix(self) -> int:
        """CLS + registers: the tokens before the patch grid."""
        return self.model.num_prefix_tokens

    @property
    def grid(self) -> tuple[int, int]:
        return self.model.patch_embed.grid_size

    @property
    def blocks(self) -> int:
        return len(self.model.blocks)


def load_vit(name: str = "dinov2-s-reg", weights: Path | None = None, pretrained: bool = True,
             img_size: int = 224) -> ViT:
    """``weights`` is a local checkpoint (timm safetensors, or Meta's original
    ``dinov2_vits14_reg4_pretrain.pth``) for machines that can't reach the Hugging
    Face hub. ``pretrained=False`` gives random weights, which the tests use."""
    kwargs = dict(pretrained=pretrained, img_size=img_size)
    if pretrained and weights is not None:
        kwargs["pretrained_cfg_overlay"] = dict(file=str(weights))
    model = timm.create_model(MODELS[name], **kwargs).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return ViT(name, model, Preprocess(IMAGENET_MEAN, IMAGENET_STD))


# --- The three operations whose backward pass the explanation methods change ---

@dataclass(frozen=True)
class Rules:
    """Backward-pass rules. The forward value is the same for every setting.

    gelu:      "exact" (true derivative), "deconv" (pass relu(grad_out), the
               DeConvNet rule ported from ReLU) or "identity" (AttnLRP: relevance
               passes through unchanged).
    layernorm: "exact" or "detach_std" (AttnLRP: treat 1/std as a constant, so
               LayerNorm is linear in x).
    matmul:    "exact" or "uniform" (AttnLRP: split relevance of a product of
               two activations evenly between them).
    """

    gelu: str = "exact"
    layernorm: str = "exact"
    matmul: str = "exact"


GRADIENT = Rules()
DECONV = Rules(gelu="deconv")
ATTN_LRP = Rules(gelu="identity", layernorm="detach_std", matmul="uniform")


class _DeconvGELU(torch.autograd.Function):
    """GELU forward; backward ignores the input and rectifies the signal,
    exactly as Zeiler & Fergus's deconvnet does at a ReLU."""

    @staticmethod
    def forward(ctx, x):
        return F.gelu(x)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.clamp_min(0)


def gelu(x: torch.Tensor, rule: str) -> torch.Tensor:
    if rule == "exact":
        return F.gelu(x)
    if rule == "deconv":
        return _DeconvGELU.apply(x)
    if rule == "identity":
        # y = x * (gelu(x) / x) with the ratio held constant: grad = gelu(x)/x,
        # so x * grad_in == y * grad_out, i.e. relevance passes through as is.
        ratio = torch.where(x == 0, torch.full_like(x, 0.5), F.gelu(x) / x)
        return x * ratio.detach()
    raise ValueError(rule)


def layer_norm(norm: nn.LayerNorm, x: torch.Tensor, rule: str) -> torch.Tensor:
    if rule == "exact":
        return norm(x)
    if rule == "detach_std":
        centered = x - x.mean(-1, keepdim=True)
        std = (centered.pow(2).mean(-1, keepdim=True) + norm.eps).sqrt()
        return centered / std.detach() * norm.weight + norm.bias
    raise ValueError(rule)


def matmul(a: torch.Tensor, b: torch.Tensor, rule: str) -> torch.Tensor:
    if rule == "exact":
        return a @ b
    if rule == "uniform":
        # Same value as a @ b; each factor gets half the gradient.
        return 0.5 * (a @ b.detach()) + 0.5 * (a.detach() @ b)
    raise ValueError(rule)


# --- The forward pass ---

@dataclass
class Trace:
    """Everything the forward pass records, one entry per block run."""

    tokens_in: torch.Tensor | None = None  # (B, N, D) block-0 input, after pos-embed and prefix
    mlp: list[torch.Tensor] = field(default_factory=list)  # (B, N, 4D) GELU outputs
    resid: list[torch.Tensor] = field(default_factory=list)  # (B, N, D) residual stream after the block
    attn: list[torch.Tensor] = field(default_factory=list)  # (B, heads, N, N) softmax attention


def attention(attn: nn.Module, x: torch.Tensor, rules: Rules) -> tuple[torch.Tensor, torch.Tensor]:
    """timm's ``Attention.forward`` with the softmax spelled out (no fused kernel)."""
    b, n, d = x.shape
    qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)
    scores = matmul(q * attn.scale, k.transpose(-2, -1), rules.matmul)
    weights = scores.softmax(dim=-1)  # (B, heads, N, N); row i = where token i reads from
    out = matmul(weights, v, rules.matmul)
    out = out.transpose(1, 2).reshape(b, n, d)
    return attn.proj(out), weights


def forward(vit: ViT, images: torch.Tensor, rules: Rules = GRADIENT, last_block: int | None = None,
            retain_attn_grad: bool = False) -> Trace:
    """Run ``images`` ([0, 1] RGB, (B, 3, 224, 224)) through blocks ``0..last_block``."""
    m = vit.model
    last_block = vit.blocks - 1 if last_block is None else last_block
    trace = Trace()

    x = m.patch_embed(vit.preprocess(images))  # (B, 256, D), row-major over the 16x16 grid
    x = x + m.pos_embed
    prefix = [t.expand(x.shape[0], -1, -1) for t in (m.cls_token, m.reg_token) if t is not None]
    x = torch.cat([*prefix, x], dim=1)
    trace.tokens_in = x

    for i, blk in enumerate(m.blocks[: last_block + 1]):
        a, weights = attention(blk.attn, layer_norm(blk.norm1, x, rules.layernorm), rules)
        if retain_attn_grad and weights.requires_grad:
            weights.retain_grad()
        x = x + blk.ls1(a)
        hidden = gelu(blk.mlp.fc1(layer_norm(blk.norm2, x, rules.layernorm)), rules.gelu)
        x = x + blk.ls2(blk.mlp.fc2(hidden))
        trace.mlp.append(hidden)
        trace.resid.append(x)
        trace.attn.append(weights)
    return trace


def patch_map(tokens: torch.Tensor, vit: ViT) -> torch.Tensor:
    """(B, N, C) token tensor -> (B, C, 16, 16) over the patch grid (prefix tokens dropped)."""
    patches = tokens[:, vit.num_prefix :]
    return patches.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], *vit.grid)
