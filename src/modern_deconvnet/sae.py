"""A TopK sparse autoencoder on one block's residual stream.

MLP neurons are one choice of "unit", but they're polysemantic: one neuron
often responds to several unrelated things. A sparse autoencoder (SAE) learns
a different basis for the residual stream, with many more directions than
dimensions, where each token uses only a few. Those directions ("features")
tend to be easier to name.

This is the TopK SAE of Gao et al., "Scaling and evaluating sparse
autoencoders" (arXiv 2406.04093)::

    pre = relu(W_enc (x - b_dec) + b_enc)        # (F,) one score per feature
    f   = keep the k largest entries of pre, zero the rest
    x̂   = W_dec f + b_dec                        # reconstruction

trained on plain MSE. TopK fixes sparsity at exactly k, so there is no L1
coefficient to tune. We train on patch tokens only (not CLS or registers,
whose norms are much larger) from the same image subset, which is small
enough for a CPU.
"""

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, n_features: int, k: int):
        super().__init__()
        self.k = k
        self.register_buffer("scale", torch.tensor(1.0))  # inputs are divided by this
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Linear(d_model, n_features)
        w = F.normalize(torch.randn(n_features, d_model), dim=1)
        self.W_dec = nn.Parameter(w)  # (F, D), unit-norm rows: one direction per feature
        self.encoder.weight.data.copy_(w)  # tied init, as in Gao et al.
        self.encoder.bias.data.zero_()

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """Dense, non-negative feature scores. Differentiable in ``x``, which the
        projection methods need; equals ``encode(x)`` wherever a feature is active."""
        return F.relu(self.encoder(x / self.scale - self.b_dec))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.pre_acts(x)
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return (f @ self.W_dec + self.b_dec) * self.scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f


def train_sae(acts: torch.Tensor, expansion: int = 8, k: int = 32, steps: int = 2000, batch_size: int = 4096,
              lr: float = 1e-3, seed: int = 0) -> tuple[TopKSAE, dict]:
    """Fit a TopK SAE to ``acts`` (N, D). Returns the SAE and some fit statistics."""
    torch.manual_seed(seed)
    n, d = acts.shape
    sae = TopKSAE(d, expansion * d, k).to(acts.device)
    # Rescale so the average token has norm sqrt(D), and start b_dec at the mean.
    sae.scale.fill_(float(acts.norm(dim=1).mean() / d**0.5))
    sae.b_dec.data.copy_((acts / sae.scale).mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    fired = torch.zeros(sae.W_dec.shape[0], dtype=torch.bool, device=acts.device)
    for step in tqdm(range(steps), desc="train SAE"):
        x = acts[torch.randint(n, (batch_size,), device=acts.device)]
        x_hat, f = sae(x)
        loss = (x_hat - x).pow(2).sum(-1).mean() / x.pow(2).sum(-1).mean()  # normalized MSE
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)
            if step >= steps - steps // 10:  # liveness over the last 10% of training
                fired |= (f > 0).any(0)
    return sae, {"normalized_mse": loss.item(), "dead_fraction": float(1 - fired.float().mean())}


@torch.no_grad()
def explained_variance(sae: TopKSAE, acts: torch.Tensor, batch_size: int = 8192) -> float:
    err = sum((sae(x)[0] - x).pow(2).sum() for x in acts.split(batch_size))
    return float(1 - err / (acts - acts.mean(0)).pow(2).sum())
