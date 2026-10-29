import pytest
import torch
import torch.nn.functional as F

from modern_deconvnet.sae import TopKSAE, train_sae
from modern_deconvnet.vit import ATTN_LRP, DECONV, GRADIENT, forward, gelu, layer_norm, load_vit, matmul, patch_map
from modern_deconvnet.vit_project import METHODS, project_vit
from modern_deconvnet.vit_sanity import randomize_blocks, vit_randomization
from modern_deconvnet.vit_scan import Site, scan_vit, site_acts


@pytest.fixture(scope="module")
def vit():
    """DINOv2-S with registers, random weights, only 3 blocks to keep the tests fast."""
    torch.manual_seed(0)
    v = load_vit("dinov2-s-reg", pretrained=False)
    v.model.blocks = v.model.blocks[:3]
    for blk in v.model.blocks:  # LayerScale starts at 1e-5; make the blocks matter
        blk.ls1.gamma.data.fill_(0.5)
        blk.ls2.gamma.data.fill_(0.5)
    return v


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(1)
    return torch.rand(2, 3, 224, 224)


def test_forward_matches_timm(vit, images):
    trace = forward(vit, images)
    with torch.no_grad():
        m = vit.model
        expected = m.blocks(m.patch_drop(m._pos_embed(m.patch_embed(vit.preprocess(images)))))
    torch.testing.assert_close(trace.resid[-1], expected, atol=1e-4, rtol=1e-4)
    assert trace.tokens_in.shape == (2, 5 + 256, 384)  # CLS + 4 registers + 16x16 patches
    assert trace.mlp[0].shape == (2, 261, 1536)


def test_rules_only_change_the_backward_pass(vit, images):
    reference = forward(vit, images).resid[-1]
    for rules in (DECONV, ATTN_LRP):
        torch.testing.assert_close(forward(vit, images, rules).resid[-1], reference)


def test_gelu_rules():
    x = torch.randn(1000, requires_grad=True)
    g = torch.randn(1000)
    gelu(x, "deconv").backward(g)
    assert torch.equal(x.grad, g.clamp_min(0))  # ignores x entirely
    x.grad = None
    y = gelu(x, "identity")
    y.backward(g)
    torch.testing.assert_close(x * x.grad, y * g)  # relevance in == relevance out, per element


def test_layernorm_rule_conserves_relevance():
    norm = torch.nn.LayerNorm(16)
    norm.weight.data.normal_()  # bias stays 0: a bias would absorb relevance
    x = torch.randn(4, 16, requires_grad=True)
    y = layer_norm(norm, x, "detach_std")
    torch.testing.assert_close(y, norm(x))
    g = torch.randn(4, 16)
    y.backward(g)
    torch.testing.assert_close((x * x.grad).sum(1), (y * g).sum(1))


def test_uniform_matmul_splits_relevance_in_half():
    a = torch.randn(3, 5, requires_grad=True)
    b = torch.randn(5, 4, requires_grad=True)
    z = matmul(a, b, "uniform")
    torch.testing.assert_close(z, a @ b)
    g = torch.randn(3, 4)
    z.backward(g)
    total = (z * g).sum()
    torch.testing.assert_close((a * a.grad).sum(), total / 2)
    torch.testing.assert_close((b * b.grad).sum(), total / 2)


def test_patch_map_layout(vit):
    tokens = torch.arange(261.0).view(1, 261, 1)
    grid = patch_map(tokens, vit)
    assert grid.shape == (1, 1, 16, 16)
    assert grid[0, 0, 0, 0] == 5 and grid[0, 0, 1, 0] == 5 + 16  # row-major after the 5 prefix tokens


def test_gradient_matches_autograd(vit, images):
    site, unit, token = Site("mlp", 1), 7, torch.tensor([5 + 40, 5 + 100])
    maps = project_vit(vit, {}, images, site, unit, token, "gradient")
    x = images[:1].clone().requires_grad_()
    forward(vit, x, GRADIENT).mlp[1][0, token[0], unit].backward()
    torch.testing.assert_close(maps[:1], x.grad)


def test_projection_shapes(vit, images):
    torch.manual_seed(0)
    saes = {2: TopKSAE(384, 768, 8)}
    token = torch.tensor([5 + 40, 5 + 100])
    for site in (Site("mlp", 1), Site("sae", 2)):
        for m in METHODS:
            maps = project_vit(vit, saes, images, site, torch.tensor([3, 4]), token, m)
            assert maps.shape == {"gradient": (2, 3, 224, 224), "deconv": (2, 3, 224, 224),
                                  "attnlrp": (2, 1, 224, 224), "chefer": (2, 1, 16, 16)}[m]
            assert torch.isfinite(maps).all() and maps.abs().sum() > 0


def test_chefer_block0_is_one_head_average(vit, images):
    """With one block, R = I + mean_h(relu(dA * A)), so the unit's own token gets >= 1."""
    site, token = Site("mlp", 0), torch.tensor([5 + 17, 5 + 200])
    maps = project_vit(vit, {}, images, site, 2, token, "chefer").flatten(1)
    assert maps[0, 17] >= 1 and maps[1, 200] >= 1
    assert (maps >= 0).all()


def test_sae_trains_and_is_sparse():
    torch.manual_seed(0)
    dirs = F.normalize(torch.randn(64, 32), dim=1)
    codes = torch.rand(4000, 64) * (torch.rand(4000, 64) < 0.05)
    sae, stats = train_sae(codes @ dirs, expansion=4, k=4, steps=300, batch_size=512)
    assert stats["normalized_mse"] < 0.5
    _, f = sae(codes[:10] @ dirs)
    assert ((f > 0).sum(-1) <= 4).all()


def test_scan_finds_the_max(vit, images):
    from torch.utils.data import TensorDataset

    class Indexed(TensorDataset):
        def __getitem__(self, i):
            return self.tensors[0][i], i

    site = Site("mlp", 1)
    result = scan_vit(vit, [site], {}, Indexed(images), k=2, batch_size=1, num_workers=0)[site]
    acts = forward(vit, images).mlp[1]
    patch_max = acts[:, 5:].amax(1)  # (B, C)
    torch.testing.assert_close(result.top.values, patch_max.T.sort(dim=1, descending=True).values)
    torch.testing.assert_close(result.prefix_max, acts[:, :5].amax((0, 1)))


def test_randomization(vit, images):
    net = randomize_blocks(vit, [2])
    assert torch.equal(net.model.blocks[0].mlp.fc1.weight, vit.model.blocks[0].mlp.fc1.weight)
    assert not torch.equal(net.model.blocks[2].mlp.fc1.weight, vit.model.blocks[2].mlp.fc1.weight)
    assert torch.equal(net.model.blocks[2].ls1.gamma, vit.model.blocks[2].ls1.gamma)  # LayerScale kept

    site = Site("mlp", 2)
    r = vit_randomization(vit, {}, site, images, torch.tensor([1, 2]), torch.tensor([5 + 3, 5 + 9]),
                          ["gradient", "chefer"])
    assert r.labels == ["trained", "block2+", "block1+", "block0+"]
    assert torch.allclose(r.signed["gradient"][0], torch.ones(2))


def test_deconv_differs_from_gradient(vit, images):
    site, token = Site("mlp", 2), torch.tensor([5 + 40, 5 + 100])
    grad = project_vit(vit, {}, images, site, 3, token, "gradient")
    deconv = project_vit(vit, {}, images, site, 3, token, "deconv")
    assert not torch.allclose(grad, deconv, atol=1e-6)
