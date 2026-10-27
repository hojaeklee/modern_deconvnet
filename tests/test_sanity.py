import torch

from modern_deconvnet.sanity import (convs_below, make_units, model_randomization, randomize, sobel_edges,
                                     spearman)


def test_spearman():
    a = torch.randn(100)
    assert abs(spearman(a, a) - 1) < 1e-9
    assert abs(spearman(a, -a) + 1) < 1e-9
    assert abs(spearman(a, a.exp()) - 1) < 1e-9  # rank-based: any monotone map gives 1
    assert spearman(a, torch.zeros(100)) != spearman(a, torch.zeros(100))  # NaN for a constant map


def test_randomize_copies_and_resets_only_given_convs(tiny_vgg):
    convs = convs_below(tiny_vgg, 7)
    assert convs == [0, 3, 6]
    net = randomize(tiny_vgg, [6])
    assert torch.equal(net[0].weight, tiny_vgg[0].weight)
    assert torch.equal(net[3].weight, tiny_vgg[3].weight)
    assert not torch.equal(net[6].weight, tiny_vgg[6].weight)
    assert torch.equal(randomize(tiny_vgg, [6])[6].weight, net[6].weight)  # seeded


def test_model_randomization_steps(tiny_vgg):
    torch.manual_seed(1)
    images = torch.rand(4, 3, 32, 32)
    units = make_units(tiny_vgg, 7, images, torch.tensor([0, 3, 5, 9]), torch.tensor([2, 3, 4, 5]),
                       torch.tensor([1, 2, 3, 4]))
    r = model_randomization(tiny_vgg, 7, units, mode="cascade")
    assert r.labels == ["trained", "conv3+", "conv2+", "conv1+"]
    for m in r.signed:
        assert torch.allclose(r.signed[m][0], torch.ones(4, dtype=torch.float32))  # trained vs itself
    # The gradient depends on every weight, so re-initializing all convs must change it.
    assert r.signed["gradient"][-1].nanmean() < 0.9
    assert model_randomization(tiny_vgg, 7, units, mode="independent").labels == [
        "trained", "conv3", "conv2", "conv1"]


def test_sobel_edges():
    flat = torch.full((3, 8, 8), 0.5)
    assert sobel_edges(flat).abs().max() == 0
    step = flat.clone()
    step[:, :, 4:] = 1.0
    edges = sobel_edges(step)[0]
    assert edges[:, 3:5].min() > 0 and edges[:, :2].max() == 0
