import torch

from modern_deconvnet.project import project_unit
from modern_deconvnet.reference_deconv import reference_deconv


def test_captum_matches_hand_written_deconvnet(tiny_vgg):
    torch.manual_seed(1)
    images = torch.rand(3, 3, 32, 32)
    units = [(5, 2, 3), (0, 7, 7), (11, 4, 0)]  # (channel, h, w) per image, at the last ReLU

    channel, h, w = (torch.tensor(u) for u in zip(*units))
    captum = project_unit(tiny_vgg, 7, images, channel, h, w)

    for i, (c, hh, ww) in enumerate(units):
        expected = reference_deconv(tiny_vgg, 7, images[i : i + 1], c, hh, ww)
        assert expected.abs().sum() > 0
        torch.testing.assert_close(captum[i : i + 1], expected)


def test_deconv_differs_from_plain_gradient(tiny_vgg):
    """The ReLU override must actually be active, or we'd just be computing a gradient."""
    image = torch.rand(1, 3, 32, 32, requires_grad=True)
    tiny_vgg(image)[0, 5, 2, 3].backward()
    deconv = project_unit(tiny_vgg, 7, image.detach(), 5, torch.tensor([2]), torch.tensor([3]))
    assert not torch.allclose(deconv, image.grad)
