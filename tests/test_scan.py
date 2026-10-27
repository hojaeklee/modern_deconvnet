import torch
from torch.utils.data import TensorDataset

from modern_deconvnet.scan import scan


class Indexed(TensorDataset):
    def __getitem__(self, i):
        return self.tensors[0][i], i


def test_scan_matches_brute_force(tiny_vgg):
    torch.manual_seed(0)
    images = torch.rand(20, 3, 32, 32)
    result = scan(tiny_vgg, {"a": 4, "b": 7}, Indexed(images), k=3, batch_size=6, num_workers=0)

    with torch.no_grad():
        acts = tiny_vgg[:8](images)  # (N, C, H, W) at layer 7
    per_image_max = acts.flatten(2).max(2).values  # (N, C)
    expected = per_image_max.T.topk(3, dim=1)

    top = result["b"]
    torch.testing.assert_close(top.values, expected.values)
    # Image indices must agree wherever there are no ties (dead channels tie at 0).
    no_ties = (top.values[:, -1] > 0) & (top.values.diff(dim=1) < 0).all(dim=1)
    assert no_ties.sum() > 5
    assert torch.equal(top.image_idx[no_ties], expected.indices[no_ties])
    # The stored location really holds that value.
    c = torch.arange(acts.shape[1]).unsqueeze(1).expand_as(top.h)
    torch.testing.assert_close(acts[top.image_idx, c, top.h, top.w], top.values)
