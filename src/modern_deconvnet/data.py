"""A flat image dataset: every image file under a folder, as 224x224 [0, 1] RGB.

Any folder works (ImageNet val, Imagenette, your own photos). The default is
Imagenette's validation split (3,925 images, 10 ImageNet classes), because it
downloads quickly and needs no account.
"""

import random
import tarfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The usual ImageNet eval crop. No normalization here: models.Preprocess does it.
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


class ImageFiles(Dataset):
    """All images under ``root``, optionally a seeded random subset of ``max_images``.

    Items are ``(image, index)`` so the scan can remember which image a top
    activation came from and load it again later.
    """

    def __init__(self, root: Path, max_images: int | None = None, seed: int = 0):
        paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise FileNotFoundError(f"no images under {root}")
        if max_images is not None and max_images < len(paths):
            paths = sorted(random.Random(seed).sample(paths, max_images))
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[i]) as im:
            return TRANSFORM(im.convert("RGB")), i


def download_imagenette(dest: Path) -> Path:
    """Download and extract Imagenette's val split; returns the folder of images."""
    val_dir = dest / "imagenette2-320" / "val"
    if val_dir.exists():
        return val_dir
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "imagenette2-320.tgz"
    if not archive.exists():
        torch.hub.download_url_to_file(IMAGENETTE_URL, str(archive))
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("imagenette2-320/val")]
        tar.extractall(dest, members=members, filter="data")
    return val_dir
