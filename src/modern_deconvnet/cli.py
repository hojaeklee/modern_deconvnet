"""``modern-deconvnet``: scan a dataset, then draw one Fig. 2-style figure per layer.

    uv run modern-deconvnet --model alexnet --max-images 2000

The scan is cached in ``<out>/scan.pt``, so re-running with different
``--num-channels`` or ``--method`` only redraws the figures.
"""

import argparse
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .data import ImageFiles, download_imagenette
from .figures import layer_figure
from .models import load_model
from .scan import TopK, scan

# Captum warns every time it hooks the ReLUs, which is once per channel drawn.
warnings.filterwarnings("ignore", message="Setting backward hooks on ReLU")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="alexnet", choices=["alexnet", "vgg16", "vgg16-keras"])
    p.add_argument("--weights", type=Path, help="local .h5 for vgg16-keras (downloaded if omitted)")
    p.add_argument("--data", default="imagenette",
                   help="folder of images, or 'imagenette' to download Imagenette's val split")
    p.add_argument("--data-cache", type=Path, default=Path("data"))
    p.add_argument("--max-images", type=int, default=2000, help="random subset size (CPU budget)")
    p.add_argument("--k", type=int, default=9, help="top images per channel")
    p.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3", "layer4", "layer5"])
    p.add_argument("--method", nargs="+", default=["deconv"], choices=["deconv", "guided"],
                   help="projection(s) to draw; 'guided' is guided backprop, for comparison")
    p.add_argument("--num-channels", type=int, default=16, help="channels drawn per layer")
    p.add_argument("--seed", type=int, default=0, help="for the image subset and channel choice")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, default=None, help="default: outputs/<model>")
    return p.parse_args(argv)


def load_or_scan(args, net, dataset) -> dict[str, TopK]:
    cache = args.out / "scan.pt"
    key = dict(model=args.model, data=str(dataset.paths[0].parent), n=len(dataset), k=args.k,
               seed=args.seed, layers=args.layers)
    if cache.exists():
        saved = torch.load(cache, weights_only=False)
        if saved["key"] == key:
            print(f"reusing {cache}")
            return saved["topk"]
    layers = {name: net.layers[name] for name in args.layers}
    topk = scan(net.features, layers, dataset, k=args.k, batch_size=args.batch_size)
    torch.save({"key": key, "topk": topk}, cache)
    return topk


def pick_channels(top: TopK, n: int, seed: int) -> list[int]:
    """A seeded random sample of channels, skipping dead ones (never positive)."""
    alive = (top.values[:, 0] > 0).nonzero().flatten().tolist()
    return sorted(random.Random(seed).sample(alive, min(n, len(alive))))


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out = args.out or Path("outputs") / args.model
    args.out.mkdir(parents=True, exist_ok=True)

    net = load_model(args.model, args.weights)
    root = download_imagenette(args.data_cache) if args.data == "imagenette" else Path(args.data)
    dataset = ImageFiles(root, args.max_images, args.seed)
    print(f"{args.model}: {len(dataset)} images from {root}")

    topk = load_or_scan(args, net, dataset)
    for name in args.layers:
        channels = pick_channels(topk[name], args.num_channels, args.seed)
        for method in args.method:
            fig = layer_figure(net.features, net.layers[name], topk[name], channels, dataset,
                               title=f"{args.model} {name}", method=method)
            path = args.out / f"{name}_{method}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
