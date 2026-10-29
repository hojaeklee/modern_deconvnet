"""``modern-deconvnet-sanity``: the milestone 2 sanity checks, on the Fig. 2 units.

    uv run modern-deconvnet-sanity --model vgg16-keras --max-images 2000

Reuses the Fig. 2 scan (``<out>/scan.pt``, same flags as ``modern-deconvnet``)
and, per layer, takes the top-1 image of ``--num-units`` random live channels.
Writes ``<out>/sanity/``: one figure per layer and randomization mode,
``results.json`` and a ``summary.md`` table. See ``sanity.py`` for the tests.
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .cli import load_or_scan, pick_channels
from .data import ImageFiles, download_imagenette
from .figures import to_display
from .models import load_model
from .sanity import (ALL_METHODS, RandomizationResult, Units, convs_below, edge_similarity, make_units,
                     model_randomization, randomize, unit_sensitivity)

warnings.filterwarnings("ignore", message="Setting backward hooks on ReLU")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="alexnet", choices=["alexnet", "vgg16", "vgg16-keras"])
    p.add_argument("--weights", type=Path)
    p.add_argument("--data", default="imagenette")
    p.add_argument("--data-cache", type=Path, default=Path("data"))
    p.add_argument("--max-images", type=int, default=2000)
    p.add_argument("--k", type=int, default=9, help="must match the Fig. 2 run to reuse its scan")
    p.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3", "layer4", "layer5"],
                   help="layers to scan (match the Fig. 2 run to reuse its scan)")
    p.add_argument("--test-layers", nargs="+", default=["layer2", "layer3", "layer5"],
                   help="layers to run the sanity checks on")
    p.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=ALL_METHODS)
    p.add_argument("--modes", nargs="+", default=["cascade", "independent"], choices=["cascade", "independent"])
    p.add_argument("--num-units", type=int, default=16, help="units (channels) tested per layer")
    p.add_argument("--n-other", type=int, default=4, help="other channels per unit, for unit sensitivity")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, default=None, help="default: outputs/<model>")
    return p.parse_args(argv)


def randomization_figure(result: RandomizationResult, units: Units, title: str) -> plt.Figure:
    """Top: the maps for image 0 at every step, one row per method. Bottom: Spearman curves."""
    methods = list(result.example)
    cols = len(result.labels) + 1
    fig = plt.figure(figsize=(1.15 * cols + 1, 1.25 * len(methods) + 3.2))
    grid = fig.add_gridspec(len(methods) + 1, cols, height_ratios=[1] * len(methods) + [2.6])

    t, l, b, r = units.boxes[0]
    patch = units.images[0, :, t:b, l:r]
    for row, m in enumerate(methods):
        for col, img in enumerate([patch, *(to_display(c[None])[0] for c in result.example[m])]):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title("image" if col == 0 else result.labels[col - 1], fontsize=7, rotation=45)
            if col == 0:
                ax.set_ylabel(m, fontsize=9)

    ax = fig.add_subplot(grid[-1, :])
    x = torch.arange(len(result.labels))
    for m in methods:
        s = result.signed[m]
        median = s.nanmedian(dim=1).values
        q1, q3 = s.nanquantile(0.25, dim=1), s.nanquantile(0.75, dim=1)
        ax.plot(x, median, marker="o", label=m)
        ax.fill_between(x, q1, q3, alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x, result.labels, rotation=45, fontsize=7)
    ax.set_ylim(-0.3, 1.05)
    ax.set_ylabel("Spearman vs trained\n(signed, median + IQR)")
    ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def summary_table(results: dict) -> str:
    lines = ["| layer | method | top-conv rand. | all convs rand. | other units | edges (trained) | edges (random) |",
             "|---|---|---|---|---|---|---|"]
    for layer, r in results.items():
        for m, row in r["methods"].items():
            cells = [row.get("cascade_top"), row.get("cascade_all"), row["unit_sensitivity"]["signed"],
                     row["edges_trained"], row["edges_random"]]
            lines.append(f"| {layer} | {m} | " + " | ".join("–" if c is None else f"{c:.2f}" for c in cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out = args.out or Path("outputs") / args.model
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "sanity"
    out.mkdir(exist_ok=True)

    net = load_model(args.model, args.weights)
    root = download_imagenette(args.data_cache) if args.data == "imagenette" else Path(args.data)
    dataset = ImageFiles(root, args.max_images, args.seed)
    topk = load_or_scan(args, net, dataset)

    results = {}
    for name in args.test_layers:
        layer_idx = net.layers[name]
        top = topk[name]
        channels = pick_channels(top, args.num_units, args.seed)
        random.Random(args.seed).shuffle(channels)  # image 0 of the figure is a random unit
        channel = torch.tensor(channels)
        images = torch.stack([dataset[i][0] for i in top.image_idx[channel, 0].tolist()])
        units = make_units(net.features, layer_idx, images, channel, top.h[channel, 0], top.w[channel, 0])
        print(f"{name}: {len(channels)} units, receptive field up to {units.boxes[0]}")

        per_method = {m: {} for m in args.methods}
        for mode in args.modes:
            r = model_randomization(net.features, layer_idx, units, args.methods, mode, args.seed)
            fig = randomization_figure(r, units, f"{args.model} {name}: {mode} randomization")
            path = out / f"{name}_{mode}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            print(f"wrote {path}")
            for m in args.methods:
                per_method[m][mode] = {"labels": r.labels, "signed": r.signed[m].tolist(),
                                       "abs": r.absolute[m].tolist()}
                if mode == "cascade":
                    per_method[m]["cascade_top"] = float(r.signed[m][1].nanmedian())
                    per_method[m]["cascade_all"] = float(r.signed[m][-1].nanmedian())

        sens = unit_sensitivity(net.features, layer_idx, units, args.methods, args.n_other, args.seed)
        edges = edge_similarity(net.features, layer_idx, units, args.methods)
        rand_net = randomize(net.features, convs_below(net.features, layer_idx), args.seed)
        edges_rand = edge_similarity(rand_net, layer_idx, units, args.methods)
        for m in args.methods:
            per_method[m].update(unit_sensitivity=sens[m], edges_trained=edges[m], edges_random=edges_rand[m])
        results[name] = {"channels": channels, "methods": per_method}

    (out / "results.json").write_text(json.dumps(results, indent=1))
    table = summary_table(results)
    (out / "summary.md").write_text(
        f"# Sanity checks: {args.model}, {len(dataset)} images\n\n"
        "Median Spearman (signed) vs the trained map after randomizing the top conv / all convs; "
        "mean Spearman (signed) to other channels at the same location; "
        "mean Spearman of |map| with Sobel edges, trained and fully random net.\n\n" + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
