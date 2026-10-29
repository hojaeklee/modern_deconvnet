"""``modern-deconvnet-vit``: milestone 3, the Fig. 2 protocol on DINOv2 with registers.

    uv run modern-deconvnet-vit --max-images 500            # CPU-sized subset
    uv run modern-deconvnet-vit --max-images 3925 --sanity  # all of Imagenette val, on a GPU

Steps, each cached in ``<out>/`` so re-runs only redo what changed:

1. Train a TopK SAE on the residual stream of each ``--sae-blocks`` block
   (``sae_block{b}.pt``), from the first ``--sae-images`` images.
2. Scan the images: top-k per MLP neuron (``--mlp-blocks``) and per SAE
   feature (``scan.pt``).
3. Per site, ``<site>_top9.png`` for ``--num-units`` random live units, and
   ``<site>_unit{u}.png`` (the four-method projection panel) for the first
   ``--num-projected`` of them.
4. With ``--sanity``, cascading randomization of the projections
   (``sanity/<site>_cascade.png`` and ``sanity/summary.md``).
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset

from .data import ImageFiles, download_imagenette
from .sae import TopKSAE, explained_variance, train_sae
from .vit import MODELS, load_vit
from .vit_figures import METHOD_LABELS, method_tiles, projection_figure, resize, top9_figure, unit_images
from .vit_project import METHODS
from .vit_sanity import vit_randomization
from .vit_scan import Site, SiteScan, collect_resid, scan_vit


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2-s-reg", choices=list(MODELS))
    p.add_argument("--weights", type=Path, help="local checkpoint, if the Hugging Face hub is unreachable")
    p.add_argument("--random-init", action="store_true",
                   help="random weights: checks the pipeline runs, the figures mean nothing")
    p.add_argument("--data", default="imagenette")
    p.add_argument("--data-cache", type=Path, default=Path("data"))
    p.add_argument("--max-images", type=int, default=500, help="random subset size (CPU budget)")
    p.add_argument("--k", type=int, default=9)
    p.add_argument("--mlp-blocks", nargs="+", type=int, default=[2, 5, 8, 11], help="0-based")
    p.add_argument("--sae-blocks", nargs="*", type=int, default=[8], help="0-based; give no value to skip SAEs")
    p.add_argument("--sae-images", type=int, default=500, help="images whose patch tokens train the SAE")
    p.add_argument("--sae-expansion", type=int, default=8, help="features per residual dimension")
    p.add_argument("--sae-k", type=int, default=32, help="active features per token")
    p.add_argument("--sae-steps", type=int, default=2000)
    p.add_argument("--num-units", type=int, default=16, help="units drawn per site")
    p.add_argument("--num-projected", type=int, default=4, help="of those, units given a projection panel")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--sanity", action="store_true", help="also run cascading randomization")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None, help="default: outputs/<model>")
    return p.parse_args(argv)


def cached(path: Path, key: dict, make):
    """Load ``path`` if it was made with the same ``key``, else ``make()`` and save it."""
    if path.exists():
        saved = torch.load(path, weights_only=False)
        if saved["key"] == key:
            print(f"reusing {path}")
            return saved["value"]
    value = make()
    torch.save({"key": key, "value": value}, path)
    return value


def load_or_train_saes(args, vit, dataset) -> dict[int, TopKSAE]:
    saes = {}
    train_set = Subset(dataset, range(min(args.sae_images, len(dataset))))
    for b in args.sae_blocks:
        key = dict(model=args.model, random=args.random_init, data=str(dataset.paths[0].parent),
                   n=len(train_set), seed=args.seed, block=b, expansion=args.sae_expansion, k=args.sae_k,
                   steps=args.sae_steps)

        def make(b=b):
            acts = collect_resid(vit, b, train_set, args.batch_size, args.device)
            sae, stats = train_sae(acts, args.sae_expansion, args.sae_k, args.sae_steps, seed=args.seed)
            stats["explained_variance"] = explained_variance(sae, acts)
            print(f"SAE block {b}: {stats}")
            return {"state": sae.state_dict(), "stats": stats}

        saved = cached(args.out / f"sae_block{b}.pt", key, make)
        d = vit.model.embed_dim
        sae = TopKSAE(d, args.sae_expansion * d, args.sae_k).to(args.device)
        sae.load_state_dict(saved["state"])
        sae.stats = saved["stats"]
        saes[b] = sae.eval().requires_grad_(False)
    return saes


def pick_units(scan: SiteScan, n: int, seed: int) -> list[int]:
    """A seeded random sample of units that are positive somewhere in the dataset."""
    alive = (scan.top.values[:, 0] > 0).nonzero().flatten().tolist()
    picked = random.Random(seed).sample(alive, min(n, len(alive)))
    return picked  # unsorted: the first --num-projected are a random choice too


def registers_line(scan: SiteScan) -> str:
    """How many units fire harder on CLS/register tokens than on any patch."""
    alive = scan.top.values[:, 0] > 0
    prefix_wins = (scan.prefix_max > scan.top.values[:, 0]) & alive
    return f"{int(prefix_wins.sum())} of {int(alive.sum())} live units peak on a CLS/register token"


def sanity_figure(result, images: torch.Tensor, title: str) -> plt.Figure:
    """Top: maps for image 0 at every randomization step. Bottom: Spearman curves."""
    methods = list(result.example)
    cols = len(result.labels) + 1
    fig = plt.figure(figsize=(1.0 * cols + 1, 1.1 * len(methods) + 3))
    grid = fig.add_gridspec(len(methods) + 1, cols, height_ratios=[1] * len(methods) + [2.4])
    for row, m in enumerate(methods):
        tiles = [resize(images[:1].cpu())[0], *(method_tiles(m, x[None])[0] for x in result.example[m])]
        for col, img in enumerate(tiles):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title("image" if col == 0 else result.labels[col - 1], fontsize=7, rotation=45)
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[m], fontsize=8)
    ax = fig.add_subplot(grid[-1, :])
    x = torch.arange(len(result.labels))
    for m in methods:
        s = result.signed[m]
        ax.plot(x, s.nanmedian(dim=1).values, marker="o", label=METHOD_LABELS[m])
        ax.fill_between(x, s.nanquantile(0.25, dim=1), s.nanquantile(0.75, dim=1), alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x, result.labels, rotation=45, fontsize=7)
    ax.set_ylim(-0.3, 1.05)
    ax.set_ylabel("Spearman vs trained\n(signed, median + IQR)")
    ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out = args.out or Path("outputs") / (args.model + ("-random" if args.random_init else ""))
    args.out.mkdir(parents=True, exist_ok=True)

    vit = load_vit(args.model, args.weights, pretrained=not args.random_init)
    vit.model.to(args.device)
    vit.preprocess.to(args.device)
    root = download_imagenette(args.data_cache) if args.data == "imagenette" else Path(args.data)
    dataset = ImageFiles(root, args.max_images, args.seed)
    print(f"{args.model}: {len(dataset)} images from {root}, on {args.device}")

    saes = load_or_train_saes(args, vit, dataset)
    sites = [Site("mlp", b) for b in args.mlp_blocks] + [Site("sae", b) for b in args.sae_blocks]
    key = dict(model=args.model, random=args.random_init, data=str(root), n=len(dataset), k=args.k,
               seed=args.seed, sites=[s.name for s in sites], sae={b: s.stats for b, s in saes.items()})
    scans = cached(args.out / "scan.pt", key,
                   lambda: scan_vit(vit, sites, saes, dataset, args.k, args.batch_size, args.device))

    summary = [f"# {args.model}{' (random init)' if args.random_init else ''}, {len(dataset)} images\n"]
    summary += [f"SAE block {b}: " + ", ".join(f"{k} {v:.3f}" for k, v in s.stats.items()) for b, s in saes.items()]
    sanity = {}
    for site in sites:
        scan = scans[site]
        units = pick_units(scan, args.num_units, args.seed)
        summary.append(f"{site.name}: {registers_line(scan)}")
        save(top9_figure(vit, saes, site, scan, units, dataset, f"{args.model} {site.name}", args.device),
             args.out / f"{site.name}_top9.png")
        for u in units[: args.num_projected]:
            title = f"{args.model} {site.name}, {'neuron' if site.kind == 'mlp' else 'feature'} {u}"
            save(projection_figure(vit, saes, site, scan, u, dataset, args.methods, title, args.device),
                 args.out / f"{site.name}_unit{u}.png")
        if args.sanity:
            sanity[site.name] = run_sanity(args, vit, saes, site, scan, units, dataset)

    if sanity:
        lines = ["| site | method | top block rand. | all blocks rand. |", "|---|---|---|---|"]
        for name, per_method in sanity.items():
            for m, (top, all_) in per_method.items():
                lines.append(f"| {name} | {m} | {top:.2f} | {all_:.2f} |")
        (args.out / "sanity" / "summary.md").write_text(
            "Median signed Spearman vs the trained map, top-1 image of each unit.\n\n" + "\n".join(lines) + "\n")
        summary += ["", *lines]
    (args.out / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


def run_sanity(args, vit, saes, site: Site, scan: SiteScan, units: list[int], dataset) -> dict:
    out = args.out / "sanity"
    out.mkdir(exist_ok=True)
    first = [unit_images(scan.top, u, dataset) for u in units]  # top-1 image of each unit
    images = torch.stack([im[0] for im, _, _ in first]).to(args.device)
    token = torch.tensor([vit.num_prefix + h[0] * vit.grid[1] + w[0] for _, h, w in first], device=args.device)
    unit = torch.tensor(units, device=args.device)
    r = vit_randomization(vit, saes, site, images, unit, token, args.methods, "cascade", args.seed)
    save(sanity_figure(r, images, f"{args.model} {site.name}: cascading randomization"),
         out / f"{site.name}_cascade.png")
    (out / f"{site.name}_cascade.json").write_text(json.dumps(
        {"units": units, "labels": r.labels, "signed": {m: v.tolist() for m, v in r.signed.items()}}))
    return {m: (float(r.signed[m][1].nanmedian()), float(r.signed[m][-1].nanmedian())) for m in args.methods}


if __name__ == "__main__":
    main()
