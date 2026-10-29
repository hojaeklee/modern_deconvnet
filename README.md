# Modern DeConvNet

Zeiler & Fergus, *Visualizing and Understanding Convolutional Networks*
([arXiv 1311.2901](https://arxiv.org/abs/1311.2901)), reproduced on modern
PyTorch. Milestone 1 is the paper's Fig. 2: for each layer, the top-9 dataset
images per feature map, the deconvnet projection of each one's strongest
activation, and the matching image patch.

The projection itself is Captum's `NeuronDeconvolution`. This repo is the
harness around it: the dataset scan, receptive-field crops and figures.

## Run it

```bash
uv sync
uv run modern-deconvnet --model alexnet                # downloads Imagenette val (~340 MB) on first run
uv run modern-deconvnet --model vgg16 --method deconv guided
uv run modern-deconvnet --model alexnet --data /path/to/imagenet/val --max-images 10000
uv run pytest
```

Figures go to `outputs/<model>/layer{1..5}_{method}.png`. The scan is cached
in `outputs/<model>/scan.pt`, so changing `--num-channels` or `--method` only redraws (`--seed`
also picks the image subset, so it rescans). On a 4-core CPU, VGG16 scans about 7 images/s and AlexNet is much
faster; `--max-images` sets the budget.

Models: `alexnet` and `vgg16` use torchvision's ImageNet weights.
`vgg16-keras` loads the original Caffe VGG16 weights from the Keras mirror
(`uv sync --extra keras`), for machines that can't reach download.pytorch.org.

## How it works

| Stage | File | What it does |
|---|---|---|
| Model | `models.py` | Model as one `nn.Sequential`: input normalization first, so projections land in [0, 1] pixel space; ReLUs made non-inplace so Captum's hooks work. Maps "layer 1..5" to ReLU indices. |
| Scan | `scan.py` | One forward pass per batch with forward hooks on all layers. Per channel, keeps the k images with the largest spatial max, and where it was. |
| Project | `project.py` | `NeuronDeconvolution` from one unit per image, using a callable `neuron_selector` so each image picks its own (c, h, w). |
| Crop | `receptive_field.py` | Closed-form receptive field (size, jump, start) of a unit, used to crop both projection and patch. |
| Draw | `figures.py` | 3x3 projections next to 3x3 patches, per channel. |
| Reference | `reference_deconv.py` | The deconvnet by hand (switches, unpool, rectify, transposed conv). Not used by the pipeline; the test checks Captum matches it exactly. |

## Example output

`docs/figures/vgg16/` has every layer for VGG16 (original Caffe weights),
2,000 Imagenette val images, 16 random live channels per layer, deconvnet and
guided backprop. For example
[layer 3, deconvnet](docs/figures/vgg16/layer3_deconv.jpg) and
[layer 5, guided backprop](docs/figures/vgg16/layer5_guided.jpg).
Regenerate with:

```bash
uv sync --extra keras
uv run modern-deconvnet --model vgg16-keras --max-images 2000 --method deconv guided
```

## What to expect

Layers 1 to 3 look like the paper: colour blobs and edges, then textures,
circles, letters. On VGG16's layers 4 and 5 the pure deconvnet projections get
noisy and look much the same across images, while guided backprop stays sharp.
That is a known property of deconvnet, not a bug here: its backward pass ignores
the forward ReLU masks, so only the pooling switches carry information about
the image (see Nie et al., [1805.07039](https://arxiv.org/abs/1805.07039)).
AlexNet, closer to the paper's network, has fewer layers for that to compound
over. The image patches still show what each unit responds to.

## Milestone 2: sanity checks

Do the projections depend on the trained network, or mostly on the image?
`sanity.py` runs the tests from Adebayo et al., *Sanity Checks for Saliency
Maps* ([1810.03292](https://arxiv.org/abs/1810.03292)) on the same units the
Fig. 2 harness draws, with the plain gradient as a control:

- **Model randomization**: re-initialize conv weights, top-down (*cascade*)
  or one at a time (*independent*), and project the same unit of the same
  image again. Score: Spearman rank correlation with the trained map, signed.
- **Unit sensitivity**: project other channels at the same location. Similar
  maps mean the projection says little about *which* unit it came from.
- **Edge baseline**: correlation of |map| with a Sobel edge map of the patch.

```bash
uv run modern-deconvnet-sanity --model vgg16-keras --max-images 2000   # reuses the Fig. 2 scan
```

Results for VGG16 (16 units per layer, 2,000 Imagenette images) are in
`docs/figures/vgg16/sanity/`, e.g. [layer 5, cascade](docs/figures/vgg16/sanity/layer5_cascade.jpg).
Median signed Spearman with the trained map:

| layer | method | top conv randomized | all convs randomized | other channels, same spot |
|---|---|---|---|---|
| layer3 | deconv | 0.94 | 0.37 | 0.91 |
| layer3 | guided | 0.98 | 0.67 | 0.96 |
| layer3 | gradient | -0.10 | -0.02 | -0.00 |
| layer5 | deconv | 1.00 | 0.35 | 1.00 |
| layer5 | guided | 1.00 | 0.49 | 1.00 |
| layer5 | gradient | -0.04 | -0.00 | -0.06 |

Deconvnet and guided backprop fail the test the way Adebayo et al. and Nie
et al. ([1805.07039](https://arxiv.org/abs/1805.07039)) predict. At layer 5
you can re-initialize the top 7 of 13 convs and the map barely moves, and
every channel at a location gives the same map. The pictures come from the
image and the lower layers, not from the unit. The gradient passes: it changes
as soon as any weight does. Even with *every* conv random, deconv and guided
maps still correlate 0.35 to 0.5 with the trained ones, because max-pool
switches and (for guided) ReLU masks are computed from the image.

## Milestone 3: DINOv2 with registers

The same protocol on a vision transformer. Units are (a) the MLP hidden
neurons of a block (the 1536 GELU outputs in ViT-S) and (b) features of a
sparse autoencoder trained here on one block's residual stream. For each unit:
the top-9 images with the unit's per-token activation laid over each, then a
projection panel comparing four ways back to the image.

```bash
uv run modern-deconvnet-vit --max-images 500             # CPU subset: ~8 min on 4 cores, 5 of them SAE training
uv run modern-deconvnet-vit --max-images 3925 --sanity   # all of Imagenette val, GPU recommended
uv run modern-deconvnet-vit --random-init --max-images 64 --sae-steps 200   # plumbing check, no download
```

Weights come from the Hugging Face hub via timm
(`vit_small_patch14_reg4_dinov2.lvd142m`). Where the hub is blocked, download
Meta's `dinov2_vits14_reg4_pretrain.pth` some other way and pass `--weights`.
Images are 224x224, so the patch grid is 16x16 (patch size 14).

Output in `outputs/dinov2-s-reg/`: `<site>_top9.png` per site (e.g.
`block8_mlp`, `block8_sae`), `<site>_unit<u>.png` projection panels,
`summary.md` (SAE fit, and how many units peak on a CLS/register token rather
than a patch) and, with `--sanity`, `sanity/` with the cascading
randomization test from milestone 2.

| Stage | File | What it does |
|---|---|---|
| Model | `vit.py` | Loads DINOv2 from timm and writes its forward pass out by hand (patchify, position, CLS + 4 registers, 12 pre-norm blocks with LayerScale). The backward rule of GELU, LayerNorm and the attention matmuls is a parameter (`Rules`); the forward value never changes. |
| SAE | `sae.py` | TopK sparse autoencoder (Gao et al., [2406.04093](https://arxiv.org/abs/2406.04093)) on patch tokens of one block's residual stream. 8x expansion, k = 32. |
| Scan | `vit_scan.py` | Top-k images per unit over the 16x16 patch grid, reusing `scan.TopK`. Also the largest activation on CLS/register tokens. |
| Project | `vit_project.py` | The four projections, each from the unit's strongest patch token. |
| Draw | `vit_figures.py` | Top-9 heatmap grids and the projection panels. No receptive-field crop: attention makes every token's receptive field the whole image. |
| Sanity | `vit_sanity.py` | Re-initializes transformer blocks (Linear and LayerNorm; LayerScale kept) and reuses `sanity.similarity`. |

The four projections:

- **gradient**: d(unit)/d(pixels), the control.
- **GELU deconv**: the deconvnet's ReLU rule (backward passes `relu(grad)`,
  ignoring the input) applied at every GELU. A hack: a ViT has no max-pool
  switches, and LayerNorm and softmax keep their true derivatives.
- **Chefer**: gradient-weighted attention rollout (Chefer, Gur & Wolf,
  [2103.15679](https://arxiv.org/abs/2103.15679)), one value per patch.
- **AttnLRP**: Achtibat et al., [2402.05602](https://arxiv.org/abs/2402.05602),
  as gradient x input with its rules swapped in: GELU passes relevance through,
  LayerNorm's 1/std is held constant, the two attention matmuls split relevance
  evenly. Written by hand (the `lxt` package targets Hugging Face models and
  pulls in transformers and bitsandbytes); `tests/test_vit.py` checks each rule
  conserves relevance. It's the plain (epsilon = 0) variant, without the gamma
  rule the AttnLRP authors suggest for less noisy ViT maps.

## Reading the code

Read in pipeline order: `models.py`, `scan.py`, `project.py` with
`reference_deconv.py` beside it, `receptive_field.py`, `figures.py`, then
`cli.py` to see them wired together. The tests are short and double as
specifications: `test_project.py` (Captum equals the hand-written
deconvnet), `test_receptive_field.py` (closed-form RF equals the region with
nonzero gradient) and `test_scan.py` (streaming top-k equals brute force).
For milestone 3, read `vit.py` first (it doubles as a DINOv2 architecture
summary), then `vit_scan.py`, `vit_project.py`, `sae.py`; `test_vit.py`
checks the hand-written forward against timm and each AttnLRP rule on its own.
