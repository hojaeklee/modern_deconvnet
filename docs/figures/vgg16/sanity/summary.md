# Sanity checks: vgg16-keras, 2000 images

Median Spearman (signed) vs the trained map after randomizing the top conv / all convs; mean Spearman (signed) to other channels at the same location; mean Spearman of |map| with Sobel edges, trained and fully random net.

| layer | method | top-conv rand. | all convs rand. | other units | edges (trained) | edges (random) |
|---|---|---|---|---|---|---|
| layer2 | deconv | 0.70 | 0.29 | 0.55 | 0.26 | 0.22 |
| layer2 | guided | 0.92 | 0.76 | 0.85 | 0.30 | 0.21 |
| layer2 | gradient | 0.01 | -0.07 | -0.03 | 0.20 | 0.23 |
| layer3 | deconv | 0.94 | 0.37 | 0.91 | 0.22 | 0.20 |
| layer3 | guided | 0.98 | 0.67 | 0.96 | 0.25 | 0.20 |
| layer3 | gradient | -0.10 | -0.02 | -0.00 | 0.16 | 0.20 |
| layer5 | deconv | 1.00 | 0.35 | 1.00 | 0.08 | 0.06 |
| layer5 | guided | 1.00 | 0.49 | 1.00 | 0.16 | 0.08 |
| layer5 | gradient | -0.04 | -0.00 | -0.06 | -0.04 | 0.07 |
