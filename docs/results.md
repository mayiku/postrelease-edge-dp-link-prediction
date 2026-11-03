# Results

All values are medians over independent edge-DP releases. The
six-dataset/split Cora-Citeseer runs use 10 releases each.
BlogCatalog uses 5 releases.

## Cross-split AUC

| Dataset | Split | Public | Joint | Placebo | Joint-public | Joint-placebo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 0 | 0.72365 | 0.83255 | 0.71364 | +0.11104 | +0.11820 |
| Cora | 1 | 0.70226 | 0.83443 | 0.69233 | +0.13327 | +0.13891 |
| Cora | 2 | 0.72313 | 0.83641 | 0.71921 | +0.11625 | +0.11875 |
| Citeseer | 0 | 0.81520 | 0.86959 | 0.81073 | +0.05599 | +0.05950 |
| Citeseer | 1 | 0.78669 | 0.84400 | 0.78361 | +0.05507 | +0.05500 |
| Citeseer | 2 | 0.81144 | 0.86492 | 0.81269 | +0.05392 | +0.05300 |
| BlogCatalog | 0 | 0.52846 | 0.86524 | 0.53097 | +0.33710 | +0.33476 |

Joint beats public-only and placebo in `70 / 70` releases.

## BlogCatalog confirmation

```text
public median AP   = 0.01486
joint median AP    = 0.25845
placebo median AP  = 0.01506
```

BlogCatalog was selected before observing edge-label utility and
served as the first previously untouched dataset.

## Fixed-fusion baseline

The original fixed rank fusion was:

```text
0.75 * released topology rank + 0.25 * public cosine rank
```

It passed on Cora split 2 (`0.7451` vs `0.7007`) but failed on
Citeseer split 1 (`0.7025` vs `0.7869`). The post-release model
removes this regime failure by learning the topology contribution.
