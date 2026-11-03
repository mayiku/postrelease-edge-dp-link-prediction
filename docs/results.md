# Results

All values are medians over independent edge-DP releases. The
six-dataset/split Cora-Citeseer runs use 10 releases each.
BlogCatalog uses 5 releases.

## Cross-split AUC

| Dataset | Split | Public | Joint | Placebo | Joint-public | Joint-placebo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 0 | 0.72365 | 0.83298 | 0.71442 | +0.11202 | +0.12262 |
| Cora | 1 | 0.70226 | 0.83339 | 0.69253 | +0.13316 | +0.13760 |
| Cora | 2 | 0.72313 | 0.84204 | 0.71684 | +0.11809 | +0.12474 |
| Citeseer | 0 | 0.81520 | 0.86785 | 0.81026 | +0.05195 | +0.05549 |
| Citeseer | 1 | 0.78669 | 0.84389 | 0.78242 | +0.05414 | +0.05478 |
| Citeseer | 2 | 0.81144 | 0.86431 | 0.81289 | +0.05448 | +0.05190 |
| BlogCatalog | 0 | 0.52846 | 0.87073 | 0.53257 | +0.34259 | +0.33722 |

The current 8-dimensional topology configuration beats public-only and
placebo in `70 / 70` releases.

## BlogCatalog confirmation

```text
public median AP   = 0.01441
joint median AP    = 0.26248
placebo median AP  = 0.01530
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

## Lower-budget boundary

Five independent releases were used per lower-budget probe.

| Dataset | Split | epsilon | Public | Joint | Placebo | Joint-placebo | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Cora | 2 | 1.0 | 0.29690 | 0.33594 | 0.31698 | +0.00387 | fail |
| Cora | 2 | 2.0 | 0.38844 | 0.54033 | 0.40758 | +0.13705 | pass |
| Cora | 2 | 3.5 | 0.72313 | 0.84204 | 0.71684 | +0.12474 | pass |
| Citeseer | 1 | 1.0 | 0.29295 | 0.40550 | 0.34406 | +0.03918 | fail |
| Citeseer | 1 | 2.0 | 0.77209 | 0.77000 | 0.75804 | +0.01196 | fail |
| Citeseer | 1 | 3.5 | 0.78669 | 0.84389 | 0.78242 | +0.05478 | pass |
| BlogCatalog | 0 | 1.0 | 0.52451 | 0.62668 | 0.52157 | +0.08919 | pass |
| BlogCatalog | 0 | 3.5 | 0.52846 | 0.87073 | 0.53257 | +0.33722 | pass |

Low-budget failure can involve both information loss in randomized
response and instability of the debiased-label training objective.
