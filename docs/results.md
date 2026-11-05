# Results

All values are medians over independent edge-DP releases. The
six-dataset/split Cora-Citeseer runs use 10 releases each.
BlogCatalog uses 5 releases.

The native DPLP reproduction uses different datasets and a different
interface. It is reported separately in
[`dplp-native-reproduction.md`](dplp-native-reproduction.md) and
[`../results/dplp_native.csv`](../results/dplp_native.csv).

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
| Cora | 2 | 2.0 | 0.38844 | 0.54033 | 0.40758 | +0.13705 | relative-only |
| Cora | 2 | 3.5 | 0.72313 | 0.84204 | 0.71684 | +0.12474 | pass |
| Citeseer | 1 | 1.0 | 0.29295 | 0.40550 | 0.34406 | +0.03918 | fail |
| Citeseer | 1 | 2.0 | 0.77209 | 0.77000 | 0.75804 | +0.01196 | fail |
| Citeseer | 1 | 3.5 | 0.78669 | 0.84389 | 0.78242 | +0.05478 | pass |
| BlogCatalog | 0 | 1.0 | 0.52451 | 0.62668 | 0.52157 | +0.08919 | pass |
| BlogCatalog | 0 | 3.5 | 0.52846 | 0.87073 | 0.53257 | +0.33722 | pass |

The relative gate is insufficient at low absolute utility. Cora at
`epsilon=2` meets the relative margin while retaining only `0.540`
joint AUC, so it is not counted as a practical pass.

## Low-budget diagnosis

Released-validation checkpoint selection at `epsilon=1`:

| Dataset | Joint | Placebo | Joint-placebo | Joint > placebo |
| --- | ---: | ---: | ---: | ---: |
| Cora | 0.38875 | 0.32240 | +0.04876 | 5/5 |
| Citeseer | 0.40710 | 0.43247 | -0.02537 | 2/5 |
| BlogCatalog | 0.59203 | 0.51550 | +0.08160 | 4/5 |

Validation selection does not restore usable Cora/Citeseer utility.

Nonprivate true-label diagnostic at `epsilon=1`:

| Dataset | Public | Joint | Placebo | Joint-placebo |
| --- | ---: | ---: | ---: | ---: |
| Cora | 0.73337 | 0.73067 | 0.71834 | +0.00828 |
| Citeseer | 0.79003 | 0.78832 | 0.78840 | -0.00108 |
| BlogCatalog | 0.53281 | 0.66217 | 0.53273 | +0.12577 |

At `epsilon=1`, Cora/Citeseer released topology has little useful
structure even when target labels are supplied to the diagnostic.
BlogCatalog still retains structure. The dominant low-budget failure
is therefore information loss in the released graph.

## Low-budget release-domain sweep

Cora split 2 at `epsilon=1`, three releases per domain:

| Release candidates/node | Public | Joint | Placebo | Joint-placebo |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 0.29758 | 0.48692 | 0.31380 | +0.19635 |
| 100 | 0.29324 | 0.45522 | 0.33154 | +0.12204 |
| 200 | 0.29571 | 0.36747 | 0.33794 | +0.00662 |

Smaller domains increase relative lift but fail the `0.65` absolute
utility gate. Release-domain reduction alone does not solve the
low-budget regime.

## Private-topology reference

The reference uses the full private graph and true labels, but the same
public workload and 8-dimensional topology statistics.

| Dataset | Split | Public | Private-topology reference | Post-release joint |
| --- | ---: | ---: | ---: | ---: |
| Cora | 2 | 0.72897 | 0.85605 | 0.84204 |
| Citeseer | 1 | 0.78932 | 0.81251 | 0.84389 |
| BlogCatalog | 0 | 0.53250 | 0.88163 | 0.87073 |

The reference is not a strict upper bound because the post-release
model uses released labels and a different feature geometry.
