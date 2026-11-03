# P1 Confirmatory Results

Protocol: pre-registered before running Cora salt 3 and Citeseer
salt 3. Code commit: `76b2438`. Each dataset uses 100 independent
edge-DP releases.

## Results

| Dataset | Public | Fixed fusion | Placebo | Joint | Joint-public | Joint-placebo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.74089 | 0.48840 | 0.73471 | 0.83894 | +0.09759 | +0.10464 |
| Citeseer | 0.82665 | 0.52908 | 0.82421 | 0.87144 | +0.04397 | +0.04710 |

Pairwise wins for joint:

| Dataset | vs. public | vs. fixed fusion | vs. placebo |
| --- | ---: | ---: | ---: |
| Cora | 100/100 | 100/100 | 100/100 |
| Citeseer | 100/100 | 100/100 | 100/100 |

Both datasets cleared the pre-registered `+0.02` margins and `0.70`
absolute AUC floor.

## Isolation checks

- train/test target intersections: `0`;
- train/validation target intersections: `0`;
- validation/test target intersections: `0`;
- training labels: released train bits from `G_DP`;
- test target bits used for training: `false`.

The complete lock was re-hashed after the run and passed.
