# Mechanism Ablation

Three independent releases per dataset were used for exploratory
mechanism attribution.

## Median AUC

| Dataset | Public | Full F3 | Linear | No overlap | No context | Placebo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.7288 | 0.8527 | 0.6852 | 0.7215 | 0.8557 | 0.7193 |
| Citeseer | 0.7907 | 0.8476 | 0.6866 | 0.7830 | 0.8474 | 0.7895 |
| BlogCatalog | 0.5284 | 0.8652 | 0.5348 | 0.8417 | 0.8623 | 0.5337 |

## Median Average Precision

| Dataset | Full F3 | Linear | No overlap | No context | Placebo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.4111 | 0.1830 | 0.1331 | 0.4205 | 0.1285 |
| Citeseer | 0.3540 | 0.2710 | 0.2285 | 0.3600 | 0.2381 |
| BlogCatalog | 0.2562 | 0.0138 | 0.0753 | 0.2480 | 0.0151 |

## Findings

1. F3 is materially better than a linear model on the same inputs.
2. The common-neighbor / overlap features are the dominant topology
   signal.
3. The binary context indicator is redundant.
4. A row-shuffled topology placebo falls back to the public-only level.

The confirmed main configuration uses eight topology features:

```text
size + degree + overlap
```
