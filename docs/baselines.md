# Baseline Status

## PriDyG

The paper is `arXiv:2608.04255`. No public implementation was found in
the repository search performed for this project. PriDyG combines
dynamic private aggregation with an LLM text branch; a static
topology-only rewrite would not be an official reproduction.

Status: **not directly runnable**.

## DPLP

The official code is public and targets USAir, Celegans, Yeast, and PB.
The repository does not include its generated split or training
matrices. A protocol-equivalent native reconstruction was run on those
four datasets and is reported separately in
[`dplp-native-reproduction.md`](dplp-native-reproduction.md). Those
numbers are not mixed with the Cora/Citeseer/BlogCatalog post-release
table.

Status: **native datasets independently rerun; not directly comparable
to the post-release interface without adaptation**.

If adapted, the baseline will be named `DPLP-style` and will document:

- degree projection from the private training graph;
- private path or neighborhood subgraph extraction;
- DP training and sensitivity accounting;
- the exact public target workload used for evaluation.

Published AUC values from a different dataset or workload will not be
inserted into a common table.

### DPLP-style nonprivate headroom gate

A SEAL/DPLP-style path-2 GCN adapter was run for three seeds and 50
epochs on the fresh salt-3 splits.

| Dataset | Random-negative AUC median | Hard-negative AUC median | P1 public-only |
| --- | ---: | ---: | ---: |
| Cora | 0.7370 | 0.7095 | 0.7409 |
| Citeseer | 0.7203 | 0.7084 | 0.8266 |

The nonprivate adapter does not beat the public-only baseline on either
split. Therefore a DP version of this adapter is not a priority: noise
can only reduce its headroom further.

This is not a negative result about the official DPLP paper. It is a
negative result about this workload-specific adapter. The separate
native reproduction is reported in
[`dplp-native-reproduction.md`](dplp-native-reproduction.md).

## Implemented references

- public feature-only F3;
- row-shuffled topology placebo;
- fixed topology/public rank fusion;
- label-RR F3;
- private-topology nonprivate reference.
