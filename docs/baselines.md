# Baseline Status

## PriDyG

The paper is `arXiv:2608.04255`. No public implementation was found in
the repository search performed for this project. PriDyG combines
dynamic private aggregation with an LLM text branch; a static
topology-only rewrite would not be an official reproduction.

Status: **not directly runnable**.

## DPLP

The official code is public, but its released experiment pipeline
targets USAir, Celegans, Yeast, and PB and requires its own processed
split and dependency setup. The current project uses public
Cora/Citeseer/BlogCatalog target workloads and an edge-DP graph-release
interface.

Status: **not directly comparable without adaptation**.

If adapted, the baseline will be named `DPLP-style` and will document:

- degree projection from the private training graph;
- private path or neighborhood subgraph extraction;
- DP training and sensitivity accounting;
- the exact public target workload used for evaluation.

Published AUC values from a different dataset or workload will not be
inserted into a common table.

## Implemented references

- public feature-only F3;
- row-shuffled topology placebo;
- fixed topology/public rank fusion;
- label-RR F3;
- private-topology nonprivate reference.
