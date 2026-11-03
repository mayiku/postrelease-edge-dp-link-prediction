# Privacy Model

## Protected unit

One undirected graph edge under add/remove adjacency.

## Public inputs

- node identifiers;
- public node features;
- target workload construction;
- fixed public constants.

## DP output

The only private-data output is the randomized-response bit vector over
the public release domain:

```text
epsilon = 3.5
delta   = 0
```

Jaccard denoising, topology recomputation, label debiasing, model
training, fusion, and evaluation are post-processing of that output
and public information.

## Outside the current claim

- node-level DP;
- attribute-level DP;
- private query workloads;
- dynamic graph updates;
- unaudited arbitrary production queries against the raw graph.

## Required implementation invariant

After release, the training and prediction path must not read:

- the original adjacency matrix;
- private edge labels;
- private validation or test labels for model selection.

Test labels may only be used by the trusted evaluator after predictions
are fixed.
