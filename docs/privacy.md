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

The F3 model's training bits are obtained by querying
`pair in released_edges` on the already released graph. This is not a
second label-release mechanism. The output metadata records the label
source and a post-processing epsilon of zero.

If a standalone label-RR model releases training labels separately,
that is a different scenario and must not be added to the graph-release
pipeline unless both outputs are actually published and jointly
accessible.

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

## Separated interfaces

The reproduction package provides:

```text
release_graph.py
  raw private graph -> release artifact

train_postrelease.py
  release artifact only -> scores

evaluate.py
  scores + private test labels -> metrics
```

`train_postrelease.py` intentionally has no raw-data CLI argument. The
release artifact records epsilon, delta, p, q, the release seed, the
candidate-domain size, and the train/validation/test intersection
audit.
