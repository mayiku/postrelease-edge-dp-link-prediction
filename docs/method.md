# Method

## Inputs

Public:

- unordered node IDs;
- public node features;
- target workload construction.

Private:

- graph edge existence.

## Candidate domains

- target workload: top-20 public-feature neighbors per node;
- release domain: top-400 public-feature neighbors per node.

Both domains are constructed without reading the private graph.

## Edge-DP release

For every unordered pair in the release domain:

```text
Pr[release 1 | edge present] = exp(3.5) / (1 + exp(3.5))
Pr[release 1 | edge absent]  = 1 / (1 + exp(3.5))
```

Each input coordinate changes with at most one private edge, so the
whole released bit vector satisfies edge-level `epsilon=3.5, delta=0`.

## Public denoising

The released graph is post-processed by a public Jaccard top-K rule
with target average degree 4. The output is a sanitized graph
`G_release`.

## Released topology block

Nine features are recomputed from `G_release`:

```text
0    context size
1-4  endpoint out/in degrees
5-7  common-out, common-in, normalized overlap
8    binary context-presence indicator
```

The target pair's own edge is excluded from its topology support.

## Post-release model

Training labels are the released train-workload bits. Randomized
response is debiased with posterior pseudo-targets and posterior-derived
sample weights.

The F3 model is:

```text
public pair features + released topology -> 64 ReLU -> 1
```

Architecture and optimization:

- Adam;
- 2000 steps;
- batch size 128;
- learning rate 0.001;
- positive weight 10;
- fixed final step.

No private validation or test information is used for checkpoint
selection.

## Planned simplification

Mechanism ablations show that feature 8 is redundant. The next
configuration therefore freezes topology columns `[0,1,2,3,4,5,6,7]`
and must be reconfirmed before replacing the 9-feature results.
