# Pipeline Versions

Numbers from different post-processing pipelines are not directly
interchangeable.

## P0: fixed rank fusion

Private output:

- one edge-DP randomized-response graph at `epsilon=3.5, delta=0`.

Post-processing:

- public Jaccard denoising;
- topology centroid;
- fixed `0.75/0.25` topology/public rank fusion.

Result on the original confirmatory split:

- Cora split 2: `0.7451`;
- Citeseer split 1: `0.7025`;
- Citeseer public floor under the original baseline definition:
  `0.7869`.

Conclusion: the fixed pipeline has a pre-registered Citeseer boundary
failure.

## P1: post-release learned F3

Private output:

- the same one edge-DP randomized-response graph.

Post-processing:

- public Jaccard denoising;
- topology recomputation;
- debiased train bits read from the released graph;
- F3 trained on public features plus 8 topology features.

Result:

- Cora split 2 joint: `0.84204`;
- Citeseer split 1 joint: `0.84389`.

P1 changes the post-processing model class. It does not reuse or
reverse the P0 result. The Citeseer behavior changes because the P1
model learns a feature-dependent topology contribution.

Fresh pre-registered P1 splits:

| Dataset | Split | Public | Fixed fusion | Joint | Runs joint > fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 3 | 0.7409 | 0.4884 | 0.8389 | 100/100 |
| Citeseer | 3 | 0.8266 | 0.5291 | 0.8714 | 100/100 |

The P1 confirmatory run used 100 independent releases per dataset and
passed its absolute, relative, and workload-isolation gates.

## Baseline labels

| Baseline | Training labels | Privacy state |
| --- | --- | --- |
| raw cosine | none | zero budget |
| P1 public-only F3 | released train bits from `G_DP` | same `epsilon=3.5` release |
| label-RR F3 | separately released train labels | separate `epsilon=3.5` scenario |
| nonprivate public F3 | true train labels | nonprivate |
| private-topology reference | true train labels plus full private graph | nonprivate |

In particular:

- `0.7231 / 0.7867 / 0.5285` are P1 released-label public-only F3
  values;
- `0.7290 / 0.7893 / 0.5325` are nonprivate true-label public F3
  values;
- old fixed-fusion floors must not be described as P1 floors.

## Accounting rule

The main P1 pipeline consumes only the graph-release budget:

```text
epsilon_total = epsilon_graph = 3.5
delta_total   = 0
```

Jaccard, topology recomputation, label debiasing, and F3 training are
post-processing of `G_DP` and public inputs. No second label-release
mechanism is run inside P1.

If the standalone label-RR F3 baseline were released together with the
graph output and both were exposed, that would require separate
composition accounting. It is a separate baseline scenario, not a
component of P1.
