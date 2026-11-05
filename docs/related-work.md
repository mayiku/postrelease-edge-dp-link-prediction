# Related Work

This note positions the current post-release link-prediction pipeline
against the three closest private graph learning systems. It is a
scope and interface comparison, not a claim that published AUC values
from different tasks and workloads are mutually comparable.

## DPLP (ICDE 2024)

DPLP studies edge-level differentially private link prediction with
subgraph GNNs. It protects the existence of one link while training a
neighborhood- or path-subgraph model, and derives the gradient
sensitivity from the number of subgraphs affected by that link. The
released artifact is the trained model, not a private graph. Its
native evaluation uses `USAir`, `C.ele`, `Yeast`, and `PB`.

Therefore, DPLP is nearest to this project on the privacy unit and
prediction task, but differs on the output interface: DPLP trains on
the private graph, whereas this project publishes one edge-DP graph
over a public candidate domain and performs all learning afterward as
post-processing.

## PrivIM (ICDE 2025)

PrivIM studies influence maximization under unbounded node-level DP.
It trains a subgraph-based GNN with dual-stage adaptive frequency
sampling and outputs a learned seed selector for influence spread. Its
evaluation covers Email, Bitcoin, LastFM, HepPh, Facebook, Gowalla,
and a large-scale Friendster stress test.

PrivIM is not a link-prediction baseline under a stronger or weaker
version of the same graph-release interface. It changes both the task
(graph-level influence maximization) and the protected unit
(node-level rather than edge-level privacy).

## PriDyG (arXiv 2026)

PriDyG formulates edge-level DP dynamic graph inference. New edges are
buffered and processed exactly once by incremental private multi-hop
aggregation; disjoint edge batches permit parallel composition, so the
total edge-DP cost does not grow with the number of model updates.
Its LLM branch reads node text, which is public under its edge-level
threat model, and a confidence gate combines the GNN and LLM scores.
Link prediction is a post-processing cosine score over privatized
representations, with randomized-response calibration on a disjoint
candidate set. The reported datasets are Cora, PubMed, `ogbn-arxiv`,
and `ogbn-products`.

### Reproducibility gap

As of 2026-11-05, the PriDyG manuscript does not link a public
implementation, and an unauthenticated GitHub repository search for
`PriDyG` returned zero repositories. This is evidence of a missing
artifact, not evidence that the method is invalid or that the authors
will not release code later.

We therefore do not include PriDyG in an empirical comparison. A
static topology-only reimplementation would omit incremental PMA,
parallel-composition accounting, LLM fusion, and streaming evaluation;
calling such a rewrite a PriDyG baseline would be misleading.

## Gap relative to this work

The current project is the only one in this set whose primary output
is a reusable edge-DP graph release over a public candidate domain,
followed by post-release link prediction trained only on that release
and public features. DPLP trains a private model directly; PrivIM
returns a node-private influence selector; PriDyG exposes a private
streaming inference service. These differences must remain explicit in
the proposal.

## Additional adjacent work not in the four-way matrix

The requested novelty matrix covers DPLP, PrivIM, and PriDyG. It is not
a complete literature clearance.

- **EdgeRefine** already combines local edge-LDP randomized response,
  Jaccard-based filtering, private graph release, and downstream GNN
  training. The remaining distinctions are central versus local DP,
  public-workload-conditioned release, and learned post-release F3.
- **PGB** studies one-shot private graph generation for multiple
  queries, so "release once and reuse" is not new by itself.
- **CF-DPGNN** and **DP-MuLP** are close edge-private link-prediction
  systems, but only abstract-level evidence is currently available.
- **AdvSGM** and older private graph-release work already cover
  reusable private representations for downstream predictions.

The proposal must therefore claim a specific interface and mechanism,
not first private graph release or first post-release link prediction.

