# Open Questions

The current evidence is sufficient to start a scoped research proposal,
but not to claim universal novelty or deployment-ready DP link
prediction.

## Still open for the proposal

1. Complete novelty clearance beyond the requested four-way matrix.
   EdgeRefine already combines local edge-LDP randomized response,
   Jaccard filtering, private graph release, and downstream GNN
   learning. PGB studies one-shot private graph generation for
   multiple queries. CF-DPGNN and DP-MuLP remain abstract-level
   evidence. The proposal must distinguish this work from those
   systems rather than claiming a broad "release-first" novelty.
2. Add a same-workload DP DPLP baseline if the project wants a direct
   model-training comparison. The official DPLP native reproduction is
   on USAir/C.ele/Yeast/PB and is reported separately; the local
   DPLP-style adapter has no nonprivate headroom on the P1 workload.
   PriDyG has no public implementation and cannot be run empirically.
3. Establish external validity beyond repeated random releases.
   The 100 confirmatory runs measure mechanism randomness on one fixed
   test set per dataset; they are not 100 independent graph samples.
   BlogCatalog remains exploratory under the stricter P1 chronology.
4. State the candidate-domain limit. The method ranks only pairs in a
   public top-20 target workload drawn from a top-400 release domain;
   it does not predict arbitrary unseen pairs or recover the full graph.
5. Explain the graph-release evaluation semantics. Test-pair
   membership is not used as a training label, but its
   randomized-response bit belongs to the published `G_DP` and may
   affect global post-processing. This is transductive release
   evaluation, not an inductive service.
6. The utility claim is regime-dependent: `epsilon=3.5` passes the
   frozen gates, while `epsilon=1` fails on Cora/Citeseer and
   `epsilon=2` remains below usable absolute utility. No dynamic graph
   or private-query workload is covered.
7. Operational separation is enforced by code, import-graph tests, and
   a training-only container, not by an adversarial OS sandbox.

## Threats to validity

- The current positive results use a relatively large `epsilon=3.5`.
- Relative low-budget gains are not enough; future gates must include a
  minimum absolute utility threshold.
- Shrinking the public release domain improved relative lift but not
  usable absolute utility at `epsilon=1`.
- The public candidate domain can exclude many graph edges.
- Cora and Citeseer splits were used during exploratory development.
  BlogCatalog is the first untouched graph, but it is one additional
  dataset.
- The post-release F3 training budget and architecture are not yet
  compared against a broad model family.
- Published aggregate evaluation uses private test labels in the
  trusted evaluator. Those metrics are research evidence, not part of
  a deployable prediction API's DP claim.
