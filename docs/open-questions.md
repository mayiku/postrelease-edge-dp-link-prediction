# Open Questions

## Must resolve before a proposal

1. Compare against DPLP and PriDyG under a common workload, privacy
   unit, and evaluation protocol.
2. Add an executable post-release data-path audit that fails if model
   code touches the original graph.
3. Measure candidate-domain coverage and its effect on utility.
4. Separate low-budget information loss from debiased-label training
   instability. Current `epsilon=1/2` probes show graph-dependent
   boundaries.

## Threats to validity

- The current positive results use a relatively large `epsilon=3.5`.
- The public candidate domain can exclude many graph edges.
- Cora and Citeseer splits were used during exploratory development.
  BlogCatalog is the first untouched graph, but it is one additional
  dataset.
- The post-release F3 training budget and architecture are not yet
  compared against a broad model family.
