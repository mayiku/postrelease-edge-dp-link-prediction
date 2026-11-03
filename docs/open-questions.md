# Open Questions

## Must resolve before a proposal

1. Reconfirm the 8-feature model across all seven dataset/split
   settings.
2. Compare against DPLP and PriDyG under a common workload, privacy
   unit, and evaluation protocol.
3. Add an executable post-release data-path audit that fails if model
   code touches the original graph.
4. Measure candidate-domain coverage and its effect on utility.
5. Test performance under lower `epsilon` and lower prevalence.

## Threats to validity

- The current positive results use a relatively large `epsilon=3.5`.
- The public candidate domain can exclude many graph edges.
- Cora and Citeseer splits were used during exploratory development.
  BlogCatalog is the first untouched graph, but it is one additional
  dataset.
- The post-release F3 training budget and architecture are not yet
  compared against a broad model family.
