# Open Questions

## Must resolve before a proposal

Completed: P1 confirmation on fresh Cora/Citeseer salt-3 splits passed
100/100 release gates on each dataset.

1. Compare against DPLP and PriDyG under a common workload, privacy
   unit, and evaluation protocol. PriDyG has no public implementation
   in the checked surface; any DPLP adaptation must be labelled
   DPLP-style rather than official.
   The current DPLP-style nonprivate adapter lacks headroom against the
   public-only baseline, so its DP version is not prioritized.
2. Split release/training/evaluation into separate processes whose file
   permissions enforce the post-release boundary. Current output-level
   workload and label-source audits pass, but the implementation still
   lives in one process.
3. Measure candidate-domain coverage and its effect on utility.
4. Improve low-budget graph release. Oracle-label diagnostics show
   released topology loses most useful structure on Cora/Citeseer at
   `epsilon=1`; checkpoint selection alone does not fix this.

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
