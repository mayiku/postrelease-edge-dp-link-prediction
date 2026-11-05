# Evidence Audit

Audit date: 2026-11-05.

## Checks passed

- The P1 protocol, implementation, tests, and requirements SHA256
  values match `R21-p1-confirm-v1-lock.json`.
- The locked Cora and Citeseer `.mat` files are absent from the repo,
  but redownloading the public SEAL files reproduces both locked
  SHA256 values exactly.
- The 100-run Cora and Citeseer JSON files match the published
  confirmatory medians and win counts.
- A separated-pipeline rerun at Cora salt 3, draw 98, reproduces the
  main-pipeline scores with absolute AUC differences at most
  `1.11e-16` for public, joint, placebo, and fixed fusion.
- `postrelease_lp.py` and `postrelease_core.py` have identical source
  for `canonical`, `edge_target_priority`, `posterior_and_weights`,
  `rank01`, `topology_features`, and `train_model`.
- All local unit/import-boundary tests pass. The latest pushed CI run
  passed `unit-tests`, `import-boundary`, and `container-boundary`.

## Required claim qualifications

- One hundred releases are one hundred randomizations over the same
  graph and test workload, not one hundred independent sampled graphs.
- The test pair is excluded from its own topology support, but its
  randomized-response bit is part of the published graph and may enter
  global denoising. The result is a transductive graph-release
  evaluation, not an inductive unseen-edge service.
- The interface ranks public target candidates only. It does not
  answer arbitrary private-graph queries.
- The confirmed operating point is `epsilon=3.5, delta=0`. Lower
  budgets do not meet the absolute utility gates on Cora/Citeseer.
- The four-way novelty matrix is not a complete literature clearance;
  EdgeRefine and PGB remain closest additional references.
