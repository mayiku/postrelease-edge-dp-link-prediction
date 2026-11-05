# Novelty Matrix

One-page scope comparison. Entries describe what each system actually
releases or answers; they are not a common-budget leaderboard.

| Work | Task | Privacy model | Interface type | Datasets | Reproducibility status |
| --- | --- | --- | --- | --- | --- |
| **This work (P1)** | Link prediction after one private graph release; public target workload `top-20` features and release domain `top-400` | Edge-level `(epsilon, delta)`; main P1 release uses `epsilon=3.5, delta=0`; downstream F3 training is post-processing | One-shot released graph over a public candidate domain, then public-feature + released-topology training and pair ranking | Cora and Citeseer (confirmatory); BlogCatalog (exploratory) | Public code, frozen P1 protocol, isolation audits, 100-release confirmatory CSVs, and CI boundary checks |
| **DPLP (ICDE 2024)** | Subgraph-GNN link prediction | Edge-level DP over add/remove of one link | Trains on the private graph and releases/query-serves the learned link predictor; no private graph release | USAir, C.ele, Yeast, PB | Official code public; native `.mat` files public via SEAL; native 10-run rerun completed with reconstructed processed inputs, with the reproducibility boundary documented |
| **PrivIM (ICDE 2025)** | Influence maximization / seed selection | Unbounded node-level DP; the paper notes adaptation to edge-level DP is possible, but that is not its main evaluation | Trains a private subgraph GNN and outputs a seed set or seed-selection model | Email, Bitcoin, LastFM, HepPh, Facebook, Gowalla; Friendster scale test | Official code and data directories public at `TechReport2025/PrivIM`; not independently rerun in this project |
| **PriDyG (arXiv 2026)** | Dynamic graph inference: node classification and link prediction | Edge-level DP; incremental PMA uses parallel composition over disjoint edge batches, with max-composition for randomized-response calibration | Streaming/query inference service combining private GNN representations with public-text LLM scores | Cora, PubMed, `ogbn-arxiv`, `ogbn-products` | No public implementation linked in the manuscript or found by repository search as of 2026-11-05; not empirically compared |

## What is actually new here

The novelty is not the invention of edge-level DP, randomized
response, or post-processing. It is the release-first interface:

1. the private graph is queried once on a public candidate domain;
2. the resulting edge-DP release is reusable by later models;
3. public features and released topology are combined only after the
   DP output is fixed;
4. the training/evaluation boundary is enforced by separate phases and
   an import-graph test.

This differs from direct private model training (DPLP), node-private
influence selection (PrivIM), and a dynamic private inference service
(PriDyG).

## What is not claimed

- PriDyG is not reported as an empirical baseline because its public
  artifact is missing.
- Published DPLP results on USAir/C.ele/Yeast/PB are not mixed with the
  Cora/Citeseer/BlogCatalog post-release table.
- PrivIM is not described as an edge-level DP baseline: its primary
  privacy unit and task both differ.
- The current P1 evidence establishes utility under the tested
  candidate-domain and public-feature assumptions; it does not justify
  arbitrary private query access, node-level DP, or dynamic updates.

