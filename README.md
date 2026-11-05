# Post-release Learned Fusion for Edge-Private Link Prediction

Research log for edge-level differentially private link prediction on
graphs with public node features.

> Status: exploratory research, not peer reviewed. Results below are
> frozen experiments with recorded protocols and controls, but the
> work is still being prepared for a formal proposal and paper.

## Research question

Can one edge-DP release of a public-candidate graph support useful link
prediction when all later computation is restricted to:

- the released graph;
- public node identifiers and node features;
- public target-workload construction?

The current answer is **yes in the tested regimes**, provided that the
released graph is used by a learned post-release model rather than a
fixed topology/public rank blend.

## Method

```mermaid
flowchart LR
    A[Public target workload<br/>top-20 feature neighbors] --> B[Public release domain<br/>top-400 feature neighbors]
    B --> C[Edge-level randomized response<br/>epsilon = 3.5, delta = 0]
    C --> D[Public Jaccard denoising]
    D --> E[Released topology features<br/>size, degree, overlap]
    A --> F[Public pair features]
    C --> G[Debiased released labels]
    E --> H[Post-release F3]
    F --> H
    G --> H
    H --> I[Link ranking<br/>AUC / AP / Recall@K]
```

The post-release model does not read the original private graph or
private test labels. Training on the released graph and public
features is therefore post-processing of the edge-DP output.

## Headline results

Exploratory median AUC over independent edge-DP releases:

| Dataset | Split | Public only | Joint F3 | Topology placebo | Joint - placebo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 0 | 0.7237 | 0.8330 | 0.7144 | +0.1226 |
| Cora | 1 | 0.7023 | 0.8334 | 0.6925 | +0.1376 |
| Cora | 2 | 0.7231 | 0.8420 | 0.7168 | +0.1247 |
| Citeseer | 0 | 0.8152 | 0.8679 | 0.8103 | +0.0555 |
| Citeseer | 1 | 0.7867 | 0.8439 | 0.7824 | +0.0548 |
| Citeseer | 2 | 0.8114 | 0.8643 | 0.8129 | +0.0519 |
| BlogCatalog | 0 | 0.5285 | 0.8707 | 0.5326 | +0.3372 |

The joint model beats the public-only and placebo models in all
`70 / 70` measured releases across these seven dataset/split settings.

### Confirmatory P1 results

Fresh, pre-registered splits with 100 releases each:

| Dataset | Split | Public | Fixed fusion | Placebo | Joint | Joint-public | Joint-placebo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 3 | 0.7409 | 0.4884 | 0.7347 | 0.8389 | +0.0976 | +0.1046 |
| Citeseer | 3 | 0.8266 | 0.5291 | 0.8242 | 0.8714 | +0.0440 | +0.0471 |

Joint beat public-only, matched fixed fusion, and placebo in
`100 / 100` releases on both datasets. Workload train/valid/test
intersections were zero, and test target bits were not used in F3
training.

The public features alone are weak on BlogCatalog (`AUC 0.5285`) but
strong on Citeseer (`AUC 0.7867-0.8152`). The released topology adds
signal in both regimes.

## Privacy-utility boundary

Five-release probes at lower budgets show a graph-dependent boundary:

| Dataset | Split | epsilon=1 | epsilon=2 | epsilon=3.5 |
| --- | ---: | --- | --- | --- |
| Cora | 2 | fail | relative-only, unusable | pass |
| Citeseer | 1 | fail | fail | pass |
| BlogCatalog | 0 | pass | not needed | pass |

At `epsilon=1`, Cora and Citeseer public-only learning also collapses
because released labels and topology are much noisier. At `epsilon=2`,
Cora meets the relative gate but remains far below usable absolute
utility (`joint AUC 0.540`). At `epsilon=3.5`, all settings are usable.

Oracle-label diagnostics isolate the bottleneck: at `epsilon=1`, giving
the model true target labels restores public-only learning, but
released topology adds almost no signal on Cora/Citeseer. BlogCatalog
retains structural signal. Low-budget failure is therefore primarily
information loss, not only checkpoint selection.

Shrinking the release domain also does not restore usable utility:
Cora at `epsilon=1` has joint AUC about `0.487 / 0.455 / 0.367` for
`U=50 / 100 / 200`, all below the pre-registered absolute floor.

## Private-topology reference

A nonprivate reference computes the same 8 topology features from the
full private graph and trains F3 with true labels:

| Dataset | Public | Private-topology reference | Post-release joint |
| --- | ---: | ---: | ---: |
| Cora split 2 | 0.7290 | 0.8560 | 0.8420 |
| Citeseer split 1 | 0.7893 | 0.8125 | 0.8439 |
| BlogCatalog | 0.5325 | 0.8816 | 0.8707 |

This is a reference, not a strict upper bound: released labels and
released topology induce a different training geometry.

## Why a learned post-release model?

A fixed `0.75 * topology_rank + 0.25 * public_rank` fusion passed on
Cora but failed on Citeseer, where public features were much stronger.
Replacing the fixed blend with an F3 trained on released labels and
public features removed that boundary.

## Mechanism ablation

The released topology block contains:

```text
size      = [0]
degree    = [1,2,3,4]
overlap   = [5,6,7]
context   = [8]
```

Findings:

- F3 outperforms a linear readout by `+0.16` to `+0.34` median AUC.
- Removing the overlap group loses `-0.02` to `-0.12` median AUC and
  much more Average Precision.
- Removing the context indicator `[8]` does not hurt and can slightly
  improve results.

The confirmed current configuration uses the 8-dimensional topology
block `[0,1,2,3,4,5,6,7]`.

## Privacy scope

This project currently claims **edge-level DP only**:

- one undirected edge add/remove is the privacy unit;
- upper-triangle randomized response uses `epsilon=3.5`, `delta=0`;
- all downstream model training reads only the released graph and
  public side information;
- node-level DP and attribute-level DP are outside the claim.

## Repository contents

- [`src/postrelease_lp.py`](src/postrelease_lp.py): minimal end-to-end
  reproduction implementation.
- [`src/postrelease_core.py`](src/postrelease_core.py): pure
  post-processing operations with no graph-file access.
- [`scripts/private_topology_upper_bound.py`](scripts/private_topology_upper_bound.py):
  nonprivate same-statistic reference.
- [`scripts/release_graph.py`](scripts/release_graph.py): private-data
  release phase.
- [`scripts/train_postrelease.py`](scripts/train_postrelease.py):
  artifact-only training phase.
- [`scripts/evaluate.py`](scripts/evaluate.py): trusted evaluator phase.
- [`scripts/run_separated.py`](scripts/run_separated.py): runs all
  three phases as separate subprocesses.
- [`Dockerfile.train`](Dockerfile.train): trainer image containing no
  raw graph loader or release implementation.
- [`docs/reproduce.md`](docs/reproduce.md): install and run commands.
- [`docs/method.md`](docs/method.md): release interface and model.
- [`docs/pipelines.md`](docs/pipelines.md): versioned result semantics
  and baseline labels.
- [`docs/results.md`](docs/results.md): full result tables.
- [`docs/privacy.md`](docs/privacy.md): threat model and accounting.
- [`docs/mechanism.md`](docs/mechanism.md): ablations and interpretation.
- [`docs/baselines.md`](docs/baselines.md): DPLP/PriDyG comparability
  status.
- [`docs/related-work.md`](docs/related-work.md): task/interface/scope
  comparison with DPLP, PrivIM, and PriDyG.
- [`docs/novelty-matrix.md`](docs/novelty-matrix.md): one-page novelty
  and reproducibility matrix.
- [`docs/open-questions.md`](docs/open-questions.md): unresolved work.
- [`results/`](results/): compact, sanitized CSV results.

The AST-based import-boundary test rejects a training dependency chain
that reaches `src.postrelease_lp`, `src.raw_graph`,
`load_graph_data`, `graph_edges`, `scipy.io`, or `loadmat`. GitHub
Actions runs this check on every push and pull request.

The CI also builds `Dockerfile.train` and asserts that raw graph modules
are physically absent from the trainer image.

## Quick start

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.postrelease_lp \
  --dataset cora \
  --data /path/to/cora.mat \
  --split-salt 2 \
  --release-draws 10 \
  --topology-dimension 8 \
  --output outputs/cora_split2.json
```

The implementation defaults to the simplified 8-dimensional topology
block identified by the mechanism ablation.

## Data

Experiments use public link-prediction datasets distributed with SEAL:
Cora, Citeseer, and BlogCatalog. Raw datasets and machine-specific
artifacts are not redistributed in this repository.

## Next steps

1. Compare against DPLP and PriDyG under a common measurement protocol.
2. Audit the post-release data path so production code cannot read the
   raw graph.
3. Separate low-budget information loss from training instability.
4. Complete the research proposal after the baseline and privacy
   closure are stable.
