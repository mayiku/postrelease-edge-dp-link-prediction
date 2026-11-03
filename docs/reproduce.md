# Reproduce

## Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `.mat` input must contain:

- `net`: an undirected adjacency matrix;
- `group`: a node-feature matrix.

These are the formats used by the public SEAL link-prediction datasets.

## Run

```bash
python -m src.postrelease_lp \
  --dataset cora \
  --data /path/to/cora.mat \
  --split-salt 2 \
  --release-draws 10 \
  --topology-dimension 8 \
  --output outputs/cora_split2.json
```

The command runs three paired models on every release:

1. public feature-only F3;
2. public + released topology F3;
3. public + row-shuffled topology placebo F3.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Scope

The released graph is the only private-data output. Model training,
topology recomputation, placebo construction, and evaluation are
post-processing of that output and public information.
