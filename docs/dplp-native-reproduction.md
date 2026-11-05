# DPLP Native Reproduction

This is a separate native-dataset track for DPLP-PS. Its rows are not
interchangeable with the Cora/Citeseer/BlogCatalog post-release results.

## Scope and provenance

- Official code:
  `marlowe518/Awesome-Differential-Privacy-and-Meachine-Learning`,
  branch `20230731`, commit `be0ee5e4f4364d661963022de17873529f99c3ab`.
- Native adjacency matrices: SEAL `Python/data/{USAir,Celegans,Yeast,PB}.mat`,
  SEAL commit `ca1f019a15fb0c21796042165b4e6bee73981dd3`.
- Split construction: the author's
  `GNN.link_prediction.utils.do_edge_split`, with the code's fixed split
  seed `234`, `val_ratio=0.05`, and `test_ratio=0.10`.
- Training seed: the DPLP script's default `random_seed=999`.
- Native datasets: USAir, C.ele (`Celegans.mat`), Yeast, and PB.

The official repository does not publish the generated
`*_split_edges` or `*_train.mat` files. The reconstruction therefore
uses the public raw matrices and the author's split function. Each
`*_train.mat` contains only the symmetric training edge set from
`split_edge["train"]`; validation and test positives are absent. This
avoids feeding test context into the test-subgraph extractor.

## Configuration

All runs use:

```text
epsilon=4
delta=1e-5
epochs=50
runs=10
lr=0.01
max_norm=1
num_layers=3 (script default)
```

| Dataset | Hops `k` | Degree threshold `theta` | Batch size |
| --- | ---: | ---: | ---: |
| USAir | 2 | 60 | 128 |
| Celegans / C.ele | 2 | 40 | 128 |
| Yeast | 4 | 40 | 128 |
| PB | 2 | 60 | 1024 |

The per-dataset `k` and `theta` values follow the paper/author notebook,
not the script's single global defaults.

## Environment and reproducibility patch

The local rerun uses Python 3.9 with:

```text
torch 1.13.1+cpu
torch_geometric 2.2.0
torch_sparse 0.6.16+pt113cpu
ogb 1.3.5
numpy 1.21.5
scipy 1.10.1
scikit-learn 1.2.1
```

One numerical guard is applied in
`privacy_analysis/RDP/rdp_convert_dp.py`: an RDP value in
`(-1e-8, 0)` is rounded to zero. The unmodified code rejects an
approximately `-8.6e-12` round-off value at the binary search's
initial `sigma=10000`. The patch changes no split, sensitivity,
subgraph, optimizer, or privacy formula.

The exact patch is
[`patches/dplp-rdp-near-zero.patch`](../patches/dplp-rdp-near-zero.patch).

Yeast extraction uses eight independent subgraph workers. The worker
does not change the split or the deterministic one-hop sampling
configuration (`ratio_per_hop=1.0`, no per-hop cap).

## Results

The rerun summary is stored in
[`results/dplp_native.csv`](../results/dplp_native.csv). Percentages are
AUC at the validation-selected epoch, aggregated over all ten runs
without removing low-utility optimization basins.

| Dataset | Published DPLP-PS, epsilon=4 | Rerun mean +/- sample SD | Rerun median |
| --- | ---: | ---: | ---: |
| USAir | 93.74 +/- 0.04 | 92.38 +/- 2.03 | 93.05 |
| C.ele | 84.12 +/- 0.29 | 81.12 +/- 5.19 | 83.91 |
| Yeast | 92.02 +/- 1.18 | 91.24 +/- 2.35 | 91.64 |
| PB | 90.75 +/- 0.24 | 91.65 +/- 0.01 | 91.65 |

The medians reproduce the published utility regime within about 0.9
AUC points on every dataset. PB is the most stable. USAir and C.ele
show bimodal training behaviour: several runs are close to the paper,
while a smaller number fall into low-utility basins. The median is
therefore reported alongside the unscreened mean.

This audit does not establish that the authors cherry-picked runs. The
original processed split artifacts are unavailable, and the reported
experimental protocol may have selected a stable configuration through
unreleased preprocessing or repeated tuning. It does establish that
the public code plus reconstructed native inputs is not a deterministic
one-command reproduction of the published point estimates.

## Artifact hashes

```text
USAir_split_edges
230158B1E61283A9F901F3DEB3F4522AA3C991327C5D00E9CA7C78278A8C18F1
Celegans_split_edges
D5F2EC52CE8824EF7AD960451F5FDC14244432DC29106CFCEF97522465CF5CBC
Yeast_split_edges
148902AE082214AC5682A7116C50B5411D5D5CA50F8229F2973C205FB6AD622E
PB_split_edges
DE340FED9A4AAFF8F03ABE5594C80A0900DAD40DD14DF56A8FCAC7D1F56E7361

USAir_train.mat
664EC97377D50BDAAB504352843B5EA0D40E6567249CE89010FF333FA65FCF6A
Celegans_train.mat
574AB7922568163F48C1D915E159642B5B89DCAD653EF5771153DE0F1F655D7C
Yeast_train.mat
8D4B987E0B179FD6D4F8F7495DC2BB3C95A1F191300F9062D26502800BEFDEC0
PB_train.mat
1453469A9FB72A15CF45184D299ADD761D5832044EADB2BAD77DACE0C09EFC95
```

