# Low-budget Diagnosis

## Question

Is failure at `epsilon=1` caused by noisy-label optimization or by
loss of graph information in the DP release?

## Checkpoint selection

Released validation bits are post-processing of the DP graph and can
select a checkpoint without extra privacy cost.

At `epsilon=1`, released-validation selection produced:

| Dataset | Joint AUC | Placebo AUC | Joint > placebo |
| --- | ---: | ---: | ---: |
| Cora | 0.38875 | 0.32240 | 5/5 |
| Citeseer | 0.40710 | 0.43247 | 2/5 |
| BlogCatalog | 0.59203 | 0.51550 | 4/5 |

Checkpoint selection did not restore usable Cora/Citeseer utility.

## Oracle-label diagnostic

This diagnostic is explicitly nonprivate and is not part of the method.
It uses true target labels while retaining the topology computed from
the `epsilon=1` released graph.

| Dataset | Public | Joint | Placebo | Joint-placebo |
| --- | ---: | ---: | ---: | ---: |
| Cora | 0.73337 | 0.73067 | 0.71834 | +0.00828 |
| Citeseer | 0.79003 | 0.78832 | 0.78840 | -0.00108 |
| BlogCatalog | 0.53281 | 0.66217 | 0.53273 | +0.12577 |

## Conclusion

At `epsilon=1`, Cora/Citeseer released topology contains little useful
incremental signal, even with clean labels. The main bottleneck is
information loss in randomized-response graph release, not merely
training instability. BlogCatalog is an exception where topology
survives.

At `epsilon=2`, Cora's released-label model can meet a relative gate
while its absolute AUC remains only `0.540`. Future gates must include
an absolute utility floor.
