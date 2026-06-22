# Position Predictor — Progress across versions: football WR

Best model per version (head-to-head vs market on covered rows) against the compute and data volume it took. Use this to judge whether a version's ranking gains justified its added cost.

## Ranking quality (best model vs market)

| version | best model | Spearman | Weighted τ | P@12 | P@24 | Δ Spearman vs market |
|---|---|---|---|---|---|---|
| v1 | ridge | 0.748 | 0.712 | 0.52 | 0.69 | -0.019 |
| v2 | lasso | 0.745 (eval set changed †) | 0.747 | 0.56 | 0.69 | -0.046 |

> † **Eval set changed** between versions (different test folds — e.g. excluding COVID 2020), so cross-version Spearman is not directly comparable at that boundary; compare against the market column (re-scored on each version's own folds) instead.

## Cost: compute & data volume

| version | wall (s) | total fit (s) | features | labeled rows | Spearman / fit-s |
|---|---|---|---|---|---|
| v1 | 88.6 | 55.5 | 81 | 3532 | 0.66 |
| v2 | 93.0 | 59.2 | 81 | 3203 | 0.63 |

_Snapshots live in `reports/versions/<stem>/<version>/` (committed; the big regenerable CSVs stay gitignored)._
