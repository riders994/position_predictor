# Position Predictor — Progress across versions: football QB

Best model per version (head-to-head vs market on covered rows) against the compute and data volume it took. Use this to judge whether a version's ranking gains justified its added cost.

## Ranking quality (best model vs market)

| version | best model | Spearman | Weighted τ | P@6 | P@12 | Δ Spearman vs market |
|---|---|---|---|---|---|---|
| v1 | lasso | 0.629 | 0.518 | 0.50 | 0.62 | -0.066 |
| v2 | xgboost | 0.552 (eval set changed †) | 0.531 | 0.56 | 0.67 | -0.110 |

> † **Eval set changed** between versions (different test folds — e.g. excluding COVID 2020), so cross-version Spearman is not directly comparable at that boundary; compare against the market column (re-scored on each version's own folds) instead.

## Cost: compute & data volume

| version | wall (s) | total fit (s) | features | labeled rows | Spearman / fit-s |
|---|---|---|---|---|---|
| v1 | 64.2 | 37.0 | 73 | 1456 | 0.86 |
| v2 | 129.3 | 36.3 | 73 | 1338 | 0.81 |

_Snapshots live in `reports/versions/<stem>/<version>/` (committed; the big regenerable CSVs stay gitignored)._
