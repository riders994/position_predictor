# Position Predictor — Progress across versions: football RB

Best model per version (head-to-head vs market on covered rows) against the compute and data volume it took. Use this to judge whether a version's ranking gains justified its added cost.

## Ranking quality (best model vs market)

| version | best model | Spearman | Weighted τ | P@12 | P@24 | Δ Spearman vs market |
|---|---|---|---|---|---|---|
| v1 | xgboost | 0.706 | 0.654 | 0.55 | 0.72 | -0.026 |
| v2 | xgboost | 0.706 (+0.000 vs prev) | 0.654 | 0.55 | 0.72 | -0.026 |
| v3 | xgboost | 0.711 (+0.006 vs prev) | 0.658 | 0.58 | 0.72 | -0.021 |
| v4 | xgboost | 0.749 (eval set changed †) | 0.644 | 0.53 | 0.74 | -0.059 |

> † **Eval set changed** between versions (different test folds — e.g. excluding COVID 2020), so cross-version Spearman is not directly comparable at that boundary; compare against the market column (re-scored on each version's own folds) instead.

## Cost: compute & data volume

| version | wall (s) | total fit (s) | features | labeled rows | Spearman / fit-s |
|---|---|---|---|---|---|
| v1 | 244.4 | 166.3 | 80 | 2975 | 0.17 |
| v2 | 273.2 | 198.4 | 80 | 2975 | 0.17 |
| v3 | 273.6 | 199.2 | 81 | 2975 | 0.18 |
| v4 | 268.9 | 215.7 | 81 | 2728 | 0.65 |

_Snapshots live in `reports/versions/<stem>/<version>/` (committed; the big regenerable CSVs stay gitignored)._
