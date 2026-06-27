# Position Predictor — Progress across versions: football TE

Best model per version (head-to-head vs market on covered rows) against the compute and data volume it took. Use this to judge whether a version's ranking gains justified its added cost.

## Ranking quality (best model vs market)

| version | best model | Spearman | Weighted τ | P@12 | P@24 | Δ Spearman vs market |
|---|---|---|---|---|---|---|
| v1 | elasticnet | 0.723 | 0.728 | 0.60 | 0.77 | -0.059 |

## Cost: compute & data volume

| version | wall (s) | total fit (s) | features | labeled rows | Spearman / fit-s |
|---|---|---|---|---|---|
| v1 | 66.8 | 38.1 | 81 | 2079 | 0.98 |

_Snapshots live in `reports/versions/<stem>/<version>/` (committed; the big regenerable CSVs stay gitignored)._
