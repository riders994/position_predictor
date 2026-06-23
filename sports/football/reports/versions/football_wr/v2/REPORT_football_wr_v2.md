# Position Predictor — Results: football WR v2

_**WR v2** · migrate-nflreadpy @ 8c77d9f_

_Generated 2026-06-23._ Test seasons **[2019, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 7 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `elasticnet` (val_weighted, 20-yr window) — Spearman **0.803 ± 0.041**, Precision@12 0.57, MAE 2.47 PPG (full eligible universe).
- **Best baseline:** `linear` — Spearman 0.782 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.800 vs our best `elasticnet` 0.776 — the model does **not** beat the market on overall rank.
  On **Precision@12 (tier-1 / WR1)**: market 0.65 vs best `xgboost` 0.60 — model trails market.
  On **Precision@24 (tier-2 / WR2)**: market 0.65 vs best `elasticnet` 0.65 — model trails market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.771 vs `elasticnet` 0.746 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.803 | 0.803 | 0.803 |
| elasticnet | recency_weighted | 0.801 | 0.801 | 0.801 |
| elasticnet | val_weighted | 0.803 | 0.803 | 0.803 |
| lasso | mean | 0.801 | 0.801 | 0.801 |
| lasso | recency_weighted | 0.801 | 0.800 | 0.800 |
| lasso | val_weighted | 0.801 | 0.802 | 0.802 |
| lightgbm | mean | 0.791 | 0.795 | 0.795 |
| lightgbm | recency_weighted | 0.791 | 0.792 | 0.792 |
| lightgbm | val_weighted | 0.791 | 0.795 | 0.795 |
| random_forest | mean | 0.787 | 0.787 | 0.787 |
| random_forest | recency_weighted | 0.787 | 0.787 | 0.787 |
| random_forest | val_weighted | 0.787 | 0.787 | 0.787 |
| ridge | mean | 0.802 | 0.799 | 0.799 |
| ridge | recency_weighted | 0.797 | 0.794 | 0.794 |
| ridge | val_weighted | 0.802 | 0.799 | 0.799 |
| xgboost | mean | 0.796 | 0.797 | 0.797 |
| xgboost | recency_weighted | 0.795 | 0.796 | 0.796 |
| xgboost | val_weighted | 0.796 | 0.797 | 0.797 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2022 | 185 | 145 | 0.78 | 0.828 | 0.75 |
| 2023 | 165 | 109 | 0.66 | 0.795 | 0.50 |
| 2024 | 171 | 161 | 0.94 | 0.815 | 0.67 |
| 2025 | 176 | 115 | 0.65 | 0.763 | 0.67 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 (WR1) | Precision@24 (WR2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.800 | 0.771 | 0.65 | 0.65 |
| elasticnet | 0.776 | 0.746 | 0.56 | 0.65 |
| ridge | 0.768 | 0.744 | 0.56 | 0.62 |
| lasso | 0.773 | 0.741 | 0.54 | 0.65 |
| random_forest | 0.744 | 0.733 | 0.58 | 0.62 |
| lightgbm | 0.754 | 0.733 | 0.58 | 0.60 |
| xgboost | 0.757 | 0.729 | 0.60 | 0.62 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.773, P@12 0.58; **without**: Spearman 0.766, P@12 0.57.
- **Decision:** keep the `ngs_efficiency` block — it improves (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 4.49 | 0.798 |
| gbm_poisson | 4.96 | 0.798 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`elasticnet`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 4 | 6 | 7 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| Spearman | 0.805 | 0.806 | 0.802 | 0.799 | 0.799 | 0.806 |


## Data volume

- **Feature rows:** 5400 (3646 labeled with a next-season target)
- **Features:** 81 columns
- **Seasons:** 1999–2025 (26 seasons)
- **Labeled rows per era:** boxscore 1885, snaps 599, ngs 1162

## Compute & efficiency

- **Experiment wall-clock:** 122.5s (total model fit time 70.3s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=7). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| elasticnet | val_weighted | 0.15 | 2792 | 81 | 0.803 | 0.57 | 5.49 |
| elasticnet | mean | 0.07 | 2792 | 81 | 0.803 | 0.57 | 11.79 |
| lasso | val_weighted | 0.21 | 2792 | 81 | 0.802 | 0.55 | 3.75 |
| lasso | mean | 0.10 | 2792 | 81 | 0.801 | 0.55 | 7.89 |
| elasticnet | recency_weighted | 0.08 | 2792 | 81 | 0.801 | 0.57 | 10.57 |
| lasso | recency_weighted | 0.11 | 2792 | 81 | 0.800 | 0.55 | 7.15 |
| ridge | val_weighted | 0.14 | 2792 | 81 | 0.799 | 0.57 | 5.57 |
| ridge | mean | 0.06 | 2792 | 81 | 0.799 | 0.57 | 13.47 |
| xgboost | mean | 1.86 | 2792 | 81 | 0.797 | 0.60 | 0.43 |
| xgboost | val_weighted | 2.71 | 2792 | 81 | 0.797 | 0.60 | 0.29 |
| xgboost | recency_weighted | 1.60 | 2792 | 81 | 0.796 | 0.60 | 0.50 |
| lightgbm | val_weighted | 1.70 | 2792 | 81 | 0.795 | 0.58 | 0.47 |
| lightgbm | mean | 0.94 | 2792 | 81 | 0.795 | 0.58 | 0.84 |
| ridge | recency_weighted | 0.06 | 2792 | 81 | 0.794 | 0.55 | 13.27 |
| lightgbm | recency_weighted | 0.97 | 2792 | 81 | 0.792 | 0.60 | 0.81 |
| random_forest | recency_weighted | 2.82 | 2792 | 81 | 0.787 | 0.58 | 0.28 |
| random_forest | mean | 2.83 | 2792 | 81 | 0.787 | 0.58 | 0.28 |
| random_forest | val_weighted | 5.55 | 2792 | 81 | 0.787 | 0.58 | 0.14 |

## Figures

![Spearman by window](figures/report_football_wr_windows.png)

![Model vs market](figures/report_football_wr_vs_market.png)
