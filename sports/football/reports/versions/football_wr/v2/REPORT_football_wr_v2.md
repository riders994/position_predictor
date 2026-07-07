# Position Predictor — Results: football WR v2

_**WR v2** · primary @ 2bc2224-dirty_

_Generated 2026-07-06._ Test seasons **[2019, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 7 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `elasticnet` (mean, 20-yr window) — Spearman **0.804 ± 0.042**, Precision@12 0.57, MAE 2.65 PPG (full eligible universe).
- **Best baseline:** `linear` — Spearman 0.786 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.802 vs our best `elasticnet` 0.777 — the model does **not** beat the market on overall rank.
  On **Precision@12 (tier-1 / WR1)**: market 0.65 vs best `lightgbm` 0.62 — model trails market.
  On **Precision@24 (tier-2 / WR2)**: market 0.67 vs best `random_forest` 0.67 — model trails market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.774 vs `elasticnet` 0.748 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.804 | 0.804 | 0.804 |
| elasticnet | recency_weighted | 0.803 | 0.802 | 0.802 |
| elasticnet | val_weighted | 0.804 | 0.804 | 0.804 |
| lasso | mean | 0.803 | 0.802 | 0.802 |
| lasso | recency_weighted | 0.801 | 0.801 | 0.801 |
| lasso | val_weighted | 0.803 | 0.802 | 0.802 |
| lightgbm | mean | 0.790 | 0.794 | 0.794 |
| lightgbm | recency_weighted | 0.791 | 0.793 | 0.793 |
| lightgbm | val_weighted | 0.790 | 0.795 | 0.795 |
| random_forest | mean | 0.786 | 0.788 | 0.788 |
| random_forest | recency_weighted | 0.787 | 0.789 | 0.789 |
| random_forest | val_weighted | 0.786 | 0.788 | 0.788 |
| ridge | mean | 0.802 | 0.801 | 0.801 |
| ridge | recency_weighted | 0.799 | 0.796 | 0.796 |
| ridge | val_weighted | 0.802 | 0.801 | 0.801 |
| xgboost | mean | 0.795 | 0.797 | 0.797 |
| xgboost | recency_weighted | 0.795 | 0.796 | 0.796 |
| xgboost | val_weighted | 0.795 | 0.797 | 0.797 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2022 | 185 | 145 | 0.78 | 0.830 | 0.75 |
| 2023 | 165 | 109 | 0.66 | 0.794 | 0.58 |
| 2024 | 171 | 161 | 0.94 | 0.817 | 0.58 |
| 2025 | 176 | 115 | 0.65 | 0.767 | 0.67 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 (WR1) | Precision@24 (WR2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.802 | 0.774 | 0.65 | 0.67 |
| elasticnet | 0.777 | 0.748 | 0.58 | 0.65 |
| ridge | 0.769 | 0.747 | 0.56 | 0.62 |
| lasso | 0.774 | 0.745 | 0.56 | 0.65 |
| xgboost | 0.762 | 0.740 | 0.60 | 0.62 |
| random_forest | 0.750 | 0.740 | 0.58 | 0.67 |
| lightgbm | 0.758 | 0.730 | 0.62 | 0.64 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.774, P@12 0.57; **without**: Spearman 0.768, P@12 0.60.
- **Decision:** drop the `ngs_efficiency` block — no top-12 gain out-of-fold, so the ngs era collapses to the snaps schema (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 4.49 | 0.798 |
| gbm_poisson | 4.98 | 0.814 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`elasticnet`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 4 | 6 | 7 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| Spearman | 0.806 | 0.808 | 0.804 | 0.799 | 0.799 | 0.806 |


## Data volume

- **Feature rows:** 5400 (3646 labeled with a next-season target)
- **Features:** 81 columns
- **Seasons:** 1999–2025 (26 seasons)
- **Labeled rows per era:** boxscore 1885, snaps 599, ngs 1162

## Compute & efficiency

- **Experiment wall-clock:** 65.1s (total model fit time 45.8s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=7). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| elasticnet | mean | 0.07 | 2792 | 81 | 0.804 | 0.57 | 11.99 |
| elasticnet | val_weighted | 0.12 | 2792 | 81 | 0.804 | 0.57 | 6.67 |
| lasso | val_weighted | 0.17 | 2792 | 81 | 0.802 | 0.55 | 4.80 |
| lasso | mean | 0.08 | 2792 | 81 | 0.802 | 0.55 | 9.75 |
| elasticnet | recency_weighted | 0.07 | 2792 | 81 | 0.802 | 0.55 | 12.18 |
| ridge | val_weighted | 0.05 | 2792 | 81 | 0.801 | 0.55 | 16.45 |
| ridge | mean | 0.02 | 2792 | 81 | 0.801 | 0.55 | 36.40 |
| lasso | recency_weighted | 0.08 | 2792 | 81 | 0.801 | 0.55 | 9.52 |
| xgboost | mean | 1.11 | 2792 | 81 | 0.797 | 0.58 | 0.72 |
| xgboost | val_weighted | 1.33 | 2792 | 81 | 0.797 | 0.58 | 0.60 |
| ridge | recency_weighted | 0.02 | 2792 | 81 | 0.796 | 0.53 | 31.98 |
| xgboost | recency_weighted | 0.84 | 2792 | 81 | 0.796 | 0.57 | 0.95 |
| lightgbm | val_weighted | 3.13 | 2792 | 81 | 0.795 | 0.60 | 0.25 |
| lightgbm | mean | 2.55 | 2792 | 81 | 0.794 | 0.60 | 0.31 |
| lightgbm | recency_weighted | 1.80 | 2792 | 81 | 0.793 | 0.58 | 0.44 |
| random_forest | recency_weighted | 1.59 | 2792 | 81 | 0.789 | 0.57 | 0.50 |
| random_forest | val_weighted | 2.76 | 2792 | 81 | 0.788 | 0.57 | 0.29 |
| random_forest | mean | 1.41 | 2792 | 81 | 0.788 | 0.57 | 0.56 |

## Figures

![Spearman by window](figures/report_football_wr_windows.png)

![Model vs market](figures/report_football_wr_vs_market.png)
