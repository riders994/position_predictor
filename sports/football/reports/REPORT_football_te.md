# Position Predictor — Results: football TE v1

_**TE v1** · te-model-v1 @ 20ce892-dirty_

_Generated 2026-06-26._ Test seasons **[2019, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 5 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `lasso` (val_weighted, 20-yr window) — Spearman **0.744 ± 0.069**, Precision@12 0.57, MAE 1.83 PPG (full eligible universe).
- **Best baseline:** `linear` — Spearman 0.726 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.782 vs our best `elasticnet` 0.723 — the model does **not** beat the market on overall rank.
  On **Precision@12 (tier-1 / TE1)**: market 0.69 vs best `xgboost` 0.65 — model trails market.
  On **Precision@24 (tier-2 / TE2)**: market 0.80 vs best `random_forest` 0.80 — model beats market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.756 vs `elasticnet` 0.728 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.728 | 0.737 | 0.737 |
| elasticnet | recency_weighted | 0.717 | 0.721 | 0.721 |
| elasticnet | val_weighted | 0.729 | 0.739 | 0.739 |
| lasso | mean | 0.741 | 0.744 | 0.744 |
| lasso | recency_weighted | 0.735 | 0.735 | 0.735 |
| lasso | val_weighted | 0.741 | 0.744 | 0.744 |
| lightgbm | mean | 0.727 | 0.731 | 0.731 |
| lightgbm | recency_weighted | 0.727 | 0.728 | 0.728 |
| lightgbm | val_weighted | 0.727 | 0.732 | 0.732 |
| random_forest | mean | 0.740 | 0.737 | 0.737 |
| random_forest | recency_weighted | 0.737 | 0.735 | 0.735 |
| random_forest | val_weighted | 0.740 | 0.737 | 0.737 |
| ridge | mean | 0.647 | 0.653 | 0.653 |
| ridge | recency_weighted | 0.627 | 0.631 | 0.631 |
| ridge | val_weighted | 0.648 | 0.654 | 0.654 |
| xgboost | mean | 0.734 | 0.735 | 0.735 |
| xgboost | recency_weighted | 0.730 | 0.728 | 0.728 |
| xgboost | val_weighted | 0.734 | 0.734 | 0.734 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2022 | 99 | 72 | 0.73 | 0.775 | 0.75 |
| 2023 | 96 | 46 | 0.48 | 0.778 | 0.67 |
| 2024 | 96 | 90 | 0.94 | 0.770 | 0.67 |
| 2025 | 107 | 60 | 0.56 | 0.807 | 0.67 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 (TE1) | Precision@24 (TE2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.782 | 0.756 | 0.69 | 0.80 |
| elasticnet | 0.723 | 0.728 | 0.60 | 0.77 |
| random_forest | 0.719 | 0.722 | 0.60 | 0.80 |
| xgboost | 0.709 | 0.718 | 0.65 | 0.77 |
| lasso | 0.722 | 0.710 | 0.58 | 0.78 |
| lightgbm | 0.714 | 0.709 | 0.60 | 0.76 |
| ridge | 0.635 | 0.665 | 0.58 | 0.72 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.696, P@12 0.57; **without**: Spearman 0.694, P@12 0.58.
- **Decision:** drop the `ngs_efficiency` block — no top-12 gain out-of-fold, so the ngs era collapses to the snaps schema (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 4.03 | 0.799 |
| gbm_poisson | 4.41 | 0.798 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`lasso`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| Spearman | 0.743 | 0.740 | 0.739 | 0.738 | 0.731 | 0.728 |


## Data volume

- **Feature rows:** 3027 (2079 labeled with a next-season target)
- **Features:** 81 columns
- **Seasons:** 1999–2025 (26 seasons)
- **Labeled rows per era:** boxscore 1058, snaps 350, ngs 671

## Compute & efficiency

- **Experiment wall-clock:** 66.8s (total model fit time 38.1s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=5). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| lasso | val_weighted | 0.15 | 1588 | 81 | 0.744 | 0.57 | 5.03 |
| lasso | mean | 0.07 | 1588 | 81 | 0.744 | 0.57 | 10.72 |
| elasticnet | val_weighted | 0.12 | 1588 | 81 | 0.739 | 0.60 | 6.34 |
| random_forest | mean | 1.61 | 1588 | 81 | 0.737 | 0.60 | 0.46 |
| random_forest | val_weighted | 3.36 | 1588 | 81 | 0.737 | 0.60 | 0.22 |
| elasticnet | mean | 0.05 | 1588 | 81 | 0.737 | 0.62 | 13.80 |
| lasso | recency_weighted | 0.07 | 1588 | 81 | 0.735 | 0.57 | 10.01 |
| random_forest | recency_weighted | 1.58 | 1588 | 81 | 0.735 | 0.57 | 0.46 |
| xgboost | mean | 1.01 | 1588 | 81 | 0.735 | 0.63 | 0.73 |
| xgboost | val_weighted | 1.86 | 1588 | 81 | 0.734 | 0.63 | 0.39 |
| lightgbm | val_weighted | 1.06 | 1588 | 81 | 0.732 | 0.60 | 0.69 |
| lightgbm | mean | 0.56 | 1588 | 81 | 0.731 | 0.60 | 1.30 |
| xgboost | recency_weighted | 1.07 | 1588 | 81 | 0.728 | 0.63 | 0.68 |
| lightgbm | recency_weighted | 0.52 | 1588 | 81 | 0.728 | 0.58 | 1.39 |
| elasticnet | recency_weighted | 0.06 | 1588 | 81 | 0.721 | 0.58 | 12.57 |
| ridge | val_weighted | 0.10 | 1588 | 81 | 0.654 | 0.57 | 6.53 |
| ridge | mean | 0.05 | 1588 | 81 | 0.653 | 0.57 | 13.93 |
| ridge | recency_weighted | 0.04 | 1588 | 81 | 0.631 | 0.55 | 14.34 |

## Figures

![Spearman by window](figures/report_football_te_windows.png)

![Model vs market](figures/report_football_te_vs_market.png)
