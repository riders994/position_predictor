# Position Predictor — Results: football RB

_Generated 2026-06-16._ Test seasons **[2020, 2021, 2022, 2023, 2024]**, eligibility cutoff **g\* = 4 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `xgboost` (val_weighted, 20-yr window) — Spearman **0.752 ± 0.031**, Precision@12 0.53, MAE 2.62 PPG (full eligible universe).
- **Best baseline:** `linear` — Spearman 0.720 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.732 vs our best `xgboost` 0.706 — the model does **not** beat the market on overall rank.
  On **Precision@12** (the draftable top tier): market 0.60 vs `xgboost` 0.55.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.689 vs `ridge` 0.663 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.741 | 0.746 | 0.746 |
| elasticnet | recency_weighted | 0.744 | 0.749 | 0.749 |
| elasticnet | val_weighted | 0.740 | 0.747 | 0.747 |
| lasso | mean | 0.741 | 0.745 | 0.745 |
| lasso | recency_weighted | 0.745 | 0.748 | 0.748 |
| lasso | val_weighted | 0.741 | 0.744 | 0.744 |
| lightgbm | mean | 0.733 | 0.746 | 0.746 |
| lightgbm | recency_weighted | 0.732 | 0.737 | 0.737 |
| lightgbm | val_weighted | 0.733 | 0.745 | 0.745 |
| random_forest | mean | 0.733 | 0.742 | 0.742 |
| random_forest | recency_weighted | 0.732 | 0.738 | 0.738 |
| random_forest | val_weighted | 0.733 | 0.742 | 0.742 |
| ridge | mean | 0.737 | 0.747 | 0.747 |
| ridge | recency_weighted | 0.745 | 0.749 | 0.749 |
| ridge | val_weighted | 0.738 | 0.747 | 0.747 |
| xgboost | mean | 0.739 | 0.751 | 0.751 |
| xgboost | recency_weighted | 0.738 | 0.746 | 0.746 |
| xgboost | val_weighted | 0.739 | 0.752 | 0.752 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2020 | 101 | 73 | 0.72 | 0.487 | 0.58 |
| 2021 | 105 | 90 | 0.86 | 0.749 | 0.75 |
| 2022 | 96 | 89 | 0.93 | 0.818 | 0.58 |
| 2023 | 93 | 81 | 0.87 | 0.803 | 0.42 |
| 2024 | 92 | 91 | 0.99 | 0.804 | 0.67 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 |
|---|---|---|---|
| market_ecr _(market)_ | 0.732 | 0.689 | 0.60 |
| ridge | 0.704 | 0.663 | 0.65 |
| lasso | 0.702 | 0.657 | 0.63 |
| elasticnet | 0.703 | 0.657 | 0.62 |
| xgboost | 0.706 | 0.654 | 0.55 |
| random_forest | 0.691 | 0.650 | 0.55 |
| lightgbm | 0.696 | 0.649 | 0.52 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.711, P@12 0.45; **without**: Spearman 0.708, P@12 0.47.
- **Decision:** drop the `ngs_efficiency` block — no top-12 gain out-of-fold, so the ngs era collapses to the snaps schema (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 4.33 | 0.782 |
| gbm_poisson | 4.57 | 0.773 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`ridge`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Spearman | 0.745 | 0.748 | 0.731 | 0.714 | 0.714 |


## Figures

![Spearman by window](figures/report_football_rb_windows.png)

![Model vs market](figures/report_football_rb_vs_market.png)
