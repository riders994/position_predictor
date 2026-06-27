# QB Injury-Risk List — 2026

_Projected QB starters ranked by injury/availability risk — **draft a backup** for the High tier. Model-only (ECR/ADP are benchmarks)._

> **Read the tiers, not the raw number.** The QB games model regresses toward a backup-heavy pool mean, so it ranks risk well (clears-cutoff AUC ~0.90) but under-predicts absolute games for everyone. `risk_tier` is **relative to the projected-starter cohort** (High = riskiest quartile).
>
> **Caveat:** the model reads rushing/workload as injury exposure, so durable high-usage QBs (e.g. Josh Allen, Lamar Jackson) can be flagged riskier than their track record warrants — they are outliers who sustain that load. Treat a long clean availability history as a discount on the model's ranking.

## Risk signal (leak-safe backtest)

Chosen: **`availability_model`** (best clears-cutoff AUC at g\*=7; `winner`=`availability_model`).

| signal | games MAE | clears AUC | folds |
|---|---|---|---|
| availability_model | 3.33 | 0.895 | 5 |
| durability_3yr | 3.80 | 0.876 | 5 |
| prior_games | 3.75 | 0.859 | 5 |

## QBs most likely to miss time (draft a backup)

| # | QB (QB rank) | proj PPG | risk tier | draft a backup? |
|---|---|---|---|---|
| 1 | Kyler Murray (24) | 12.8 | High | **yes** |
| 2 | Jayden Daniels (15) | 15.1 | High | **yes** |
| 3 | Aaron Rodgers (22) | 13.7 | High | **yes** |
| 4 | Jaxson Dart (12) | 15.6 | High | **yes** |
| 5 | Joe Burrow (19) | 14.1 | High | **yes** |
| 6 | C.J. Stroud (17) | 14.7 | High | **yes** |
| 7 | Josh Allen (1) | 18.7 | Moderate | — |
| 8 | Brock Purdy (20) | 13.7 | Moderate | — |
| 9 | Daniel Jones (21) | 13.7 | Moderate | — |
| 10 | Bryce Young (23) | 13.1 | Moderate | — |
| 11 | Patrick Mahomes (8) | 16.6 | Moderate | — |
| 12 | Sam Darnold (18) | 14.1 | Moderate | — |
| 13 | Matthew Stafford (7) | 16.6 | Moderate | — |
| 14 | Jalen Hurts (2) | 18.3 | Moderate | — |
| 15 | Dak Prescott (14) | 15.1 | Moderate | — |
| 16 | Lamar Jackson (4) | 17.6 | Moderate | — |
| 17 | Justin Herbert (10) | 16.2 | Moderate | — |
| 18 | Drake Maye (3) | 17.6 | Moderate | — |
| 19 | Baker Mayfield (11) | 15.9 | Lower | — |
| 20 | Trevor Lawrence (5) | 16.9 | Lower | — |
| 21 | Jordan Love (16) | 14.7 | Lower | — |
| 22 | Jared Goff (9) | 16.4 | Lower | — |
| 23 | Bo Nix (6) | 16.8 | Lower | — |
| 24 | Caleb Williams (13) | 15.5 | Lower | — |

