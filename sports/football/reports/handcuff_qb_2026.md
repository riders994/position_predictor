# QB Injury-Risk List — 2026

_Scoring: **ppr**. Projected QB starters ranked by injury/availability risk — **draft a backup** for the High tier. Model-only (ECR/ADP are benchmarks)._

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
| 1 | Marcus Mariota (29) | 11.7 | High | **yes** |
| 2 | Kyler Murray (24) | 12.8 | High | **yes** |
| 3 | Justin Fields (25) | 12.6 | High | **yes** |
| 4 | Mac Jones (31) | 11.3 | High | **yes** |
| 5 | Geno Smith (32) | 10.9 | High | **yes** |
| 6 | Jayden Daniels (15) | 15.1 | High | **yes** |
| 7 | Aaron Rodgers (22) | 13.7 | High | **yes** |
| 8 | Tyler Shough (26) | 12.2 | High | **yes** |
| 9 | J.J. McCarthy (30) | 11.5 | Moderate | — |
| 10 | Jaxson Dart (12) | 15.6 | Moderate | — |
| 11 | Joe Burrow (19) | 14.1 | Moderate | — |
| 12 | C.J. Stroud (17) | 14.7 | Moderate | — |
| 13 | Tua Tagovailoa (27) | 12.2 | Moderate | — |
| 14 | Daniel Jones (21) | 13.7 | Moderate | — |
| 15 | Josh Allen (1) | 18.7 | Moderate | — |
| 16 | Brock Purdy (20) | 13.7 | Moderate | — |
| 17 | Bryce Young (23) | 13.1 | Moderate | — |
| 18 | Patrick Mahomes (8) | 16.6 | Moderate | — |
| 19 | Sam Darnold (18) | 14.1 | Moderate | — |
| 20 | Matthew Stafford (7) | 16.6 | Moderate | — |
| 21 | Jalen Hurts (2) | 18.3 | Moderate | — |
| 22 | Lamar Jackson (4) | 17.6 | Moderate | — |
| 23 | Dak Prescott (14) | 15.1 | Moderate | — |
| 24 | Justin Herbert (10) | 16.2 | Moderate | — |
| 25 | Drake Maye (3) | 17.6 | Lower | — |
| 26 | Cam Ward (28) | 12.0 | Lower | — |
| 27 | Baker Mayfield (11) | 15.9 | Lower | — |
| 28 | Trevor Lawrence (5) | 16.9 | Lower | — |
| 29 | Jordan Love (16) | 14.7 | Lower | — |
| 30 | Jared Goff (9) | 16.4 | Lower | — |
| 31 | Bo Nix (6) | 16.8 | Lower | — |
| 32 | Caleb Williams (13) | 15.5 | Lower | — |

