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
| 1 | Kirk Cousins (29) | 12.1 | High | **yes** |
| 2 | Marcus Mariota (26) | 12.3 | High | **yes** |
| 3 | Justin Fields (32) | 11.8 | High | **yes** |
| 4 | Mac Jones (30) | 12.0 | High | **yes** |
| 5 | Geno Smith (31) | 11.9 | High | **yes** |
| 6 | Jayden Daniels (18) | 15.4 | High | **yes** |
| 7 | Aaron Rodgers (25) | 12.5 | High | **yes** |
| 8 | Tyler Shough (27) | 12.3 | High | **yes** |
| 9 | Jacoby Brissett (28) | 12.2 | Moderate | — |
| 10 | Jaxson Dart (13) | 16.2 | Moderate | — |
| 11 | Joe Burrow (23) | 13.8 | Moderate | — |
| 12 | C.J. Stroud (20) | 14.8 | Moderate | — |
| 13 | Tua Tagovailoa (24) | 12.9 | Moderate | — |
| 14 | Josh Allen (1) | 20.4 | Moderate | — |
| 15 | Daniel Jones (16) | 15.9 | Moderate | — |
| 16 | Brock Purdy (21) | 14.1 | Moderate | — |
| 17 | Bryce Young (19) | 15.2 | Moderate | — |
| 18 | Matthew Stafford (6) | 17.9 | Moderate | — |
| 19 | Sam Darnold (17) | 15.4 | Moderate | — |
| 20 | Patrick Mahomes (12) | 16.7 | Moderate | — |
| 21 | Lamar Jackson (2) | 19.9 | Moderate | — |
| 22 | Jalen Hurts (3) | 19.1 | Moderate | — |
| 23 | Dak Prescott (14) | 16.1 | Moderate | — |
| 24 | Justin Herbert (9) | 17.3 | Moderate | — |
| 25 | Drake Maye (5) | 18.7 | Lower | — |
| 26 | Cam Ward (22) | 13.9 | Lower | — |
| 27 | Baker Mayfield (10) | 17.1 | Lower | — |
| 28 | Trevor Lawrence (4) | 18.9 | Lower | — |
| 29 | Jordan Love (15) | 16.1 | Lower | — |
| 30 | Jared Goff (11) | 17.0 | Lower | — |
| 31 | Bo Nix (7) | 17.8 | Lower | — |
| 32 | Caleb Williams (8) | 17.4 | Lower | — |

