# NBA Archetypes (Phase 1) — k=12 soft GMM (PCA-whitened)

_Soft Gaussian-mixture on **PCA-whitened** z-scored play-STYLE features (per-36 rates + shot profile + tendencies), fit on eligible **E2+E3** (modern game) and assigned to all eligible 2013+ seasons. Whitening decorrelates the collinear style features so membership is genuinely **soft**. The hard label is argmax; **names are provisional** (seed-tied) pending soft→hard consolidation._

Fit pool: **2981** player-seasons · BIC 71127. Assigned (all eligible): **4239**.
Softness: median top_prob **0.866**, **40%** of player-seasons are blends (top_prob < 0.8).
Stability: **63%** keep their archetype year-over-year (N=3044 consecutive pairs; vs ~8% random).

## Archetypes

| # | archetype | n | signature (top + / −, season-z) | exemplars (most minutes) |
|---|---|---|---|---|
| 0 | **Off-Ball Wing** | 384 | +oreb_rate +0.5, fg3a_rate +0.3, fg3_pct +0.1 · −ast36 -0.8, tov36 -0.6 | Andrew Wiggins, Jabari Smith Jr., Kelly Oubre Jr. |
| 1 | **3-and-D Wing** | 613 | +fg3a_rate +1.1, fg3a36 +0.8, fg3_pct +0.4 · −ft_rate -0.8, fta36 -0.7 | OG Anunoby, Gary Trent Jr., Anfernee Simons |
| 2 | **High-Usage Primary** | 111 | +tov36 +1.8, ast36 +1.8, fga36 +0.7 · −oreb_rate -0.7, ft_pct -0.5 | Julius Randle, Luka Doncic, LeBron James |
| 3 | **Scoring Combo Guard** | 499 | +fga36 +1.1, pts36 +0.9, ft_pct +0.7 · −oreb36 -0.6, blk36 -0.6 | Tyrese Maxey, Fred VanVleet, Kyrie Irving |
| 4 | **Rim-Running Center** | 130 | +fg_pct +2.4, oreb36 +2.3, scoring_eff +1.8 · −fg3_pct -3.2, fg3a_rate -2.0 | Rudy Gobert, Clint Capela, Steven Adams |
| 5 | **Foul-Drawing Iso Scorer** | 54 | +fta36 +2.7, pts36 +2.0, ft_rate +1.9 · −fg3a_rate -1.0, fg3a36 -0.5 | DeMar DeRozan, James Harden, Jimmy Butler |
| 6 | **Two-Way Forward** | 360 | +dreb36 +1.0, blk36 +0.7, pts36 +0.5 · −stl36 -0.4, ast_tov -0.4 | Pascal Siakam, Miles Bridges, Jayson Tatum |
| 7 | **Slashing Non-Shooter** | 101 | +oreb36 +1.7, fg_pct +1.6, ft_rate +1.4 · −fg3a_rate -1.7, fg3a36 -1.7 | Amen Thompson, Domantas Sabonis, Ben Simmons |
| 8 | **Lead Playmaker** | 277 | +ast_tov +1.3, ast36 +1.1, stl36 +0.6 · −dreb36 -0.6, oreb36 -0.6 | James Harden, Mikal Bridges, Fred VanVleet |
| 9 | **Interior Big** | 209 | +oreb36 +1.6, oreb_rate +1.4, fg_pct +1.2 · −fg3a36 -1.2, fg3a_rate -1.2 | Scottie Barnes, Evan Mobley, LaMarcus Aldridge |
| 10 | **Connector Wing** | 231 | +stl36 +1.6, ast_tov +0.6, ast36 +0.3 · −pts36 -0.9, fga36 -0.8 | Josh Hart, Lonzo Ball, Robert Covington |
| 11 | **Non-Shooting Center** | 12 | +fg3_pct +5.0, fg_pct +2.5, scoring_eff +2.3 · −fg3a_rate -1.9, fg3a36 -1.9 | Hassan Whiteside, Jakob Poeltl, Mark Williams |

## Cross-season stability (YoY persistence)

Share of players who keep an archetype the next season — validation, and the must-beat baseline for the Phase-3 predictor. Distinctive roles are stickiest; the low-signal middle churns most (a soft→hard consolidation candidate).

| archetype | YoY persistence |
|---|---|
| 3-and-D Wing | 0.75 |
| Rim-Running Center | 0.74 |
| Foul-Drawing Iso Scorer | 0.71 |
| Scoring Combo Guard | 0.67 |
| Two-Way Forward | 0.66 |
| Slashing Non-Shooter | 0.58 |
| Off-Ball Wing | 0.57 |
| Interior Big | 0.56 |
| High-Usage Primary | 0.55 |
| Lead Playmaker | 0.53 |
| Connector Wing | 0.52 |
| Non-Shooting Center | 0.00 |

## Notes
- Membership table (`data/processed/nba_archetype_membership.parquet`) carries the full probability vector `p0..pK-1` + `entropy` (blend-iness) per player-season — the soft input for Phase 2 (composition) and Phase 3 (the predictor target).
- PCA-whitening decorrelates the collinear style features → genuinely soft membership and higher YoY stability than a raw full-cov GMM. Next: soft→hard consolidation of the low-signal middle archetypes + finalize names.
