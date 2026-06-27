# NBA Archetypes (Phase 1) — k=12 soft GMM (PCA-whitened)

_Soft Gaussian-mixture on **PCA-whitened** z-scored play-STYLE features (per-36 rates + shot profile + tendencies), fit on eligible **E2+E3** (modern game) and assigned to all eligible 2013+ seasons. Whitening decorrelates the collinear style features so membership is genuinely **soft**. The hard label is argmax; **names are provisional** (seed-tied) pending soft→hard consolidation._

Fit pool: **2631** player-seasons · BIC 63610. Assigned (all eligible): **3889**.
Softness: median top_prob **0.847**, **43%** of player-seasons are blends (top_prob < 0.8).
Stability: **61%** keep their archetype year-over-year (N=2787 consecutive pairs; vs ~8% random).

## Archetypes

| # | archetype | n | signature (top + / −, season-z) | exemplars (most minutes) |
|---|---|---|---|---|
| 0 | **Foul-Drawing Iso Scorer** | 42 | +ft_rate +1.7, fta36 +1.5, scoring_eff +1.1 · −fg3a36 -0.7, fg3a_rate -0.7 | DeMar DeRozan, James Harden, Jimmy Butler |
| 1 | **Lead Playmaker** | 160 | +ast_tov +1.8, ast36 +1.3, stl36 +0.4 · −scoring_eff -0.8, dreb36 -0.8 | Tyrese Maxey, Tyrese Haliburton, Chris Paul |
| 2 | **Scoring Combo Guard** | 239 | +ast36 +1.1, fga36 +1.0, pts36 +0.9 · −dreb36 -0.8, oreb36 -0.7 | Fred VanVleet, Kyrie Irving, Tyrese Maxey |
| 3 | **Connector Wing** | 150 | +stl36 +0.6, ast_tov +0.4, oreb_rate +0.2 · −ft_pct -1.2, pts36 -0.8 | Josh Hart, Mikal Bridges, Scottie Barnes |
| 4 | **Volume Wing Scorer** | 277 | +fg3a36 +0.9, fga36 +0.8, ft_pct +0.7 · −oreb_rate -0.8, oreb36 -0.7 | Paul George, Coby White, Bradley Beal |
| 5 | **Two-Way Forward** | 276 | +stl36 +1.2, tov36 +0.4, ast36 +0.4 · −shooting_eff -0.6, fg_pct -0.3 | Pascal Siakam, OG Anunoby, Fred VanVleet |
| 6 | **3-and-D Wing** | 440 | +fg3a_rate +1.0, fg3a36 +0.6, fg3_pct +0.5 · −ft_rate -0.9, fta36 -0.8 | Mikal Bridges, OG Anunoby, Gary Trent Jr. |
| 7 | **Balanced Wing** | 438 | +oreb_rate +0.3, fg3a_rate +0.2, dreb36 +0.2 · −ast36 -0.7, ast_tov -0.6 | Miles Bridges, Andrew Wiggins, Scottie Barnes |
| 8 | **Interior Big** | 204 | +oreb36 +1.9, fg_pct +1.6, oreb_rate +1.5 · −fg3a_rate -1.6, fg3a36 -1.6 | Anthony Davis, Bam Adebayo, Evan Mobley |
| 9 | **High-Usage Primary** | 108 | +fta36 +2.2, pts36 +2.1, tov36 +1.8 · −fg3a_rate -0.6, oreb_rate -0.5 | Julius Randle, Luka Doncic, LeBron James |
| 10 | **Rebounding Big-Forward** | 179 | +dreb36 +1.1, blk36 +1.1, oreb36 +0.5 · −ast_tov -0.4, ast36 -0.2 | Kevin Durant, Domantas Sabonis, Jalen Johnson |
| 11 | **Rim-Running Center** | 118 | +fg_pct +2.4, oreb36 +2.2, scoring_eff +1.8 · −fg3_pct -3.2, fg3a_rate -1.9 | Rudy Gobert, Ben Simmons, Clint Capela |

## Cross-season stability (YoY persistence)

Share of players who keep an archetype the next season — validation, and the must-beat baseline for the Phase-3 predictor. Distinctive roles are stickiest; the low-signal middle churns most (a soft→hard consolidation candidate).

| archetype | YoY persistence |
|---|---|
| Rim-Running Center | 0.78 |
| Interior Big | 0.70 |
| High-Usage Primary | 0.69 |
| Scoring Combo Guard | 0.67 |
| Lead Playmaker | 0.65 |
| 3-and-D Wing | 0.65 |
| Foul-Drawing Iso Scorer | 0.59 |
| Balanced Wing | 0.58 |
| Volume Wing Scorer | 0.56 |
| Rebounding Big-Forward | 0.51 |
| Two-Way Forward | 0.51 |
| Connector Wing | 0.45 |

## Notes
- Membership table (`data/processed/nba_archetype_membership.parquet`) carries the full probability vector `p0..pK-1` + `entropy` (blend-iness) per player-season — the soft input for Phase 2 (composition) and Phase 3 (the predictor target).
- PCA-whitening decorrelates the collinear style features → genuinely soft membership and higher YoY stability than a raw full-cov GMM. Next: soft→hard consolidation of the low-signal middle archetypes + finalize names.
