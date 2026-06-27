# NBA Archetypes (Phase 1) — k=12 soft GMM

_Soft Gaussian-mixture on z-scored play-STYLE features (per-36 rates + shot profile + tendencies), fit on eligible **E2+E3** (modern game) and assigned to all eligible 2013+ seasons. Style space is continuous, so membership is **soft** (probabilities); the hard label is argmax. **Names are provisional** (seed-tied) pending soft→hard consolidation._

Fit pool: **2631** player-seasons · BIC 40710. Assigned (all eligible): **3889**.

## Archetypes

| # | archetype | n | signature (top + / −, season-z) | exemplars (most minutes) |
|---|---|---|---|---|
| 0 | **Connector Wing** | 194 | +oreb_rate +0.9, stl36 +0.6, oreb36 +0.5 · −pts36 -0.8, fga36 -0.8 | Josh Hart, Scottie Barnes, Dyson Daniels |
| 1 | **Two-Way Forward** | 223 | +dreb36 +0.9, blk36 +0.8, oreb36 +0.5 · −ast_tov -0.6, stl36 -0.5 | Harrison Barnes, Jalen Johnson, Karl-Anthony Towns |
| 2 | **Volume Perimeter Scorer** | 310 | +fga36 +1.0, fg3a36 +0.8, pts36 +0.7 · −oreb36 -0.6, oreb_rate -0.6 | Fred VanVleet, Kyrie Irving, Miles Bridges |
| 3 | **Non-Shooting Interior Big** | 162 | +oreb36 +1.9, fg_pct +1.7, oreb_rate +1.5 · −fg3a36 -1.6, fg3a_rate -1.6 | Evan Mobley, Taj Gibson, Jarrett Allen |
| 4 | **Movement Shooter** | 325 | +fg3a_rate +1.0, fg3a36 +0.8, ft_pct +0.5 · −ft_rate -0.9, fta36 -0.8 | Fred VanVleet, Lonzo Ball, Buddy Hield |
| 5 | **Rim-Running Center** | 135 | +fg_pct +2.3, oreb36 +2.3, scoring_eff +1.7 · −fg3_pct -3.2, fg3a_rate -1.9 | Rudy Gobert, Andre Drummond, Clint Capela |
| 6 | **Floor General** | 183 | +ast_tov +1.5, ast36 +1.1, stl36 +0.8 · −pts36 -0.8, fga36 -0.8 | James Harden, Lonzo Ball, Kyle Lowry |
| 7 | **High-Usage Shot Creator** | 224 | +fga36 +1.8, pts36 +1.8, fta36 +1.4 · −oreb_rate -0.8, oreb36 -0.6 | DeMar DeRozan, Kyrie Irving, Tyrese Maxey |
| 8 | **3-and-D Wing** | 309 | +fg3a_rate +0.6, fg3_pct +0.3, fg3a36 +0.1 · −pts36 -0.7, fta36 -0.7 | Mikal Bridges, Robert Covington, Keegan Murray |
| 9 | **Slashing Forward** | 149 | +fta36 +1.6, pts36 +1.4, dreb36 +1.3 · −fg3a_rate -1.1, fg3a36 -0.8 | Pascal Siakam, Jimmy Butler, Giannis Antetokounmpo |
| 10 | **Balanced Combo Wing** | 303 | +ft_pct +0.3, fg3a36 +0.3, fga36 +0.3 · −oreb36 -0.5, blk36 -0.5 | Mikal Bridges, James Harden, Fred VanVleet |
| 11 | **Corner Specialist** | 114 | +fg3a_rate +1.2, ast_tov +0.4, fg3_pct +0.3 · −pts36 -1.5, fga36 -1.5 | P.J. Tucker, Royce O'Neale, Dorian Finney-Smith |

## Notes
- Membership table (`data/processed/nba_archetype_membership.parquet`) carries the full probability vector `p0..pK-1` + `entropy` (blend-iness) per player-season — the soft input for Phase 2 (composition) and Phase 3 (the predictor target).
- Continuous style space → low silhouette is expected; soft membership is the design, not a defect. Next: cross-season stability + soft→hard consolidation/naming.
