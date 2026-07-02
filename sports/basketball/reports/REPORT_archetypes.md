# NBA Archetypes (Phase 1) — k=13 soft GMM (PCA-whitened)

_Soft Gaussian-mixture on **PCA-whitened** z-scored play-STYLE features (per-36 rates + shot profile + tendencies), fit on eligible **E2+E3** (modern game) and assigned to all eligible 2013+ seasons. Whitening decorrelates the collinear style features so membership is genuinely **soft**. The hard label is argmax; **names are provisional** (seed-tied) pending soft→hard consolidation._

Fit pool: **2981** player-seasons · BIC 66522. Assigned (all eligible): **4239**.
Softness: median top_prob **0.773**, **54%** of player-seasons are blends (top_prob < 0.8).
Stability: **57%** keep their archetype year-over-year (N=3044 consecutive pairs; vs ~8% random).

## Archetypes

| # | archetype | n | signature (top + / −, season-z) | exemplars (most minutes) |
|---|---|---|---|---|
| 0 | **Rim-Running Center** | 261 | +fg_pct +2.2, oreb36 +2.1, scoring_eff +1.6 · −fg3a_rate -1.8, fg3a36 -1.8 | Rudy Gobert, Clint Capela, Steven Adams |
| 1 | **Lead Scoring Guard** | 319 | +fta36 +1.0, pts36 +1.0, fga36 +0.9 · −blk36 -0.6, oreb36 -0.6 | Damian Lillard, Tyrese Maxey, Devin Booker |
| 2 | **Stretch Forward** | 304 | +fg3a_rate +0.6, fg3a36 +0.4, fg3_pct +0.3 · −oreb_rate -0.6, ft_rate -0.6 | Miles Bridges, Jabari Smith Jr., Tobias Harris |
| 3 | **Low-Usage Wing** | 171 | +fg3a_rate +0.3, oreb_rate +0.3, ft_rate +-0.0 · −ast36 -0.7, ft_pct -0.7 | Kelly Oubre Jr., Lauri Markkanen, Kyle Kuzma |
| 4 | **Foul-Drawing Iso Scorer** | 36 | +fta36 +1.9, ft_rate +1.8, scoring_eff +1.2 · −fg3a_rate -0.7, fg3a36 -0.7 | DeMar DeRozan, Pascal Siakam, Jimmy Butler |
| 5 | **High-Usage Engine** | 88 | +tov36 +2.0, fta36 +2.0, pts36 +2.0 · −oreb_rate -0.7, fg3a_rate -0.5 | Julius Randle, Luka Doncic, LeBron James |
| 6 | **Perimeter Stopper** | 194 | +stl36 +1.7, oreb_rate +0.5, ast_tov +0.3 · −pts36 -0.7, fga36 -0.7 | Amen Thompson, OG Anunoby, Scottie Barnes |
| 7 | **Movement Shooter** | 402 | +fg3a_rate +1.1, fg3a36 +0.8, fg3_pct +0.8 · −tov36 -0.8, ft_rate -0.8 | OG Anunoby, Gary Trent Jr., Mikal Bridges |
| 8 | **Rebounding Forward** | 244 | +oreb_rate +0.9, oreb36 +0.8, dreb36 +0.3 · −ast36 -0.8, ast_tov -0.7 | Keegan Murray, P.J. Tucker, Nikola Vucevic |
| 9 | **Lead Playmaker** | 299 | +ast_tov +1.4, ast36 +1.4, stl36 +0.7 · −dreb36 -0.7, oreb36 -0.7 | Fred VanVleet, Dejounte Murray, Jrue Holiday |
| 10 | **Non-Scoring Playmaker** | 93 | +stl36 +0.9, ast36 +0.7, ast_tov +0.5 · −ft_pct -1.5, pts36 -1.0 | Josh Hart, Ben Simmons, Lonzo Ball |
| 11 | **Wing Shot-Creator** | 344 | +fga36 +0.8, fg3a36 +0.8, ft_pct +0.7 · −oreb36 -0.6, fg_pct -0.6 | Tyrese Maxey, Fred VanVleet, Kyrie Irving |
| 12 | **Rim-Protecting Big** | 226 | +dreb36 +1.4, blk36 +1.3, oreb36 +0.9 · −ast_tov -0.7, fg3a_rate -0.6 | Pascal Siakam, Kevin Durant, Anthony Davis |

## Cross-season stability (YoY persistence)

Share of players who keep an archetype the next season — validation, and the must-beat baseline for the Phase-3 predictor. Distinctive roles are stickiest; the low-signal middle churns most (a soft→hard consolidation candidate).

| archetype | YoY persistence |
|---|---|
| Rim-Running Center | 0.82 |
| High-Usage Engine | 0.73 |
| Lead Playmaker | 0.69 |
| Movement Shooter | 0.61 |
| Lead Scoring Guard | 0.60 |
| Rim-Protecting Big | 0.59 |
| Foul-Drawing Iso Scorer | 0.58 |
| Non-Scoring Playmaker | 0.51 |
| Wing Shot-Creator | 0.49 |
| Perimeter Stopper | 0.49 |
| Rebounding Forward | 0.46 |
| Stretch Forward | 0.42 |
| Low-Usage Wing | 0.28 |

## Notes
- Membership table (`data/processed/nba_archetype_membership.parquet`) carries the full probability vector `p0..pK-1` + `entropy` (blend-iness) per player-season — the soft input for Phase 2 (composition) and Phase 3 (the predictor target).
- PCA-whitening decorrelates the collinear style features → genuinely soft membership and higher YoY stability than a raw full-cov GMM. Next: soft→hard consolidation of the low-signal middle archetypes + finalize names.
