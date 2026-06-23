# Data Dictionary

Definitions for source fields and engineered features. Populated as the pipeline is
implemented. Every engineered feature must (a) be computed only from seasons ≤ *N*
(no leakage) and (b) be listed here with its formula and source.

## Identity / keys

| Field | Description | Source |
|---|---|---|
| `player_id` | nflverse GSIS / canonical id | `load_ff_playerids` |
| `player_name` | Display name | rosters |
| `season` | Feature season *N* | — |
| `team` | Team in season *N* | rosters |

## Label & next-season signals (Stage 2 — `data/build.py`)

| Field | Description | Formula |
|---|---|---|
| `target_ppg_next` | Next-season PPR PPG (regression target) — **defined only for `status_next == active`** | total PPR pts in *N+1* ÷ games in *N+1* |
| `ppr_points_next` | Next-season total PPR points (0 when out the whole season) | sum of weekly PPR pts in *N+1* |
| `games_next` | Next-season games played (availability label, §4.3); 0 if out all year, NaN if censored | count of *N+1* regular-season game rows |
| `status_next` | How *N+1* is classified (see below) | `active` / `injured_out` / `retired` / `censored` |
| `retired_next` | Player never appears again after *N* (a zero-availability outcome) | `status_next == retired` |
| `snap_share_next` | Next-season snap share (0 if out all year; NaN if censored or no snap data) | see snap share below, season *N+1* |
| `eligible_next__g{G}` | Games-cutoff candidate flags (nullable; NA when censored) | `games_next ≥ G` for each `G` in `eligibility.candidate_games_played` |
| `eligible_next__s{S}` | Snap-share-cutoff candidate flags (nullable; NA when censored or pre-2012/unmapped) | `snap_share_next ≥ 0.S` for each `S` in `eligibility.candidate_snap_share` (e.g. `__s30` = 0.30) |

**`status_next` — separating "out" from "gone"** (a full-season miss has *no* weekly rows;
verified against real data, e.g. J.K. Dobbins 2021). Using *future* seasons:
- `active` — played in *N+1*; `games_next` > 0 and the PPG target is defined.
- `injured_out` — no *N+1* rows but the player returns in a later season → `games_next = 0`,
  a genuine availability-zero training example for the §4.3 model; PPG target is NaN.
- `retired` — no *N+1* rows and the player never returns → **treated as a type of injury**
  (`games_next = 0`), flagged via `retired_next`. We do not model "un-retirement". NB: many
  marginal one-season backs land here (functionally "gone"), which is signal for §4.3.
- `censored` — *N+1* exceeds the last **observed** season → unknown future; all `_next`
  labels NaN; excluded from supervised training. The boundary is
  `min(data.latest_completed_season, max season present in the data)`, so if the configured
  latest season is not yet published by nflverse (e.g. the just-finished season), the trailing
  feature season's rows are correctly `censored` rather than mislabeled `retired`. These rows
  remain available as **prediction inputs** for the live next-season board.

> The *chosen* eligibility rule is derived analytically in the eligibility step
> (PROJECT_PLAN §4.2); Stage 2 materialises both the candidate **games** and **snap-share**
> grids so downstream rank metrics can be reported across cutoffs for robustness.

**Excluded seasons (`data.exclude_seasons`, e.g. `[2020]` COVID).** `apply_season_exclusion`
removes an anomalous season from supervised use in *both* roles without dropping players:
rows whose **label** season is excluded are kept as history but get `_next` labels nulled and
`status_next = excluded_season`; rows **in** an excluded season are dropped entirely, so it never
feeds a prediction and is hopped over by the per-player multi-year feature windows (no leakage into
neighbouring years). Consequence: the season *after* an excluded one loses its label fold too
(predicting it would need the excluded season's features). 2020 is excluded because its outcome is
near-unrankable (preseason ECR Spearman ~0.49 vs ~0.73–0.82 elsewhere) and COVID absences would
otherwise mislabel availability/wear. See PROMPT_LOG entry 024.

## Fantasy position eligibility (Stage 2) — not NFL designation

A player-season is kept if the player is **fantasy-eligible** at the target position, which
can differ from the NFL roster `position`.

| Field | Description | Source |
|---|---|---|
| `is_eligible` | Fantasy-eligible at the target position (filter; dropped from output) | resolved below |
| `eligibility_source` | Which signal decided eligibility | `sleeper` or `nflverse_fallback` |

- **Primary:** Sleeper `fantasy_positions` array (`data/raw/sleeper_players.parquet`, fetched
  from `api.sleeper.app/v1/players/nfl`), joined on `gsis_id`. Treated as a career-level
  attribute (Sleeper is a *current* snapshot, not per-season history).
- **Fallback:** nflverse `position_group` (already folds FB→RB) for players Sleeper doesn't
  cover — notably pre-2010, and the ~⅔ of the Sleeper universe lacking a `gsis_id`.

## Season-*N* production (Stage 2, aggregated from weekly box scores)

Per `(player_id, season)`, regular-season only. Summed volume columns are kept defensively
(only those present in the nflverse weekly frame): `carries, rushing_yards, rushing_tds,
rushing_first_downs, receptions, targets, receiving_yards, receiving_tds,
receiving_first_downs, fantasy_points, fantasy_points_ppr` (+ fumbles lost).

| Field | Description | Formula |
|---|---|---|
| `games` | Regular-season games played in *N* | count of weekly rows (`season_type == 'REG'`) |
| `ppr_points` | Season-*N* total PPR points | `fantasy_points_ppr` summed |
| `ppg` | Season-*N* PPR points-per-game | `ppr_points / games` |
| `touches` | Season-*N* touches | `carries + receptions` |
| `snap_share` | Season-*N* offensive snap share (2012+), snaps-weighted over games played | `Σ offense_snaps / Σ team_offense_snaps` (team total = `offense_snaps / offense_pct`) |
| `snaps` | Season-*N* total offensive snaps (2012+) | `Σ offense_snaps` (REG only) |
| `snaps_per_game` | Snaps per game played (2012+) | `snaps / games with ≥1 snap` |
| `age_at_season_start` | Age in years on Sept 1 of season *N* | `(date(N, 9, 1) − birth_date) / 365.25` |
| `years_exp`, `height`, `weight`, `draft_number`, `rookie_year`, `entry_year` | Player attributes | seasonal rosters |

> `age_at_season_start` is computed from `birth_date` (a career constant), **not** the
> roster's coarser `age` field, since the RB age cliff makes precise age a headline feature.

## Feature blocks (Stage 4 — `features/build.py`)

Blocks match the era schemas (PROJECT_PLAN §6.3); `features/build.py` emits a block→columns
map (`*_feature_blocks.json`) so `eras.feature_columns_for_era` selects each era model's
inputs. All features are season-*N*-only (no leakage); deltas/slopes/baselines use **prior**
seasons via `shift(1)` within player. Divide-by-zero → NaN. Era availability noted per block.

| Block | Era avail. | Columns |
|---|---|---|
| `production` | 1999+ | `ppg, ppr_points, rushing_yards, receiving_yards, total_yards, rushing_tds, receiving_tds, total_tds, receptions, finish_ppr_rank, finish_ppg_rank` (finish = within-season rank, 1=best) |
| `volume` | 1999+ | `carries, targets, receptions, touches` and their `_pg` per-game rates; `weighted_opportunities` (=carries+2·targets), `wo_pg` |
| `team_context` | 1999+ | `rush_att_share, target_share, touch_share` (vs season-*N* team totals), `team_run_rate, team_plays_pg` (pace). Traded players measured vs last team. |
| `efficiency` | 1999+ | `yards_per_carry, yards_per_rec, yards_per_target, yards_per_touch, catch_rate, rush_td_rate, rec_td_rate, rush_fd_rate, rec_fd_rate, ppr_per_touch, rush_epa_per_att, rec_epa_per_target` |
| `player_attrs` | 1999+ | `age` (=age_at_season_start), `age_sq` (age-cliff non-linearity), `bmi`, `years_exp, height, weight`, `draft_capital` (UDFA→261 sentinel), `is_undrafted` |
| `availability` | 1999+ | `games, games_missed` (vs 16/17-game `season_length`), `availability_rate, avail_rate_3yr`, `career_touches, touches_3yr, touches_pg_3yr` (wear). Also feed the §4.3 model. |
| `trajectory` | 1999+ | `ppg_delta1, touches_pg_delta1, target_share_delta1, yards_per_carry_delta1` (YoY), `ppg_slope3` (OLS slope of last ≤3 PPG), `seasons_played` |
| `regression_mean` | 1999+ | `td_per_touch`, and `*_vs_prior` gaps of `td_per_touch / ppg / yards_per_carry` vs the player's expanding **prior** mean — flag unsustainable production |
| `snap_usage` | 2012+ | `snap_share, snaps_per_game, snap_share_delta` (NaN before 2012) |
| `ngs_efficiency` | 2016+ | rushing: `ryoe_per_att, rush_pct_over_expected, ngs_efficiency, avg_time_to_los, pct_attempts_8plus_box`; receiving: `yac_above_expected, avg_separation`; coverage flags `has_ngs_rush, has_ngs_rec`. NGS only covers qualified players (~73% of top-36, ~6% of fringe), and missing NGS rushing ≈ a receiving-profile back — so the flags are **informative**. Whether the NGS values earn their place is settled by the §7.3 ablation; trees take raw NaN, linear models impute + use the flags. |
| **QB passing blocks** (used when `position==QB`, in place of the rushing/receiving `production`/`volume`/`efficiency`/`ngs_efficiency` above; QB scores off passing per nflverse `fantasy_points_ppr`) ||
| `production` (QB) | 1999+ | `ppg, ppr_points, passing_yards, passing_tds, interceptions, completions, rushing_yards, rushing_tds, total_tds` (=passing+rushing TDs), `finish_ppr_rank, finish_ppg_rank` |
| `volume` (QB) | 1999+ | `attempts, completions, dropbacks` (=attempts+sacks), `passing_air_yards` and their `_pg` per-game rates |
| `efficiency` (QB) | 1999+ | `completion_pct, yards_per_attempt, yards_per_completion, pass_td_rate, int_rate, sack_rate` (per dropback), `air_yards_per_att, pass_fd_rate, pass_epa_per_db` |
| `rushing` (QB) | 1999+ | mobile-QB rushing — `carries, rushing_yards, rushing_tds, carries_pg, rush_yards_pg, yards_per_carry, rush_epa_per_att` (a real, persistent QB fantasy edge) |
| `ngs_passing` (QB) | 2016+ | `cpoe` (completion % over expected), `expected_completion_pct, avg_time_to_throw, aggressiveness, avg_air_yards_to_sticks, avg_intended_air_yards, ngs_passer_rating`; coverage flag `has_ngs_pass`. Same informative-missingness logic as `ngs_efficiency`. |
| `air_yards` (RB/WR) | 1999+ | **Dormant — measured no-gain, not in the default pipeline.** Receiving air-yards opportunity: `receiving_air_yards`, `adot` (depth of target), `air_yards_pg`, `air_yards_share` (vs team air yards), `racr` (rec yds / air yds), `wopr` (=1.5·target_share + 0.7·air_yards_share). Tested in the air-yards back-apply (RB v4 / WR v2 probe) → flat for WR, slightly negative for RB trees: WOPR is collinear with the `target_share`/`targets` already in `team_context`/`volume`, so no incremental signal. `add_air_yards` + plumbing kept (tested) for re-use. See PROMPT_LOG entry 023. |
| `offseason` | 1999+ | **Season-*N+1* preseason context known by Sept 1** (the one block that reads N+1 data — see leakage note), **position-agnostic**: `changed_team_next` (moved teams N→N+1), `rookie_drafted_next`/`rookie_draft_capital_next` (best overall pick of a rookie **at the player's position** the N+1 team drafted; sentinel 300 = none)/`rookie_count_next`, `room_prior_workload_next` (proven season-N workload — `touches` for RB, `targets` for WR via `features.offseason_workload_col` — of the *other* same-position players on the N+1 roster; the competition in the room, self excluded), `room_size_next`, **`vacated_workload_next`** (RB-v3: season-N workload on the player's N+1 team that **departed** — opportunity that opened up; self-returning workload excluded). Quantifies the roster/draft dynamics ECR reacts to; the continuous capital/competition signals carry the weight. From `draft_picks` + `rosters`. |

> **Leakage note for `offseason`:** every other block uses data only through season *N*. This
> block deliberately reads the *N+1* **preseason** state (the April draft + the preseason roster),
> all of which is settled by the **Sept 1** cutoff — before any season-*N+1* game is played — so it
> is available at draft time and is *not* future leakage. It mirrors the information the ECR
> benchmark already has. `team_next` uses the primary N+1 roster team (a rare midseason trade is an
> approximation). The block is parameterized by `experiment.position` +
> `features.offseason_workload_col` so it generalizes across positions (§9).

> Feature backlog (see PROJECT_PLAN §13 roadmap for status):
> - **Red-zone/goal-line touches & route participation** need play-by-play — **out of scope**
>   (user decision, reaffirmed 2026-06-23); not pursued.
> - **PFR advanced stats** — deferred to a later enrichment pass (needs a scraper).
> - **Combine athletic testing** is fetched/cached (`combine`) **but not yet joined** — open,
>   in-scope item.

---

## EDA outputs (Stage 6 — `eda/analyze.py`, `scripts/run_eda.py`)

Read-only analysis of the processed matrix; **no new columns** are added to the dataset. The
notebooks (`notebooks/football/rb/01_coverage.ipynb`, `02_distributions_target_stability.ipynb`)
and the headless script call the same library functions and emit these regenerable artifacts to
`reports/results/eda_<sport>_<position>_*.csv` (+ `_summary.json`) and `reports/figures/eda_*.png`:

| Artifact | Function | What it shows |
|---|---|---|
| `universe` | `season_universe` | per-season RB count, `status_next` breakdown, **target coverage** (share with observed N+1), mean games/PPG, era. Exposes the trailing-season censoring. |
| `block_coverage` | `block_coverage` | mean non-null fraction of each feature block per era; confirms the nested era schema (snap_usage 2012+, ngs_efficiency 2016+). |
| `column_coverage` | `column_coverage_by_season` | per-season non-null fraction of era-gated columns → era boundaries drawn from data. |
| `ngs_tier_coverage` | `ngs_coverage_by_tier` | NGS coverage by within-season PPR finish tier (≈75% top-36 vs ≈10% RB37+) — the §7.3 ablation rationale. |
| `distributions` | `distribution_summary` | mean/std/skew/quantiles/zero-share of target + key features (right-skew → rank metrics). |
| `target_stability` | `target_stability` | per-season + pooled Spearman/Pearson of PPG_N vs PPG_N+1 and the **persistence-baseline** MAE/RMSE (§7.1 must-beat floor). |
| `regression_to_mean` | `regression_to_mean` | mean PPG_N+1 by season-N PPG quintile + `shrink` (mean-reversion strength). |

---

## Experiment outputs (Stage 8 — `eval/experiment.py`, `scripts/run_experiment.py`)

Walk-forward results over the fixed test block × {10,20,30}-yr windows × models (§6–§8). No
columns are added to the dataset; tidy tables are written to
`reports/results/experiment_<sport>_<position>_*.csv` (+ `_summary.json`), keyed by
(sport, position, model, window, combine, cutoff, fold):

| Artifact | What it holds |
|---|---|
| `ranking` | one row per (model_type, model, window_years, combine, cutoff_games, **test_season**): MAE/RMSE/R², Spearman/Kendall, **`weighted_tau`** (top-weighted Kendall τ — errors near rank #1 count most), NDCG@k & Precision@k (k=12/24/36). `era_weights` records the combiner's per-era weights. |
| `ranking_aggregate` | the above **mean ± sd across folds** per (model, window, combine, cutoff) + `n_folds`. |
| `recency` | headline Spearman by **window** per (model, combine, cutoff) — the §6.2 recency-bias read (does 10 yr ≈/> 30 yr?). |
| `ngs_ablation` | the `ngs` era model **with vs without** the `ngs_efficiency` block on the g* board (P@12/24/36 + Spearman/NDCG) and the `keep_ngs_block` decision (§7.3). |
| `availability` | the §7.4 count model vs the prior-games baseline per fold: games MAE/RMSE + clears-cutoff AUC/PR-AUC. |
| `benchmark` | preseason market (−ECR) scored per (test_season, cutoff) on the eligible∩ranked universe, with `n_eligible`/`n_market_ranked`/`coverage`. |
| `benchmark_comparison` | **head-to-head** on the identical market-ranked rows (longest window, default combiner, g*): each era-ensemble model vs `market_ecr` — Spearman + **`weighted_tau`** (top-weighted) + Precision@12 (tier-1) + Precision@24 (tier-2). The "do we beat the market?" test, per tier. |
| `summary.json` | headline best model/combine/window at g*, its Spearman/P@12/MAE, and the best baseline it beats. |

## Market benchmark (Stage 9 — `data/benchmark.py`, `data/external/market_<sport>_<position>.parquet`)

Preseason ADP/ECR is a **benchmark only**, never a feature (§7.4). The committed reference table is
built from the open DynastyProcess FantasyPros archive — each season's **latest preseason**
redraft-overall scrape, restricted to the position and mapped to `gsis_id` via the nflverse ID
crosswalk (`fantasypros_id` → `gsis_id`):

| Field | Meaning |
|---|---|
| `player_id` | nflverse `gsis_id` (joins to the dataset; the merge is on **label** season N+1). |
| `season` | the season the ranking predicts (label season Y). |
| `market_ecr` | raw expert-consensus rank (lower = better); the experiment scores `−market_ecr`. |
| `market_rank` | dense positional rank within the season. |
| `scrape_date` | the preseason snapshot date (provenance). |

## Report (Stage 9 — `eval/report.py`, `scripts/make_report.py` → `reports/REPORT_<sport>_<position>.md`)

Markdown assembly of the result tables (headline, recency, market head-to-head, NGS ablation,
availability, eligibility-cutoff sensitivity) + two figures (`report_*_windows.png`,
`report_*_vs_market.png`). Pure rendering — no modeling — so it always reflects the latest run.

> Leakage controls (§6.4): training windows are bounded above at `min(test)−1−horizon`; imputation
> is fit inside each estimator pipeline on training rows only; the eligibility cutoff is **applied,
> not refit**, inside folds and reported across the whole candidate grid for robustness.
