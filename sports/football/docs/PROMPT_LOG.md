# Prompt Log — Football

A chronological record of every prompt related to the **football** modeling project, with notes on the
response. Maintained for **reproducibility and decision provenance** — anyone should be able to trace
*why* this project looks the way it does.

> **Scope.** This is the football-domain view. The repo-wide superset (all sports + repo-level
> infrastructure prompts) lives in the root [`docs/PROMPT_LOG.md`](../../../docs/PROMPT_LOG.md).
> Entry numbers are **global** and shared with the primary, so they are stable for cross-referencing but
> **non-contiguous here** where repo-level entries (e.g. the reorg, the log system) are kept to the
> primary only.
>
> Convention: append every football prompt to **both** this log and the primary, quoted **in full**;
> keep response notes concise but specific (decisions, files, rationale).

---

## Entry 001 — Project kickoff & scope

**Date:** 2026-06-15

**Prompt (full text):**

> I want to start a new project. I want to build a plan with you that will include
> options for data sources, model types, recommended data sources, recommended
> pipeline, features, models, scoring stats to use in analysis, and any other things
> needed including EDA. For the purposes of reproducibility, make sure there is a
> document that includes the full text of every prompt I submit and some notes on your
> response to that prompt. Make sure to include this one.
> I want it run like a proper reproducible experiment. With data fetching, EDA
> notebooks, feature analysis and selection, and modeling and results.
> This will become a project on github so I will also want to make sure we have proper
> project structure and README and anything else.
> The goal of this project is to use past data on pro sports players to predict their
> fantasy rank next season. I want to produce a separate process for each sport and make
> sure each position is modeled separately for accuracy. We're going to start with
> football and the running back position.
> Lets start with you asking any additional questions you have and dig into what the
> project will look like

**Response notes:**
- Confirmed the framing: supervised, per-position, per-sport; start NFL RB.
- Asked two rounds of clarifying questions (see decisions captured in Entries 002–003).
- Created this prompt log per the explicit reproducibility requirement, including this
  first prompt verbatim.

---

## Entry 002 — Clarifying questions, round 1

**Date:** 2026-06-15

**Prompt (full text — answers to round-1 questions):**

> - **Scoring format:** PPR
> - **Target construction:** "We want to look purely at PPG ranking and filter out anyone
>   playing fewer than a certain game count or snap count when evaluating final season
>   ranking. A high PPG low game player still has value if those points come at the right
>   time. Determining the cutoffs for snap count or game count should be part of the
>   process of every football model."
> - **Player pool:** Returning players only
> - **Data source:** nfl_data_py / nflverse

**Response notes:**
- Set target = **next-season PPR PPG**, ranked among eligible players.
- Elevated the **games/snap eligibility cutoff to a first-class, data-driven pipeline
  step** required of every football model (reliability analysis + sensitivity grid),
  rather than a hardcoded threshold. Documented in PROJECT_PLAN §4.
- Scope = returning players; rookies deferred. Source = nflverse.

---

## Entry 003 — Clarifying questions, round 2

**Date:** 2026-06-15

**Prompt (full text — answers to round-2 questions):**

> - **Tooling:** uv + Jupyter
> - **History:** "For every model, I want to look at a 10, 20, and 30 year timeline to
>   compare the results. This way I can judge recency bias for each position. If there
>   are limitations to the data we can try to accommodate for them but the focus should
>   be on this comparative analysis."
> - **Market data:** Benchmark only

**Response notes:**
- Tooling = uv (pyproject + uv.lock) + Jupyter; library code, not notebook logic.
- Established the **10/20/30-year comparative training-window design** to measure recency
  bias per position, with a fixed recent test block (PROJECT_PLAN §6.2). Documented data
  limits: snaps 2012+, NGS 2016+, nflverse box scores 1999+ (30yr is data-capped).
- ADP/ECR used as a **benchmark to beat**, not a feature.
- Wrote `docs/PROJECT_PLAN.md`, this log, `README.md`, scaffolded the repo
  (`src/`, `notebooks/`, `scripts/`, `config/`, `data/`, `reports/`, `tests/`),
  `pyproject.toml`, `.gitignore`, and the RB experiment config.

---

## Entry 004 — Resolving §12 open questions

**Date:** 2026-06-15

**Prompt (full text):**

> for the open questions in s12: I like the last 5 seasons for k. I like the idea of
> adding PFR advanced stats later. We'll ignore the rookie model as a future track, I'm
> not interested in pursuing.

**Response notes:**
- Resolved PROJECT_PLAN §12: test block **k = last 5 seasons**; **PFR advanced stats =
  later enrichment block**; **rookie model dropped entirely** (not a future track).
- Updated §1.1 scope row to mark rookies out of scope. Position ordering after RB
  (WR → QB → TE) remains the only open item.

---

## Entry 005 — Build begins; quantitative-only features + injury model

**Date:** 2026-06-15

**Prompt (full text):**

> Go ahead and build. One thing I want to keep in mind for feature engineering: I want to
> try to keep this as based on numbers as possible. So take coaching changes and similar
> qualitative aspects out even if this affects the quality of the results. There may be
> ways to build proxies via snap count or other stats. I would like to make sure that
> injuries are taken into account, so some way of predicting reinjury or total games
> yielded in the N+1 season will need to be part of the football process at the least

**Response notes:**
- Added a **quantitative-only feature principle** (PROJECT_PLAN §5): no qualitative/
  hand-labeled inputs; replaced the "coaching/scheme change flag" with measurable stat
  proxies (YoY team run/pass, personnel/pace, snap- & route-share deltas). Set
  `features.quantitative_only: true` in config.
- Added a **required availability/injury model** (PROJECT_PLAN §4.3, §7.3, §8): a parallel
  count model predicting N+1 games played, gating projected eligibility and reporting
  reinjury risk; configured under `availability_model` in `config/football_rb.yaml`.
- Began the build with the **fetch stage**: `src/position_predictor/utils/{config,io}.py`,
  `src/position_predictor/data/fetch.py`, and `scripts/fetch_data.py` (config-driven,
  cached + manifested nflverse pulls, lazy import, `--dry-run`). Tested in `tests/test_fetch_plan.py`.
- **Stage 2 (build dataset)** added: `src/position_predictor/data/build.py` +
  `scripts/build_dataset.py`. Aggregates RB weekly box scores (regular season) into a
  `(player_id, season)` table (games, PPR points, PPG, volume), joins roster attributes
  (age/exp/size), attaches the **no-leakage next-season target** (`target_ppg_next`) plus
  availability signals (`games_next`), and materialises the **eligibility candidate games
  grid** (`eligible_next__g{G}`). Returning-players-only is enforced structurally (rows exist
  only for season-*N* feature seasons; non-returners get NaN target). Pure logic unit-tested
  in `tests/test_build_dataset.py` (9/9 tests pass overall). `docs/data_dictionary.md` updated
  with the produced fields.

---

## Entry 006 — Stage 2 design review

**Date:** 2026-06-15

**Prompt (full text):**

> Which design decisions need to be evaluated? What in stage 2 needs to be reviewed?

**Response notes:**
- Surfaced Stage 2 review points: (1) "games played" = weekly row-count definition needing
  empirical validation; (2) a full-season miss being *dropped* rather than labeled
  `games_next = 0`, conflating injury with leaving the league; (3) snap-share eligibility
  dimension deferred; (4) position by NFL designation vs fantasy eligibility; (5) `age` from
  the fuzzy roster field; (6) multi-team season handling.
- Catalogued open design decisions for later stages: era-limited data × 10/20/30yr windows
  and whether to pull `pbp` (Stage 4); cutoff derivation operationalisation (Stage 5);
  walk-forward mechanics, recency-study honesty caveat, ranking approach (Stage 8); and ADP/
  ECR benchmark acquisition (Stages 7–9).

---

## Entry 007 — Verify games-played; injury vs retirement; fantasy eligibility; age

**Date:** 2026-06-15

**Prompt (full text):**

> verify the games-played behavior against real data. for players who miss games, some might
> be retired or semi-retired so use future seasons played as an indicator for this to separate
> those who are "gone" temporarily. don't try to predict 'unretirement' but maybe consider
> retirement a type of injury. for positions, make sure the position is determined by fantasy
> eligibility rather than nfl designation. make sure to compute age-at-season-start.
> [follow-up] and use a subagent to figure out an alternative data source for fantasy eligibility

**Response notes:**
- **Verified games-played against real nflverse data:** the weekly row-count (REG only)
  matches the seasonal `games` column **100%** over a 2-season sample. Confirmed a full-season
  miss has **no** weekly rows (J.K. Dobbins 2021), validating review point (2).
- **Injury vs retirement:** rewrote `attach_next_season_target` to classify `status_next` ∈
  {`active`, `injured_out`, `retired`, `censored`} using *future* seasons. Retirement is folded
  in as a zero-availability outcome (`games_next = 0`, `retired_next = True`); no
  "un-retirement" is modelled. PPG target is defined only for `active`. Verified end-to-end
  (Dobbins 2020 → `injured_out`; Gore/McCoy → `retired`).
- **Fantasy eligibility (subagent):** launched a research subagent which recommended the
  **Sleeper `/v1/players/nfl`** endpoint (`fantasy_positions`, join on `gsis_id`). Verified the
  API live (12.2k players, 14.6 MB, no auth). Added a `sleeper_players` fetch dataset and
  `resolve_fantasy_eligibility` (Sleeper primary → nflverse `position_group` fallback, with an
  `eligibility_source` provenance column), since only ~⅓ of Sleeper players carry a `gsis_id`.
  Spelled out the *current-snapshot / no historical eligibility* caveat in the data dictionary.
- **Age-at-season-start:** `attach_roster_attributes` now computes `age_at_season_start` from
  `birth_date` as of Sept 1 of season *N*, replacing the roster's coarse `age` field.
- Tests rewritten/expanded (`tests/test_build_dataset.py`, 7 cases) — full suite **11/11 pass**.
  `docs/data_dictionary.md` updated. (Environment note: this restored shell has no `uv`/pandas;
  verification used a throwaway venv at `/tmp/ppvenv` — not part of the repo.)

---

## Entry 008 — Fix environment; snap-share crosswalk

**Date:** 2026-06-15

**Prompt (full text):**

> Can you fix the environment issue and then do the snap-share crosswalk

**Response notes:**
- **Environment fixed for real:** installed `uv` (0.11.21) and ran `uv sync`, creating the
  project `.venv` + `uv.lock` from `pyproject.toml`. Added a missing runtime dependency
  **`requests`** (the Sleeper loader needs it; would have broken `make fetch`). Added
  `.python-version` (3.12) per the reproducibility checklist. `uv run pytest` now works in the
  project's own environment (no more throwaway venv).
- **Snap-share crosswalk:** snaps key on `pfr_player_id`, not the gsis `player_id`. Added
  `src/position_predictor/data/snaps.py` — `crosswalk_pfr_to_gsis` (via `import_ids`
  `pfr_id`↔`gsis_id`) and `build_player_season_snaps`, which filters to REG (`game_type`;
  playoffs are WC/DIV/CON/SB, not "POST"), crosswalks, and computes a snaps-weighted season
  `snap_share` (= Σ snaps / Σ team snaps, team total recovered as `offense_snaps/offense_pct`),
  plus `snaps` and `snaps_per_game`. **Verified 100% pfr→gsis mapping** for RB snap rows on a
  recent season.
- **Integrated into Stage 2:** `attach_snap_share` joins season-*N* snap share; the next-season
  attach now carries `snap_share_next` (0 when out all year, NaN if censored/pre-2012); and
  `label_eligibility_grid` now emits the **snap-share candidate flags** `eligible_next__s{S}`
  (e.g. `__s30`=0.30) alongside the games flags — completing the §4.2 two-dimension cutoff grid.
- Smoke-tested end-to-end on real data: snap share 100% populated for 2012+ RBs; Derrick Henry
  2022 = 0.67 snap share, 40.5 snaps/game. Tests added (`tests/test_snaps.py` + build cases) —
  full suite **18/18 pass**. `docs/data_dictionary.md` updated.

---

## Entry 009 — Model eras separately; window→era-model routing

**Date:** 2026-06-15

**Prompt (full text):**

> model eras separately and make sure that an X-year window model runs data through each era
> model appropriately

**Response notes:**
- Resolved the era-limited-data decision (snaps 2012+, NGS 2016+ × the 10/20/30-yr windows) in
  favour of **era-separate modeling** rather than imputing/masking structurally-absent features.
- Defined **eras** = contiguous season spans sharing a feature schema: `boxscore` (1999–2011),
  `snaps` (2012–2015, +snap_usage), `ngs` (2016+, +ngs_efficiency). Schemas are **nested** so an
  older-era model can score a modern player on its feature subset. Added the `eras` +
  `era_modeling` config blocks to `config/football_rb.yaml`.
- Confirmed with the user that a row's era is keyed by its **feature season N** (`route_by`).
- Built `src/position_predictor/eras.py`: `load_eras` (validates contiguity + nested schemas),
  `assign_era`, `train_window_bounds` (no-leakage feature-season bounds with horizon
  subtracted), and **`era_segments_for_window`** — maps an X-year window to the per-era
  season segments it trains on, i.e. exactly the era models that compose that window and the
  data routed to each. Default window combiner = `val_weighted` (configurable).
- Documented the architecture in PROJECT_PLAN **§6.3** (era-separate modeling + window
  composition) and **§7.3** (era ensemble); renumbered leakage controls to §6.4 / availability
  model to §7.4 and fixed cross-refs. Tests in `tests/test_eras.py` (8 cases) — full suite
  **25/25 pass**, ruff clean.
- Stage-8 will implement the per-era estimators + combiner against this contract; Stage-4
  features will emit the block→columns map that `feature_columns_for_era` consumes.

---

## Entry 010 — Stage 4: engineered features (block-organised, era-aware)

**Date:** 2026-06-16

**Prompt (full text):**

> yes, proceed to stage 4

**Response notes:**
- Built **Stage 4** (`src/position_predictor/features/build.py` + `scripts/build_features.py`):
  interim player-season table + raw weekly/NGS → processed modeling matrix, organised into the
  10 feature **blocks** matching the era schemas. Each `add_*` block returns the columns it
  produces; the orchestrator writes the **block→columns map** (`*_feature_blocks.json`) that
  `eras.feature_columns_for_era` consumes for era-aware feature selection.
- Blocks: `production, volume, team_context, efficiency, player_attrs, availability, trajectory,
  regression_mean` (all 1999+), `snap_usage` (2012+), `ngs_efficiency` (2016+). Added a
  `team_context` block to the era config (all eras) for §5 team proxies (usage shares, pace,
  run-rate). Extended Stage-2 aggregation to sum `rushing_epa`/`receiving_epa` (powers EPA/play).
- **No-leakage** enforced and tested: trajectory deltas/slopes and regression-to-mean baselines
  use only prior seasons (`shift(1)` within player); within-season finish ranks use season-N
  rows only; first-season rows are NaN for prior-dependent features.
- Hardened `utils.io.write_parquet`: nflverse `jersey_number` is a mixed-type object column that
  Arrow refuses to serialise — now coerces only mixed object columns to string on a write
  failure and retries (would otherwise break `make fetch` on wider season ranges).
- Verified end-to-end on real data (2009–2023): 2,432 RB player-seasons, **73 feature columns**;
  era selection nests correctly (boxscore 63 → snaps 66 → ngs 73); snap_share 0% pre-2012, NGS
  0% pre-2016 as designed. Tests added (`tests/test_features.py`, 11 cases) — full suite
  **36/36 pass**, ruff clean. `docs/data_dictionary.md` feature-block section filled in.

---

## Entry 011 — NGS sparsity: coverage flags + planned ablation

**Date:** 2026-06-16

**Prompt (full text):**

> Should we consider either ignoring the NGS stats since they're so sparsely populated or only
> including NGS players since they are the high-volume rushers?
> [follow-up] add the flags now and bake the ablation into stage 8

**Response notes:**
- Measured NGS coverage by tier (2016–2023): ~73% for top-36 RBs vs ~6% for RB49+; the overall
  27% is dominated by fantasy-irrelevant fringe backs. A top-12 back lacking NGS *rushing* is
  typically a **receiving-profile** RB (NGS rushing has an attempt threshold) — so the
  missingness is **signal, not noise**.
- Rejected both proposed options: dropping NGS discards real signal for ~73% of ranked backs;
  restricting to NGS players drops ~25% of RB1s (receiving backs) and breaks the eligibility
  universe / era comparability.
- **Implemented:** added `has_ngs_rush` / `has_ngs_rec` coverage flags to the `ngs_efficiency`
  block (`add_ngs_efficiency`), so every model type can use the coverage signal (trees also take
  raw NaN natively). Tests added (`tests/test_features.py`, now 13 cases) — suite green, ruff clean.
- **Baked the ablation into Stage 8** (PROJECT_PLAN §7.3): the `ngs` era model is run with and
  without the `ngs_efficiency` block, compared on top-k precision (P@12/24/36) + Spearman/NDCG;
  keep the block only if it improves out-of-fold, else the `ngs` era collapses to the `snaps`
  schema. Generalised as a reusable feature-block ablation harness.

---

## Entry 012 — Stage 5: derive the eligibility cutoff

**Date:** 2026-06-16

**Prompt (full text):**

> proceed to stage 5

**Response notes:**
- Built **Stage 5** (`src/position_predictor/eligibility/cutoff.py` + `scripts/run_eligibility.py`):
  derives the games/snap ranking-eligibility cutoff (PROJECT_PLAN §4.2) via three reads —
  (1) **split-half PPG reliability** per games count *k* (correlation of two disjoint *k*-game
  samples; primary), (2) **coverage vs purity** across the candidate games grid, (3) snap-share
  coverage (secondary, 2012+). Writes chosen rule + full grid + reliability/coverage figure to
  `reports/`.
- Added config knobs `eligibility.reliability_target` (0.70) and `reliability_k_max` (8; split-half
  needs 2·k games so the curve is testable to ~8 on 16/17-game seasons).
- **Result on real data (1999–2023): `g* = 4`** — smallest games count where reliability ≥ 0.70
  (0.738 at G=4; curve 0.45→0.80, stabilising ~0.80 by G=6–8). G≥4 retains **98.5%** of fantasy
  production vs 79% at G≥12. Recorded `chosen_games_played: 4` in config and PROJECT_PLAN §12.
- Tests in `tests/test_eligibility.py` (6 cases incl. reliability recovers known signal / ≈0 for
  pure noise) — suite green, ruff clean. `reports/results` is gitignored (regenerable outputs).

---

## Entry 013 — Stage 6: EDA (coverage, distributions, target stability)

**Date:** 2026-06-16

**Prompt (full text):**

> [resume] … do 6 first, make sure you do things in order

**Response notes:**
- Clarified that pipeline **Stage 3 (target)** was never skipped — PROJECT_PLAN §3 specifies it
  `(in build)`, i.e. it lives inside `data/build.py` (`attach_next_season_target`,
  `target_ppg_next`, `status_next`, eligibility grids), not as a separate script.
- **Ran the upstream pipeline in order on real data** (fetch → build → features). Surfaced and
  fixed a **fetch robustness bug**: `nfl_data_py` aborts the whole `weekly`/`seasonal` request
  if any single season's release asset is missing, so the unpublished **2025** `player_stats`
  asset 404'd and wiped out 1999–2024 box scores. Added a **per-year fallback** in
  `data/fetch.py` (`_load_resilient`): try the fast batch, and on failure retry season-by-season,
  keeping what exists and recording skipped seasons in the manifest. Result: `weekly`/`seasonal`
  fetched 1999–2024, 2025 cleanly skipped.
- **Built Stage 6** (`src/position_predictor/eda/analyze.py` + `scripts/run_eda.py` + `make eda`):
  pure, tested functions for the three §3 reads — **coverage** (`season_universe`,
  `block_coverage`, `column_coverage_by_season`, `ngs_coverage_by_tier`), **distributions**
  (`distribution_summary`), **target stability** (`target_stability`, `regression_to_mean`) —
  writing tidy `reports/results/eda_*.csv` + `_summary.json` and two `reports/figures/eda_*.png`.
  Two notebooks (`01_coverage`, `02_distributions_target_stability`) call the same library code
  (no copy-paste logic) and were **executed end-to-end** with their outputs/figures committed.
- **EDA caught a data-labeling bug** (its whole point): the latest feature season (2024) was
  classified entirely `retired` because the censoring boundary came from config
  (`latest_completed_season: 2025`) while the data only reached 2024 — so 2024's absent N+1 read
  as "left the league" instead of "unknown future". Fixed in the **build orchestrator** (not the
  pure function, which a unit test calls with an explicit boundary): clamp the boundary to
  `min(config latest, max season present)`. 2024's 146 rows now correctly become `censored`;
  `active`/`injured`/target counts unchanged. Updated `data_dictionary.md` accordingly.
- **Findings (1999–2024, 2,975 modelable RB seasons):** year-over-year PPG is strongly
  autocorrelated (pooled Spearman ≈0.66, Pearson ≈0.71; persistence-baseline MAE ≈3.16 PPG);
  NGS coverage ≈75% for top-36 vs ≈10% for RB37+ (confirms the §7.3 missing-is-signal rationale);
  clear mean reversion in the PPG tails (top quintile 16.4→13.7 vs grand mean 7.0). These ground
  the §7.1 baselines and the rank-based evaluation objective.
- Tests `tests/test_eda.py` (7 known-signal cases) — full suite **51/51 pass**, ruff clean.
  Updated README status + `data_dictionary.md` (EDA outputs table). Next: Stage 8 modeling.

---

## Entry 014 — Stage 8: walk-forward experiment (era ensemble, ablation, availability)

**Date:** 2026-06-16

**Prompt (full text):**

> proceed to stage 8

**Response notes:**
- Built **Stage 8** end-to-end against the existing era contract (`eras.train_window_bounds` /
  `era_segments_for_window` / `feature_columns_for_era`):
  - `eval/metrics.py` — regression (MAE/RMSE/R²), ranking (Spearman/Kendall/NDCG@k/Precision@k for
    tiers 12/24/36), availability (games MAE/RMSE + clears-cutoff AUC/PR-AUC). All pure.
  - `models/baselines.py` (§7.1: persistence, smoothed_history, mean_reversion, linear) +
    `models/zoo.py` (§7.2 estimator factory: ridge/lasso/elasticnet/random_forest/lightgbm/xgboost
    with the right NaN regime per family; + §7.4 Poisson availability model).
  - `models/era_ensemble.py` (§6.3/§7.3) — fit one estimator per era on its nested schema; combine
    by `val_weighted` (weight ∝ each era model's held-out-tail Spearman), `mean`, `recency_weighted`.
  - `eval/experiment.py` + `scripts/run_experiment.py` (`make experiment`, `--fast`) — walk-forward
    over the fixed test block × {10,20,30}-yr windows × models; eligibility cutoff **applied not
    refit**, reported across the candidate grid; the required **NGS-block ablation**; the
    availability model vs prior-games baseline. Writes 6 tidy tables keyed by
    (sport, position, model, window, combine, cutoff, fold) to `reports/results/experiment_*`.
- **Fixed two feature-matrix bugs surfaced by the model fits** (both real data-quality issues):
  (1) `draft_capital` was a *string* dtype (nflverse `draft_number` is numeric-valued string) —
  coerced to numeric in `features/build.py`; (2) `feature_columns_for_era` returns **duplicate**
  columns because blocks overlap (e.g. `receptions` ∈ production & volume) — deduped in the
  model-matrix consumers (LightGBM rejects duplicate feature names). Rebuilt features.
- **Results (RB v1, test 2020–2024, g\* = 4):** era ensemble Spearman ≈ **0.741** (lightgbm/xgboost/
  ridge, 20–30-yr window) beats best baseline `linear` 0.720 and persistence 0.691; MAE ≈ 2.69 vs
  3.14. Recency: 20≈30 yr (data-capped), ~10 yr close behind, `val_weighted` ≈ equal era weights →
  older-era relationships still predict modern RBs. **NGS ablation:** with-NGS does *not* beat
  without on P@12 → `ngs` collapses to the `snaps` schema for v1 (flags kept). **Availability:**
  Poisson edges prior-games on clears-AUC (≈0.79 vs 0.78), not on games MAE. Recorded all in
  PROJECT_PLAN §12. Full run ≈ 75 s.
- Tests `tests/test_metrics.py` + `test_era_ensemble.py` + `test_experiment.py` (13 cases) — full
  suite **64/64 pass**, ruff clean. Updated README + `data_dictionary.md` (experiment outputs).
  Next: Stage 9 reporting + ADP/ECR market benchmark.

---

## Entry 015 — Stage 9: market benchmark + results report

**Date:** 2026-06-16

**Prompt (full text):**

> Go ahead!  (proceed to Stage 9)

**Response notes:**
- **Acquired the market benchmark** (§2.3, §7.4) — the genuinely hard part. Probed sources and
  used the open **DynastyProcess FantasyPros ECR** archive (`db_fpecr.parquet`). Built
  `data/benchmark.py` + `scripts/fetch_benchmark.py` (`make benchmark`): take each season's
  **latest preseason** redraft-overall scrape, restrict to RB, map `fantasypros_id → gsis_id`
  via the already-fetched nflverse `ids` crosswalk, and write a small **committed**
  `data/external/market_football_rb.parquet` (800 player-seasons, 2020–2025; coverage of the
  eligible universe 72%→99%).
- **Scored the market head-to-head** by extending `eval/experiment.py`: `_attach_market` joins
  preseason ECR onto each row by its **label** season (market never enters as a feature — model
  columns come only from the block map). Two new tables: `benchmark` (market metrics + coverage
  per fold on eligible∩ranked) and `benchmark_comparison` (each era model **vs** `market_ecr` on
  the *identical* market-ranked rows, longest window + default combiner).
- **Built Stage 9 reporting** (`eval/report.py` + `scripts/make_report.py`, `make report`):
  renders `reports/REPORT_football_rb.md` (headline, recency table, market head-to-head, NGS
  ablation, availability, eligibility-cutoff sensitivity) + two figures.
- **Headline / honest verdict:** the model's 0.74 Spearman was on the *full* eligible universe;
  on the **identical rows the market ranks**, the **market wins overall rank** (Spearman ≈0.732 vs
  best model ≈0.690) — but the model **matches/edges the market on Precision@12** (ridge 0.63 vs
  0.60), the draftable tier, using **no** market data. The benchmark's apples-to-apples design is
  exactly what surfaced this (the full-universe number flatters the model via easy-to-rank fringe
  backs). Eligibility-cutoff robustness: Spearman stable g*=4→6, degrades only at g≥8. Recorded in
  PROJECT_PLAN §12.
- Tests `tests/test_benchmark.py` (5 cases: preseason-scrape selection, crosswalk/scoring, safe
  missing-file) — full suite **69/69 pass**, ruff clean. Updated README (status → RB v1 complete),
  `data_dictionary.md` (benchmark + report outputs), Makefile (`benchmark` target). Next: richer
  usage features to challenge the market, then WR → QB → TE.

---

## Entry 016 — Top-weighted ranking metric (weighted Kendall τ)

**Date:** 2026-06-16

**Prompt (full text):**

> Is there a way to re-score the accuracy so that it counts less if the player is farther away
> from the number 1 rank?

**Response notes:**
- Yes — this is **top-weighted ranking**. Noted we already compute NDCG@k (log-discount,
  top-emphasis) and added the most direct answer: **`weighted_tau`** =
  `scipy.stats.weightedtau`, a rank correlation whose pairs are weighted by rank position with a
  hyperbolic decay, so a swap near #1 costs far more than the same swap deep on the board.
  Verified: a top-two swap drops weighted τ to 0.886 while a bottom-two swap stays 0.984, whereas
  **Spearman scores both swaps identically** (0.988).
- Added it **additively** (nothing removed) to `eval/metrics.ranking_metrics`; threaded through
  `_aggregate_ranking` and the market `benchmark_comparison`, and surfaced in the report headline
  + market head-to-head table. Made `_aggregate_ranking` robust to metric columns being absent.
- **Effect on the market verdict (RB v1):** the top-weighted gap is *narrower* than the uniform
  one but the market still leads on rank order — Weighted τ market **0.689** vs best model
  (`ridge`) **0.662** (Spearman was 0.732 vs 0.690); our model still wins the strict top tier
  (Precision@12 0.63 vs 0.60). Best model flips by metric (xgboost on Spearman, ridge on
  weighted τ / P@12), all reported.
- Test `test_weighted_tau_penalizes_top_more_than_bottom` (+ monotonic check) — full suite
  **70/70 pass**, ruff clean. Re-ran experiment + report. Updated `data_dictionary.md`.

---

## Entry 017 — Offseason features (Sept-1 cutoff) to challenge ECR

**Date:** 2026-06-16

**Prompt (full text):**

> I don't want to blend. At draft time I want to look at what the model says, what ECR says,
> compare the two, and make decisions. For offseason signals, let's use september 1st as the
> cutoff date for any offseason activity. Let's see what quantifying them does to our model
> performance.

**Response notes:**
- **No blending** — ECR stays a benchmark to compare against, never a feature (unchanged). The
  ask is to let the *model* see offseason activity up to a **Sept 1 cutoff** — legitimately
  available at draft time, exactly the information ECR has, and not leakage (predates any
  season-*N+1* game).
- Added an **`offseason` feature block** (`features/build.add_offseason`) from `draft_picks` +
  `rosters`, attached to each season-*N* row about its *N+1* prediction: `changed_team_next`,
  `rookie_rb_drafted_next` / `rookie_rb_capital_next` (best overall pick of a RB the N+1 team
  drafted, sentinel 300) / `rookie_rb_count_next`, `backfield_prior_touches_next` (proven
  season-N workload of the *other* RBs in the N+1 room, self excluded), `backfield_rb_count_next`.
  Wired into all three eras (available 1999+; draft/rosters back to 1999, and 2025 already
  present so the live 2024→2025 board gets real signal). Documented the Sept-1 no-leakage
  exception in PROJECT_PLAN §5 + `data_dictionary.md`.
- **Impact (the point of the exercise):** on the market head-to-head, model Spearman **0.690 →
  0.706** (market 0.732 — closed ~40% of the gap), Precision@12 **0.633 → 0.650** (now clear of
  market 0.60); full-universe Spearman **0.74 → 0.752**. The **top-weighted** score barely moved
  (weighted τ 0.662 → 0.663) — offseason signals re-rank the *middle* of the board (committee
  backs gaining/losing a job), not the elite. Feature importance: the continuous
  `backfield_prior_touches_next` and `rookie_rb_capital_next` carry the weight; the binary flags
  are minor. Verified e.g. Derrick Henry 2023→2024: changed_team=1 (TEN→BAL), rookie drafted at
  pick 165, 173 proven competition touches in a 6-RB room.
- Tests `test_add_offseason_quantifies_n_plus_1_context` + `test_add_offseason_sentinel_and_none_safe`
  — full suite **72/72 pass**, ruff clean. Re-ran experiment + report; updated PROJECT_PLAN §5/§12.

---

## Entry 018 — Generalize to WR (skip TE)

**Date:** 2026-06-17

**Prompt (full text):**

> Branch off of the fresh primary. We won't do anything with the pbp data. Let's move on to WR
> and then QB. I think TE is worth skipping due to the small number of fantasy relevant TE's per
> season.

**Response notes:**
- Branched `wr-model` off `primary`. Confirmed the §9 design: build / eras / CV / metrics / report
  were already config-driven by `experiment.position`; the **only** RB-hardcoded code was the
  `offseason` block. **Parameterized `add_offseason`** by `position` + `workload_col` with
  position-agnostic columns (`changed_team_next`, `rookie_drafted/draft_capital/count_next`,
  `room_prior_workload_next`, `room_size_next`); added `features.offseason_workload_col` (RB
  `touches`, WR `targets`). Two harness fixes surfaced by WR: (1) always score the chosen `g*`
  even if it's off the candidate grid (`cutoff_grid |= {g*}`); (2) drop all-NaN feature columns at
  era-model fit time (kills the rushing-NGS-for-WR imputer warnings; no-op for RB).
- Built `config/football_wr.yaml` and ran the full pipeline (build → features → eligibility → eda
  → benchmark → experiment → report). **WR eligibility re-derived: g\* = 7 games** (vs RB 4) — WR
  split-half reliability never clears 0.70 (peaks 0.694 @ k=7); WR scoring is noisier, ranking
  harder.
- **WR results (test 2020–2024):** best model `ridge` Spearman **0.752** full-universe (> linear
  0.732). Vs market on its ranked rows: market 0.767 vs 0.748 (Spearman), 0.733 vs 0.713
  (weighted τ) — market leads overall rank as with RB, **but the tree models beat the market on
  Precision@12 (0.583 vs 0.567)**. **NGS-block ablation keep=True for WR** (P@12 0.53 vs 0.50) —
  receiving separation/air-yards is real WR signal, unlike RB where NGS collapsed.
- WR ECR benchmark fetched (982 player-seasons; ~76% coverage of eligible WRs). Tests:
  `test_add_offseason_position_aware_workload` (WR currency = targets, RB picks excluded) — full
  suite **73/73 pass**, ruff clean. Updated PROJECT_PLAN §9/§12, `data_dictionary.md`, README. TE
  skipped per the prompt; QB next.

---

## Entry 019 — RB v3: back-apply WR learnings (new iterative process)

**Date:** 2026-06-17

**Prompt (full text):**

> push the WR branch now and then lets start work on RB v3. I want this to be the process. After
> finishing model N, recheck models N-1 .. 1 and apply learnings to see what can be gained.
> [follow-up] can you branch off of the fresh primary instead

**Response notes:**
- Pushed `wr-model` (PR #2, since merged). Saved the **iterative back-apply process** to memory
  (after model N, revisit N-1..1 with N's learnings). Re-based `rb-v3` off the fresh `primary`
  (which now includes the merged WR generalization) per the follow-up.
- **Applied two WR learnings to RB, measuring each against the captured v2 baseline (ridge head-to-head
  Spearman 0.704 / wτ 0.663 / P@12 0.650; full-universe xgboost 0.752):**
  1. **Vacated-opportunity feature** (`vacated_workload_next`) — the workload that *left* the
     player's N+1 team (complement to room competition). **Real gain**: xgboost head-to-head
     Precision@12 0.55 → **0.58**, full-universe P@12 0.53 → 0.57 / Spearman 0.752 → 0.756; concentrated
     in the tree models at the top tier (where we compete with ECR). Linear models flat. Verified
     the team-level logic (DEN vacated 465 touches after 2022, shared by its 2023 backs).
  2. **Receiving-NGS-for-RB** — WR showed receiving NGS helps. Probed the RB ngs-era model with
     rushing-only / receiving-only / no-NGS subsets → **identical** top-12 (0.450) & Spearman
     (~0.712). NGS does not move RB ranking (opportunity dominates). **Rejected** — not
     productionized. The process keeping only what measurably helps.
- Feature is position-agnostic, so also applied + re-run for WR. Test
  `test_add_offseason_vacated_opportunity` — full suite **74/74 pass**, ruff clean. Updated
  PROJECT_PLAN §12, `data_dictionary.md`. Next big bet for the top-of-board gap: learning-to-rank.

---

## Entry 020 — Top-weighted-training probe + Precision@24

**Date:** 2026-06-17

**Prompt (full text):**

> I was going to suggest weight training, so let's start there. Also, I think there might be value
> in looking at Precision@24 as a metric as well, since drafting an rb2 can be just as important as
> an rb1.

**Response notes:**
- **Cheap learning-to-rank probe (weight training toward the top).** Added
  `top_weighted_sample_weights` + an `EraEnsemble(top_weighted=...)` path (config
  `models.top_weighted_training`, `--top-weighted` CLI flag; routes a per-row weight to each
  estimator incl. sklearn pipelines). Weight = within-season NDCG-style finish discount, rescaled
  to mean 1. **Result: REJECTED.** vs the v3 baseline it nudged `weighted_tau` +0.006–0.011 for
  linear models (its design target) but **hurt Precision@12 by −0.03 to −0.05** and was flat
  elsewhere — chasing the smooth top-weighted score trades away real top-12 hits. Machinery kept
  but **off by default**. This de-risks (lowers expected payoff of) full LambdaMART.
- **Precision@24 surfaced** in `benchmark_comparison` (already computed; now reported per the top
  two tiers) + the report headline/table. **Finding:** we **beat** the market on the RB1 tier
  (P@12 0.65 vs 0.60) but **trail** on the RB2 tier (**P@24 ~0.72 vs 0.75**) — the market's broad
  consensus ranks the middle of the board better than us. A genuine blind spot the metric exposed.
- Tests `test_top_weighted_sample_weights` + `test_top_weighted_ensemble_fits_and_predicts` —
  full suite **76/76 pass**, ruff clean. Restored canonical (top-weighted-off) RB+WR results +
  reports. Updated PROJECT_PLAN §12, `data_dictionary.md`.

---

## Entry 021 — Versioned, comprehensive reports (per-version snapshots + cross-version progress)

**Date:** 2026-06-17

**Prompt (full text):**

> Where are we at now? [...] What happened to RB v2? [...] Is it at least identifiable in reports?
> I want to make sure reports are comprehensive [...] can you make sure there are individual reports
> for each version so we can analyze progress between versions and relative gains of models vs
> compute/efficiency/data volume

**Response notes:**
- **Reports were not version-identifiable** (only a generation date) and **overwrote in place** — no
  history. Added `experiment.version` to each config (RB `v3`, WR `v1`); the report header now stamps
  `**RB v3** · <branch> @ <shortsha>[-dirty]` (`_git_provenance()` in `report.py`).
- **Instrumented the experiment for cost.** `run_experiment` now records a `cost` table (per
  model×window×combine: `fit_seconds`, `n_train_rows`, `n_features`) and a `data_volume` +
  `compute` block in the summary (feature/labeled rows, per-era counts, season span, total fit +
  wall-clock). Report gained **Data volume** and **Compute & efficiency** sections — the latter
  joins per-model fit-seconds to ranking quality (Spearman/fit-s). Surfaced the key efficiency
  finding: **ridge gets ~0.747 Spearman & P@12 0.63 at ~0.1s fit vs xgboost 0.756 / P@12 0.57 at
  ~11s — ~100× cheaper for ~1% less Spearman and better top-12.**
- **Per-version snapshots.** `make report` now archives a committed snapshot to
  `reports/versions/<stem>/<version>/` (report .md + summary.json + cost/benchmark_comparison/recency
  CSVs). `reports/versions/` is git-tracked (only `reports/{figures,results}/*` are ignored).
- **Cross-version progress report** (`eval/progress.py`, `scripts/make_progress.py`, `make
  progress`) → `reports/versions/<stem>/PROGRESS_<stem>.md`: best-model-vs-market metrics and
  compute/data-volume per version with deltas.
- **Backfilled v1 & v2 from git.** Worktrees at c2cd394 (v1) / a42466e (v2), regenerated each
  version's *features* with its own code, then ran the current instrumented harness against them
  (isolates the version's feature-set effect under consistent instrumentation). **Finding:** RB
  v1 ≡ v2 (the offseason block shipped inside the v1 commit; v1→v2 was the WR generalization +
  harness fixes, no RB feature change); v3's single `vacated_workload` feature (80→81 cols, flat
  compute) lifts head-to-head Spearman 0.706→0.711 / P@12 0.55→0.58 and narrows the market gap
  −0.026→−0.021.
- Tests `test_progress.py` (`_version_key`, `_best_market_row`) + `test_git_provenance_format` —
  full suite **80/80 pass**, ruff clean on changed files (notebook lint errors pre-date this work).

---

## Entry 022 — QB v1 (third position; passing feature family)

**Date:** 2026-06-17

**Prompt (full text):**

> commit and push so I can apply the PR [then] start the QB work from the fresh primary

**Response notes:**
- Merged RB v3 (PR #3) into primary; branched `qb-model` off fresh primary. QB was a bigger lift
  than WR: the whole feature pipeline was rushing/receiving-only. The **target needed no work** —
  nflverse `fantasy_points_ppr` already scores passing (4-pt pass TDs), so QB PPG flows through the
  existing build.
- **Data layer:** added `ngs_passing` to `fetch.py` (CPOE/time-to-throw/aggressiveness, 2016+);
  added passing box-score columns to `SEASON_SUM_COLS` (completions, attempts, passing_yards/tds,
  interceptions, sacks, passing_air_yards, passing_first_downs, passing_epa); generalized
  `ngs_season()` with a `passing` branch.
- **Passing feature family** (`features/build.py`), dispatched on `position=="QB"`:
  `add_passing_production`, `add_passing_volume` (incl. dropbacks=att+sacks), `add_passing_efficiency`
  (cmp%, Y/A, TD/INT/sack rate, EPA/dropback), `add_qb_rushing` (mobile-QB rushing — kept), and
  `add_ngs_passing`. Shared blocks (player_attrs, availability, snap_usage, trajectory,
  regression_mean, offseason) reused as-is (all column-defensive). No `team_context` for QB (a
  starter ≈ the team's pass game). Offseason competition currency = **pass attempts**.
- **Config / eligibility:** `config/football_qb.yaml` (tiers `[6,12,24]` = QB1-elite/QB1/QB2; eras
  map the NGS era to `ngs_passing`). Stage-5 reliability **never clears 0.70, peaks 0.614 @ k=7**
  (QB PPG noisier than RB g\*=4 / WR g\*=7 — TD & game-script variance) → **g\*=7**.
- **Results (QB v1):** best model random_forest **Spearman 0.697** (full universe) > baseline 0.635;
  MAE 4.18 PPG. vs market: trails on overall rank (0.629 vs 0.695 covered) and on the elite **QB1
  tier (P@6 0.50 vs 0.53)**, but **edges market at the QB2/streamer tier (P@12 0.67 vs 0.65)** —
  same shape as RB/WR (model competitive in the startable middle, market sharp at the very top).
- **Genericization fixes** (QB exposed report hardcoding): `report.py` headline + benchmark
  sections and `progress.py` now derive precision tiers from the configured `precision_at_<k>`
  columns and label them by position (QB1/QB2, RB1/RB2 …) instead of hardcoded `precision_at_12/24`
  + "RB1". Verified identical RB/WR output (no snapshot churn).
- Tests: 7 QB feature-block tests in `test_features.py` — full suite **87/87 pass**, ruff clean.

---

## Entry 023 — Air-yards receiving features (back-apply from QB) → REJECTED

**Date:** 2026-06-17

**Prompt (full text):**

> Let's start the next step from the fresh primary [then] also add the air-yards piece for RBs
> since we're doing ppr scoring

**Response notes:**
- **Back-apply step after QB** ([[iterative-backapply-process]]): QB v1 leaned on air-yards, and air
  yards are a canonical *receiving* opportunity signal that RB/WR didn't use. Hypothesis: add
  receiving air-yards depth/share + WOPR to the skill positions (position-agnostic, RB **and** WR
  per the PPR point — pass-catching backs). Built on branch `air-yards-receiving`.
- Added `receiving_air_yards` to `SEASON_SUM_COLS`, `team_air_yards` to `team_season_context`, and
  `add_air_yards` (RB/WR): `adot` (depth of target), `air_yards_pg`, `air_yards_share`, `racr`
  (air-yards conversion), `wopr` (=1.5·target_share + 0.7·air_yards_share). Wired into RB+WR eras
  (1999+, all eras), bumped RB v3→v4 / WR v1→v2 for the probe.
- **Measured vs baselines — REJECTED (no gain):**
  - **WR v1→v2:** every model within ±0.001 Spearman, weighted_tau identical (0.712), P@12 flat
    (0.52), P@24 flat-to-down. Zero incremental signal.
  - **RB v3→v4:** linear flat; **trees slightly hurt** (xgboost Spearman 0.711→0.701, P@12
    0.58→0.57).
  - **Why:** WOPR is by construction a linear combo of `target_share` + `air_yards_share`, and
    `target_share`/`targets` are already features — air yards add ~no information beyond existing
    target volume; the redundant columns nick the trees. Same lesson as the rejected
    receiving-NGS-for-RB and top-weighted-training probes: keep only what measurably helps.
- **Resolution:** reverted configs (back to RB v3 / WR v1) + the `build_features` dispatch + report
  artifacts — **no model change**. Kept `add_air_yards` + data plumbing **dormant** (docstring notes
  it's out of the default pipeline) + a unit test, so the capability is preserved and re-enabling is
  a 2-line change (dispatch + era block). Net branch diff = dormant function + `SEASON_SUM_COLS` /
  `team_air_yards` plumbing + test + docs.
- Tests: `test_add_air_yards_depth_share_and_wopr` — full suite **88/88 pass**, ruff clean.

---

## Entry 024 — Exclude the COVID-2020 season (RB v4 / WR v2 / QB v2)

**Date:** 2026-06-17

> (Sequenced after the air-yards rejection, Entry 023, on its own branch.)

**Prompt (full text):**

> The 2020 covid season plays a big role in messing with the data. [...] I want to take that season
> out of the analysis in some way. What would you recommend? [...later, after I quantified that
> removing every 2020 *player* empties the 2021 fold and biases survivors toward inexperienced
> players] Sure, that makes sense about the survivorship bias. Move forward with the third option.

**Response notes:**
- Recommended (and the user picked) excluding 2020 **as a season in both roles** — label *and*
  feature season — while **keeping the full player population** (rejecting the more extreme
  "remove every player who played in 2020," which I showed empties the 2021 test fold and biases
  survivors toward players who hadn't debuted by 2020 → survivorship bias). The subtle win: also
  threading the exclusion through the multi-year feature windows so 2020 doesn't leak into
  neighbouring years' rolling/career/trajectory features.
- **Implementation** (`data/build.apply_season_exclusion`, config `data.exclude_seasons: [2020]`):
  (1) rows whose **label** season is 2020 (i.e. 2019 rows) keep their place as history but have
  `_next` labels nulled + `status_next=excluded_season` (drop out of supervised use); (2) rows
  **in** 2020 are dropped entirely, so the season never predicts and — being absent — is naturally
  hopped over by per-player `shift`/`rolling`/`expanding` (verified: a back's `career_touches` goes
  2019→2021, `ppg_delta1` at 2021 is measured vs 2019). Calendar-merge target made this clean.
  Unavoidable cost (flagged): the **2021 label fold** is also lost (predicting it needs 2020
  inputs). Test block → the 5 most-recent **clean** label seasons `[2018,2019,2022,2023,2024]`.
- Applied to all three positions (RB v3→v4, WR v1→v2, QB v1→v2). **g\* is robust** — re-derived
  eligibility unchanged (RB 4, WR 7, QB 7); RB reliability is actually a touch higher without 2020.
- **Key analytical finding — the clean eval is *less* flattering, and that's the point.** Excluding
  2020 **widens** the model's gap to the market on overall rank (RB Δ −0.021→−0.059, WR
  −0.019→−0.046, QB −0.066→−0.110): 2020 was the fold where the *market itself* failed (preseason
  ECR Spearman ~0.49 vs ~0.73–0.82 elsewhere), so it had been propping up the model's *relative*
  standing. On non-COVID data the market is a clearly stronger overall-rank predictor than the
  contaminated numbers implied — **but the model's top-tier edge holds or improves** (WR P@12
  0.52→0.56 & beats market on P@24; QB beats market P@12 0.67 vs 0.61; RB still beats P@12).
- Because v-latest is scored on **different folds** than v1–v3, added a **fold-change guard** to
  `progress.py`: when a version's `test_label_seasons` differ from the prior version's, the
  "Δ vs prev" is replaced with "eval set changed †" + a footnote, so the Spearman jump isn't
  misread as a model gain. Compare against the (re-scored) market column instead.
- Tests: `test_apply_season_exclusion_*` in `test_build_dataset.py` — full suite **89/89 pass**,
  ruff clean.

---

## Entry 025 — Make the models usable: keeper-league assistant

**Date:** 2026-06-18

**Prompt (full text):**

> Now that we've finalized our models, let's make them usable. First use case: keeper leagues. I
> should be able to run a script that takes in a list of players and what pick they were drafted at
> and get as output the predicted rank for next season and a prioritized keeper list

**Response notes:**
- First **inference/serving** path (everything prior was walk-forward eval). Two pieces:
  - **`eval/projection.py`** — `project_position` fits the configured era ensemble on all labeled
    history and predicts the latest, **censored** feature season (2024→2025 live board): projected
    PPR PPG + within-position rank. Default model `ridge` (stable, strong top-of-board precision),
    swappable via `projection.model`. Sanity-checked: top RBs = Saquon/Gibbs/Bijan, WRs =
    Chase/St. Brown/Jefferson, QBs = Daniels/Lamar/Allen — a credible 2025 board. `scripts/project.py`
    + `make project`.
  - **`eval/keeper.py` + `scripts/keeper.py`** — turns per-position projections into one
    cross-position **VORP** board (value over replacement), then ranks the user's keepers by
    **surplus = pick paid − projected board slot**. Replacement levels are league-configurable per
    the user's spec: **format ∈ {1qb, sf, 2qb}** (QB started-slots/team 1.0/1.7/2.0) × **8–16 teams**,
    flex filled from the best remaining RB/WR. Name resolution is punctuation/suffix-insensitive with
    a conservative fuzzy fallback; **TE/K/DST and unmatched names are reported as *unscored*** (TE was
    never modeled).
- **Design choices** (confirmed with the user): VORP-surplus prioritization (not raw PPG — handles
  positional scarcity); exact-drafted-pick cost; league format a **runtime setting**. Kept it
  **model-only** — ECR is not used in the keeper math (stays a benchmark). Verified the format toggle
  works: superflex makes QBs the top keepers (Daniels QB1, +89 at pick 95); switching to 10-team 1QB
  raises QB replacement (15.3→17.4) and drops Daniels' board slot 6→11 while RB/WR rise.
- Scope note surfaced: the cross-position board covers **RB/WR/QB only**, so projected pick slots are
  mildly compressed (missing ~1–2 TEs/round); documented, not hidden.
- Tests: `tests/test_keeper.py` (replacement-level flex allocation, format depth, VORP/board,
  name resolution, surplus+sort) — full suite **96/96 pass**, ruff clean. Example input at
  `examples/keepers_example.csv`; README usage section added.

---

## Entry 026 — Redraft-league assistant (new-season draft board)

**Date:** 2026-06-22

> _Original prompt not captured at the time; reconstructed from PR #8 (`7be2237`)._ Build the second
> serving use case after the keeper tool: a workflow that checks the just-completed season is
> published, refreshes only what's stale, and projects the coming season into a draft board.

**Response notes:**
- Second **serving** path. One workflow: (a) gate on nflverse having published the completed season,
  (b) refresh only stale caches, (c) project the coming season into a top board (default 20 QB /
  50 RB / 75 WR).
- **`data/availability.py`** — `check_season_available` (per-dataset network probe with row-count
  floors so a *partial* release still fails the gate) and `datasets_needing_refresh` (manifest-driven
  stale detection; always-refresh `ids`/`sleeper_players`/`draft_picks`). Both reused later by the
  postseason tool.
- **`eval/redraft.py`** — `run_redraft` orchestrates gate → fetch → per-position
  build/features/`project_position` → combined board. `latest_completed_season` overridden at runtime
  via a **deepcopied Config** so committed YAMLs are untouched. Top-N is "returning players *within*
  top N" — projections stay pure (no ECR blend, per [[no-blend-ecr-benchmark]]); the list is trimmed
  by `estimate_rookie_count` (draft class inside the latest `db_fpecr` board, graceful fallback to 0).
- `scripts/redraft.py` + `make redraft [SEASON=]`; gate failure is a hard exit 1; writes
  `reports/redraft_<season>.csv` (gitignored). Tests: `test_availability.py` + `test_redraft.py`
  (no network/model). **107 pass.**

---

## Entry 027 — Postseason report (grade model / ECR / ADP vs actual finish)

**Date:** 2026-06-22

> _Original prompt not captured; reconstructed from PR #9 (`fa939dc`)._ Third (final football) serving
> use case: after a season completes, grade the model's preseason projection against ECR, ADP, and the
> actual finish.

**Response notes:**
- For a completed season Y: per player, the model's **preseason** projection for Y, the **ECR** and
  **ADP** going into Y, the **actual** Y finish, plus a summary of how well each source called it.
- **Leak-safety** is the crux: `eval/projection.py` `project_position` gains a `feature_season` param —
  it boards that season and trains only on labels `season < feature_season`, so a backtest never
  trains on the answer. No-op in live mode (latest season censored), so keeper/redraft/project callers
  are unchanged. The invariant is pinned by a test.
- **`data/adp.py` (new)** — historical PPR ADP from the **FantasyFootballCalculator** API (`db_fpecr`
  carries no ADP). `fetch_ffc_adp` + `build_adp_benchmark` map FFC names → `gsis_id` by normalized
  name+position (reusing `keeper._norm`), dense-rank within position, cache to
  `data/external/adp_<stem>.parquet`. _(This FFC dependency is exactly what bites in Entry 030.)_
- **`eval/postseason.py`** — `build_postseason_report`: gate (reused from redraft) → optional refresh →
  per-position build/features/preseason projection → outer-join model/ECR/ADP/actual; scores each
  source vs actual (Spearman, Precision@tier, mean abs rank error) on its covered-and-eligible rows,
  plus hits/busts/value/reach highlights. `scripts/postseason.py` + `make postseason [SEASON=]`.
- Tests: `test_adp.py` + `test_postseason.py` (incl. the leak-safe projection invariant). **115 pass.**

---

## Entry 029 — Migrate nflverse source: nfl_data_py → nflreadpy (data now 1999–2025)

**Date:** 2026-06-23

> _Original prompt not captured; reconstructed from PR #11 (`8c77d9f`, `f650829`, `551642a`,
> `f41ceb9`)._ `nfl_data_py` is deprecated and frozen at 2024; migrate to the maintained `nflreadpy`
> loader and bring the data current.

**Response notes:**
- `nfl_data_py` is deprecated upstream and frozen at the `player_stats` release (stops at 2024).
  Switched the fetch registry to **`nflreadpy`** (nflverse's maintained loader, serving
  `stats_player`).
- **`data/fetch.py`** — mapped every loader to its nflreadpy equivalent. nflreadpy returns **polars**
  frames; each loader converted to pandas and renamed `stats_player` columns back to the canonical
  schema (`passing_interceptions`→`interceptions`, `sacks_suffered`→`sacks`, `team`→`recent_team`,
  roster `gsis_id`→`player_id`), so the migration is contained to this module and the cached schema is
  unchanged. _(The pandas conversion here is what Entry 030 later removes.)_
- Re-pulled all 11 datasets (manifests now `source=nflreadpy`) and **added 2025**. Schema-scope review
  confirmed the swap is neutral: every column the build/features/snaps code consumes survives, the
  reg-season filter and NGS `week==0` conventions hold, and a full RB rebuild reproduces v4 exactly
  (4464 rows, 3064 labeled). Metric shifts vs prior reports are driven by **+2025 / dropped-2018 fold**,
  not the source.
- Added `test_fetch_loaders.py` (fakes nflreadpy, asserts loader wiring + renames, no network).
- `f650829`: regenerated WR/QB v2 reports on the new cache (WR Spearman 0.73→0.80, QB 0.49→0.57 — both
  +2025-driven; both still beat market). `551642a`: refreshed PROJECT_PLAN/data_dictionary roadmap
  (pbp marked out of scope). `f41ceb9`: a subagent suite review trimmed 3 weak tests (net −2, no
  coverage lost) → **119 pass.**

---

## Entry 030 — Cleanup session: crash post-mortem, polars fetch, ADP fallback

**Date:** 2026-06-23

**Prompt (full text):** _(verbatim, sequential prompts from this session)_

> The terminal window shut down while you were running the last process. Let's figure out what
> happened. […] I asked you to fix the football readme. it references the 2024 postseason report, and
> it should reference the 2025 one instead. when running the report, you and the terminal window
> crashed. let's try to figure this out. […] commit the README fix. We'll look into the ADP issue
> later, it might just be an update they need to push. […] Is there any way to reduce the memory usage
> and reduce chance of OOM errors in future runs? […] Skip the pandas copy and see where else in the
> codebase pandas can be skipped in favor of other packages. Sometimes it has high memory overhead.
> […] Rethinking the FP thing. If our ECR is coming from FP via nflverse, how are we sourcing ECR for
> 2025 season? […] yes, check what ADP files DynastyProcess publishes […] check if Sleeper exposes a
> preseason ADP endpoint […] verify FFC 2025 is genuinely gone or just unpublished […] implement the
> FFC→FantasyPros fallback and rerun the 2025 postseason report

**Response notes:**
- **Crash post-mortem.** The terminal died mid-run. Diagnosis: the *report generation* had actually
  completed (all 5 RB-v4 report files written in a 1-second batch); the regenerated diff was pure
  timing/branch-stamp noise (identical model metrics), so it was reverted. The real crash trigger was
  separately found: `make postseason SEASON=2025` default-refreshes via `fetch_all(1999…2025,
  overwrite=True)` — a full re-download whose peak memory plausibly OOM'd the 15 GiB / 512 MiB-swap
  box. Running `--no-refresh` (cache already current) completes cleanly.
- **README/USAGE 2025 fix** (`9495b35`, + USAGE.md follow-up): postseason example `SEASON=2024`→`2025`.
- **Memory: keep the nflverse fetch in polars** (`014dba2`). Loaders no longer materialize a pandas
  copy (undoing the conversion added in Entry 029); `write_parquet` is now polymorphic (polars writes
  natively, strictly typed → the mixed-object coercion path is pandas-only). Cache + all downstream
  pandas readers unchanged. Removes the ~2× peak on wide multi-season pulls. Pandas-skip survey: only
  bulk paths matter — `build.py`/`features/build.py` flagged as the next tier (deferred); eval/serving
  paths handle small tables, left as-is.
- **ADP source investigation** (the bulk of the session). FFC returned no 2025 ADP. Walked the option
  tree and ruled out dead ends with evidence: ECR actually comes from the **DynastyProcess** archive
  `db_fpecr.parquet` (not nflverse — nflverse only supplies the `fantasypros_id`→`gsis_id` crosswalk),
  pulled live; **DynastyProcess publishes no ADP file**; **Sleeper exposes no aggregate-ADP endpoint**
  (per-draft picks only). Verified FFC 2025 is a **permanent hole** (`status:Error, "No ADP data
  found."` across all formats/sizes, while 2023/2024 full and 2026 already live), and FantasyPros
  **does** have 2025 (~336 players, consensus AVG).
- **Implemented FFC→FantasyPros fallback** (`f57d036`): `fetch_fantasypros_adp` parses the FP overall
  board with **stdlib regex** (no lxml/bs4 dep — none installed); `build_adp_benchmark` tries FFC then
  falls back, records `source` in the match dict; report labels the source. 2025 ADP now grades:
  **QB 40/40, RB 85/92, WR 106/110** matched. Merged as PR #12. **121 pass.**

---

## Entry 033 — Add the TE model (v1) + wire TE through the serving tooling

**Date:** 2026-06-26

**Prompt (full text):** _(verbatim, sequential prompts from this session)_

> Is git fresh and up to date? Let's start the TE model. I think we need it to make the tooling layer
> more complete.
>
> [clarifying answers] P@k board depth = **TE1/TE2 [12, 24]** (drop the fantasy-irrelevant TE3 tier);
> scope = **returning-only** (consistent with RB/WR/QB v1).
>
> [follow-up] add a PROMPT_LOG entry for this, this piece should be automatic. It's the whole point of
> the project.

**Response notes:**
- **Git hygiene first.** Was on the already-merged `docs-prompt-log-backfill` branch with
  `origin/primary` 4 commits ahead and regeneration churn in the tree (an RB v4 re-run with *identical*
  model metrics — only timestamps/hashes/wall-clock differed). Discarded the churn, fast-forwarded
  `primary` to `20ce892`; left the unrelated untracked `data/raw/_manifests/pbp.json` alone. Branched
  `te-model-v1`.
- **TE v1 was a pure-config add** — TEs are pass-catchers, so they reuse WR's receiving feature
  pipeline with **zero** build/feature code. New `config/football_te.yaml`: workload = `targets`, P@k
  tiers `[12,24]` (TE1/TE2), returning-only, same era/CV/model grid as WR.
- **Eligibility g\*=5.** TE split-half PPG reliability **clears 0.70 at k=5** (0.713) — TEs are *more*
  stable than WR (never clears, g\*=7) or QB (g\*=7) once they play 5+ games. (Validates the "add a
  position = new config + re-derive eligibility" claim for any position in an existing scoring family.)
- **Results.** Best model `lasso` (a *linear* model wins) — Spearman **0.744 ± 0.069**, P@12 0.57,
  MAE 1.83 PPG. Does **not** beat preseason ECR overall (0.782) but **edges it on the TE2 tier**; NGS
  block dropped (no out-of-fold gain — consistent with RB/WR).
- **Tooling completeness (the stated goal).** Wired TE through every multi-position consumer:
  **redraft** (`DEFAULT_TOP_N["TE"]=24`, `_DRAFT_POS_TO_FANTASY`, `rookie_names`, `--top-te`),
  **postseason** grading (`DEFAULT_CONFIGS`), and **keeper** (dedicated **1-TE slot**, *not*
  flex-eligible — `MODELED_POS`/`DEFAULT_ROSTER`/`starters`). Added
  `test_te_dedicated_slot_replacement`; updated README + PROJECT_PLAN (removed the stale "TE skipped"
  notes). **122 tests pass, ruff clean.**
- **Committed** `2ed3819` on `te-model-v1` (`pbp.json` deliberately excluded); pushed, PR pending.
- **This entry is the follow-up's point:** prompt-logging is the project's reproducibility backbone and
  must be **automatic** — appended here without being asked, going forward, for every prompt.

---

## Entry 035 — Handcuff-selection tool (RB backups to draft by starter injury risk)

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> Code merged. I have a new tool to add to the football section. Handcuff Selection. Using the injury
> model we should be able to predict who will miss games due to injury and using that whose backups to
> draft
>
> [clarifying answers] Risk signal = **compute both (availability model + durability baselines) and
> pick the backtest winner**; scope = **RB only**; output = **global handcuff board**.

**Response notes:**
- **Serving use-case #5.** `eval/handcuff.py` + `scripts/handcuff.py` + `make handcuff [SEASON=]` →
  `reports/handcuff_<season>.{md,csv}`. RB-only; model-only (ECR/ADP stay benchmarks, never inputs).
- **Definition.** No depth-chart dataset is fetched, so the **model defines the depth chart**: per
  team, starter = top projected RB, handcuff = next. Score = **contingent upside** = (starter − backup
  projected PPG) × the starter's projected **miss share**. Ranked by contingent upside.
- **Risk signal chosen by a leak-safe backtest** (the user's "pick best") among the §7.4 availability
  model + transparent durability baselines (prior-games, 3yr-availability index), by clears-cutoff AUC.
- **Empirical reversal of the earlier read.** I had flagged that the availability GBM loses to a naive
  prior-games baseline (true in the *windowed* experiment folds). With **full expanding-history
  training** it actually **wins** (AUC 0.813 vs 0.784, MAE 4.31 vs 4.52) — *after* a bug fix:
- **Bug found + fixed.** The `offseason` block's `_next` features (room competition / vacated workload)
  are **all-zero for the live board's N+1 horizon**, which collapsed the Poisson prediction to ~0.1
  games for *every* starter (risk wasn't differentiating anyone). Excluded the `offseason` block from
  the availability feature set — it's opportunity context, not durability, so the fix is also correct
  in principle. Risk now varies sensibly (CMC 5.4 / Kamara 9.8 / Bijan 3.4 expected games missed).
- **2026 live board sanity:** top handcuffs Brian Robinson (CMC), Tyler Allgeier (Bijan), Jaylen Wright
  (Achane); Kamara's age pushes his backup up the board. README / PROJECT_PLAN / Makefile updated.
  **129 tests pass, ruff clean.** Branch `handcuff-tool`, committed + pushed, PR pending.

---

## Entry 036 — Offseason-bug follow-ups: availability-model fix, degenerate-board warning, log fix

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> Do both and fix up the prompt log issue mentioned above

_(Context: "both" = (a) warn when the offseason block is degenerate on a live board, (b) drop the
offseason block from the experiment's availability model; "the prompt log issue" = the root log's
template had lost its `<!--` comment wrapper. All following an investigation of "does the bug affect
the results of the other models?")_

**Response notes:**
- **First answered "does it affect the other models?" by measurement.** Experiment/headline metrics:
  **unaffected** (they train/score on historical boards where the offseason `_next` features are
  populated). Live PPG projections (project/keeper/redraft + handcuff's own projection):
  **negligible** — projection model is `ridge` (linear), so zeroed features ≈ neutral (Spearman 0.998
  zeroed-vs-dropped, 49/50 top-50 unchanged). The tree-based availability model was the **only** one
  materially hit. Root cause = a **data gap**: `rosters` is cached only through 2025, and the offseason
  `_next` features need the N+1 (2026) roster join.
- **(a) Degenerate-board warning.** New `offseason_degenerate(df, block_columns, season)` helper —
  flags when **every** offseason column is constant across players (all-zero *or* sentinel defaults
  like draft-capital 300 / room-size 1; the "all-zero" check alone missed those). Wired as a build-time
  warning (`build_features`) + a serving-time `warnings.warn` in `project_position` (covers
  project/keeper/redraft/handcuff). Verified: fires on the live 2025 board, silent on healthy 2024.
- **(b) Dropped offseason from the experiment's availability model** (`_availability_over_folds`), to
  match the handcuff risk model. **Honest result:** in the *windowed* experiment the gbm still trails
  the prior-games baseline — the handcuff "win" came from expanding full-history training, not the
  offseason exclusion. But clears-AUC improved for 3/4 positions (RB 0.736→0.778, WR 0.798→0.814,
  TE 0.775→0.798; QB 0.887→0.881, a slight dip but already the best). Updated only the `gbm_poisson`
  line in the 4 **live** reports (surgical, no timing churn; version snapshots left immutable).
- **(prompt-log fix)** Restored the root log's missing `## Entry 087 — serving brief for the home-site dashboard (football models behind a web UI)

**Date:** 2026-08-21

**Prompt (full text):**

> [cross-session message from the Claude session working in `/home/weezy/activity/home-site`,
> the LAN home dashboard (FastAPI + Jinja2 + htmx, no Node/build step), in plan mode:]
>
> Rohan asked me to work with you: he wants the first set of tools on the dashboard to be webapps
> wrapping the successful football models you two built in `/home/weezy/activity/position_predictor`.
> ... (1) which models are actually "successful" enough to put behind a UI and which to skip;
> (2) for each: a one-line plain-English description, the exact Python entry point (module +
> function + repo-relative file), inputs with names/types/units/ranges/defaults and the full value
> list for any categorical, the exact output shape with per-field meaning and what to format as a
> percentage/probability/rank, and latency + whether it needs a warm model in memory;
> (3) where trained model files live and whether inference needs a dataset on disk or the network;
> (4) what position_predictor depends on and whether it's pip-installable;
> (5) whether an existing CLI already wraps predictions.
> Home-site apps are small FastAPI routers rendering a form and a result card — fast, synchronous,
> single prediction per submit. Anything >2s per call it would handle differently.

**Response notes:**
- Read-only reconnaissance plus **measured** timings (nothing was answered from the docs alone).
  Deliverable: `/tmp/claude-1000/shared/position_predictor_serving_brief.md` (309 lines) + a
  condensed reply. No source changes; one temporary `football_te_standard` artifact build was
  timed and deleted, tree left clean.
- **The structural fact that drives their design: there are no serialized models.** No
  `.pkl`/`.joblib` anywhere in the repo — every serving tool refits the era ensemble at call time
  off `data/processed/*_features.parquet`. That reads as fatal for a request handler and isn't:
  cold process 1.9 s, **warm single position 0.17 s**, all four 0.62 s. So the caching unit is the
  *process*, not a model file — import at FastAPI startup and stay warm.
- Ranked the four positions honestly against their own reports (walk-forward Spearman vs the
  must-beat baseline): RB 0.819/0.803, WR 0.803/0.782, TE 0.744/0.726, QB 0.572/0.529. All four
  clear baseline; none beats ECR on overall rank. Told them to ship all four and *label* QB rather
  than hide it, and restated the no-blend rule (ECR/ADP shown separately, never an input).
- **Found a live defect while enumerating the inputs.** `project_position` falls back to
  `config.get("projection.model", "ridge")` and **no config file sets `projection.model`** — so
  every board projects with ridge. Harmless at RB/WR/QB (−0.004 to −0.015 Spearman), but at TE
  ridge is 0.654 against lasso's 0.744: a 0.09 gap, the largest ridge-vs-best spread of any
  position. Handed them the explicit per-position map
  (`QB=lasso, RB=random_forest, WR=elasticnet, TE=lasso`) so the web layer passes `model=`.
  *Worth fixing at the source too — adding `projection.model` to the four YAMLs would stop the
  next caller inheriting the same silent TE downgrade.*
- Triaged what **not** to build, which was as useful as the recommendations: `qb_breakout`
  (shelved, and its headline is negative — a UI would imply a signal the study says isn't there),
  `bestball_calibrate` (null result, λ=0 everywhere), `medstaff` (complete, but a research
  pipeline with no per-request predict function), `redraft`/`postseason` (network + season-gated →
  cron the CSV, don't call it from a handler).
- Latency triage against their 2 s budget: projection + keeper are 0.17–0.65 s (**synchronous, no
  network**); handcuff RB is 4.9 s and QB injury-risk 2.8 s, because each walk-forward-backtests
  three risk signals before boarding. Both depend only on (config, season, signal, scoring) — all
  static until the cache refreshes — so they precompute at startup rather than needing a job queue.
- Inference surface is small and local: ~9 MB of feature parquet + `_feature_blocks.json` + the
  config YAMLs. The 341 MB `data/raw/` is **not** needed at predict time. Flagged that `data/` is
  git-ignored, so those files exist on this machine only — fine for a LAN box, not for a clone.
- Passed on the freshness caveat as a UI banner rather than a footnote: the 2025 `offseason` block
  is all-zero (next season's roster join un-refreshed), so `project_position` raises a
  `UserWarning` and the live 2026 board lacks offseason signal. Better surfaced than swallowed.
- Environment: the root `pyproject.toml` *is* pip-installable and does expose `position_predictor`,
  but it drags in the basketball stack (`nba_api`, `sportsdataverse`, `fantraxapi`,
  `yahoo_fantasy_api`) and `nflreadpy`, none of which the serving path imports. Recommended the
  `sys.path` insert over installing or subprocessing — the scripts write CSVs into `reports/` as a
  side effect, which a web request must not do.
- **Noted for us:** the football README headline still claims "Spearman ≈ 0.74–0.75 (RB/WR)",
  which the current reports (0.819 / 0.803) have overtaken. Told them the reports are the source of
  truth; the README line should be corrected on the next football touch.

---

## Entry 088 — fix `_norm`, then serve the best model per position on next season's rosters

**Date:** 2026-08-21

**Prompt (full text):**

> fix _norm first, then regenerate the artifacts from the best models. it looks like the handcuff
> report is still using old rosters.

**Response notes:**
- Two commits: `3cae5ef` (name matching) then `b84db02` (models + rosters + regeneration), in the
  order asked. The roster observation was correct and turned out to have **two independent
  causes**, one data and one code — fixing either alone would have left the board wrong.
- **`_norm` needed two fixes, not the one that was queued.** The single-letter collapse handles
  `AJ Brown` ≡ `A.J. Brown`, but it does nothing for `V Jones` — there is no *run* of initials to
  fold. That case needed the suffix strip **anchored to the end**: unanchored, the `v` alternative
  in `\b(jr|sr|ii|iii|iv|v)\b` ate a standalone "v" anywhere in the name, so `V Jones` normalised
  to the bare surname `jones` and then fuzzy-matched any Jones on the board. Both landed;
  `Sammy Watkins IV` still strips correctly and `Robert V Smith` is now left intact.
- Why fix rather than tighten the cutoff: `get_close_matches(n=1)` returns the *best* candidate
  above 0.88, not a safe one, so an initialled name at ~0.94 was competing on ratio against the
  whole 616-row board. Raising the cutoff would have traded a mis-resolve for a miss. Collapsing
  removes those names from fuzzy contention entirely.
- **Jr./Sr. left deliberately colliding**, with a test that says so. Dropping the suffix is exactly
  what lets a typed "Brian Thomas" reach "Brian Thomas Jr."; an asymmetric rule would guard against
  nothing in a pool with no father/son pair and cost every future reader a rule to carry.
- Verified the whole change is a **no-op on real data**: zero `_norm` collisions across the board
  before or after, and the eight-row `keepers_example` still reproduces the shipped keeper board to
  the number. That is the point — these fail by resolving to a *confident wrong row*, not by
  erroring, so "nothing changed" is the expected result and the tests are the guard.
- **`projection.model` was never set in any config**, so `project_position` fell back to `"ridge"`
  and every shipped board — `project.py`, `keeper.py`, `redraft.py`, `handcuff.py` — was a ridge
  board regardless of what the experiment picked. Cheap at RB/WR/QB (−0.004 to −0.015 Spearman);
  **TE ridge is 0.654 against lasso's 0.744**. Fixed in the four configs rather than at the
  callers, which is what makes it reach every consumer including the home-site dashboard that
  reads the CSVs and cannot pass `model=` at all.
- **Nothing in the pipeline had ever fetched next season's rosters.** `add_offseason` reads the
  season *N+1* preseason roster, but `datasets_needing_refresh` only asks whether a dataset covers
  `feature_season` (N) — rosters cached through N satisfied that and were skipped, `always_refresh`
  covered `draft_picks` but not `rosters`. So the `team_next` join produced all-NaN and every
  offseason feature collapsed to a constant. **A degenerate block, not a stale one**: no amount of
  tightening an N-based staleness rule would ever have caught it. nflverse *had* published 2026
  rosters (2930 rows) — we simply never asked. Post-fetch the block is live: 85–91% of the board
  carries a next-season team, 31 of 157 RBs changed clubs.
- **The handcuff board grouped by `recent_team`**, so even with N+1 rosters cached it paired a
  starter with *last* season's backup. `team_next` now survives `add_offseason` as context — never
  a feature, since the returned block list is the only thing the model sees — and the board prefers
  it with a `recent_team` fallback for players with no N+1 roster row (unsigned FAs).
- **Impact, measured by rebuilding the board both ways: 9 of 26 starters had the wrong handcuff.**
  CMC → Brian Robinson becomes Isaac Guerendo; Bijan → Tyler Allgeier becomes Brian Robinson;
  Travis Etienne → Bhayshul Tuten becomes Alvin Kamara; Kenneth Walker III → Zach Charbonnet
  becomes Kareem Hunt. Three teams appear that were absent (HOU/NO/WAS), two drop off (CAR/JAX).
- **No experiment re-run, and the committed reports still stand.** The offseason block was
  degenerate only for the *latest* feature season — the live board — because every test fold's N+1
  rosters were already cached by the time that fold's features were built. So the Spearman numbers
  in `REPORT_football_*.md` are unaffected; only the serving artifacts were regenerated.
- Updated `test_datasets_needing_refresh`, which had encoded the old assumption (rosters current
  through N ⇒ not stale). Replaced with a test named for the trap and explaining why an N-based
  rule cannot express an N+1 dependency.
- Credit where due: the initials gap was found by the home-site dashboard session porting this
  matcher and measuring it, and the Jr./Sr. edge was entirely its find.
- 552 tests pass (+7 across the two commits), ruff clean.

---

<!-- Template for new entries:` opener (the
  `-->` was orphaned, so the template was rendering as a literal entry).
- Added `test_offseason_degenerate`. **130 tests pass, ruff clean.** Branch `handcuff-tool` (same PR).

---

## Entry 037 — QB handcuff mode: injury-risk list (+ starter-training and outlier findings)

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim, sequential)_

> For the QB handcuff tool, it's sufficient to generate a list of QBs who are likely to get injured and
> miss multiple games. That way someone who drafts them knows to draft a backup
>
> Can the model be refined by focusing only on QBs projected to be starters?
>
> Josh Allen could also be an outlier. He's elite for a reason.

**Response notes:**
- **QB mode = injury-risk list, not a backup board.** A QB backup rarely inherits standalone value, so
  the QB deliverable is just projected starters ranked by injury/availability risk, flagging who to
  handcuff. `eval/handcuff.py`: refactored shared setup into `_board_with_risk`, added
  `build_injury_risk_list` / `run_injury_risk` / `render_injury_markdown`; `scripts/handcuff.py`
  dispatches by position (RB → contingent-upside board; else → list). `make handcuff
  CONFIG=config/football_qb.yaml` → `reports/handcuff_qb_<season>.{md,csv}`. The QB availability model
  is genuinely strong (clears-AUC **0.895**, beats both baselines).
- **"Refine by focusing only on projected starters?" — tested, answer is no for *training*.**
  Restricting the availability model's *training* rows to starters slightly *lowers* AUC (0.848→0.833
  g≥10, 0.808 g≥7 — less data hurts) and doesn't fix the conservatism (durable QBs still ~14 predicted
  vs ~17 actual). The backup rows help the model learn. Correct version of the idea = filter the
  *scoring* population to projected starters (which the list already does via `top_starters`), not the
  training population.
- **Absolute games are biased low** (the QB pool is backup-heavy → regression to a ~6–7 game mean), so
  the list uses **relative quartile tiers** (High = riskiest 25%), not the raw `exp_games_missed`. The
  markdown leads with tiers and omits the raw number.
- **"Josh Allen could be an outlier" — agreed, and handled.** The model reads rushing/workload as
  injury exposure, so it initially put Allen in a (tercile) High tier. Switched tiers to **quartiles**
  → Allen and Lamar correctly drop to Moderate, and the High set becomes all-defensible (Murray,
  Daniels, Rodgers, Dart, Burrow, Stroud). Added an explicit report caveat that durable high-usage QBs
  are outliers the model can over-flag — *not* hard-coding any player. Added 2 tests. **132 pass, ruff
  clean.** Branch `handcuff-tool`.

---

## Entry 055 — New football project: the late-breakout QB (cohort + labels)

**Date:** 2026-07-25

**Prompt (full text):**

> Let's start a new football project. We're in the era of the late-breakout QB. I want to see what
> we can do to identify them before they happen. Let's start by analyzing QBs that have done this
> and then look at player archetypes, drafting situations, and other factors. While their pro
> statistics should contribute to identifying the archetype, I want a limitation of this analysis
> to be that it focuses mainly on college and high school stats for any final modeling.

**Follow-up decisions (answers to clarifying questions):**

> - **Breakout definition:** Fantasy — first top-12 PPG season. *"Let's make it top 20 for superflex
>   relevance. Also QB15 tends to be good draft value"*
> - **Late threshold:** Both Y4+ and team-change flags (cross-tab the four cells)
> - **High-school data:** Recruiting profile + best-effort HS stats scrape. *"This is research, so
>   partial data is better than none."*

**Response notes:**
- **New project** `sports/football/src/qb_breakout/` (sibling to `position_predictor`, same tree,
  reuses its nflverse caches and `utils.io` paths). Branch `qb-late-breakout`.
- **Data-source recon before planning.** ESPN's recruiting API is open (no key) and returns HS
  name/city/state, national/position/state/region rank, ESPN grade, and camp measurables
  (40/3-cone/shuttle/vertical). College PBP 2003+ is free via cfbfastR-data; `sportsdataverse.cfb`
  is already a dependency. CFBD needs a free key (401 without). **Flagged up front:** no free bulk
  source for HS box scores exists — the HS layer is fundamentally a *recruiting profile*, with the
  scrape as partial-coverage augmentation.
- **Key finding — the user's stated definition doesn't capture the user's own examples.** Built
  "first top-20 PPG season" as asked, then checked it against the named archetypes: **Baker
  Mayfield's 2018 rookie season ranks exactly 20th** and Tannehill's 2014 ranks 10th, so both
  classify as *on-time*. Built two alternatives and scored all three against 11 unambiguous cases:
  - `late` (first top-20) — misses Mayfield, Tannehill
  - `late_qb1` (first top-12) — wrongly calls **Tom Brady** late (first top-12 in year 6)
  - **`late_sustained`** (top-20 in ≥2 of 3 seasons from the breakout) — correct on all 11
  Shipped `late_sustained` as the modelling target, other two as diagnostics. Peak-timing was also
  tested and rejected: it flags 51/93 QBs because nearly everyone peaks late (Brady, Burrow, Allen).
- **Three label bugs found and fixed while building:** (1) PFR vs nflverse team codes (GNB≠GB) made
  Rodgers/Cassel/Alex Smith read as having changed teams — added a franchise crosswalk that also
  collapses relocations (SD→LAC, STL→LA, OAK→LV); (2) entry season imputed from first *observed*
  season made Testaverde (1987 #1 pick) a 1999 rookie — now resolved draft → `rosters.entry_year`
  → first season, dropping the unresolvable; (3) no censoring — recent entrants were counted as
  failures.
- **Central result (§4 of the report):** through 3 NFL seasons, future late breakouts look far more
  like busts than like on-time breakouts (median PPG 5.97 vs 4.18 / 15.68; median early rank at the
  unranked floor for both late and never). **Early NFL production does not separate them** — which
  is the empirical justification for the user's pre-NFL-only constraint.
- **Shipped:** `labels/cohort.py` (3 label variants, entry resolution, two-way censoring),
  `labels/teams.py` (franchise crosswalk), `eda/describe.py` (draft-situation base rates, trough
  profile, timing distribution, pending watchlist), `scripts/qb_breakout_cohort.py`,
  `docs/QB_BREAKOUT_PLAN.md`, `reports/REPORT_qb_breakout_cohort.md`. 21 new tests, 211 pass,
  ruff clean.
- **Cohort:** 331 QBs entering 1999+; 69 ever broke out; **17 late breakouts**; 28 censored.
  Draft-capital base rate is the actionable one: R1 top-10 breaks out often but rarely *late*
  (10% of its breakouts), while R4-7 and undrafted are 50%/67% late-given-breakout.

---

## Entry 056 — QB breakout: top-15 tier (two-bar rule) + drop the high-school layer

**Date:** 2026-07-26

**Prompt (full text):**

> Top 20 now seems too big. What does it look like for top 15?

> Let's skip high school data since it's so sparse

**Response notes:**
- **Top-15 alone breaks the label — the tier and the sustain rule compound.** Swapping 20→15 while
  keeping "top-N in ≥2 of 3" *deleted* the archetypes: Mayfield, Geno Smith, Alex Smith, Jordan
  Love and Garoppolo all stopped being breakouts at all. Cause: Mayfield's good run is 17/4/19 and
  Geno's is 9/21/16 — one top-15 season each, so neither holds two in any 3-year window.
- **Fix — two bars for two claims,** which also reconciles the user's own two remarks (top-20 for
  superflex, "QB15 tends to be good draft value"): a **top-15 season triggers** the breakout
  (quality), **top-20 in ≥2 of 3 confirms** it held (relevance). Trigger high, confirm lower.
  Implemented as `rank_qb_seasons(breakout_rank=15, sustain_rank=20)` → `is_breakout` +
  `is_startable`; `first_sustained_breakout` triggers on the first, holds on the second (falls back
  to single-bar if `is_startable` is absent).
- **Scored 14 undisputed careers: 13 correct.** Baker Y7 (2024, rank 4 — his real breakout, was Y6),
  Tannehill Y8, Geno Y10, Alex Smith Y9 (was Y5), Rodgers/Cousins/Love late; Brady/Allen/Burrow/
  Purdy/Foles on-time. Darnold still censored (window open). **Garoppolo now never breaks out**
  (career-best rank 18) — reported as a visible consequence of the QB15 bar, not hidden.
- Cohort at the new tier: **59 ever broke out, 16 late** (was 69/17). Kyle Orton + Garoppolo drop
  out; Jameis Winston enters.
- **High-school layer dropped, and the numbers justify it.** Re-ran the recruiting join at top-15:
  still only **6** late breakouts with a HS profile. Measured the reach trade-off — the
  late-breakout cohort is **bimodal** (2001–2005 and 2011–2020, dead zone between), and recruiting's
  2006-class floor cuts off the entire earlier cluster:
  | earliest entry | cohort | ever | late |
  |---|---|---|---|
  | 2010 (recruiting reach) | 190 | 33 | 7 (6 matched) |
  | 2005 (college PBP reach) | 244 | 42 | **11** |
  | 2004 | 257 | 46 | **12** |
  Going college-only **nearly doubles the positives**. HS box-score scrape **cancelled**; the built
  recruiting artifacts are **retained but demoted** to an optional covariate rather than deleted.
- **Report prose is now computed from the tables** rather than hardcoded, so the trough numbers
  cannot drift from the tier. Central result survives the change: late breakouts median early PPG
  6.0 vs 4.48 (busts) / 15.75 (on-time), early rank at the unranked floor for both late and never.
- Plan rewritten: §2.2 layer table, new §2.3 (HS set aside), §2.4 join chain, §3.2 (one bar cannot
  define a breakout), §5.1 N ladder → `ever_breakout` n≈42 / late n=11. 4 new tests (25 total for
  labels), 165 football tests pass, ruff clean.

---

## Entry 057 — QB breakout stage 3: college production layer from cfbfastR play-by-play

**Date:** 2026-07-26

**Prompt (full text):**

> Start

**Response notes:**
- Built the **primary pre-NFL evidence layer** from cfbfastR play-by-play (public GitHub parquet,
  no API key): `data/college.py` (aggregation + career shape), `data/college_link.py` (cohort join),
  `scripts/qb_breakout_college.py`, `REPORT_qb_breakout_college.md`.
- **Play-by-play over a season-stats table, for two concrete reasons.** (1) EPA per dropback and
  success rate exist at all. (2) **Sacks separate cleanly from rushing** — NCAA box scores charge
  sack yardage to rushing, so Mayfield's 2015 reads as 405 official rushing yards where he actually
  gained 604 on rush plays and lost the rest on 39 sacks. That is the difference between measuring
  mobility and measuring pass protection. Verified the aggregation against his real line
  (3,689 yds / 36 TD / 7 INT vs actual 3,700 / 36 / 7).
- **Three source boundaries found by building it, all real, none of them fetch bugs:**
  - **Floor 2004, not 2002/2003.** Those two files exist but ship an older, thinner schema (366
    cols vs 405) with no `completion` / `pass_td` / `rush_td` / `EPA_success` / team-name fields.
  - **Ceiling 2021.** cfbfastR-data publishes no further and `sportsdataverse` reads the same repo.
    Costs nothing for *fitting* (last college season 2022+ → NFL entry 2023+ → right-censored) but
    means the layer **cannot score today's prospects**. Closing it needs a CFBD key or an ESPN
    loader — flagged as the one outstanding data decision.
  - **2009 types `rush` as Float64** where every other season is Boolean; added dtype normalisation.
- **Truncation caught and flagged rather than hidden.** Careers straddling 2004 are clipped —
  Aaron Rodgers reads as *one* college season / 274 attempts because only 2004 is in range.
  `college_career_truncated` marks them (30 matched, 2 late breakouts); `career_*` is unusable for
  those rows, `final_*` is not, which is now the documented argument for leaning on final-season
  form.
- **Join is name + timing, with school as tiebreaker not filter.** cfbfastR carries no player ID,
  so `draft_picks.cfb_player_id` has nothing to join to. Requiring school to match would discard
  players over spelling ("Michigan St." vs "Michigan State"), and that variance is worse for small
  programs — i.e. exactly where late-round and undrafted QBs come from. Ambiguity refused, not
  guessed.
- **Results:** 3,419 QB-seasons (>=50 dropbacks) across 1,684 college careers. Cohort match
  **215/331 overall, 88% for 2005+ entrants**, and **unbiased across outcomes** (late 82% / never
  88% / on-time 94%). **9 late breakouts carry a college profile vs 6 for high school** — Rodgers,
  Alex Smith, Tyrod Taylor, Cousins, Tannehill, Geno, Winston, Mayfield, Love. `ever_breakout` = 38
  matched, which is the modelling sample for stage 6.
- **Bugs fixed while building:** column-name collisions between the two career frames
  (`first_season`, `last_season`, `career_games` mean different things on each side) — college side
  now prefixed `college_*`; and an empty rushing frame producing null-typed join keys, which
  crashed aggregation for any slice with no rush plays.
- 23 new tests (246 total), ruff clean. Plan updated: §2.2 boundaries, §2.4 join chain rewritten
  (cfb_player_id is not usable), §4.2 what stage 3 established, §5.1 N ladder → ever_breakout 38 /
  late 9.

## Entry 058 — QB breakout stage 4: CFBD extension + a comparability test before splicing

**Date:** 2026-07-26

**Prompt (full text):**

> Here's a CFBD api key: [redacted — stored at `~/.config/cfbd/api_key`, chmod 600, outside the repo]

**Response notes:**
- **Key handling first.** Written to `~/.config/cfbd/api_key` (0600), outside the repository.
  `data/cfbd.py:api_key()` reads `CFBD_API_KEY` first, then that file, and exits with instructions
  if neither exists. Verified by grep that the key string appears nowhere in the tree — not in
  code, reports, manifests or commit messages.
- **This closes the one outstanding data gap.** cfbfastR-data stops at 2021, so the college layer
  could describe history but not score a current prospect — the thing the project exists to do.
  CFBD covers 2013-present. Fetched **3,511 QB-seasons (2013-2025, ≥50 attempts)** →
  `data/raw/cfbd_qb_seasons.parquet`.
- **Endpoint dead end, recorded so it isn't retried:** CFBD `/plays` carries `ppa` but **no player
  name or id fields** — the player appears only inside free-text `playText`. So play-level
  aggregation the way cfbfastR allows is not possible here; the layer is built from
  `/stats/player/season` + `/ppa/players/season` instead, and **games played is absent rather than
  guessed**. PPA player-season aggregates begin 2013; passing stats are thin before that.
- **The point of the stage is the comparability test, not the fetch.** Concatenating two
  differently-derived series would put a source discontinuity at the 2021/2022 boundary that a
  model would read as signal. So the two sources were joined on the 1,774-row 2013-2021 overlap
  and measured:
  - **Volume agrees**: attempts / yards / TDs correlate **0.98-0.99** — the same events counted twice.
  - **Rushing attempts differ by ~16 per season**, and that gap correlates **0.78 with cfbfastR's
    sack count**. CFBD follows the NCAA convention of charging sacks as rushing attempts; our
    cfbfastR layer deliberately does not. The discrepancy *is* the convention, not an error.
  - **Efficiency does not agree**: `ppa_pass` vs `pass_epa_per_db` correlates only **0.735**, on a
    scale whose level shift exceeds the target's own mean.
- **Calibration reported with the error it carries.** `fit_calibration` returns
  `noise_ratio = resid_sd / target_sd` — the share of between-player spread the mapping *fails* to
  reproduce. rush_share **0.216**, td_rate 0.273, YPA 0.279, completion_pct 0.429, int_rate 0.649,
  **PPA→EPA 0.679**. rush_share is the clean case: slope 1.01, intercept −0.043 (exactly the sack
  correction), r² 0.95.
- **Consequence — a two-tier feature design, which is a real modelling constraint:**
  - **Portable** (native and near-identical in both): completion %, YPA, TD rate, rush share, volume.
    A model on these can be **fit on history and pointed at this year's class**.
  - **cfbfastR-only**: EPA per dropback, success rate, adjusted yards per dropback — better
    features, 2004-2021 only, so a **historical instrument**. Stage 6 fits both and reports what
    portability costs; if the gap is large, that is itself the finding.
  - Calibrated values are written as `*_est` with `efficiency_is_estimated=1` so an estimate can
    never be mistaken downstream for a measurement.
- **Incidental finding worth keeping:** cfbfastR averages only **8.8 games per QB-season** — its
  play-by-play has game gaps, so its *totals* understate while its *rates* are fine. Reinforces the
  existing preference for `final_*` over `career_*` features.
- **Bug fixed:** `/stats/player/season` omits a stat type entirely when a player has none of it, so
  a whole column could be missing for a season and derived columns raised `KeyError: 'pass_yds'`.
  The frame is now reindexed to a guaranteed schema.
- 10 new tests (**256 total**), ruff clean. Plan gains §2.5 (extension, calibration table, two-tier
  design); stages table renumbered (the cancelled HS scrape is now 3b) with CFBD as stage 4.

## Entry 059 — QB breakout stage 5: college archetypes, and a data defect found on the way

**Date:** 2026-07-26

**Prompt (full text):**

> Go for it

**Response notes:**

**A defect in stage 3, found before building on top of it.** Sanity-checking the college
QB-season aggregate against known real-world rates showed cfbfastR **does not populate every play
flag in every season**, and an unpopulated flag aggregates to a clean **zero rather than a null**:

- `int` is absent in **2004 and 2006–2013** (observed rate 0.000–0.004 vs a real 2.5–3.3%)
- `sack` is absent in **2013** (0.001 vs a real 6–7% of dropbacks); sack plays are missing from
  the play-by-play entirely that year, so dropbacks collapse to attempts

This is the most dangerous shape a defect can take — nothing is missing, nothing errors, and every
QB who played 2006–2013 simply appears never to have thrown an interception. Cam Newton, Russell
Wilson, Andy Dalton, Nick Foles, Kirk Cousins and Ryan Tannehill all sat in that window, and
"never turns it over" would have been the loudest false archetype in this stage.

Fixed in `college.py` with `FLAG_FLOORS` + `mask_unreliable_flags()`: a **minimum plausible
season-level prevalence** is checked at build time and the affected columns nulled when a flag
fails — not a hard-coded year list, since upstream could backfill and the same failure could
appear in a season not yet published. `season_flag_quality()` reports the evidence, each season
carries `int_recorded`/`sacks_recorded`, and a career INT total is nulled unless *every* season in
it recorded them. The repair is idempotent and was applied to the cached aggregate.

**It also invalidated a stage-4 conclusion, in the good direction.** The CFBD calibration had
`int_rate` at noise ratio **0.649** — written off as "rare events, too noisy to carry". A fifth of
the overlap was 2013, regressing real CFBD values against fabricated zeros. With those rows nulled
the same calibration is **0.253**, one of the tightest in the table, and INT rate moves into the
portable tier. *A disagreement between two sources is a hypothesis about one of them, not a
verdict on both.*

**Stage 5 proper — archetypes.** New `qb_breakout/archetypes/` package,
`scripts/qb_breakout_archetypes.py`, `REPORT_qb_breakout_archetypes.md`.

- **Style, not quality — enforced by measurement, not intention.** Clustering on efficiency
  returns a leaderboard with four bins. The check is the share of within-season EPA variance
  falling *between* archetypes: **0.56** clustering on quality stats (control), **0.31** with
  `completion_pct` in, **0.14** for the set kept, **0.03** for mobility alone. Completion
  percentage is an accuracy measure but also the most quality-loaded rate a QB has, so it was cut
  and kept as a profile column.
- **Final four features:** `rush_share`, `rush_yds_per_att`, `rush_td_share`,
  `yards_per_completion` (depth-of-target proxy — dividing yards by *completions* rather than
  attempts removes the accuracy term). All portable-tier, so the taxonomy can be assigned to a
  current prospect. Nothing per-game, deliberately: CFBD exposes no games-played, and a per-game
  feature would be un-assignable to exactly the prospects this exists to judge.
- **Z-scored within season.** Without it the first split found is a decade, not a style.
- **Fitted on ≥150 dropbacks, assigned to everything** — the basketball project's attempt-gating
  lesson back-applied. Thin seasons still get an archetype; they just don't get to pull a centroid.
- **The algorithm recovered a 2×2 it was never told to look for**: at k=4 the centroids land one
  per quadrant of mobility × depth. Quick-Game Pocket Passer (Stroud, Foles, Kellen Moore,
  Haskins), Downfield Pocket Passer (Bradford, Winston, Burrow), Short-Game Runner (Colt McCoy,
  Locker), Downfield Dual-Threat (Kyler Murray, RG3, Mariota, Hurts).
- **Names derive from centroid position, not the k-means label integer** — refit with a different
  seed and **99.1%** of seasons keep their name, where label integers permute wholesale. Directly
  back-applies the basketball refit's renaming cascade.
- **Honest about k:** silhouette peaks at k=2 and never exceeds 0.24. Style is a continuum; k=4 is
  a partition of it chosen because every cluster has an obvious name, and the report says so.

**Result, with its limits stated:** among 211 matched QBs with a settled outcome, `pocket_quick`
produces a sustained breakout at **0.05 (3/60)** vs `runner_downfield` at **0.28 (9/32)**,
non-overlapping intervals, permutation p = **0.034**. Among the 38 who did break out, archetype
does **not** predict lateness (p = 0.064). Reported with Wilson intervals and a permutation null
rather than a chi-square, because the cells are single-digit.

**Two bugs of my own, both caught by tests I wrote to fail:**
- The outcome cross-tab first used `ever_breakout` — the **single-bar top-15 diagnostic** (52
  matched), not the project's two-bar label (38). It inflated every rate on the page. Outcomes are
  now specs carrying the population they are defined over, and `late_sustained` is labelled as
  conditional on having broken out at all rather than reported as a rate over everyone.
- The report's conclusion paragraph asserted "not evidence of anything" while the computed p was
  0.030. The verdict prose is now **generated from the p-values** (`build_verdict`), the same
  discipline the cohort script already uses.
- A zero-variance season produced **NaN, not null**, in the z-score — NaN passes a null check and
  reaches k-means. Fixed in both the standardiser and the usability filter.

13 new tests (**216 football tests**), ruff clean. Plan gains §2.6 (missing flags), §4.3 (what
stage 5 established), corrected §2.5 calibration table; stages table has 5 done, 6 next.

## Entry 060 — QB breakout stage 6: the pre-NFL-only model, and a negative result worth having

**Date:** 2026-07-26

**Prompt (full text):**

> Move on to the nest phase

**Response notes:**

New `qb_breakout/model/` package (`features.py` tiers + sample rule, `fit.py` evaluation),
`scripts/qb_breakout_model.py`, `REPORT_qb_breakout_model.md`, `qb_breakout_scores.parquet`.

**Defining the sample turned out to be the first real decision.** A 2024 entrant who has not
broken out has not *failed* to break out — labelling him a negative teaches the model that recent
profiles do not work, and there were 34 such rows. Restricted to **5+ NFL seasons of opportunity**
(90% of sustained breakouts happen by year 5, 95% by year 7): **177 QBs, 35 breakouts**. Reported
with a sensitivity table rather than asserted — the base rate is flat at ~0.18 from a 3-season
threshold to an 11-season one, which is the evidence the cut removes false negatives without
reshaping the outcome.

**Also had to extend `add_career_features` with portable `final_*` columns** (`final_attempts`,
`final_td_rate`, `final_int_rate`, `final_yards_per_completion`, `final_rush_td_share`) so the
portable tier was real rather than approximately real.

**Results:**
- **College production carries real signal.** CV AUC **0.696** vs a label-permutation null of
  **0.493 ± 0.071** (p < 0.005, 200 shuffles with the whole pipeline re-run each time). Temporal
  split (train early entrants → score later ones) holds at **0.725**.
- **Portability is free — stage 4's question, answered.** Portable tier **0.696** vs full tier
  **0.689**; the efficiency features add *nothing detectable*, well inside the ±0.018 fold spread.
  The project ends with a model that can score current prospects, at no measured cost. Gradient
  boosting doesn't help either (fitted and reported so nobody wonders).
- **What it reads is workload and role, not efficiency**: final-season attempts, *not* being a
  quick-game pocket passer, rush share, archetype stability — all sign-stable under bootstrap.
  That ordering is exactly why EPA adds nothing.

**The draft benchmark, and the trap in it.** Draft position is never a feature — the project's
standing convention (model and market as independent opinions, never blended) applies here as it
does to ECR. As a benchmark it scores **0.890** against the model's 0.696. But a fantasy breakout
requires snaps and *the draft allocates snaps*: first-rounders average **85 career games** and
break out at **0.565**, day-three/undrafted average **18 games** and break out at **0.023**. The
draft partly **causes** the outcome it predicts, so beating it is not the standard a college-only
model should be held to — a model that did beat it would be suspect.

**The fair test is within a draft band, and it fails where it matters.** Among first-round picks
the college model is at **chance (0.496, n=46, 26 breakouts)** — once the league has committed a
first-rounder, his college box score says nothing about which of those picks hits. R2-3 looks
better (0.675) on 7 positives; day-three has 2 breakouts and cannot be evaluated.

**The conclusion is negative and stated as such:** if late breakouts are visible before the NFL,
they are **not visible in college production**. Of the 35 breakouts in the sample, 8 were late —
enough to describe, not enough to fit. Stage 5 hit the same wall from the archetype direction.
What is untested is **context**: competition faced, supporting cast, scheme — the conditions a QB
produced under rather than the totals he produced.

**Testing focused on the ways a small-N model lies**, not on accuracy it isn't entitled to: that
unsettled outcomes stay out, that censored ones stay out, that draft position never becomes a
feature, that the portable tier is a strict subset of the full one differing by exactly the
efficiency block, that shuffled labels score ~0.5 (a null that isn't centred means a leak), and
that out-of-fold scores cover every QB exactly once.

15 new tests (**231 football tests**), ruff clean. Plan gains §4.4; stages table now 1–6 all done.

## Entry 061 — QB breakout stage 7: drafting team and regime, bounded by arithmetic

**Date:** 2026-07-26

**Prompt (full text):**

> Let's look at drafting team, also GM of drafting team as factors as well

**Response notes:**

Closes the last thread from the project's first prompt — "player archetypes, **drafting
situations**, and other factors". New `qb_breakout/situation/`, `scripts/qb_breakout_situation.py`,
`REPORT_qb_breakout_situation.md`.

**Checked the power before building the analysis, and it changed the design.** 140 drafted QBs
with a settled outcome across 32 franchises = a **median of 4 QBs and 1 breakout per team**; seven
teams have zero, one has three. A table of raw team breakout rates would show a 0–60% spread, all
noise, and would be the most quotable artifact in the repo. **So the module never computes one** —
there's a test asserting the output frame has no `rate` column.

**What it computes instead: observed minus expected given draft capital.** Each QB gets an
out-of-fold P(breakout) from his pick alone (log-scaled pick + undrafted flag); a regime's
expectation is the sum. Teams differ hugely in the picks they spend on QBs, so a raw comparison is
mostly a comparison of draft position. Null simulated by drawing each QB from his own probability
— no normal approximation, which is wrong at these counts.

**Results — nothing detectable:**
- Franchise: dispersion of observed-minus-expected **p = 0.128** (28 franchises ≥3 QBs)
- Head coach at draft: **p = 0.089** (15 coaches ≥3 QBs)
- Two coaches clear p < 0.05 individually (Shanahan 3/3 vs 1.29 expected; Harbaugh 2/3 vs 0.34)
  against **0.8 expected by chance** from 15 uncorrected comparisons — the report states that
  arithmetic next to the table rather than leaving it to the reader.
- Belichick: 6 QBs, 0 breakouts, 0.95 expected — the most-quoted-shaped cell in the table, and not
  significant either.

**The power calculation is the actual output.** A typical franchise drafted 4 QBs expecting ~1
breakout; to clear 5% it needed **3** — about **2 extra breakouts above expectation across its
entire draft history**. No plausible front-office effect is that large. That converts "we found
nothing" into "**nothing this size or smaller was findable**", which is a different and more
useful claim.

**Head coach is available; GM is not.** nflverse publishes coaches via `load_schedules`
(home_coach/away_coach, 1999+, 177 coaches) — stacked home+away, mid-season changes resolved to
whoever coached most games, since the draft is a spring decision. **No free structured
GM-by-team-season dataset exists** (searched; Wikipedia has it per team page). Built
`attach_gm()` + `--gm-table` so a supplied CSV is a one-command job, and said plainly in the
report that it wouldn't change the conclusion — GM tenures are *shorter* than franchise histories,
so the cells shrink.

**Team-code gotcha again:** draft frames use PFR codes (GNB, SDG), schedules use nflverse codes
(GB, LAC). Both sides canonicalised through `labels/teams.py` before joining; there's a test.

**Framing note:** this stays out of the pre-NFL player model, on the same rule that keeps draft
position out — situation is the league's decision about the player, not evidence about him. It is
analysed as a benchmark-side factor.

**Test-driven correction to my own test:** `test_random_group_labels_do_not_look_like_a_team_effect`
failed on one seed at p=0.03. Checked calibration across 30 seeds first — mean p 0.545, 3% below
0.05, so the implementation was right and the *test* was a coin-flip that fails 1 run in 20. Now
asserts the distribution across 10 seeds (median p > 0.2, ≤2 below 0.05).

**Conclusion the report reaches:** this question cannot be answered with 177 QBs, and a career
event with 35 instances is the wrong instrument for a situation effect. The right follow-up is a
**different outcome, not more factors** — snaps in years 1–3, starts, or per-season fantasy points
give a data point per season instead of one per career. And note §4.4 already showed the pick
itself predicts at AUC 0.890 largely *because it allocates opportunity*: a franchise's influence
on a QB's career mostly **is** the pick it spent, which is measured, not missing.

15 new tests (**246 football tests**), ruff clean. Plan gains §4.5 and stage 7.

---

## Entry 062 — Medical-staff injury grades: new football project, planned

**Date:** 2026-08-03

**Prompt (full text):**

> New idea for a football research project. Injury grades of medical staff. We should develop a
> system to grade each team's medical staff over the last 3 years and 5 years. The grade should
> consider reinjury risk. Maybe separate grades for position groups.

**Response notes:**
- **Fourth project in the repo**, planned as a sibling package `sports/football/src/medstaff/`
  mirroring `qb_breakout` conventions. Plan doc `docs/MEDSTAFF_PLAN.md`.
- **Feasibility probed before designing** (all measured, not assumed): `nflreadpy.load_injuries`
  covers **2009–2025**; injury rows join **~99–100%** to `load_rosters_weekly` on
  `(gsis_id, week)` → rosters_weekly is the availability spine (all positions, gsis-keyed).
  **Rejected `snap_counts` as the spine** — keyed on `pfr_player_id`, combined crosswalk only
  **81.7%**, and starts 2012.
- **⚠️ Regime break at 2016 found in probing:** `report_status` null 3–6% (2009–15) → 39% (2016)
  → ~53% (2017–25); the league dropped "Probable". Body part (coalesced) and `practice_status`
  are ~0% null throughout, so **all outcomes are built on the regime-invariant columns**.
- **Power is good, unlike qb_breakout:** median **412 episodes/team over 5yr** (254 over 3yr),
  vs the ~4 QBs/franchise that made the sibling's stage 7 unpowered. Raw "Out" per team 2021–25
  spans **98 (MIA) → 277 (TEN)**, sd 46 vs Poisson ~13 — real dispersion to decompose.
- **User decisions:** whole roster (all positions, 8 groups) · headline = **return-speed ×
  recurrence 2×2** (separates good rehab from rushing players back) · grades **always publish**
  with a signal-vs-luck verdict leading (explicitly NOT gated/suppressed) · team-window grade +
  optional `--staff-table` CSV hook (no free head-athletic-trainer source — same problem as GM).
- **Design choices that reverse the obvious**, adopted after a design review: **not a GBM** for
  the expectation models (a flexible learner eats the residual variance being measured →
  penalised discrete-time logistic hazard, shallow GBM as robustness check only);
  **leave-one-team-out**, not random k-fold; fit incidence **with and without prior-injury
  history and report the pair as a bound** (a bad staff manufactures fragile-looking players, so
  adjusting for prior injuries adjusts away its own effect); **informative censoring is
  gameable** so `P(returns at all)` and `released_while_injured` become their own outcomes;
  whole-factor permutation as primary inference + player-block bootstrap null.
- **Framing to lead with:** this data cannot identify "medical staff" — the residual bundles
  athletic training, S&C, sports science, coach practice intensity, GM roster taste and scheme.
  It grades a **team availability system**. Stated as a non-goal, not a caveat.
- Standing decision `PROJECT_PLAN.md:151` ("no injury-report text") is **unchanged** — this is a
  separate project and nothing feeds back into the ranking models.

---

## Entry 063 — Cross-position-group injury signature: the scheme instrument

**Date:** 2026-08-03

**Prompt (full text):**

> One thing to look at across position groups for a team: same injury tendencies. Focus on knee,
> ankle, back, hip, concussion (scheme)

**Response notes:**
- **User's idea became stage 5 and is the strongest attribution instrument in the project.**
  Logic: if a team's excess sits in ONE body part and appears across position groups sharing
  nothing except the building, that implicates a common cause (contact policy, S&C, surface,
  protocol); if confined to one group it's roster/scheme/luck for that group.
- **It also sidesteps the project's worst confounder**: body-part *composition* is
  **disclosure-robust** — a club that lists everyone inflates numerator and denominator alike.
- **Statistic is a variance decomposition, not a grade**: team main effect vs team×group
  interaction per body part. Team × bodypart × position-group cells are tiny (**median 4
  episodes over 5yr**), so cells are never graded; concordance aggregates across the 8 groups.
- **Exploratory probe (2021–25, off/def concordance across 32 teams, crude exposure
  denominator), recorded so the real run is confirmatory rather than fishing:**

  | part | share-based (signature net of level) | rate-based (level+signature) |
  |---|---|---|
  | concussion | 0.405 (p=.022) | 0.404 (p=.022) — **only one stable in both** |
  | knee | −0.027 | 0.335 (p=.061) — rate-only ⇒ level artifact |
  | back | 0.433 (p=.013) | 0.172 — share-only ⇒ compositional |
  | ankle | 0.258 | 0.174 |
  | hip | 0.209 | 0.106 |

- **The disagreement between specs IS the finding**, so both are always reported. Knee flipping
  positive only under rate is the signature of a *level* effect (knee is the largest category so
  it tracks overall burden); back flipping the other way means its share result was compositional.
- **Multiplicity preregistered:** 5 parts ⇒ Bonferroni 0.01, and **no probe result clears it**.
  Concussion designated the **primary a-priori hypothesis** (practice contact policy and tackling
  technique are documented coach decisions; concussion reporting is protocol-mandated hence least
  disclosure-contaminated). Other four labelled exploratory.
- **Strongest causal handle designed in:** does the signature **follow the head coach** across
  franchises? Coaches available 1999+ from `load_schedules`. It is the one design element that
  breaks the shared-roster confound (offense and defense aren't fully independent — an old roster
  is old on both sides).

---

## Entry 064 — medstaff stage 1: dataset registration, body-part taxonomy, data report

**Date:** 2026-08-03

**Prompt (full text):**

> start on stage 1

**Response notes:**
- Branch `medstaff-grades` off fresh `primary`. **NB `qb-late-breakout` is still unmerged**, so
  its `src/qb_breakout` is absent here (only stale `__pycache__` on disk) and its PROMPT_LOG
  entries **055–061** are not on primary — this entry continues from **062** so the global
  numbering stays collision-free when that PR lands.
- **3 datasets registered** in `data/fetch.py` (one `Dataset(...)` line each, existing
  `L()`/`_clip_seasons`/`_load_resilient`/`_write_manifest` machinery unchanged): `injuries`
  (2009+, 90,752 rows), `rosters_weekly` (2002+, 906,378 rows), `schedules` (7,548 rows).
- **New package** `src/medstaff/` + `data/` subpackage: `taxonomy.py`, `positions.py`,
  `teams.py`, `ingest.py`, facade `__init__.py`. Registered in `[tool.hatch...] packages`.
- **`taxonomy.py` is the substance.** 292 distinct raw body-part strings → 16 groups. Two rules
  do the work: **non-injury is checked first on the raw string** (so
  `Ankle [Not Injury Related - Personal, Thursday Only]` is excluded, not counted as an ankle),
  and among body-part matches **earliest match wins, longest match breaks ties** — earliest
  encodes "the primary injury is listed first" (`Foot/Wrist/Hip` → foot), longest keeps
  `hip flexor` in soft-tissue instead of collapsing to `hip`, **structurally rather than by rule
  ordering**. Handles laterality prefixes, case, plurals, bracket clauses and free-text sentences.
- **Result: 99.8% of rows map** — only 200 rows (0.2%) land in `other`, and **no single unmapped
  string exceeds 0.1%** of rows.
- **`_normalize_keys` fix:** the caches disagreed on join-key dtype — `rosters_weekly` lands
  `season`/`week` as f64 (nflreadpy concatenates multi-season pulls with `diagonal_relaxed`,
  which widens ints once any season has a null) while `injuries` keeps i32; polars refused the
  join. Both pinned to Int32 at load.
- **Stage-1 report `REPORT_medstaff_data.md` measures the three design-deciding properties:**
  (1) the **2016 regime break reproduced exactly** — `report_status` null 3.0–6.0% pre-2016,
  **39.2% in 2016**, 48.7–54.7% after, while coalesced body part is ≤0.1% null and
  `practice_status` ≤0.7% in every season; (2) **spine join 99.2%** overall — 97.4–98.2% for
  2009–2015 but **99.9–100% from 2016**; (3) per-club disclosure indices incl.
  `questionable_play_rate`.
- **Sanity checks against real-world rates** (the `qb_breakout` flag-defect lesson): distinct
  concussed players/season **90–183**, matching the known 113–179 range for 2015–25; and
  **zero clean-zero season × focal-group cells** — the exact signature that defect presented as.
- 71 new tests (**203 football tests**, 261 repo-wide), ruff clean.

---

## Entry 065 — Shelve the QB-breakout path; medstaff onto its own branch

**Date:** 2026-08-03

**Prompt (full text):**

> Branch the work on qb-late-breakout as a separate path, shelving it for later. Then switch to
> a new branch for this work

**Response notes:**
- **`qb-late-breakout` shelved, not abandoned.** Verified fully pushed first (local ==
  `origin/qb-late-breakout` at `f67d2ba`, nothing unpushed), so nothing could be lost. Added a
  local marker branch **`shelf/qb-late-breakout`** at the same commit. Deliberately **not
  pushed** — the work is already preserved on `origin`, and pushing would create a remote branch
  that wasn't asked for.
- **medstaff moved to a fresh branch `medstaff`** carrying the stage-1 commit `ec23f0d`; the
  earlier `medstaff-grades` name deleted (same commit, one branch now).
- **Topology:** both paths fork from `primary` @`2bc2224` and share no commits, so shelving one
  has no effect on the other.
- **Numbering consequence recorded:** with `qb-late-breakout` parked, entries **055–061** stayed
  claimed but unlanded, leaving a gap on `primary`. Kept medstaff at **062–064** rather than
  filling it — renumbering into 055 would collide the moment the branch unshelved. (Entry 066
  then closed the gap properly.)
- Corrected two figures in the project memory that came from early probes and were superseded by
  stage 1: spine join is **99.2%**, not the 100% a 2023-only probe showed; body-part vocabulary is
  **292 strings** across full history, not the 172 seen in 2021–25 alone.

---

## Entry 066 — Land the QB-breakout decision trail and findings on primary

**Date:** 2026-08-03

**Prompt (full text):**

> Wait, let's go back to the breakout model. Those prompts need to land somewhere, let's fix
> that now.

**Response notes:**
- **The problem:** shelving the branch stranded the *provenance*, not just the code. Prompt
  logging is this repo's reproducibility backbone (Entry 001), so a complete seven-stage project
  whose decision trail exists only on an unmerged branch is a real gap.
- **What was stranded** — all documentation and findings, **no code**: both `PROMPT_LOG.md`
  copies (+429 lines each, entries 055–061), `QB_BREAKOUT_PLAN.md` (467 lines), and the seven
  `REPORT_qb_breakout_*.md` stage reports carrying the actual results.
- **Chosen (user, from three options): docs + findings only.** Cherry-picked onto
  `qb-breakout-docs` off `primary`. The code stays shelved as decided in Entry 065. Rejected
  alternatives: merging the whole branch (would reverse the shelving decision one turn after it
  was made) and renumbering medstaff into 055–057 (breaks chronology — the QB prompts genuinely
  came first).
- **README `Run it` section deliberately NOT copied** from the branch: it documents commands for
  `scripts/qb_breakout_*.py`, which do not exist on `primary`, so landing it verbatim would have
  documented commands that fail. Wrote a short shelved-status section instead, carrying the
  headline finding.
- **Status banners added** to `QB_BREAKOUT_PLAN.md` and `README.md` so a reader on `primary` is
  told up front that `src/qb_breakout/`, `scripts/qb_breakout_*.py` and the 103 tests are on the
  `qb-late-breakout` branch, not here.
- **Known cost, accepted:** when `qb-late-breakout` eventually merges it will conflict on these
  files; the resolution is to take `primary`'s side. Recorded here so it isn't rediscovered.
- **⚠️ Gotcha found and fixed while doing this — the log has TWO template markers.** Entry 037's
  prose quotes the literal `<!-- Template for new entries:` opener (it documents an earlier
  prompt-log repair), so appending via a first-match replace inserts new entries at **line ~1179,
  inside that entry**, instead of at the end of the file. This silently corrupted both this
  entry pair *and* medstaff's 062–064, which had already been committed. **Always insert at the
  LAST occurrence** (`str.rfind`), and check ordering afterwards rather than trusting the write.
- Primary's log is now contiguous **001–061**; medstaff's **062–064** land with that branch.

---

## Entry 067 — medstaff stage 2: episodes, and a 2021 comparability break

**Date:** 2026-08-03

**Prompt (full text):**

> Start stage 2
>
> [interrupted mid-probe to land the QB-breakout docs, then]
>
> resume stage 2 with the roster status signal

**Response notes:**
- **Plan rule changed before implementing, on measurement.** The plan said "any week with snaps
  terminates an episode". Measured instead: players on `RES`/`INA`/`DEV`/`CUT` take a snap in
  **~0.02%** of player-weeks, so weekly roster status separates played from did-not-play almost
  perfectly — gsis-keyed, no crosswalk, back to 2002. `snap_counts` keys on `pfr_player_id`,
  crosswalks at **81.7%** from 2012, and would have misread the **21%** of `ACT` player-weeks
  with zero snaps: a healthy scratch or deep backup is **available**, not injured. Snaps deferred
  to stage 3 as an intensity covariate.
- **Episode rules.** Only an `ACT`-and-undesignated week terminates; `INA`/`RES`/bye all
  continue, which absorbs report noise for free (listed weeks 5 and 7, inactive week 6 → one
  spell, not two). Only an *impaired* week (real body part, or reserve) starts one, so a healthy
  scratch never creates an episode.
- **Two design calls the plan left open.** (a) **Concurrent injuries are not split** — the report
  carries one primary part and clubs flip it weekly, so splitting would manufacture episodes far
  more often than catch a real second injury; the spell keeps its opening group and sets
  `group_changed`. (b) **Body part carries forward within a spell, never across one** — a global
  forward-fill was written and then deleted, because it would let an unrelated week-10 IR stint
  inherit a week-2 knee across a return, and long IR spells matter most.
- **⚠️ THE FINDING — a 2021 comparability break, same shape as the cfbfastR flag defect.**
  First run showed `used_reserve` at **2 episodes total across 2012–2019** then **1,059/season
  from 2021**, and episodes jumping 2,106 → 3,385. IR was not invented in 2021: two
  `rosters_weekly` fields are unpopulated earlier. `status_description_abbr` carries **0.0%**
  R-codes for 2012–2015 (51% null in 2016), and `status == "INA"` is 2.8k rows across 2012–2019
  vs 16.8k across 2021–2025 while `ACT` falls **0.86 → 0.59** — the gameday active/inactive split
  is simply absent. **Fix:** reserve reads from `status` (stable 6–14% every season), never the
  abbr codes; absence outcomes restricted to **2021–2025**. That window *is* the 5-year grading
  window, and sits wholly inside the post-COVID 17-game era and post-2016 reporting regime — one
  homogeneous regime, not a compromise. `roster_status_regime_table()` + tests pin both halves.
- **⚠️ Reserve/COVID-19 (`R59`) excluded.** 2021 only, 725 player-weeks, **exactly zero** every
  other season; carries `RES` without being an injury. Left in it pushed 2021's reserve share to
  0.46 vs ~0.31 for 2022–25 — i.e. it would have made every club look worse at medicine in year
  one of the five-year window. Also fixed a smaller leak where a stray `Illness`/`non_injury`
  report row could label a reserve spell (63 episodes → 0).
- **Result: 14,420 episodes / 221,313 player-weeks, 2021–2025**, now homogeneous (2,791–2,948
  per season; reserve share 0.29–0.34; mean games missed 3.2–3.4). **451 episodes per club over
  5 years** (336–564), sd 58 against Poisson 21 — **2.7× wider than chance**, which is the
  dispersion stage 4 has to decompose.
- **Recurrence at k=6 games: 9.3% overall.** Knee **13.4%** (n=1,250 returns) > ankle 9.9% >
  back 8.0% > concussion **5.2%** > hip 5.0%. Concussion lowest is mechanistically right —
  protocol-mandated clearance. Risk set is **returns, not episodes** (60.5% resolved); the clock
  counts **games available**, not calendar weeks, so a week-17 return has two games of exposure
  rather than a free pass.
- **Known limitation recorded:** 21.6% of episodes have an `unknown` body part — spells that
  opened on a bare reserve week and never picked up a report row. They count for availability but
  are excluded from body-part analyses, which matters for stage 5.
- 33 new tests, one per edge case (bye, trade, roster gap, practice squad, season boundary,
  never-returns, group flip, severity, all recurrence horizons). **294 repo-wide**, ruff clean.
- Removed `FIRST_MODEL_SEASON` rather than silence its unused-import warning: a constant named
  "first model season" that is no longer the first model season is a trap.

---

## Entry 068 — medstaff stage 3: exposure, and a position-biased ID crosswalk

**Date:** 2026-08-03

**Prompt (full text):**

> start stage 3

**Response notes:**
- **`src/medstaff/exposure/`** + `scripts/medstaff_exposure.py` + `REPORT_medstaff_exposure.md`.
  Risk set = **130,139 player-weeks** (2021–2025) with confounders attached.
- **Risk-set definition:** a week only counts as exposure if a new injury could have *started*
  in it. Weeks inside an open spell are dropped (counting them turns one long absence into many
  weeks of injury-free exposure), as are practice-squad weeks and byes.
- **⚠️ THE FINDING — the ID crosswalk is position-biased.** `snap_counts` keys on
  `pfr_player_id`; the two obvious crosswalk sources fail *specifically on linemen*:
  `rosters_weekly.pfr_id` is null for **99.8% of OL rows** (vs 9–24% elsewhere), and `ids` is a
  **fantasy** table that resolved **2 of 352** distinct linemen. Symptom that exposed it: OL mean
  snap share came out **0.032** against QB 0.780 — linemen play nearly every offensive snap, so
  it had to be wrong. **Why it mattered more than missing data:** position correlates with body
  part, and body part is exactly what the stage-5 signature analysis compares, so position-biased
  missingness would have arrived looking like a finding. **Fix:** registered `players`
  (`load_players`), ~12% null for linemen and ~11% for skill players — unbiased. **OL snap
  coverage 2.4% → 83.2%**, mean snap share 0.558. QB coverage is 47% and that is correct:
  inactive backup weeks stay in the risk set (a scratch can still get hurt in practice) and have
  no snap row.
- **Surface is dirty free text:** `"grass "` with a trailing space is a distinct value from
  `"grass"` (93 vs 612 rows) and `""` means missing. Normalised to grass/turf with turf brands
  collapsed — brand is not an injury-risk distinction. 17 clubs play home on grass, 15 on turf.
- **Prior injury history ships as its own table, deliberately.** It reaches back to **2009** from
  the *report* (comparable across that span even though the roster fields are not) and is
  **strictly prior seasons** so it cannot leak the outcome. It is the one covariate that is
  partly the staff's own output — a poor system manufactures players who look fragile — so stage
  4 must fit incidence **with and without** it and report the pair as a **bound**, not a point.
- **Bug found by a test:** an all-null `birth_date` column arrives typed Null and raised
  `- not allowed on date and null`; now cast explicitly to Date. Real data has 0.8% nulls mixed
  in so it never surfaced in the live run — only a synthetic fixture caught it.
- `years_exp` clipped at 22 (one 2023 roster row claims 29 — the row is real exposure, the number
  is not). Coverage: surface 96.3%, home surface/rest/indoor 100%, age 99.3%, BMI 99.9%.
- 26 new tests (**320 repo-wide**), ruff clean.

---

## Entry 069 — medstaff stage 4: expectation models, and a risk-set bug they exposed

**Date:** 2026-08-03

**Prompt (full text):**

> start stage 4

**Response notes:**
- **`src/medstaff/expected/`** (`models.py` + `datasets.py`), `scripts/medstaff_expected.py`,
  `REPORT_medstaff_expected.md`. Penalised discrete-time logistic hazard with splines on age and
  week; **leave-one-team-out** folds; **club identity guarded out of the design matrix** (raises).
- **The load-bearing test:** inject a synthetic club effect of known size, assert **LOTO recovers
  it while random k-fold shrinks it**. LOTO is the more expensive scheme and the whole design
  rests on it being necessary, so necessity is demonstrated rather than asserted.
- **⚠️ BUG the first run exposed: incidence base rate came out 0.000.** The stage-3 risk set
  excluded weeks inside an episode **including the onset week itself**, so the event week was
  never at risk. In survival terms the event week is the **last at-risk week**. Fixed to exclude
  only `onset_week+1 .. end_week`; risk set 130,139 → **144,552** player-weeks and base rate
  0.000 → 0.100. Caught only because a zero base rate is obviously impossible — nothing errored.
- **Results (LOTO, 2,000-draw nulls):** incidence 0.100 base / perm p **0.025** / intraclass
  0.867 · duration 0.163 / **0.0005** / 0.882 · recurrence 0.018 / **0.058** / 0.440 ·
  returns_at_all 0.545 / **0.0045** / 0.625.
- **THE PATTERN: the evidence runs opposite to attributability.** Recurrence — most plausibly
  owned by a medical staff, downstream of the return-to-play decision, least contaminated by
  luck — is the **only component that fails its permutation test**. Incidence — *least*
  attributable (conditioning, scheme, surface, luck) — separates clubs most strongly. Consistent
  with clubs differing mainly in exposure and IR usage rather than medicine. A caution, not a
  finding; stage 6 decides whether any of it is stable.
- **The with/without-history bound is TIGHT** (intraclass 0.859 vs 0.867), so the §5.3 ambiguity
  is not load-bearing in practice. Could not have been known in advance — it is why the bound was
  built rather than a covariate choice being made.
- **⚠️ Caveat added to the report: `intraclass` is NOT "share attributable to the medical
  staff."** It only says the spread is not Poisson noise; the surviving variance can still be
  roster construction, disclosure, scheme or stadium — and the deliberately restrained model
  (§5.1) leaves uncaptured composition *in* the residual by design. Most likely misreading.
- **Nulls:** whole-factor permutation (reassigns whole *players* to clubs) is the primary
  inference — one test, no multiplicity. Per-club p-values use the **more conservative** of
  independent-Bernoulli and **player-block bootstrap**; the block null is verified wider by test
  when within-player weeks are correlated. Uncorrected p's reported with expected chance hits
  stated inline.
- Calibration tracks the diagonal for incidence and recurrence. Runtime ~6 min at 2,000 sims.
- 22 new tests (**342 repo-wide**), ruff clean.

---

## Entry 070 — medstaff stage 6: reliability, and what actually persists

**Date:** 2026-08-03

**Prompt (full text):**

> Do stage 6

**Response notes:**
- Run **before** the grades on purpose: a reader who stops here has the honest answer.
  `src/medstaff/validate/`, `scripts/medstaff_reliability.py`, `REPORT_medstaff_reliability.md`.
- **Two tests because they fail differently.** **Split-half** splits each club's **players**,
  never its rows (a row split would put one fragile player on both sides and manufacture
  agreement out of one man's hamstring), Spearman-Brown corrected. **Temporal** grades 2021–23
  and tests 2024–25 — the harder test, and the one a reader cares about, since a grade is only
  useful if it says something going forward.
- **RESULT — 2 of 5 components pass the preregistered gate:**

  | component | split-half | temporal | temporal p | perm p | gate |
  |---|---|---|---|---|---|
  | incidence (no hist) | 0.708 | 0.293 | 0.104 | 0.020 | FAIL — temporal |
  | incidence (with hist) | 0.701 | 0.192 | 0.293 | 0.033 | FAIL — temporal |
  | **duration** | 0.827 | **0.529** | 0.002 | 0.001 | **PASS** |
  | recurrence | 0.414 | **0.078** | 0.670 | 0.057 | FAIL — temporal + permutation |
  | **returns-at-all** | 0.664 | **0.397** | 0.024 | 0.003 | **PASS** |

- **THE HEADLINE IS THE GAP BETWEEN THE TWO TESTS.** Split-half is high everywhere (0.41–0.83) —
  the residuals are internally consistent, not measurement noise — but temporal is much lower.
  For most components the residual is a property of **a period**, not a carry-forward property of
  the club. Real, but not something to put on next season's board.
- **What is stable is NOT medicine.** The two passing components — duration and returns-at-all —
  are largely *how a club uses IR and times a return*: an operational/roster-policy signature.
  **Recurrence, the one outcome most plausibly owned by a medical staff, is weakest on every
  test.** Combined with stage 4 (where recurrence was also the only component failing its
  permutation test), the consistent story is that clubs differ in availability *management*, not
  in medicine.
- **Consequence for stage 7, now written into the plan:** the board separates clubs by
  **availability-management policy**, not quality of medicine — a narrower claim than "medical
  staff grades" and the one the evidence supports.
- **The gate LABELS, it does not suppress** (per the locked decision): grades publish either way,
  failing components marked at every appearance rather than in a footnote.
- **Removed hardcoded constants:** stage 6 first carried stage 4's permutation p-values as
  retyped literals. Now recomputed from stage 4's own parquet — a retyped number goes stale
  silently the first time the upstream stage is rerun.
- 11 new tests (**353 repo-wide**), ruff clean. Runtime ~45s.

---

## Entry 071 — medstaff stage 5: the cross-group signature holds for concussion

**Date:** 2026-08-03

**Prompt (full text):**

> start stage 5

**Response notes:**
- The user's own idea (Entry 063), now run properly. `src/medstaff/signature/`,
  `scripts/medstaff_signature.py`, `REPORT_medstaff_signature.md`. Needed **body-part-specific
  expectations**, which stage 4 did not produce (its incidence model predicts *any* onset) — so
  five extra leave-one-team-out fits, one per focal part. ~4 min.
- **RESULT — the preregistered primary hypothesis holds:**

  | part | composition | p | rate | p | probe |
  |---|---|---|---|---|---|
  | **concussion** | **+0.437** | **0.012** | **+0.445** | **0.011** | 0.405 / 0.404 |
  | back | +0.368 | 0.038 | +0.402 | 0.023 | 0.433 / 0.172 |
  | ankle | +0.361 | 0.042 | +0.216 | 0.234 | 0.258 / 0.174 |
  | knee | +0.197 | 0.279 | +0.224 | 0.218 | −0.027 / 0.335 |
  | hip | +0.106 | 0.562 | +0.150 | 0.412 | 0.209 / 0.106 |

- **Concussion clears α=0.05 in BOTH specs and is strongest in both.** Named in advance ⇒ tested
  at 0.05, not the corrected 0.010 (applying the family correction to a preregistered primary
  would discard the point of naming one). **It barely moved when the probe's crude denominator
  was replaced with a proper exposure-adjusted model** — that stability is the strongest thing
  that can be said for it. **Of the 4 exploratory parts, 0 clear 0.010.**
- **⚠️ Fixed a preregistration error in my own report logic:** the first draft applied Bonferroni
  to the primary hypothesis too, which contradicts the preregistration. Also fixed a generated
  sentence that called concussion "the only part" clearing 0.05 in both specs and then listed two
  (back does as well).
- **The variance decomposition agrees via a different statistic:** `share_common` (club main
  effect vs club×group interaction) is highest for **concussion 0.299** and lowest for **knee
  0.089** — concussion's excess is the most spread across position groups, knee's the most
  group-specific. Exactly the predicted ordering, from an independent statistic.
- **The knee probe result did NOT replicate, and that is the probe being corrected** — its
  rate-only 0.335 came from a disclosure-contaminated denominator (distinct listed players).
  Properly adjusted, knee is weakly positive in both and significant in neither.
- **Composition spec is disclosure-robust** by construction (a club that lists everyone inflates
  numerator and denominator alike) — a real advantage over every level-based measure in the
  project. Tested: composition must be invariant to overall level while rate must not be.
- **`unknown` body parts (21.6% of episodes) are EXCLUDED, not pooled** — concentrated in long
  absences, would swamp the parts compared.
- **Coach-follows is power-bounded as designed:** only **4** coaches ran two clubs in the window.
  Vrabel near-identical across TEN 0.073 / NE 0.074; the other three move both ways. At n=4 this
  is the power bound reported instead of a result.
- 14 new tests (**367 repo-wide**), ruff clean.

---

## Entry 072 — medstaff stage 7: the grades, on a forced-rank curve

**Date:** 2026-08-03

**Prompt (full text):**

> Let's produce the grades. Top 3 are A, next 5 are B, next 8 are C, next 8 are D, next 8 are F.
> After running these reports for a few years, we'll re-calibrate so that a certain score yields
> a certain grade, rather than the tiered curve I established.

**Response notes:**
- **Final stage.** `src/medstaff/grades/` (compose + staff), `scripts/medstaff_grades.py`,
  `REPORT_medstaff_grades.md`. 22 tests (**389 repo-wide**), ruff clean. Runs in ~2s.
- **User-specified forced-rank quota: 3 A · 5 B · 8 C · 8 D · 8 F** — sums to exactly 32.
  Implemented as `LETTER_QUOTAS`, with the **recalibration path built in from the start**:
  `ABSOLUTE_BANDS` + `assign_letters(bands=...)` already accept an absolute score-to-grade table,
  so the future switch is a parameter, not a rewrite. Tested both paths.
- **Flagged, not argued:** a forced curve assigns three A's and eight F's to *any* input including
  noise — it orders, it does not test. There is a test documenting exactly that. The user had
  already anticipated it by asking for absolute thresholds later.
- **The two separation numbers, which must be quoted together:**
  **0 of 32** clubs are separable from the club immediately below them (adjacent-pair separation
  is a strict test in a 32-club field), but **17 of 32** are separable from the **league average**.
  So the extremes are real relative to the mean; the fine ordering is not. Added
  `separated_from_average` after the adjacent-pair statistic came back empty and uninformative.
- **17 of 32 get the same letter on the 3-year window** — the most direct stability check on the
  board itself, and consistent with stage 6 finding most signal is period-specific.
- **Weights ∝ measured split-half reliability** (self-limiting: a component with no signal gets
  ~0 weight without anyone deciding it should). Reported beside the **attributability prior**,
  which orders the components **almost exactly the other way round** — recurrence is most
  attributable and least reliable, duration least attributable and most reliable. No way to
  satisfy both; that tension is the honest content of the weights table.
- **Board (5yr, 2021–25):** A = LA, CHI, CIN · F = BUF, DAL, GB, NE, MIA, SEA, KC, CLE.
- **Report opens with stage 6's verdict**, not the letters: this grades a **team availability
  system**, and stage 6 narrowed it further to **availability-management policy, not medicine**
  (only duration and returns-at-all persist; recurrence does not).
- Position-group cells shrunk toward their own club's value and **never ranked** — they ship in
  the parquet rather than as a report table so they are not read as an ordering.
- **No raw rate reaches the board** (tested) — the direct analogue of the sibling project's
  no-raw-rate test.
- `--staff-table` hook works and canonicalises relocated codes; without it the report says plainly
  it is grading franchises, since no free trainer-by-season source exists.
- **PROJECT COMPLETE: all 7 stages done.**

---

## Entry 073 — bye weeks were splitting every long absence; hamstring split out

**Date:** 2026-08-03

**Prompt (full text):**

> Interesting that SF and NO ranked to high. They were the two teams that inspired the analysis.
>
> Are we overlooking players who miss whole seasons?
>
> Are we looking at hamstring injuries as another specific category? Or is that baked in somewhere else
>
> fix the bug and split hamstring out

**Response notes:**
- **⚠️⚠️ MAJOR BUG, surfaced by the user's whole-season question.** `rosters_weekly` **omits bye
  weeks**; the episode builder read any missing week as off-roster, so one 18-week IR spell became
  two episodes (1–13, 15–18) with the first censored `off_roster`. **12,904 of 17,833
  player-seasons** had an interior missing week, **96% entirely the club's bye**. Symptom: **max
  `games_missed` in the whole dataset was 13** and **no episode reached 15 weeks**, despite 696
  player-seasons with ZERO available weeks. Nothing errored.
- **Fix:** `_fill_interior_weeks` materialises every interior missing week and labels it by whether
  the club actually played — no game ⇒ `BYE` (spell continues), game played ⇒ `OFF_ROSTER` (spell
  censors). The old `week - prev_week > 1` gap rule is gone. Genuine releases still censor.
- **Impact:** episodes 14,420 → **12,280**; max games missed 13 → **17**; episodes ≥14 games
  0 → **903**; `off_roster` censoring 3,051 → **95**; `returns_at_all` base 0.545 → **0.704**.
- **⚠️ IT REVERSED STAGE 5's HEADLINE.** Concussion off/def concordance **+0.437 (p .012) /
  +0.445 (p .011)** → **+0.238 (p .190) / +0.340 (p .057)** — the preregistered primary hypothesis
  **no longer holds** in either spec. The earlier result was substantially an artifact of
  bye-splitting inflating body-part shares club-dependently.
- **Stage 6 still passes 2 of 5 but a DIFFERENT two:** incidence now passes (0.722 / 0.348 /
  0.004), duration still does (0.807 / 0.518 / 0.001), **`returns_at_all` dropped out** (temporal
  0.397 → **0.016**), and **recurrence collapsed further** (split-half 0.414 → 0.186, temporal
  0.078 → **−0.194**, perm 0.058 → 0.110). The most attributable component is now null on
  everything.
- **Board moved:** SF C/16th → **D/18th**; NO B/4th → **B/5th**. A = LA, CHI, ATL.
- **Hamstring split into its own group** — it was **56%** of `soft_tissue_lower` (2,210/3,980 rows)
  and invisible. Kept **out of `FOCAL_GROUPS`** deliberately: the five were preregistered, and
  adding a sixth post-hoc is the fishing preregistration exists to prevent. **Measured hamstring
  recurrence 8.3%** (751 returns) — mid-pack, below knee 14.4% and shoulder 12.0%, and
  indistinguishable from the rest of soft_tissue_lower (8.3%). Worth recording *because* the
  literature predicts it should stand out.
- **On the SF/NO question that started this:** the board is not purely an adjustment artifact —
  Spearman(raw-burden rank, grade rank) = **−0.554**, so it substantially tracks games actually
  lost. NO remains the genuine outlier (14th in raw burden, 5th in grade) because it had **fewer
  onsets than expected** on the league's highest exposure — few injuries, long ones.
- **On "shouldn't this have surfaced sooner?" — yes, and the failure is nameable.** (a) Only
  *central tendency* was checked (mean 3.3 games/spell looked fine) while the **maximum** never
  was; 13 games in a 17-game season is self-evidently wrong. (b) The evidence was **printed and
  read past**: `off_roster` censoring at **21%** was in the stage-2 report. (c) The bye test
  **passed** because its fixture supplied an explicit bye row — the defect was that real data has
  none, so the fixture encoded the wrong assumption instead of testing it. Synthetic fixtures
  verify logic given an assumed shape; they cannot catch a wrong belief about the shape.
  **Remediation:** `sanity_checks()` now runs on real data at the end of stage 2 (longest spell,
  off_roster share, interior panel holes), each with a test asserting it fires on the broken
  shape.
- 10 new tests (**401 repo-wide**), ruff clean. Full pipeline rerun, stages 1→7.

---

## Entry 074 — end-of-season rest: verified excluded, and noted

**Date:** 2026-08-04

**Prompt (full text):**

> How does this analysis handle end of season rest? Saquon famously rested when he could have
> broken a rushing record due to the increased number of games.
>
> note that in the plan and prompt log

**Response notes:**
- **Why it matters:** resting starters is **systematically tied to team quality** — good clubs
  clinch early and rest more — so counting it as absence would penalise exactly the clubs least
  likely to have an availability problem. Not a random error; a directional one.
- **It is excluded**, because `is_non_injury` is checked **first, on the raw string**, before any
  body-part matching. 2021–25: **2,634 rest / coach's-decision rows, all `non_injury`**, and
  **56 of them carry an injury-looking `report_status`** (Out/Doubtful/Questionable) that would
  have been miscounted by a designation-keyed pipeline. Week 18 is the most common week (198).
- **Worked case, Saquon Barkley 2024:** weeks 14/16/17 *"Not injury related - resting player"*;
  **week 18 *"Not injury related - coach's decision"* while listed Doubtful / Limited practice**.
  Both `report_status` and `practice_status` would have flagged him as injured. **Episodes
  charged: 0.**
- **Three checks beyond the single case:** `out_other` share flat across wk16/17/18
  (0.068/0.067/0.065); episode onsets in wk18 **lower** than wk15–17 (571 vs 614–639); and the
  decisive one — **team incidence residual with vs without week 18 correlates +0.991**. Also
  Spearman(late-season rest weeks, incidence residual) = **−0.282 (p 0.118)**, n.s. and protective
  if anything, consistent with resting teams being good teams.
- **Residual limitations recorded, not hidden:** a player simultaneously nursing something real
  *and* rested is listed with a genuine body part and IS counted — unseparable in this source, and
  it concentrates in playoff teams; and rest weeks inside an already-open spell count toward
  `games_missed`. Neither moves the board; the first would matter more if incidence were weighted
  higher.
- **The lesson, and its symmetry with Entry 073:** both turn on which field to trust.
  `report_status`/`practice_status` would have miscounted the rested player; `rosters_weekly`
  week-coverage miscounted whole-season absences. The body-part text was right in one case and the
  roster wrong in the other — which is the argument for `sanity_checks()` running against **real
  data** rather than against assumptions about it.
- Documented as `MEDSTAFF_PLAN.md` §2.9. No code change — this was verification, and it passed.

---

## Entry 075 — reports were interpolating numbers but hardcoding interpretation

**Date:** 2026-08-04

**Prompt (full text):**

> Make sure the reports for this are up to date with the changes we made, then make a new PR

**Response notes:**
- **Found a systematic defect while checking, not a stale-file problem.** The reports are f-string
  markdown specifically so "the prose cannot drift from the tables" — but they interpolated the
  **numbers** while **hardcoding the interpretation**. After the Entry 073 bug fix the numbers
  updated and the claims did not.
- **Six false or self-contradictory claims across three reports**, each sitting next to a correct
  number:
  - signature: "The primary hypothesis, concussion, **holds**" (p 0.190 / 0.057 — it does not);
    "it is the **strongest part in both**" (ankle leads composition 0.389 vs 0.238); "the number
    **barely moved** when the denominator was fixed" (0.405 → 0.238); `"0 parts clear ... ()"` —
    a dangling empty list; "Back is the **only** other part positive in both specs" (ankle is too)
  - reliability: "The two components that pass are **duration** and **returns-at-all**" — after the
    fix returns-at-all fails temporal (0.016); the pair is **incidence and duration**. Also
    "returns-at-all is **close behind**".
  - grades: the same wrong pair, in the section that frames the entire board.
- **Fix: every interpretive claim is now derived**, never asserted — which hypothesis holds
  (three-way: holds / does not hold / equivocal), which part leads each specification, which
  components pass the gate, how far a result moved against its preregistration, and a
  `stability_reading` that branches on whether recurrence is among the passing set. If a claim
  cannot be computed from the frame, it does not belong in the prose. Recorded in
  `MEDSTAFF_PLAN.md` §7.
- **All seven reports regenerated** from current data. Verified by grepping the rendered markdown
  for the old claim strings — clean.
- No result changed; this was a reporting-truth defect, not an analysis one. 401 tests, ruff clean.

---

## Entry 076 — redraft boards per league: half-PPR, 2QB, and Underdog bestball

**Date:** 2026-08-04

**Prompt (full text):**

> Let's augment the redraft report. So far it's only PPR. Let's add half ppr. It's also only 1qb.
> Let's make it so we can add a roster config as well. I currently am in a 2qb league. I also want
> to explore paid bestball leagues on Underdog, which are 3wr instead of 2wr.

**Response notes:**
- **Scoring turned out not to be a display setting.** `ppg` *is* the supervised target
  (`data/build.py` -> `target_ppg_next`), so half-PPR needs its own dataset -> features -> fit; it
  cannot be rescaled from a PPR projection (that would need projected receptions, which the model
  doesn't output). Cheap part: nflverse already caches `fantasy_points`, `fantasy_points_ppr` and
  `receptions`, and the two totals differ by **exactly** one point per reception - verified
  row-wise over all 476,156 weekly rows 1999-2025 (max deviation 3.6e-15, zero nulls) - so every
  format is an exact recomputation with no new data.
- **PPR is bit-identical to before.** New `scoring.py` + `utils/naming.artifact_stem` namespace
  artifacts by format, PPR keeping the bare `football_rb` stem. First run showed 4 players off by
  0.01: summing `(points + receptions)` per season differs from summing nflverse's per-week PPR
  total in the last ULP, enough to flip a 2-decimal rounding boundary. Fixed by passing the PPR
  total straight through instead of re-deriving it -> re-verified **0.00 max difference across all
  136 baseline players**, ranks identical.
- **Leagues are committed configs** (`config/leagues/{ppr_1qb,my_2qb,underdog_bestball}.yaml`).
  `run_redraft` loops on *scoring format*, not league, so the three shipped leagues cost two
  pipeline passes (~35 s), and redraft finally has a cross-position VORP board (it had none -
  `proj_pos_rank` was within-position only). Generalised `keeper.replacement_levels` to take a
  full roster incl. QB plus `flex_positions`, which is what Underdog's TE-eligible flex needs;
  the `--format 1qb|sf|2qb` shorthand still works.
- **Board depth had to become league-derived.** A fixed top-20 QB list is meaningless in a
  10-team 2QB league where 20 QBs are *starters*; depth is now teams x slots x 1.75, floored at
  the old defaults, then scaled so the positions together cover every pick (an 18-round Underdog
  draft spends 216 picks - the first cut left the last three rounds unguided).
- **Best ball: NULL RESULT, and that's the deliverable.** Defined realized best-ball value as
  `max(0, weekly points - that week's marginal-starter score)` - downside truncated, upside
  counted, computed straight from box scores with no roster simulation. Fitting
  `proj_ppg + lambda*sigma` against it over 13 walk-forward season pairs gives **lambda = 0
  everywhere**: gains <= +0.003 Spearman, p = 0.99/0.10/0.13/0.19 (QB/RB/WR/TE). Reproduced with
  ceiling-week rate and weekly skew, several optimising at exactly 0. Mechanism: weekly sigma is
  ~0.90 rank-correlated with weekly mean at RB/WR/TE. Practical read: **rank a bestball draft by
  projected PPG; don't pay up for spike-week reputations.** Kept the machinery +
  `--bestball-lambda` override so it can be re-tested when the model changes.
- Boards behave as predicted: half-PPR costs WRs 3.02 PPG on average, TEs 2.20, RBs 1.75, QBs
  0.00; 2QB lifts the top QB from board slot 17 -> 10 (QBs in the top 24: 2 -> 8); Underdog's 3WR
  + TE-flex drops RB/WR replacement from 11.5/11.4 to 8.9/9.1.
- ECR/ADP benchmarks stay PPR-1QB-locked (no free half-PPR or 2QB source), and per the standing
  rule they remain benchmark-only - never blended into a model number.
- 464 tests (was 401 at Entry 075 -> +63), ruff clean.

---

## Entry 077 — regenerate the redraft reports per league

**Date:** 2026-08-09

**Prompt (full text):**

> regenerate the report, make it generate one per league

**Response notes:**
- Ambiguous target, resolved by inspection: `REPORT_football_*.md` is a *model-quality* report
  (walk-forward CV, market head-to-head), and a league doesn't change model quality — per-scoring
  would be the meaningful axis there, not per-league. So "the report" = the redraft board, which
  already writes per-league files; the actionable part was regenerating and clearing the stale
  single-board `reports/redraft_2026.csv` (git-ignored, untracked, superseded).
- **Found a real bug doing it.** Diffing the fresh run against the committed one showed tied
  players swapping places. VORP is rounded to 2dp so ties are common (~1 pair per 20 players), and
  `sort_values("vorp")` uses pandas' non-stable quicksort — identical inputs produced different
  `proj_overall_rank`. Ties now break on projection, then `player_id`. Verified two back-to-back
  runs are byte-identical for all three leagues.
- Underdog moved 198 -> 197 players: the rookie count is read live from the ECR board and the
  market shifted one WR rookie into the top-N. Projections unchanged.

---

## Entry 078 — could the keeper tool infer a league from its input?

**Date:** 2026-08-09

**Prompt (full text):**

> What about the keeper tool? Would I need to specify a league or would it be able to infer from
> the roster structure of the input?

**Response notes:**
- Answered: you must specify. The keeper input is `player,pick` — your keeper *candidates*, five
  rows in the example — which carries no team count, no starter slots, and no scoring. Pick
  numbers weakly bound league size at best, and keeper leagues distort draft slots by construction
  (kept players vacate their picks). Even a full-roster export would yield teams/roster_size/slots
  but never scoring.
- Surfaced the real gap the question exposed: keeper couldn't take a league at all, so it was
  pinned to `DEFAULT_ROSTER` and — worse — full-PPR projections, i.e. a confident board in the
  wrong currency for both of the user's half-PPR leagues. Wired up in Entry 080.

---

## Entry 079 — TE is flex-eligible in every league

**Date:** 2026-08-09

**Prompt (full text):**

> Is FLEX not TE eligible right now? It should be, all of my leagues accept TE for flex

**Response notes:**
- My error: I asked about Underdog's flex and confirmed TE there, then assumed "standard" RB/WR
  for `ppr_1qb` and `my_2qb` without checking. Corrected all three configs and flipped the code
  defaults (`keeper.FLEX_POS`, `LeagueConfig.flex_positions`, `redraft.DEFAULT_LEAGUE`) so the
  keeper tool and any new league YAML are TE-eligible unless they opt out.
- **Honest impact: nil.** No replacement level and no VORP changed in any league. Every team
  already starts a dedicated TE, so the flex contest is TE13+ vs RB25+/WR25+, and the TE pool
  falls off much faster — marginal TE ~9.8/8.25/8.00 PPG vs marginal flex RB/WR ~11.5/10.7/10.3.
  No TE is ever good enough to take a flex slot. It would only bite with no dedicated TE slot, or
  a materially deeper TE pool.
- The one visible effect is depth bookkeeping: the flex share splits three ways instead of two, so
  the PPR board lists 30 TEs instead of 26 (RB 54->52, WR 76->75). Projections byte-identical.

---

## Entry 080 — keeper tool wired to league configs

**Date:** 2026-08-09

**Prompt (full text):**

> yes, wire the keeper tool up to leagues

**Response notes:**
- `scripts/keeper.py` takes `--league config/leagues/*.yaml`, which sets both halves of the
  valuation: scoring (the projections are retrained against that format's target, not rescaled)
  and roster shape (replacement level). `--teams`/`--format` still work alone but are **rejected**
  alongside `--league` — silently half-overriding a league config is how you get a confident board
  for the wrong league.
- `project_positions(..., scoring=...)` re-points the per-position configs, and `ensure_features`
  builds a format's dataset/features on first use so keeper is self-sufficient on a fresh checkout.
- Added `Config.with_overrides(dotted_dict)` and rewrote `_override_season`/`_override_scoring` on
  top of it. This wasn't just tidying: `projection.py` needed the override and importing it from
  `eval.redraft` would have been circular (redraft imports `project_position`).
- **It changes real advice.** Same keeper list, PPR-1QB vs the 10-team 2QB half-PPR league:
  Jayden Daniels -2 (don't keep) -> +45 (keep); Brian Thomas Jr. +10 (keep) -> -5 (don't keep);
  Bo Nix +56 -> +89. Two of eight recommendations flip. QBs gain (20 started vs 12, replacement
  15.5 -> 13.7 PPG); receivers lose a point per reception.
- 475 tests, ruff clean.

---

## Entry 081 — markdown report for the keeper tool

**Date:** 2026-08-09

**Prompt (full text):**

> Add markdown for keeper

**Response notes:**
- `eval/keeper.py` gains `KeeperResult` + `render_markdown`, following the
  postseason/handcuff/redraft pattern, so keeper output is viewable on GitHub instead of living
  only in a CSV and terminal scrollback. Writes `reports/keeper_board[_<league>].md` next to the
  CSV (the `.md` is committed; `.gitignore` only excludes `keeper_board*.csv`).
- Report content is the decision, not a data dump: a **Verdict — keep N of M** table with an
  explicit KEEP/pass marker per player, the pick expressed in **rounds** as well as pick number
  (keeper costs get argued about in rounds), the replacement-level table that explains why a QB
  and a WR are comparable at all, and an Unscored section so unmatched names are never silently
  dropped.
- Added `slot_summary(roster)` mirroring `LeagueConfig.slot_summary` so a board run from the
  `--teams`/`--format` shorthand still states the roster shape it assumed — the markdown is a
  shareable artifact and shouldn't hide its own assumptions. Fractional superflex prints as
  `1.7QB` rather than rounding into a lie.
- 481 tests (+6), ruff clean.

---

## Entry 082 — handcuff tool adjustable to half-PPR

**Date:** 2026-08-09

**Prompt (full text):**

> Fixup handcuff.py to be adjustable to half ppr

**Response notes:**
- Mostly a CLI lever: `_board_with_risk` already routed through `artifact_stem(config)` and
  `project_position(config, ...)` after the Stage-2 namespacing work, so both were already
  scoring-aware. Added `--scoring {ppr,half_ppr,standard}` and `--league` (which contributes only
  its scoring), `ensure_features` so a format's artifacts build on first use, `scoring` carried on
  `HandcuffResult`/`InjuryRiskResult` and printed in both markdown headers, and a filename suffix
  for non-PPR formats so boards don't clobber (PPR keeps the historical name).
- **Handcuff takes a scoring format, not a league** — deliberately. It compares a starter to his
  own backup, so roster shape and replacement level never enter the calculation.
- **I wrote a wrong claim and the data caught it.** The first draft of the docstring said scoring
  "only shifts *how much* upside a handcuff carries, not who the handcuff is — the starter/backup
  pairing is a depth-chart fact." False: the handcuff is whichever backup *projects* highest, so
  halving reception value can hand the role to a different player. On the 2026 board it changes
  the identified handcuff for 2 of 27 starters — James Cook's moves from pass-catching Ty Johnson
  to Ray Davis, Derrick Henry's from Justice Hill to Keaton Mitchell. Corrected in the module and
  script docstrings and documented in USAGE.
- Other measured effects: 19 of 25 handcuff ranks move (max 4 slots), and mean contingent upside
  falls 1.29 -> 1.14 PPG because receptions are worth less, which compresses the starter-backup
  gap the whole tool is built on.
- Housekeeping surfaced by the run: the committed RB reports were named `handcuff_2026.*` while
  the code has been emitting `handcuff_rb_2026.*` — stale artifacts from an older naming scheme.
  Replaced them. Also added the missing handcuff section to docs/USAGE.md (it was absent entirely).
- 483 tests (+2), ruff clean.

---

## Entry 083 — basketball port and football focal additions: both blocked on data, recorded

**Date:** 2026-08-04 *(authored before entries 076–082; renumbered from 076 to 083 on merge,
so the number ordering is merge order, not authoring order)*

**Prompt (full text):**

> Merged. Now let's run this same analysis for basketball. Let's include elbow, achilles, and acl
> as they are all scheme related
>
> Those injuries should be added to the football list from earlier, they feel more scheme related
> in basketball
>
> ESPN has injury history for players

**Response notes:**
- **Both requests blocked by data availability, not method.** Investigated read-only before writing
  any code; outcome is two documentation commits and no pipeline.
- **⚠️ FOOTBALL: ACL is not in the NFL injury report at all.** 2021–25 rows: `elbow` 437,
  `achilles` 176, **`ACL` 0, `MCL`/`PCL`/`LCL` 0, `torn` 0**, `knee` 4,442. Clubs report
  **anatomy, not diagnosis** — ceiling is "Knee". `taxonomy.py`'s `\bacl\b` rule has **never
  fired**. The irony: ACL is the most surface-implicated injury in the literature and is exactly
  what this source cannot see.
- **Elbow and achilles are below the power floor** (`MIN_EPISODES_PER_CELL=15`): elbow 223
  episodes / median **3** per club×side, achilles 66 / **1**, against hip — the weakest part
  already focal — at 453 / **6**.
- **A proxy exists and was deliberately NOT adopted:** `knee AND never_returned` = 481 episodes,
  ~7 per cell. Recorded with its numbers *and* the argument against it — the focal five were
  preregistered, and adding parts post-hoc is what preregistration guards against (same reason
  hamstring was split out in §2.8 but kept non-focal). Noted that if ever adopted it must be an
  **overlay**, not a taxonomy group: carving it out of `knee` would remove 26% of knee's episodes
  and silently change the published round-1 result.
  **STANDING DECISION recorded: focal set unchanged.**
- **⚠️ BASKETBALL — I was WRONG about ESPN and the user was right to push back.** ESPN injury
  records are **richer than the NFL report**: `details.type` (Knee/Ankle/Hamstring/**Achilles**/…),
  `details.detail` (**Surgery/Sprain/Strain/Fracture** — an actual *diagnosis* field the NFL data
  never had), `side`, `returnDate`, plus free text. And **`type = "Rest"` is its own category (19%
  of records)** — a genuine load-management separator.
- **But there is NO history**, which is what kills it. `athletes/{id}/injuries` returns **404 for
  every athlete tested, including players in the current snapshot**, while sibling routes
  (`/eventlog`, `/statisticslog`, `/awards`) return 200 — so the resource simply does not exist.
  `espn_nba_injuries()` is a live snapshot (148 records).
- **Every other route closed:** `nba_api` **zero** injury endpoints across 274; box-score
  `COMMENT` has 4 values with no body part (**92% "Coach's Decision"**); NBA official report PDF
  **403**; Pro Sports Transactions **403 behind a Cloudflare challenge — even on robots.txt**.
  Neither was circumvented; a challenge page is an access control, not a crawl directive.
- **Community PST datasets stop ~2020** — the football-equivalent window is 2021–25, so they end
  exactly where we would need to start, and postdate the load-management era's onset.
- **Deliverable:** `sports/basketball/docs/INJURY_DATA_FEASIBILITY.md` + a PROJECT_PLAN §7 pointer.
  **Recommendation: do not build it now.** The one action with a deadline is archiving ESPN's
  snapshot daily — ~3 seasons before a 3-year window exists, and every day not archiving is
  permanently missing.
- Docs only: no code, no re-run, no report or parquet changed. 401 tests, ruff clean.

## Entry 089 — preseason-end refresh, and a fourth league

**Date:** 2026-08-28

**Prompt (full text):**

> For football, make sure all reports are current with the latest data as pre-season ends. I also
> added a new league.

(mid-run, after the league config was edited by hand:)

> Resume previous work, had to update the configs.

**Response notes:**
- **The new league config was a half-finished copy of `my_2qb.yaml`.** The values had been edited
  but the prose had not, so the file contradicted itself four ways: `name: my_2qb` (which
  `load_leagues` rejects as a duplicate — report files would collide), `label: "10-team 2QB
  half-PPR"` against `teams: 14`, `scoring: ppr` against a `# 0.5 pt / reception` comment, and
  `QB: 1` against a `# true 2QB, not superflex` comment. Filename and every edited *value* pointed
  the same way (14-team 1QB PPR), so only **scoring** was genuinely ambiguous — asked, and the
  answer was full PPR. Rewrote `name`, `label` and all the prose to match the values.
- **Registering a league is three edits, not one.** `scripts/redraft.py` carries a hardcoded
  `DEFAULT_LEAGUES`, so dropping a YAML into `config/leagues/` gets it loaded by nobody. Added it
  there, to `SHIPPED` in `tests/test_league.py`, and to the shipped list in `docs/USAGE.md`.
- **`make fetch` silently regressed the Entry 088 roster fix.** `fetch_data.py` takes its seasons
  from `cfg.seasons()` (1999–2025), so an `--overwrite` refresh **truncated `rosters` from 2026
  back to 2025** — re-breaking the `team_next` join that Entry 088 had just fixed, and in exactly
  the way that fails silently. `redraft.py` is unaffected because it calls `fetch_all` with its own
  `range(earliest, draft_season + 1)`. Restored by calling `fetch_all` directly through 2026
  (rosters 66,480 → 69,410 rows; the 2026 draft class, 257 picks, came back with it).
- **The refresh moved next-season context, not the training set.** All four positions' current
  feature parquets match the row counts recorded in their experiment summaries exactly (QB 1996,
  RB 4464, WR 5400, TE 3027, seasons 1999–2025), so `REPORT_football_*.md` are still faithful to
  the data and re-running the experiments would only churn the v4 snapshots.
- **⚠️ Regenerating `postseason_2025.md` silently swapped the ADP benchmark from FantasyPros to
  FFC.** `data/external/adp_*.parquet` is git-ignored (`.gitignore:14`), so it is a *local cache*,
  not a committed reference — and `postseason.py` calls `build_adp_benchmark(..., write=True)`
  regardless of `--no-refresh`. The June report was graded against a FantasyPros cache that no
  longer exists. Coverage and correlation both fell: QB 40→29 names (Spearman .705→**.506**),
  RB 85→63 (.755→.735), WR 106→83 (.720→.695); TE gained its first ADP row (26 names, .561).
  **Left as-is and flagged rather than patched** — this is the deferred FantasyPros-vs-FFC source
  decision, ADP is benchmark-only and never blended, and no draft board reads it. The model rows
  in the same report genuinely improved (QB .708→**.746**) from the Entry 088 per-position model
  fix, so reverting the file would have thrown that away too.
- **TE-flex is still inert, now at three flex slots.** The new league starts 1RB/2WR/1TE/**3FLEX**
  in a 14-team league — the deepest flex demand of any shipped league — and TE still finishes with
  14 started league-wide, i.e. the dedicated slots only. The 28→42 extra flex slots went entirely
  to RB (31 started) and WR (53). The USAGE note that TE-flex never bites survives the test.

## Entry 090 — why the 2QB board doesn't look like an ADP chart

**Date:** 2026-08-29

**Prompt (full text):**

> Let's talk about the 2qb draft board. This looks very different from an ADP chart right now. Why?

**Response notes:**

Diagnostic only — no code or report changed. Measured `reports/redraft_2026_my_2qb.csv` (160
rows) against the live DynastyProcess ECR (`db_fpecr.parquet`, latest 2026 scrape 2026-08-28).

- **Most of the apparent gap is benchmark mismatch, not model behaviour.** Public ADP/ECR charts
  are single-QB (`ecr_type == "ro"`). Against `ro` the board is Spearman **.696** overall and has
  **19 QBs in its top 60 vs the market's 2**. Against redraft **superflex** (`rsf`, the right
  comparison for a 2QB league) it is **.773**, with **19 QBs in the board top 60 vs 20** in ECR.
  The QB density is right; we were reading it against the wrong chart.
- `rsf` exists in the same ECR file we already download and is a **better benchmark for
  `my_2qb`/`suz_1qb`-style superflex shapes than `ro`**. `data/benchmark.py` hard-codes
  `REDRAFT_OVERALL = "ro"`; a league-aware benchmark type is a candidate follow-up (still
  benchmark-only, never blended).
- **Non-QB divergence is genuine and concentrated at WR** (Spearman .766 ex-QB; by position
  RB .909, TE .799, WR .746). Three mechanisms, all confirmed in the feature frames:
  1. **Season-total shares punish injured players.** `target_share = targets / team_targets` over
     full seasons (`features/build.py:83`), so a hurt star reads as a low-usage receiver: Nabers
     4 games → .070 share, Tyreek Hill 4 games → .063, Garrett Wilson 7 games → .125. They land
     at WR29/WR43/WR31 against ECR 45/—/53. Same shape for Bowers (12 g, TE2), Jayden Daniels
     (7 g, QB17), Burrow (8 g, QB23).
  2. **The projection is mostly last season, shrunk.** Spearman(proj_ppg, 2025 ppg) = QB .776,
     RB .921, WR .812, TE .796, with sd(proj)/sd(actual) of .85/.78/.75/.62. Ridge shrinkage
     compresses the top, so WR5–WR16 span 12.96→10.60 PPG — ordering inside that band is noise,
     while ECR keeps a talent prior. Hence Jefferson (9.38 ppg on a 30% target share in 17 games)
     at WR11 vs ECR 17, and A.J. Brown / CeeDee Lamb / DeVonta Smith similarly faded.
  3. **No role or depth-chart knowledge for 2026.** The target is PPG, so a backup's few starts
     score as a rate. The largest model-over-market gaps are all bench bodies: Ertz (board 113 vs
     ECR 674), Mariota (134/644), Fields (153/630), Mac Jones (147/593), Cousins (146/499).
     `retired_next` sums to **0** across all four 2025 feature frames, so retirees stay on the
     board too.
- **The board is also structurally short by ~23 picks.** Rookies are subtracted as counts, never
  inserted: QB 35−1, RB 50−5, WR 75−16, TE 24−1. 18 of ECR's top 60 are absent from the board's
  top 60, and the rookie-shaped holes are the most visible part of that.
- Offseason block is **live**, not degenerate (2025: 44 WR / 27 RB / 18 QB / 25 TE
  `changed_team_next`), so the PR #33 N+1 roster join is working.
- Stale local cache noted: `data/external/market_football_*.parquet` still tops out at 2025. The
  redraft path fetches ECR live, so the boards are fine; only the committed benchmark cache lags.

## Entry 091 — which chart the 2QB board should be graded against

**Date:** 2026-08-29

**Prompt (full text):**

> What chart comparison should I be doing?

**Response notes:**

Follow-up to Entry 090. Diagnostic only — no code or report changed.

- **League shape lives entirely in cross-position placement, not within-position order.** Comparing
  the 2026 `ro` (1QB PPR) and `rsf` (PPR superflex) ECR boards head to head: positional-rank
  Spearman is **QB .991, RB .9991, WR .9973, TE .9972**, while overall-rank Spearman is .9545 and
  the median QB's overall rank moves **162 → 76**. So there are two separate comparisons, not one:
  1. *Is the board shaped right for the league?* → overall/cross-position placement → **must** use
     `rsf`; `ro` is what made the board look alien in Entry 090.
  2. *Is the model any good at ranking players?* → within-position rank → chart type is nearly
     irrelevant, so the committed `ro` cache is fine. Current pos-rank agreement vs `rsf`:
     RB .911, TE .799, QB .790, **WR .747** — WR remains the weak position.
- **Two mismatches that can't be closed from this source, and shouldn't be read as model error.**
  (a) Superflex ≠ true 2QB — superflex allows punting QB2 into the flex, so genuine 2QB demand is
  *higher* than `rsf`; the board should sit slightly ahead of `rsf` on QBs, not level with it.
  (b) Every FantasyPros page in `db_fpecr.parquet` is full PPR (`ro` = `ppr-cheatsheets.php`,
  `rsf` = `ppr-superflex-cheatsheets.php`); there is **no half-PPR ECR**, so a half-PPR board will
  structurally favour TD/deep profiles over volume slot WRs versus any chart here.
- **Rejected: grading against the expert band.** ECR ships `sd`/`best`/`worst`, and comparing to the
  band rather than the point estimate looked like the principled move. It is empirically useless —
  **all 157** matched board players fall outside 2 sd (median sd 15.6 ranks overall, **7.7** in the
  top 60). `sd` measures how tightly the experts agree with each other, not forecast uncertainty,
  and it is far narrower than real year-to-year outcome spread. Do not build a diagnostic on it.
- **The comparison that actually settles anything is neither.** Rank correlation against a chart
  measures *conformity*, not accuracy. `make postseason SEASON=Y` already grades model / ECR / ADP
  against the realised finish (`eval/postseason.py`), which is the only framing consistent with the
  two-independent-opinions rule — it treats market and model as competing forecasts rather than
  making one the target.
- **The ADP leg can't do 2QB.** `data/external/adp_football_*.parquet` is
  `[player_id, season, adp, adp_pos_rank]` with no format/scoring column, so ECR `rsf` is the only
  2QB-shaped benchmark available. Also still capped at 2025, per Entry 089.
- Follow-up candidate (unchanged from Entry 090): `data/benchmark.py:19` hard-codes
  `REDRAFT_OVERALL = "ro"`; a league-aware benchmark type would let superflex leagues pull `rsf`.

## Entry 092 — league-aware market benchmark: 2QB pulls the superflex board

**Date:** 2026-08-29

**Prompt (full text):**

> make the benchmark league-aware so 2qb pulls rsf

**Response notes:**

Acts on the follow-up candidate from Entries 090/091. 567 tests pass (was 539), ruff clean.

**What changed**

- `eval/league.py` — `LeagueConfig.qb_starters` (dedicated QB slots + a QB-eligible flex) and
  `is_superflex` (`qb_starters >= 2`). Covers true 2QB *and* superflex, which the market treats
  as one shape.
- `data/benchmark.py` — added `REDRAFT_SUPERFLEX = "rsf"`, `ECR_TYPES`, `ECR_TYPE_LABELS`,
  `resolve_ecr_type` (rejects typos: an unknown type would silently empty the filter and surface
  much later as "no preseason scrapes"), `ecr_type_for_league`, and `benchmark_path`.
  `build_market_benchmark` takes `ecr_type` (falling back to `benchmark.ecr_type` in the config,
  then `ro`). **The default board keeps its un-suffixed filename** so every committed `market_*`
  table and its readers stay valid; other boards write `market_<sport>_<pos>_<type>.parquet`.
- `eval/redraft.py` — `_rookie_market_context(season, ecr_type=…)`; `run_redraft` builds one
  context per distinct board (not per league) and keys the pipeline pass on
  **(scoring, market board)** rather than scoring alone. `LeagueBoard.ecr_type` records the board
  and the report header names it ("Board: **superflex** ECR"). `rookie_note` now prefixes each
  note with its board.
- `scripts/fetch_benchmark.py` — `--league` (derive the board from a league YAML) and
  `--ecr-type` (name it directly); `Makefile` gains `LEAGUE=…` on the `benchmark` target.
- `docs/USAGE.md` — a "Which market board" subsection under `redraft.py`, plus the two mismatches
  that cannot be closed from this source.

**The chart swap itself changed nothing — and that is the measured result, not an assumption.**
At every board depth these leagues use, `ro` and `rsf` return **identical** rookie counts
(QB 1, RB 3, WR 7, TE 0 at my_2qb's 35/50/75/24). Expected from Entry 091: the rookie counter
reads *within-position* rank, where the two charts agree at ~.99. The value of the change is
correctness and the plumbing for cross-position work, not a different board today.

**What did move my_2qb's numbers was the pass split — and it fixed a real bug.** Rookie counts
are estimated at the *pooled* depth of a pass, but the report prints the *league's own* depth
beside them. my_2qb previously shared the half_ppr pass with underdog_bestball and so subtracted
rookies counted at underdog's deeper board (RB 62, WR 93, TE 35) while printing its own
(RB 50, WR 75, TE 24) — reported as "top 75 − 16 rookies" when 16 was the count at 93. Splitting
the pass leaves my_2qb pooling alone, so its counts are now self-consistent:
RB 5→3, WR 16→7, TE 1→0. Board rows are unchanged (the 160-pick VORP cut binds first); the
redraft CSVs are git-ignored and only the four `.md` headers/counts differ.

**⚠️ The same bug still bites the remaining pooled pair.** `ppr_1qb` (WR 83) and `suz_1qb`
(WR 80) still share the `ppr/ro` pass, so suz_1qb subtracts rookies counted at 83 while its
report prints 80. The honest fix is to move the rookie subtraction out of `_project_scoring` and
into the per-league step (d), where each league's own depth is known — deliberately **not** done
here: it is outside this prompt and it would change suz_1qb's board content.

**Also noted**

- `rsf` history starts in **2021**; `ro` goes back to 2020. A superflex backtest therefore has one
  fewer season.
- The `rsf` benchmark tables have **no reader yet** — the redraft path fetches ECR live, and
  experiment/postseason grade per-position off the default `ro` files (correct: Entry 091 showed
  per-position grading is insensitive to the chart). `make benchmark LEAGUE=…` was smoke-tested
  and the orphan `market_football_qb_rsf.parquet` deleted rather than committed.
- The committed `market_football_*.parquet` tables still stop at 2025 — the 2026 preseason scrape
  (2026-08-28) is inside the window and would be picked up by a `make benchmark` refresh.

## Entry 093 — rookie trim moves from the pooled pass to the league

**Date:** 2026-08-29

**Prompt (full text):**

> fix the pooled rookie count for suz_1qb too

**Response notes:**

Fixes the bug flagged at the end of Entry 092. 569 tests pass (was 567), ruff clean, boards
regenerate idempotently.

**The bug.** `_project_scoring` estimated the rookie count once per pipeline pass, at the
**pooled** depth (the deepest board any league sharing that pass asked for), and then wrote it
onto every row. Each league later trimmed to its own depth, so the number a report *printed* was
measured at another league's depth. Two failure modes, both live:

- *Wrong number displayed.* `ppr_1qb` shares the `ppr/ro` pass with `suz_1qb`, whose TE board is
  deeper (52 vs 31). The count came from depth 52 → 5 rookies, but the trim to depth 31 discarded
  it, so the report read "**31** returning players (top 31 − **5** rookies)" — internally
  impossible, and the board carried a full 31 TEs.
- *Wrong number applied.* `suz_1qb`'s own RB/WR counts were taken at `ppr_1qb`'s deeper board
  (RB 55, WR 83), charging it for rookies the market ranks below its board.

**The fix.** `_project_scoring` now returns the projection at the pooled depth **before** any
rookie subtraction, plus `[(position, proj_season)]`. `run_redraft` step (d) computes each
league's rookie count at that league's own depth, trims to `depth − rookies`, and writes
`requested_top_n` / `rookies_subtracted` (both ride on the CSV) per league. Summaries are built
from the league's own numbers rather than the pass's.

**Board effects** — all 16 league×position combinations now reconcile (verified: board count
`<= depth − rookies`, with equality except where the total-picks cap binds):

| league | change |
|---|---|
| `ppr_1qb` | TE 31 → 30 on the board; report now "top 31 − 1 rookie" (was a false "− 5") |
| `suz_1qb` | RB 51 → 49, WR 73 → 72; total 196 → **193** |
| `my_2qb` | unchanged (it already pooled alone after Entry 092) |
| `underdog_bestball` | unchanged |

`my_2qb`'s RB/WR are the only rows where the total-picks cut still binds (board 43/59 against a
kept 47/68), which is the intended meaning of the summary — what the league actually kept.

**Tests.** `test_rookie_count_uses_each_leagues_own_depth` is the direct regression: two leagues
share a pass, the single rookie sits at ECR QB8, the shallow (top-5) league subtracts 0 and the
deep (top-12) league subtracts 1. `test_board_columns_record_the_leagues_own_trim` pins the two
CSV columns to per-league values.

Docs: `USAGE.md` gains the "top N means N board slots, counted at each league's own depth" note;
the `redraft.py` module docstring says the same where it explains the top-N rule.

## Entry 094 — the draft board becomes a pick order

**Date:** 2026-08-29

**Prompt (full text):**

> now regenerate all reports if necessary. if something says "draft board" i should be trying to
> draft in that order.

**Response notes:**

577 tests pass (was 569), ruff clean, all four boards regenerate idempotently.

**Regeneration — only the four redraft boards needed it.** Verified rather than assumed: the
keeper boards regenerate **byte-identical** (`keeper.py` builds its own board from
`project_positions` and never touches the rookie trim or the benchmark), and `handcuff.py` /
`postseason.py` are decoupled from every module touched in Entries 092-093 — postseason's only
import from `redraft` is `_override_season`.

**The defect.** The report is titled *Draft Board* but omitted rookies entirely, so row N was not
pick N. Measured against the league's own market board: 1 rookie inside the top 60, 3 inside the
top 120, **10 inside my_2qb's 160 picks** (Jeremiyah Love at 59, Carnell Tate 85, Jadarian Price
89, …). Accurate at the top, drifting up to 10 slots by the end of the draft.

Checked the keeper boards for the same flaw and **left them alone**: their slots shift by at most
2, and no keep/pass verdict moves (worst case Sam LaPorta −17 → −18).

**The fix** (user chose "insert rookies as placeholder rows" over retitling or a second section):
`market_rookies()` returns the rookies the market ranks inside the draft, with the slot each takes
— ranked over **modeled positions only**, since the `ro` scrape also carries IDP rows that are not
draftable. `insert_market_rookies()` splices them into the VORP-ordered board at those slots, cuts
to `total_picks`, and renumbers `proj_overall_rank` 1..N. Rookie rows carry no model columns
(`proj_ppg`/`vorp`/`proj_pos_rank` stay null) and are tagged `source == "market_rookie"`; the
report renders them italic with an em-dash projection and their market ECR.

**Never-blend is intact.** No model number is market-derived. The market already decided *how
many* rookies each position's top-N holds; this only adds *which ones* and *where*. The
per-position tables stay returning-players-only, as they are labelled.

**A second bug fell out of it.** The subtraction counted rookies inside a *position's depth* while
the insertion counted them inside the *pick limit* — two different windows, so boards came up
short of their own draft (`ppr_1qb` 190 of 192, `suz_1qb` 208 of 210). Restricting the subtraction
to rookies the market ranks inside `total_picks` makes both read one set, which guarantees the
board fills the draft: model rows = Σ(depth − rookies) and Σdepth ≥ total_picks, so
model + inserted ≥ total_picks always. All four boards are now exactly their draft length with
contiguous ranks: ppr_1qb 192, suz_1qb 210, my_2qb 160, underdog_bestball 216.

**⚠️ CSV schema changed** (`reports/redraft_*.csv`, git-ignored but read by the home-site
dashboard): two columns appended, `source` (`model` | `market_rookie`) and `market_ecr`. Every
existing column survives, but rookie rows carry **nulls in `proj_ppg` / `vorp` / `proj_pos_rank` /
`player_id`** — a consumer assuming those are numeric must filter `source == "model"` first.

**Also.** Per-position counts now report model rows only, so they fall where the 10 inserted
rookies pushed model players past the last pick (my_2qb QB 34→30, WR 59→54, RB 43→42). The
"N returning players (top D − R rookies)" line keeps its documented meaning: what the league
actually kept after the total-picks cut.

---

## Entry 095 — the per-position headers were lying, and why the board isn't ADP

**Date:** 2026-08-30

**Prompt (full text):**

> I inspected some of the reports and they don't add up, we need to investigate root cause and
> regenerate. Look at the 2qb league, it says top 35 with 1 rookie, but only 30 are shown.
> Similarly for RB and WR. Second, the draft board seems like it's very differenet from a current
> ADP board. I'd like an explanation of why this happened.
>
> [follow-up] Also, while it's possible for no rookie TE to break the top 24 it does seem unlikely

**Response notes:**

### Part 1 — the counts (two real bugs, both introduced by Entry 094)

`### QB — 30 returning players (top 35 − 1 rookies)` — 35 − 1 = 34, not 30. Two independent
causes, both in `eval/redraft.py`:

**(a) The subtraction and the insertion read different windows.** `estimate_rookie_count` counted
rookies inside the *position's* top-`depth[pos]` on the market board; `market_rookies` inserted
every rookie inside the *overall* top-`total_picks`. Entry 094's comment claimed restricting the
*names* made the two "read the same set" — it doesn't, because the windows differ. On deep boards
they diverge badly: suz_1qb has 14 rookie WRs inside its 210-pick board but only 8 inside its
WR top-80, so 6 rookies were spliced in that no position ever paid for. The surplus was amputated
off the tail — WR/RB over-served their allotment at TE's expense (suz TE showed 42 of 52).

Fixed: `rookies = {pos: len(draftable[pos])}` — subtract exactly what gets inserted. A rookie
costs a position a slot precisely when he takes one. Per-league-ness is preserved: the window is
now the league's own `total_picks`, so a short draft still isn't charged for a rookie past its
last pick (`test_rookie_count_uses_each_leagues_own_draft`).

**(b) The header was arithmetic on intent; the table was arithmetic on content.** `board_depth`'s
`DEFAULT_TOP_N` floors can total more than the draft has picks — my_2qb drafts 160 but the floors
ask for 35+50+75+24 = 184 — and `board_depth` only rescales *up* (`total < total_picks`), never
down. `insert_market_rookies(...).head(limit)` then silently dropped 24 model rows, while the
header kept printing the requested depth. Entry 094 saw this ("counts now… fall where the
inserted rookies pushed model players past the last pick") and left the header alone.

Deliberately **not** fixed by rescaling depth down: the VORP cut is cross-position and
value-driven, which is the right cut for a pick order; a proportional rescale would substitute a
per-position-rank cut and be worse (it would trim TE24 to TE21 while keeping WR65). Instead the
second trim is now *recorded* — `PositionSummary.dropped_past_last_pick` — and rendered:

```
### QB — 30 returning players (top 35 − 1 rookies − 4 past pick 160)
```

35 − 1 − 4 = 30. Two new regression tests assert the arithmetic closes for every position of
every league (`test_position_summary_arithmetic_closes`) and that the markdown header agrees with
the row count of its own table (`test_markdown_header_matches_its_own_table`).

**Regenerated.** All four boards; `rookies_subtracted` == rookies actually on the board at every
position now. suz_1qb moved most (RB 49→46, WR 72→66, TE 42→51 — TE was the position being
starved). my_2qb board content is unchanged; only its headers were wrong.

**CSV note:** no schema change. `rookies_subtracted` *values* change (suz WR 8→14); the new
`dropped_past_last_pick` is markdown-only and derivable from the CSV as
`requested_top_n − rookies_subtracted − rows`.

### Part 2 — the follow-up: no rookie TE in the top 24 is real

Checked, not a matching bug. The 2026 class has 22 drafted TEs and they normalise/join fine; the
highest-ranked rookie TE on the superflex board is **Kenyon Sadiq at overall slot 212**, past
my_2qb's 160-pick board. The market's top-24 TEs are 24 returning players. Also swept the market's
top 160 for names that are neither on the model board nor flagged rookie — 20 names, all genuine
veterans below the model's depth or under the eligibility gate (Kyler Murray 5 games, Malik Willis
4, vs the QB gate of 7). One name-normalisation miss found and noted but not acted on: FantasyPros
"Kenny Gainwell" vs our "Kenneth Gainwell".

### Part 3 — why the board isn't an ADP board

Agreement is real but loose: overall Spearman vs the league's own ECR board is .788 (my_2qb/rsf),
.804 (ppr_1qb), .861 (suz_1qb), .842 (underdog); median absolute gap ~24 board slots. Within
position: RB .911, TE .799, QB .767, WR .749. Cross-position composition is close — the model's
160-pick 2QB board is WR60/RB45/QB31/TE24 against the market's WR58/RB50/QB34/TE18 — so this is
**not** a VORP/replacement-level miscalibration. Three drivers, in order:

1. **The availability penalty.** Regressing the projection on last-season inputs:
   `proj ≈ 0.55–0.71 × ppg2025 + ~0.20 × games2025` (R² .74–.92). Every position loads
   significantly on games played (t = 3.2–4.9), so **each missed 2025 game costs ~0.2 projected
   PPG** on top of the mean reversion. Players with ≤11 games in 2025 sit a median **+16 slots
   worse** than market; players with ≥15 games sit **−13 slots better**. That is the whole Burrow
   (model 98 / market 4, 8 games), Purdy (82 / 20, 9 games at 19.7 PPG), Jayden Daniels (48 / 5,
   7 games), Nabers, Garrett Wilson, McLaurin, Godwin cluster. The market prices them as healthy
   again; the model prices the games they missed. Neither is obviously right — but it's a
   modelling choice, not a defect.

2. **Mean reversion.** The coefficient on last-season PPG is .55–.71, so the model compresses
   toward the pool while the market extrapolates recent form. This is what puts the model *ahead*
   of ECR in backtest at RB (.860 vs .751) and WR (.861 vs .746) — see `postseason_2025.md`.

3. **Structure without role.** The offseason block *is* populated for 2026 (team change, drafted
   competition, room size, vacated workload — all 7 columns carry variance), so the model is not
   blind to the offseason. But it only adds **+.01 to +.12 R²** over last-season PPG + games, and
   it knows roster *membership*, not projected *role*. Nothing tells it Zach Ertz is a TE3 in
   2026, so his 2025 line carries him to board slot 116 against a market slot of 484 — same story
   for Kareem Hunt (146/353), Tyreek Hill (127/286), Keenan Allen (111/190).

**Backtest context (2025, trained on labels < 2025):** model Spearman beats ECR at RB (.860 vs
.751) and WR (.861 vs .746), ties TE (.802 vs .803), trails at QB (.746 vs .789) — and QB is
exactly where the availability penalty bites hardest. The divergence is the point of the "two
independent opinions" rule, but the QB gap is a live lever: an availability model that projected
*next*-season games rather than penalising last-season absence is the obvious next move.

**Tests:** 579 pass. Ruff clean.

---

## Entry 096 — the availability lever doesn't exist (three ways), and one leak found

**Date:** 2026-08-30

**Prompt (full text):**

> lets try to flip that lever before the season starts
>
> [follow-up] Doesn't handcuff use different data sources than the main position_prediction model?

**Response notes:**

Entry 095 closed by proposing a lever: *"an availability model that projected next-season games
rather than penalising last-season absence."* This entry tested it. **It doesn't work.** Nothing
was shipped; the repo is unchanged apart from Entry 095's fixes.

### The follow-up question found a leak

Handcuff reads the **same** `data/processed/{stem}_features.parquet` and the same
`{stem}_feature_blocks.json` as `eval/projection.py` — same rows, same columns, same `games_next`
label out of `data/build.py`. So no source mismatch. But it does one thing the first draft of this
study didn't: `_board_with_risk` **drops the `offseason` block** from the risk feature set.

Re-running with the block included vs excluded moved the headline from +.018 to +.002, so the
exclusion was load-bearing. Checked why:

```
team_next MISSING : n= 841   mean games_next =  0.00   P(games_next == 0) = 1.000
team_next present : n=4091   mean games_next = 10.20   P(games_next == 0) = 0.109
```

`team_next` comes from the N+1 `rosters` table, which is the **full-season** roster, not a Sept-1
snapshot. Absence from it implies zero games with probability 1. For the PPG target that is
tolerable (PPG is conditional on playing, and the block is documented as Sept-1-knowable). For the
`games_next` target it is a **deterministic leak of the label**. Every offseason column merges on
`team_next`, so the whole block carries it.

Handcuff's docstring gives the exclusion a weaker reason than it deserves ("opportunity context,
not durability, and degenerate on a live board's N+1 horizon"). The degeneracy half is also stale —
all 7 offseason columns carry variance on the 2025 live rows. **The real reason is the leak.**
Left the code alone but this is worth a comment upgrade if the block is ever reused.

### Three attempts at the lever, all negative

Walk-forward, leak-safe throughout: feature seasons 2018/2019/2021–2024, projections trained on
labels `< F`, availability signals fit on labeled rows `< F`, graded against actual F+1. n=2375
player-seasons across QB/RB/WR/TE.

**1. Multiply projected PPG by projected games.** Spearman vs actual next-season total points,
mean over positions:

| rule | QB | RB | TE | WR | ALL |
|---|---|---|---|---|---|
| ppg × availability_model | .706 | .735 | .722 | .773 | **.734** |
| ppg (current) | .713 | .723 | .717 | .774 | **.732** |
| ppg × durability_3yr | .706 | .723 | .711 | .765 | .726 |
| ppg × prior_games | .693 | .705 | .709 | .754 | .715 |

+.002 overall and **negative at QB**, which is the position the complaint was about. (Also tested
`vorp × games` cross-position: much worse, .46–.53 — multiplying a signed VORP by a count is not a
coherent currency. Rejected.)

**2. Take the games features out of the rate model.** The Entry 095 diagnostic
(`proj ≈ 0.55–0.71 × ppg + 0.20 × games`) implied a removable penalty term. It isn't one:

| arm | vs actual PPG | vs actual points |
|---|---|---|
| full (current) | .746 | .732 |
| − `games` | .746 | .732 |
| − `games`, `games_missed` | .746 | .732 |
| − whole `availability` block | .745 | .731 |

Deleting the columns changes nothing, because `games` is collinear with the rest of the vector
(volume, career touches, snaps). The descriptive coefficient was real; the causal lever behind it
was not.

**3. Calibrate the bias away.** There *is* a measurable bias — signed rank error regressed on games
missed, cluster-robust by position-season:

| cohort | MODEL | ECR |
|---|---|---|
| all players (n=1409) | **+0.66 pct/game** (p<.001) | +0.39 pct/game (p<.001) |
| top-24 by prior PPG (n=384) | +0.22 pct/game (p=**0.51**) | +0.13 pct/game (p=0.68) |

So the model does under-rate players coming off partial seasons — but ECR does too (the excess is
only ~0.27 pct/game), and **at the top of the board the effect is absent**, p=0.51 on n=384. The
Burrow/Purdy/Daniels tier shows a flat error profile (−9.1 / −8.5 / −9.0 / −6.1 percentiles across
missed-game cohorts): both sources over-rate good players generally, neither has an availability
gradient among them.

Fitting that slope on earlier seasons and applying it forward makes things **significantly worse**:

```
current    0.7572
calibrated 0.7460   (-0.0112)   1/12 folds improved, paired t p = 0.008
```

Which is the expected failure: games missed is correlated with being genuinely worse, so
"correcting" for it promotes players who were bad *and* hurt.

### Verdict

The model's harsh ranking of Burrow (98 vs market 4), Purdy (82 vs 20) and Daniels (48 vs 5) is
**not a correctable systematic error**. Historically, good-but-injured players have not beaten the
model's ranking of them. Entry 095's proposed lever was wrong and is retracted.

### The lever that might actually exist

The model has no idea whether a player is healthy *now* — that is missing **data**, not a missing
correction. `data/raw/injuries.parquet` (the medstaff source, 2009–2025, 90,752 rows) is not wired
into the position features at all, and it carries a week-1 report snapshot every season
(176–221 rows/yr), so a Sept-1 health state is historically reconstructable for training. Medstaff
already found the cross-position-group injury signature to be the useful instrument.

Not started: it needs a leak-controlled Sept-1 snapshot design, and the season starts in ~10 days.
Flagged for the user rather than begun.

**Repo unchanged this entry.** Study scripts in scratchpad only.

---

## Entry 097 — the injury data is clean, informative, and predicts nothing

**Date:** 2026-08-30

**Prompt (full text):**

> Start it

(Entry 096's closing suggestion: wire `data/raw/injuries.parquet` into the position features so
the model knows whether a player is healthy *now*, rather than only how many games he missed.)

**Response notes:**

Built it, tested it, **rejected it as a model feature**, and kept it as report context. Net change:
the four draft boards gain a "Coming off a partial season" section; **no model number moved.**

### What was built

`features/build.season_health(weekly, injuries)` — a raw aggregate alongside `team_season_context`
and `ngs_season`, keyed on `(player_id, season)`:

- **from `weekly`** — `team_games`, `games_played`, `weeks_to_season_end`, `finished_season`,
  `longest_absence`, `returned_after_absence`
- **from `injuries`** — `inj_report_weeks`, `inj_weeks_out`, `inj_dnp_weeks`, `inj_distinct`,
  `inj_out_at_end`

Two details that took the design where it went. **Absences are counted against the player's own
team's game weeks**, so a bye never reads as a missed game. And **players on IR drop off the injury
report entirely** — Nabers and Tyreek Hill have almost no `Out` rows despite ending the season
hurt — so the report is uninterpretable alone; it only works next to the weekly appearances, which
is why both sources are folded into one function.

It discriminates exactly as intended on the 2025 rows:

| player | games | weeks_to_season_end | returned | out_at_end |
|---|---|---|---|---|
| Joe Burrow | 8 | **0** | 1 | 0 |
| Brock Purdy | 9 | **0** | 1 | 0 |
| Jayden Daniels | 7 | 4 | 0 | **1** |
| Malik Nabers | 4 | 13 | 0 | 0 |

Nothing in the existing feature set separated Burrow from Daniels — `availability` counts games
missed but not *when*.

### It doesn't predict anything

Walk-forward A/B, leak-safe, 4 positions × 8 feature seasons (2016–2024, 2020 excluded), block
added to the 2012+ eras only (injuries start 2009, so the boxscore era never gets it):

| arm | vs actual PPG | vs actual points |
|---|---|---|
| none (current) | **0.7288** | **0.7129** |
| minimal: `weeks_to_season_end` + `returned_after_absence` | 0.7278 | 0.7115 |
| + `inj_out_at_end` | 0.7272 | 0.7110 |
| + report load | 0.7265 | 0.7104 |
| full block (12 cols) | 0.7250 | 0.7091 |

Monotonically worse the more you add — 4/28 folds improved on the full block. And it **does not
move the cohort it was built for**: players who missed 4+ games and *came back* stay under-rated by
+3.0 rank percentiles with or without it; players who missed 4+ and stayed out stay at +1.2. The
information is in the columns and the model finds nothing to do with it. (Both are regularised —
lasso/elasticnet/random forest — and correctly shrink these to nothing.)

Also null on the **availability** target, which is where these features should have belonged
(handcuff's `games_next` model, its own backtest metrics):

| position | games MAE without health | with health | clears-AUC without | with |
|---|---|---|---|---|
| QB | 3.332 | 3.342 | 0.895 | 0.896 |
| RB | 4.311 | 4.310 | 0.813 | 0.814 |
| WR | 4.025 | 4.012 | 0.849 | 0.850 |
| TE | 3.874 | 3.860 | 0.825 | 0.822 |

### Why it was reverted rather than left in

`eval/experiment._all_feature_columns` takes the union of **all** blocks, not the era-filtered set,
so a `health` block in the map would silently join handcuff's availability feature set even with
no era referencing it. Carrying a tested-null block that quietly changes another tool's inputs is a
cost with no benefit, so `add_health` and the orchestrator wiring are gone and the features parquet
+ block map are byte-equivalent to before. `season_health` stays, now documented as report context.

### What shipped instead

Every draft board gains a section listing board players who missed 4+ games in the feature season,
with how the season ended — the thing the model can't see and the market prices hard:

```
## Coming off a partial season

| board | player | pos | proj PPG | 2025 games | how the season ended |
|    98 | Joe Burrow     | QB | 13.76 | 8 of 17  | returned, finished the season |
|    48 | Jayden Daniels | QB | 15.41 | 7 of 17  | ruled out over the last 4 weeks |
|    90 | Malik Nabers   | WR |  9.60 | 4 of 17  | last played with 13 weeks left |
```

This is the honest landing for Entries 095–097: the model's number stays pure (no-blend rule
intact, and the tested-null result is stated in the section's own preamble), and the drafter gets
the one piece of context the projection structurally cannot carry. Markdown only — **no CSV schema
change**; `RedraftResult.health` holds the frame and never reaches the board.

### Incidental finding, not acted on

`eval/handcuff.py`'s module docstring says the availability model "does *not* beat a naive
prior-games baseline in backtests." On the current feature set it clearly does — QB clears-AUC
.895 vs .859, WR .849 vs .798, RB .813 vs .784, TE .825 vs .799 — and `backtest_risk_signals`
picks it as the winner at all four positions. The code chooses correctly by backtest; only the
prose is stale. Left alone as out of scope.

**Tests:** 586 pass (7 new: 4 on `season_health` covering byes, the came-back/still-out split,
optional injuries, and the IR dropout; 3 on the report section and `health_note`). Ruff clean.

---

## Entry 098 — twenty seasons of positional share, and the flex that never takes a tight end

**Date:** 2026-08-31

**Prompt (full text):**

> looking at the last 20 years of fantasy football data can you make a graph that shows what
> portion of points scored by startable players were scored in each season at each position

**Response notes:**

- Read-only analysis off `data/raw/weekly.parquet` (nflverse, REG only, 2006–2025). No pipeline
  code touched; the compute script lives in the session scratchpad, not the repo.
- **"Startable" is a lineup, not a top-N-per-position list.** Filled the league-wide starting
  lineup of a 12-team league — 12 QB / 24 RB / 24 WR / 12 TE by dedicated slot — then gave the 12
  FLEX slots to the best remaining RB/WR/TE, which is the same shape `DEFAULT_LEAGUE` in
  `eval/redraft.py` uses. 84 players per season. Ranked on season-total points (modal position per
  player-season, since a player-week's listed position wobbles).
- Recomputed all three scoring formats the same way `scoring.py` does — standard points plus
  `reception_value × receptions` — so the format toggle is exact, not an approximation.
- **Findings (PPR share of startable points).** RB 34.7% (2006) → 24.7% (2015) → 31.7% (2025), the
  highest since 2009. WR peaks at 43.7% in 2015 — the same season RB bottoms — and falls to 38.1%
  by 2025. QB rises 14.6% → 18.6% between 2006 and 2011 and then does nothing: every season since
  sits inside a 3.1-point band. TE is the flattest line in the study, 10.6–12.8%.
- The 2015 RB trough and WR peak are one event, not two, and the 2024–25 RB recovery is the first
  real move against it in a decade.
- **The flex is a receiver slot.** Across 240 flex slots (20 seasons × 12), a tight end never took
  one, in *any* of the three scoring formats. In PPR, receivers took 11 or 12 of 12 every season
  since 2010; in standard the count drops to 1–10, which is most of what the scoring toggle is for.
- Scoring format moves the answer more than twenty years of NFL history does: 2025 RB share is
  31.7% (PPR) vs 39.5% (standard); WR 38.1% vs 27.2%. Shipped the toggle rather than picking one.
- Delivered as an Artifact — line chart per position, scoring toggle, hover crosshair, four
  readings, and a table view. Palette is the dataviz reference categorical order (slots 1–4); the
  validator script could not be run (no `node` on this box), so the palette is used as documented
  rather than re-validated, with every series direct-labelled so identity is never colour-alone.

**Tests:** none — no repo code changed.

---

## Entry 099 — the same chart on PPG, and how much of the RB collapse was games missed

**Date:** 2026-08-31

**Prompt (full text):**

> Use ppg instead. Same cut offs as previously established in other project efforts.

**Response notes:**

- Reworked Entry 098's chart onto **points per game**, with the ranking universe restricted to the
  Stage-5 eligibility cutoffs already committed in `config/football_*.yaml`
  (`eligibility.chosen_games_played`): **QB 7, RB 4, WR 7, TE 5**. Same lineup shape — 12/24/24/12
  plus 12 FLEX — but the sort key and the share numerator are now PPG, so a position's line is its
  slice of the 84-player lineup's combined PPG.
- Games are counted as REG weekly rows per player-season, which is how `data/build.py` counts them
  (it matches the nflverse seasonal `games` column), so the cutoff bites exactly as it does in the
  pipeline. Eligible pools are comfortably deep enough to fill every slot at every position in
  every season (thinnest is QB, 35–44 eligible for 12 slots) — asserted in the script.
- Kept 2020. `exclude_seasons: [2020]` is a *supervised* exclusion; dropping it from a descriptive
  time series would just punch a hole in the middle of the chart.
- **The headline finding changed.** On season totals the RB share collapses to 24.7% (2015) from
  34.7% (2006) — a 10-point fall. Per game it only falls to 27.7%, a 7.7-point fall, and the same
  holds in half-PPR (9.6 vs 12.4) and standard (9.7 vs 13.3). **Roughly a quarter to a third of
  the "RB collapse" is games missed, not per-game production.** That is a point in favour of PPG
  being the model's target, and an argument that a season-total positional-value chart overstates
  the case against running backs.
- Other lines, PPR: WR peaks 42.9% (2014) and ends 37.2% (2025), its lowest since 2009; QB rises
  14.5% → 17.7% by 2011 and then sits inside a **2.0-point band** for fourteen seasons (tighter on
  PPG than the 3.1 the season-total cut showed); TE spans 10.5–12.7%, still the flattest.
- The flex is still never a tight end — 0 of 240 slots, in all three scoring formats, on PPG as on
  season totals. On PPG the flex takes more backs than the season-total cut did (RB up to 5 in a
  season), because a back who misses games is no longer penalised for it.
- Artifact republished in place (same URL). Fixed two rendering bugs found while editing: the
  redraw dropped the SVG `<title>` (accessibility name), and end labels collided when two series
  finish a fraction of a point apart — now decluttered with leader lines.

**Tests:** none — no repo code changed; analysis script is session-scratchpad only.
## Entry 100 — a new effort: which starting QBs get benched, and the roster column that lied

**Date:** 2026-09-07

**Prompt (full text):**

> New football effort: Predict which QBs that are starting going into the season will get benched

**Response notes:**

Fourth standalone effort, `sports/football/src/qb_benching/`, branch `qb-benching`. Plan doc
`docs/QB_BENCHING_PLAN.md`. **Stage 1 (cohort + label) is complete**; stages 2-4 (preseason
features, model, live board) are specified and not built.

### Decisions taken with the user before building

- **Label:** build the full displacement label and split it three ways — `benched` / `injured` /
  `gone` — modelling benching, carrying the injury arm as the declared comparison. (The
  alternatives offered were benching-only and any-loss-of-job.)
- **Horizon:** preseason, one row per QB-season, features known by Sept 1. Not an in-season
  weekly hazard; that is specified as a later stage whose baseline is this model.
- **Scope:** standalone effort plus report, not wired into the draft boards.

### The result

**544 opening starters, 2009-2025; 90 benched (16.5%)**, 142 losing >=3 weeks to injury, 213 for
any reason. The bar is >=3 team-games started by somebody else, with 2/3/4/6 reported as
sensitivities every time.

Face validity is good: the most-benched seasons are Shaun Hill (2016 MIN), Scott Tolzien (2017
IND), Trubisky (2022 PIT), Flacco (2022 NYJ), Eli Manning (2019 NYG), Russell Wilson (2025 NYG).

### Three data defects found, all load-bearing

1. **⚠️ Before 2021 `rosters_weekly.status` is not a weekly value** — it is the season-final
   status stamped on every one of the player's weeks. Only 1-4% of pre-2021 QB-seasons have a
   status that varies by week, against 50-72% from 2021. This is a *different and worse* defect
   than the one `medstaff` records as `FIRST_COMPARABLE_SEASON = 2021` (that the ACT/INA split is
   unpopulated). **Found because a first version of the label reported zero benchings in 2015**,
   a season in which Kaepernick visibly lost the San Francisco job: he started eight games and
   ended on IR, so all ten of his rows read `RES` — including the weeks he was starting.
2. **⚠️ Injured reserve is invisible to the injury report** (a player on it drops off entirely).
   Against the trustworthy 2021+ status, displaced weeks with no listing and no appearance split
   **195 reserve / 104 active / 20 inactive** — ~60% genuinely hurt. A report-only label calls
   them all benchings.
3. **⚠️ `schedules` and `rosters_weekly` disagree on St. Louis.** The roster feed uses `SL` for
   all 14 St. Louis seasons (2002-2015) and never `STL`. Uncanonicalised the club comparison
   fires and every Rams QB reads as playing elsewhere — which classified **Nick Foles's 2015
   benching as a departure**. `SL` was missing from `medstaff.data.teams.TEAM_ALIASES`; added
   there rather than starting a third copy of the map. Audited all three tables: `SL` was the
   only uncovered code.

### The fix, and a new dataset

Registered **`depth_charts`** in `data/fetch.py` — the one obvious nflverse endpoint the repo had
never fetched (2001-2025, 1.42M rows). It is the only weekly role signal that survives the 2021
break, and it records both facts *directly* rather than by elimination: a demotion is
`depth_team` 1 -> 2, and a reserve move is dropping off the chart. Kaepernick's 2015 reads
straight off it — rank 1 through week 9, rank 2 behind Gabbert in week 10, absent from week 11.

⚠️ **nflverse changed the feed for 2025**: through 2024 it is weekly on
`(season, week, club_code)` with a string `depth_team`; from 2025 it is a dated snapshot feed
(`dt`, `team`, `pos_abb`, integer `pos_rank`) with **no season or week column** and ~15x the
rows. `labels/depth.py::qb_depth` normalises both, resolving each team-game to the latest chart
strictly before kickoff. Side benefit: the snapshot feed starts in early August, so it carries
the preseason chart a live board needs to know who is opening the season — which also answers the
serving problem the plan had flagged (`schedules.home_qb_id` is 100% null for an unplayed year).

**What the depth chart bought, measured.** The benched:injured ratio of displaced team-games was
1.04 pre-2021 against 0.40 from 2021 — a two-and-a-half-fold discrepancy that is the injury
leakage. With the chart leading the ladder it is 0.67 vs 0.51, and the label's base rate moved
from 0.206/0.150 either side of 2021 to **0.159/0.181**, a 0.022 gap. The chart resolves **526 of
672** benched team-games directly; 146 remain a residual.

### Ladder, and why benching is a lower bound

held -> injury report -> off the chart -> demoted on the chart -> off the roster -> another club
-> (2021+ only) weekly status -> residual benched. The injury report deliberately outranks the
chart, so a QB both listed hurt and demoted is counted hurt.

### Files

New `src/qb_benching/{__init__,labels/{__init__,starters,depth,displacement,cohort}}.py`,
`scripts/qb_benching_cohort.py`, `tests/test_qb_benching_labels.py` (13 tests, each aimed at one
of the defects above), `docs/QB_BENCHING_PLAN.md`. Modified: `data/fetch.py` (one `Dataset`),
`medstaff/data/teams.py` (the `SL` alias), root `pyproject.toml` (wheel packages).

**Known limitation recorded, not fixed:** the opening starter is not always the intended starter
(Derek Anderson, 2014 CAR, covering an injured Cam Newton). Losing the job to a returning
incumbent is not a benching on the merits; stage 3 must carry a flag for it.

**Tests:** 651 pass (638 + 13), ruff clean. Reverting the `SL` alias fails exactly one test — the St. Louis one — so the regression guard is real rather than decorative.

---

## Entry 101 — the benching model, and the control that makes it believable

**Date:** 2026-09-07

**Prompt (full text):**

> build stages 2 and 3

**Response notes:**

Stages 2 (preseason features) and 3 (the model) of [[qb-benching]]. Stage 4 (live board) still
not built. Branch `qb-benching-model`.

### Stage 2 — 32 features in five blocks, all knowable by Sept 1

`prior_play` (last season's volume, efficiency, starts), `tenure` (age, experience, draft
capital), `room` (the **week-1 depth chart** — room size, the rank-2 backup's own prior
production, a drafted rookie and his pick), `team` (prior record, point differential, new head
coach), `history` (the lagged label).

`features/build.py::add_offseason` was deliberately **not** reused even though it computes almost
the same room columns: it merges on `team_next` from the N+1 *full-season* roster, which entry
096 established is a deterministic leak for absence-shaped targets. Every room column here comes
off the week-1 chart instead.

Two fixes on the way, both now pinned by tests:
- The prior-record join silently dropped every **relocated franchise** — the cohort's team codes
  are canonical (`LA`) and raw `schedules` is not (`STL`). 5.3% of rows had no prior record until
  the schedule side was canonicalised too.
- `incumbent_present` (last season's primary starter on the week-1 chart but *not* the opener) is
  the §3.5 fill-in flag. It fires correctly on Derek Anderson's 2014 Carolina season, and carries
  a 23.1% benching rate against 16.0% otherwise.

Leakage check on the lagged label: **0 mismatches over 374 rows** against the previous season's
own outcome.

### Stage 3 — the headline, and why it is the stratified one

**Pooled CV AUC 0.788** (±0.012) vs permutation null 0.497 ± 0.047, p < 0.005. Top-5-per-season
precision **45.9%** against a 16.5% base rate. Temporal split (train <=2017) **0.686** — lower,
and the number to believe.

**The stratified result goes the opposite way to `qb_breakout`'s**, which is the finding:

| prior-attempt band | n | positives | model AUC |
|---|---|---|---|
| none | 42 | 4 | 0.513 |
| <100 | 38 | 18 | 0.556 |
| 100-299 | 68 | 22 | 0.659 |
| **300+** | **396** | **46** | **0.802** |

The model is *strongest* among established starters (base rate 11.6%), and at chance in the two
small stopgap bands — the opposite of the trap the plan was written to guard against, where draft
capital went 0.890 pooled to 0.496 within band because it allocates opportunity rather than
predicting outcome. The pooled number is not coming from "this man was never really the starter".

**The control is the part that makes it worth believing.** Identical features, identical rows,
fitted against `injured_out` (142 positives): **AUC 0.528** vs null 0.505, p=0.30, at chance in
every band. Entries 096-097 already established injury is unpredictable from this data; if these
features had "predicted" it too, the benching number would be an artefact of the evaluation
rather than a fact about benching. Folded into the main report as a section rather than a
separate file.

Beats every single-column baseline: prior attempts 0.701, prior EPA/dropback 0.661, lost-the-job
-last-season 0.589, draft capital 0.588. Boosting (0.780) does not beat the C=0.1 logistic
(0.788), so the linear model stands.

⚠️ **Coefficients are collinear and are not effects** — `has_prior_season`/`is_rookie` are
near-mirrors, the three volume columns measure one thing, and the rate columns carry suppression
signs as a result (`prior_int_rate` reads negative, `prior_comp_pct` positive). Stated in the
report so nobody reads "interceptions keep a job" off the table.

⚠️ **Scope not delivered:** the plan's fourth baseline — the ranking model's own projected QB
rank via `eval/projection.py` — was **not run**. It needs a leak-safe per-season fit across 17
seasons, a much heavier dependency than the other three; `prior_attempts` stands in as the volume
baseline. Should be added before stage 4 ships a board.

### Files

New `src/qb_benching/{features,model}/`, `scripts/qb_benching_{features,model}.py`,
`tests/test_qb_benching_{features,model}.py` (19 tests: 10 leakage/join, 9 evaluation-integrity),
`reports/REPORT_qb_benching_model.md`. Plan doc gains §4.1/§4.2.

**Tests:** 670 pass (651 + 19), ruff clean.

---

<!-- Template for new entries:

## Entry NNN — <short title>

**Date:** YYYY-MM-DD

**Prompt (full text):**

> ...

**Response notes:**
- ...

-->
