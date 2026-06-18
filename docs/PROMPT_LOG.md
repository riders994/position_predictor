# Prompt Log

A complete, chronological record of every prompt submitted to the AI assistant during
this project, with notes on the response. Maintained for **reproducibility and decision
provenance** — anyone should be able to trace *why* the project looks the way it does.

> Convention: append a new entry for **every** prompt. Quote the prompt **in full**.
> Keep response notes concise but specific (decisions made, files created, rationale).

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

<!-- Template for new entries:

## Entry NNN — <short title>

**Date:** YYYY-MM-DD

**Prompt (full text):**

> ...

**Response notes:**
- ...

-->
