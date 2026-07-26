# Prompt Log — Primary (all sports + repo-level)

The **primary, chronological record of every prompt** submitted to the AI assistant across the whole
repository — every sport plus repo-level/infrastructure work — with notes on the response. Maintained
for **reproducibility and decision provenance** so anyone can trace *why* this repo looks the way it
does. This file fulfills the original Entry 001 requirement of "a document that includes the full text
of every prompt."

> **Logging convention.** Entry numbers are **global** and shared across all logs.
> - Append **every** prompt here (the primary), with the prompt quoted **in full**.
> - Also append each prompt to its sport's log (`sports/<sport>/docs/PROMPT_LOG.md`) when it relates to
>   that sport; repo-level/infrastructure prompts (e.g. reorg, the log system itself) live here only.
> - Keep response notes concise but specific (decisions, files, rationale).

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

## Entry 028 — Reorg: each sport a standalone project + USAGE.md

**Date:** 2026-06-22

> _Original prompt not captured; reconstructed from PR #10 (`c487eb3`, `b320cd7`)._ Restructure so each
> sport is its own independent project, superseding the earlier shared-engine idea; add a usage
> reference for every command.

**Response notes:**
- Decision: fantasy scoring and per-sport data shaping differ enough that a **shared modeling engine
  is the wrong abstraction**. The entire football project (code, scripts, tests, config, docs,
  notebooks, examples, reports, data) moves under `sports/football/`; a new sport will be an
  independent project beside it (not importing football's code).
- **Zero code edits** to make it work: `io.py` derives `PROJECT_ROOT` from the package location, so once
  the package sits at `sports/football/src/...`, all config/data/report paths follow automatically
  (also dropped the `reports_dir(sport)` indirection a previous attempt had added). Whole tree moved
  with `git mv` (history preserved).
- Shared dev tooling stays at repo root: one `pyproject.toml` + `uv.lock`, with `packages`/`testpaths`
  and `.gitignore` repointed at `sports/football`. **This is the move that orphaned the prompt log** —
  `docs/PROMPT_LOG.md` → `sports/football/docs/PROMPT_LOG.md`, making it football-scoped with no
  top-level log (see Entry 031). New root README = thin container index; new football README = project
  landing page.
- Follow-up `b320cd7`: **`docs/USAGE.md`** — command reference (flags, outputs, run order) for every
  stage + serving tool, linked from the football README. Verified: `uv sync`, **115 pass**, ruff clean.

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

## Entry 031 — Prompt-log audit + backfill (this entry)

**Date:** 2026-06-23

**Prompt (full text):** _(verbatim)_

> What's going on with the prompt logs? I only see the football one, there's no full one, and the
> football one stopped after the keeper tool was created. […] Branch off primary and backfill entries
> 026-029 […] Make sure to include the prompts from this session investigating the prompt log, the
> purpose of this project is to document as much as possible for future reproducibility

**Response notes:**
- **Two findings.** (1) *No "full" log:* the standalone-sports reorg (Entry 028, `c487eb3`) **renamed**
  the only repo-root `docs/PROMPT_LOG.md` to `sports/football/docs/PROMPT_LOG.md` — it didn't create a
  top-level one, so there's no repo-wide log and cross-cutting work has no home. (2) *Stale:* the log
  stopped at Entry 025 (keeper); five subsequent bodies of work were unlogged.
- I initially miscounted as four — **the redraft tool (PR #8) was also unlogged** (only an incidental
  "redraft" mention existed in Entry 015), surfaced via the chronology check. Backfilled **all five**
  (026 redraft, 027 postseason, 028 reorg, 029 nflreadpy, 030 this cleanup session) + this meta-entry
  (031).
- **Honesty constraint for reproducibility:** prompts for 026–029 weren't captured at the time, so they
  are explicitly marked *reconstructed from the commit/PR* rather than presented as verbatim quotes;
  030–031 carry the **actual** session prompts verbatim. Sourced entry detail from `git show` of each
  PR's commits.
- **Open structural question** (deferred, user hasn't decided): whether to (re)introduce a top-level
  cross-cutting `docs/PROMPT_LOG.md` with per-sport logs underneath, vs. keeping the single
  football-scoped file. Kept single-file for now.
- Branched `docs-prompt-log-backfill` off the fresh `primary`; carried the orphaned USAGE.md 2025 fix
  along (committed here).

---

## Entry 032 — Split the prompt log into a primary + per-sport logs

**Date:** 2026-06-23

**Prompt (full text):** _(verbatim)_

> I want one file with every prompt and one file in each sport directory of every prompt related to
> that sport
>
> [follow-up] Don't use the word "master", it's not inclusive. Switch to "primary"

**Response notes:**
- Per the follow-up, the superset tier is named **"primary"** (not "master") throughout the logs.
- Restructured the single football-scoped log (the reorg in Entry 028 had left it as the only log,
  with no repo-wide one) into two tiers:
  - **`docs/PROMPT_LOG.md` (this file)** — primary superset: every prompt, all sports + repo-level.
  - **`sports/<sport>/docs/PROMPT_LOG.md`** — per-sport, filtered to that sport's domain work.
- **Entry numbers are global** and shared across both tiers, so a number means the same entry
  everywhere and cross-references hold. Per-sport files are therefore non-contiguous where repo-level
  entries are skipped.
- Classified **028 (reorg), 031 (log audit), 032 (this)** as repo-level/infrastructure → primary only;
  all football-domain entries (001–027, 029, 030) appear in both. No content lost — the primary is a
  strict superset and history is in git.
- Maintenance convention recorded in both headers: football prompts append to **both** the primary and
  the football log; repo-level prompts to the primary only.

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

## Entry 034 — Prompt-log convention: write the primary log first

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> It's better to do the main prompt log first. A prompt could apply to multiple sports.

**Response notes:**
- Convention refinement: when logging a prompt, **write the repo-wide primary `docs/PROMPT_LOG.md`
  first**, then fan out to the relevant per-sport log(s). Rationale (user's): a single prompt can apply
  to **multiple sports**, so the primary is the canonical superset that owns the entry number; per-sport
  copies are derived from it.
- Repo-level/infrastructure prompt → **primary only** (no per-sport copy), consistent with Entries
  028/031/032.

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
- **(prompt-log fix)** Restored the root log's missing `<!-- Template for new entries:` opener (the
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

## Entry 038 — Commit report .md files (un-ignore so they're publicly viewable)

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> Remove the md files for reports from the gitignore so they can be viewed by the public

**Response notes:**
- The only report `.md` still git-ignored was `sports/*/reports/postseason_*.md` (everything else —
  `REPORT_*.md`, `versions/**`, `handcuff_*.md` — was already committed). Removed that `.gitignore`
  rule so postseason reports render on GitHub; the regenerable serving **CSVs** stay ignored (the user
  asked for the `.md` files). Committed the now-tracked `postseason_2025.md` and updated the
  `.gitignore` header comment to say report `.md` files are committed.
- Repo-level/infra change → **primary log only** (per the 028/031/032/034 convention).
- Branch `publish-report-md`, stacked on `handcuff-tool` to keep the prompt log linear (primary only
  has through Entry 034; 035–037 are still on `handcuff-tool`, so a branch off primary would collide).

---

## Entry 039 — Basketball project kickoff & 3-phase plan

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim, sequential planning prompts; clarifying-question answers noted)_

> Both merged. We're going to start the basketball project off of a fresh primary. I want to start
> with a planning session and I'll lay out my plan after you refresh git
>
> This is a complex process. First, I want to identify position archetypes. This is inspired by the
> idea of their being 11-14 actual positions in basketball (rather than the traditional five) defined
> by their style of play. Second, we will use these archetypes to experiment with team configurations
> and try to identify groupings of players that fit certain archetypes that lead to successful fantasy
> seasons.
>
> Phase 3 is building an archetype predictor. Based on past data for a player, predit their
> archetype(s) for next season
>
> I want to focus on the modern game, so let's only consider players from the last 2 eras when
> developing the player archetypes. We can pull player data from previous eras for phase 2 and phase 3
> to increase depth of dataset. We start with soft archetypes and then group them into hard archetypes
> after EDA. Furthest back we should go is 2013+. We'll use 2 sample leagues to source examples of team
> configurations for the league type we will be optimizing. Rookies out for now, that will be a
> separate model since it's based on college/high school data + probably more soft data around draft
> coverage.
>
> Here are the IDs for the two fantrax leagues: blk3bn3clw9njuhc wserh14rmbbpqtcg. that should be
> enough to grab data from the API (or using the python package). I like those era boundaries. We can
> reevaluate if the results aren't good. Phase 3 is the final phase for now. We'll finish this up
> before coming up with anything else for basketball.
>
> You won't need the cookie, those leagues are public
>
> [clarifying-question answers] Phase 2 = **fantasy roster construction**; format = **9-cat roto/H2H**;
> data = **layered hoopR/sportsdataverse → Basketball-Reference → nba_api**.

**Response notes:**
- Scaffolded the standalone `sports/basketball/` project (docs only so far) and wrote
  `docs/PROJECT_PLAN.md` — the design of record. A **new problem shape** vs football (clustering +
  composition + predictor, not next-season rank regression); reuses the repo's reproducibility /
  config / prompt-log conventions.
- **3 phases.** (1) Archetype discovery — unsupervised **soft GMM** membership → consolidate to a
  hard taxonomy after EDA, ~11–14 archetypes. (2) Archetype composition → **9-cat** fantasy success,
  with real labels from the two Fantrax leagues (category-portfolio / punt-aware roster
  construction). (3) Next-season archetype predictor — supervised, **leak-safe N→N+1**, persistence
  baseline, age/trajectory the key signal. Production flow: predict archetypes → score compositions →
  draft.
- **Eras / scope:** floor **2013**. E1 2013–17 / E2 2017–20 / E3 2020–present (provisional). **Phase
  1 = E2+E3** (modern only); **Phase 2/3 = all 2013+**. Returning players only; rookies deferred.
- **Data:** layered hoopR → Basketball-Reference → nba_api. Fantrax leagues are **public** → no auth.
- Repo-wide + per-sport basketball logs created; this is the kickoff entry in both. Branch
  `basketball-plan`. Planning only — no pipeline code yet.

---

## Entry 040 — Basketball data layer: scaffold + Stage-1 fetch

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> Merged. Lets' get started.
>
> [follow-up] You won't need the cookie, those leagues are public

**Response notes:**
- **Scaffolded the standalone `sports/basketball/` project** mirroring football: dirs (config / data /
  reports / scripts / src / tests / notebooks), package **`nba_archetypes`**, shared utils (`io.py`
  path-depth-based, `config.py`) reused from football's pattern. Registered the package + tests in the
  root `pyproject.toml` (single shared env); added a basketball Makefile.
- **Data deps added + validated via `uv sync`:** `sportsdataverse` (base hoopR loaders), `nba_api`
  (granular play-type/tracking, later), `fantraxapi` (Phase-2 leagues). All installed clean; football
  unaffected.
- **Stage-1 fetch built + run** (`data/fetch.py` + `scripts/fetch_data.py`, registry + cache +
  manifests like football). Live pull 2014–2025 (season = ending year; 2014 = 2013-14 floor):
  `player_season_stats` 272,763×15 (long format, pivot in build), `rosters`, `team_season_stats`
  cached + manifested. `shots`/`player_boxscore` are large/opt-in (`--include-large`).
- **Fantrax validated:** league `blk3bn3clw9njuhc` reads **without auth** (public confirmed) — 14
  teams; `standings` / `team_roster` / `scoring_period_results` / `position_counts` available for
  Phase 2.
- 3 no-network tests, ruff clean; football's 132 still pass. Branch `basketball-data-layer`.
- **Next:** `rosters` loader returned only 537 rows (likely latest-season-only — investigate in
  build); build the long→wide player-season + style features; add nba_api/shots enrichment + a
  Fantrax fetch module.

---

## Entry 041 — Basketball build + style-feature stage

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> Proceed into the build/feature stage

**Response notes:**
- **Stage 2 (`data/build.py`):** pivot the long ESPN `player_season_stats` → **wide per-(athlete,
  season)** (one ESPN season row per player, verified — no multi-team dupes). Parse the
  `"made-attempted"` combo fields out of `display_value` (`value` is null for those). Eligibility =
  rotation players (`min_minutes_pg` 15 / `min_games` 20, config). **6,361 player-seasons, 3,889
  eligible.**
- **Stage 3 (`features/build.py`):** **19 style features** in 5 blocks — scoring (pts/fga/fta per-36),
  shot_profile (`fg3a_rate`, `ft_rate`, fg3a36, shooting splits + scoring/shooting-eff), playmaking
  (ast36/tov36/ast_tov), rebounding (oreb36/dreb36/`oreb_rate`), defense (stl36/blk36). All **per-36
  (style not volume)** and **z-scored within season over eligible players** (era-relative); each
  season tagged E1/E2/E3. **Phase-1 clustering pool (eligible & E2+E3) = 2,631 player-seasons.**
- **Validated style separation:** Gobert `fg3a_rate 0.0` / `blk36 2.76` (rim-runner), Curry `0.55`
  3-pt reliance, Chris Paul `ast36 11.0` (pure PG), Jokić `ast36 8.5` (playmaking big). Z-scores clean
  (per-season mean 0 / std 1; 0.5% NaN).
- 8 no-network tests (combo parse, era mapping, div-guard, block-map disjoint); ruff clean; **140
  total** (132 football + 8 basketball). Branch `basketball-data-layer`.
- **Next:** Phase 1 — GMM soft clustering on the `*_z` features (k≈11–14, BIC/silhouette), name +
  stability-check the archetypes, then consolidate to a hard taxonomy after EDA.

---

## Entry 042 — Phase 1: archetype discovery (soft GMM, k=12)

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> Merged. Now what

**Response notes:**
- **k-selection EDA:** BIC has a relative plateau at k≈11–12 (then jumps at k≥13); silhouette is
  uniformly low (0.04–0.07) — *expected*: NBA style is continuous/overlapping, which is exactly why
  soft GMM + soft→hard is the right frame. With the ~11–14 / "13 positions" target → **k=12**.
- **Fixed a NaN trap:** dropping rows with any NaN `*_z` excluded no-3PT bigs (Gobert's `fg3_pct` is
  undefined) — the very archetype we want. Now **impute NaN→0** (neutral) instead of dropping; pool
  stays 2,631 (fit) / 3,889 (assigned).
- **`archetypes/discover.py`:** GMM fit on **eligible E2+E3** (2,631), soft membership **assigned to
  all eligible 2013+** (3,889, so E1 carries archetypes for Phases 2/3). Membership parquet carries
  the full prob vector `p0..p11` + `entropy` (blend-iness) — the soft input for Phase 2/3.
- **12 coherent, named archetypes** (provisional, seed-tied): Rim-Running Center (Gobert/Capela),
  3-and-D Wing (Mikal Bridges/Covington), Floor General (low-usage facilitator), High-Usage Shot
  Creator (DeRozan/Kyrie), Movement Shooter (Hield/Lonzo), Slashing Forward (Giannis/Siakam),
  Non-Shooting Interior Big (Mobley/Allen), Corner Specialist (PJ Tucker), Two-Way Forward, Volume
  Perimeter Scorer, Connector Wing, Balanced Combo Wing. Same players recur across seasons in the same
  archetype (good stability signal). Report → `reports/REPORT_archetypes.md`.
- 11 tests, ruff clean, **143 total**. Branch `basketball-archetypes`.
- **Next:** cross-season stability check + soft→hard consolidation/naming, then Phase 2 (Fantrax
  league configs → archetype composition vs 9-cat success).

---

## Entry 043 — Phase 1 iteration: PCA-whiten (soft membership) + stability

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> Keep iterating on phase 1

**Response notes:**
- **Diagnosed:** the raw full-cov GMM was **overconfident** (median `top_prob` 1.0, only 1% blends) —
  the "soft" membership was effectively hard, because the style features are collinear (3-pt-ness,
  usage). Compared covariance types: full has the best BIC by far but is overconfident; tied is softer
  but fits much worse.
- **Fix = PCA-whiten → GMM.** Decorrelating to ~90% variance (8 PCs) before the GMM gives genuinely
  **soft** membership (median `top_prob` 0.85, **43% of player-seasons are blends**) *and* raises YoY
  **stability 0.55 → 0.61**. Profiles are still computed on the original z-features, so interpretation
  is unaffected; cluster indices changed, so the 12 names were re-derived (sharper now: Foul-Drawing
  Iso Scorer = DeRozan/Harden/Butler; Lead Playmaker (Haliburton/CP3) split from Scoring Combo Guard
  (Kyrie); High-Usage Primary = Luka/LeBron; Rim-Running Center = Gobert/Capela).
- **Added cross-season stability** (`archetype_stability`): **61%** keep their archetype YoY (vs ~8%
  random) — validation + the Phase-3 must-beat baseline. Per-archetype: Rim-Running Center stickiest
  (0.78), Interior Big / High-Usage Primary ~0.70; the low-signal middle (Connector Wing 0.45, Two-Way
  Forward / Rebounding Big-Forward 0.51) churns most → soft→hard consolidation candidates. Report now
  surfaces softness + a stability table.
- 13 tests, ruff clean, **145 total**. Branch `basketball-archetypes`.
- **Next:** soft→hard consolidation of the low-signal middle (e.g. Balanced Wing catch-all);
  optional shot-zone enrichment (`load_nba_shots`); then Phase 2.

---

## Entry 044 — Defer Phase-1 polish to a roadmap; ship Phase 1

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> Let's mark these as future possible improvements for the model. We can come back later. Setup the PR
> for me to merge and we can start phase 2

**Response notes:**
- Recorded the deferred Phase-1 levers in **PROJECT_PLAN §8 "Future improvements"** (soft→hard
  consolidation of the low-signal middle; shot-zone enrichment via `load_nba_shots`; nba_api
  play-type/tracking; `rosters` multi-season fix; era-boundary revisit) — tracked, none blocking.
  Phase 1 ships as-is: k=12 PCA-whitened soft GMM, 43% blends, 61% YoY stability.
- Branch `basketball-archetypes` finalized + pushed for merge (PR via compare URL). Phase 2 starts off
  a fresh `primary` after merge: Fantrax fetch module (two public leagues → settings + rosters +
  standings) → archetype composition vs 9-cat success.

---

## Entry 045 — Phase 2 start: optimal-lineup success labels via max_pf

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> For Phase 2, some of these teams didn't set their lineups so win/loss records might not be useful.
> Use this python package to get estimates of how the season would go if everyone sets their lineup:
> https://github.com/riders994/basketball_max_pf. Phase 1 is merged.
>
> [follow-up] Package is available in pypi at max-pf

**Response notes:**
- **Why:** actual W/L is contaminated — some managers never set lineups, so realized results
  under-report roster quality. `max_pf` recomputes each team's optimal-lineup outcome.
- **Added `max-pf` (2.0.0).** It's purpose-built for **9-cat** fantasy basketball; its 9 categories
  match ours exactly; it vendors `fantraxapi` and is literally tested against these league IDs (public,
  no auth). API: `run({"platform":"fantrax","league_id":...}, methodology="hindsight", nash=True) ->
  list[TeamSeason]`. `TeamSeason` = `team_id, name, periods, actual_pf, m1/m2/m3_pf` — `actual_pf` =
  category wins realized; `m1/m2/m3_pf` = **optimal-lineup** ceilings under the 3 objectives
  (catwins/zscore/raw); `nash=True` = the **mutual ceiling** ("if *everyone* optimizes").
- **Built `data/fantrax.py::fetch_success_labels` + `scripts/fetch_fantasy.py` + `make fantasy`** →
  `data/processed/fantrax_success.parquet` (`league_id, team, actual_pf, m1/m2/m3_pf, lineup_gap`).
  The Phase-2 success label = optimal `m1_pf` (not `actual_pf`); `lineup_gap = m1_pf − actual_pf`
  quantifies points left on the table.
- **Validated live:** 30 team-seasons across both leagues; `lineup_gap` flags the unset-lineup teams
  (e.g. ChampagniePapi 6.5→8.0, Clingan On 6.0→8.0 over a 1-week probe). Full-season pull via
  `make fantasy`.
- 14 tests (max_pf.run monkeypatched), ruff clean, **146 total**. Branch `basketball-phase2`.
- **Next:** rosters aren't in `TeamSeason` → build the roster pull (fantraxapi) + name-match fantasy
  rosters → NBA players → Phase-1 archetype membership = **per-team archetype composition**, then
  relate composition → the max_pf success labels.

---

## Entry 046 — Harden the success-label fetch (Fantrax rate-limit)

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> Your previous work was halted by token limits. Please resume.

**Response notes:**
- On resume: the full-season `make fantasy` pull had **failed** — Fantrax **rate-limited** the heavy
  full-season request volume (`"Invalid Request"` raised in max_pf's per-period
  `get_team_roster_info`). The 1-week path is validated, so this is a data-pull robustness issue, not
  core logic. (Code increment from Entry 045 was already committed/pushed — nothing lost.)
- **Hardened `data/fantrax.py`:** extracted `_run_league` with **retry + backoff** (30s × attempt) and
  a **warm box-score cache** (`cache_dir`, config `fantasy.cache_dir=.cache`, gitignored), plus
  **per-league isolation** (one league failing no longer aborts the batch; the manifest records
  `leagues_succeeded`). 1 test passes, ruff clean.
- Re-running the full-season pull in the background with the warm cache + backoff.
- **Next (unchanged):** roster pull (fantraxapi) + name-match fantasy rosters → NBA players → Phase-1
  archetype membership = per-team archetype composition → relate to the max_pf success labels.

---

## Entry 047 — Phase 2: roster composition + M2 grading + name resolution

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> Let's keep going with the roster/comp build now. For the max_pf success labels, prioritize
> M2 > M3 > M1 as the metrics to grade off of. M2 assumes both teams set their lineup, M3 assumes
> maximal management to set lineup, and M1 is just based on actual player competition.
>
> and make sure to look for other name resolution problems between the datasets. Sometimes fonts can
> be different etc.

**Response notes:**
- **Grading = M2 > M3 > M1** (corrected the meanings I'd guessed): `m1` = vs opponents' *actual*
  lineups; `m2` = **both teams optimize** (mutual/Nash ceiling) — the realistic "everyone manages";
  `m3` = maximal management. Set `primary_pf = m2_pf` and `lineup_gap = m2_pf − actual_pf`.
- **Built Phase-2 composition** (`eval/compose.py` + `scripts/compose.py` + `make compose`):
  `fetch_rosters` (fantraxapi `team_roster`, retry/backoff), `team_composition` = name-match each
  team's roster → ESPN archetype membership for the league's season (`season` inferred from
  `roster.period_date`) → **soft archetype shares** `comp_<arch>` (mean membership) + hard counts
  `n_<arch>` + `match_rate`; `build_phase2_table` joins the max_pf success labels.
- **Name resolution hardened (the "fonts" issue = diacritics):** `_norm` now NFKD-folds accents
  (Jokić/Dončić/Şengün/Jović → ascii) on *both* sides, plus punctuation/suffix stripping; added a
  config `fantasy.name_aliases` escape hatch for nicknames and an **unmatched-player diagnostic** that
  `make compose` prints, so residual mismatches surface against real data.
- 19 basketball tests (accent folding, alias, composition math, unmatched), ruff clean, **151 total**.
  Branch `basketball-phase2`.
- **Pending:** the full-season success pull is still running (~53 min, cache 920 MB, progressing) and
  holds Fantrax, so the **live roster fetch + actual mismatch report** runs next once it frees up.

---

## Entry 048 — Extend the archetype pipeline to 2026 + re-derive names

**Date:** 2026-06-27

**Prompt (full text):** _(verbatim)_

> Proceed with the 2026 extension and re-derive names

**Response notes:**
- **Live roster fetch revealed the dominant Phase-2 mismatch was a season gap, not names:** the two
  leagues are different seasons (`blk3...`=2025, `wserh14...`=2026), but the archetypes only went to
  2025. (Name resolution itself is solid — 2025 league 93.7%; accent-folding + 3 nickname aliases.)
- **Extended to 2026** (`data.latest_completed_season` 2025→2026; 2025-26 is complete as of mid-2026).
  Re-ran fetch (`player_season_stats` 273k→298k rows) → build → features (clustering pool 2631→2981)
  → discover. Uses ESPN + local compute only, so no conflict with the running Fantrax success pull.
- **Re-derived the 12 names** (PCA-whitened GMM re-fit reshuffles indices): Off-Ball Wing, 3-and-D
  Wing, High-Usage Primary, Scoring Combo Guard, Rim-Running Center, Foul-Drawing Iso Scorer, Two-Way
  Forward, Slashing Non-Shooter, Lead Playmaker, Interior Big, Connector Wing, and a tiny (n≈12)
  **Non-Shooting Center** `fg3_pct`-artifact cluster (flagged as a consolidation candidate). Re-fit
  healthy: 40% blends, **YoY stability 0.61→0.63**. Membership now covers **2014–2026**.
- **Both leagues now compose** (cached rosters): overall **88%** (2025 93.7%, 2026 83.5%). The lower
  2026 rate is the injury-heavy 2025-26 (Tatum/Lillard/Murray/Beal/VanVleet missed the season) + rookies
  — legitimate eligibility exclusions, not name bugs.
- 151 tests, ruff clean. Branch `basketball-phase2`.
- **Next:** finish the success pull, then `make compose` (join composition → M2 success) → the
  composition-vs-success modeling.

---

## Entry 049 — Yahoo redraft history as Phase-2 augmentation data

**Date:** 2026-06-28

**Prompt (full text):** _(verbatim, with follow-up clarifications)_

> So these two IDs are for a dynasty league. At ~/activity/notebooks/woj_tokens.json there are
> credentials for a Yahoo! use that you can use with the yahoo_fantasy_api python package. That has
> more data

> [clarif.] The two IDs we've been using in previous sessions. I re-did the last command because I
> inadvertently quit the session. … [dynasty = the Fantrax league; Yahoo = personal redraft history:
> a middle-school NBA league + a ~10-year end-of-college NBA league + some football]. … Goal = extra
> Phase-2 examples. Roster snapshot = season-long average. Success label = category-win rate. …
> For the 24-25 fantrax season … it takes a long time to run a season that isn't cached — skip that
> season and move on.

**Response notes:**
- **New Phase-2 data source: the user's Yahoo NBA redraft history** (more team-config→success examples
  than the 2 Fantrax dynasty leagues). Surveyed 25 Yahoo leagues; **split sport by game-key prefix** —
  `Game.league_ids()` is NOT sport-filtered (nba/nfl returned identical 25). Usable corpus = the
  long-running **"H2H Cat One"** league, **exactly 9-cat** (FG%,FT%,3PTM,PTS,REB,AST,ST,BLK,TO):
  8 seasons 2015-16…2023-24 (`353.l.37110`…`428.l.5686`). Excluded: 2014/2016 = H2H **Points**;
  2004/2005 = pre-2014 (outside archetype window); 4-team side league; the user's **NFL** leagues
  ("This League is Roman!"/"DMV"). **Season off-by-one:** Yahoo labels by START year, archetypes by
  END year → **+1** on the join.
- **Built the Yahoo Phase-2 path** mirroring Fantrax (`data/yahoo.py` + `scripts/fetch_yahoo.py` +
  `scripts/compose_yahoo.py` + `make fetch-yahoo`/`compose-yahoo`): OAuth from
  `~/activity/notebooks/woj_tokens.json`; `fetch_team_weeks` (per team × regular-season week roster,
  per-league cache + backoff); `tally_category_wins`/`fetch_labels` (per-week `stat_winners` →
  category-win rate, ties=0.5, + final standings); **`season_long_composition`** = archetype
  membership **weighted by weeks-on-roster** (reuses `_norm` + `fantasy.name_aliases`).
- **Validated end-to-end on one league** (2023-24 Sauron's): season=2024 ✓, `cat_win_rate` mean
  exactly 0.500 (symmetric) ✓, weeks-weighted match rate **96.9%** (unmatched = injury/suspension DNPs
  like Morant/Simmons/Rob Williams — legit eligibility exclusions), comp shares sum ~1.0, clean join.
- Added deps `yahoo_fantasy_api` + `yahoo-oauth` (root `pyproject.toml`); **8 new tests** (off-by-one,
  regular-week filter, cat-win tally incl. ties/playoff-skip, weeks-weighted composition), ruff clean.
- **Dropped** the 24-25 Fantrax `max_pf` cache idea (`~/activity/basketball_max_pf`) per user — that
  season stays unlabeled (left-join handles the gap). Branch `basketball-phase2`.
- **Next (user-run, heavy):** `make fetch-yahoo` (~1800 calls, cached/resumable) → `make compose-yahoo`
  → fold the ~88 Yahoo team-seasons into the composition-vs-success modeling.

---

## Entry 050 — Phase-2 modeling → simulation diagnosis (representation is the bottleneck)

**Date:** 2026-06-28

**Prompt (full text):** _(verbatim, condensed across the arc)_

> [pull it] make fetch-yahoo … make compose-yahoo … start the Phase-2 modeling with the 88 examples …
> Commit this, then let's try final standings rank. Is there another way to rank? … One other path to
> consider … bootstrap teams by looking at mock draft articles close to the beginning of each season …
> [draft prior = Both in parallel] … Reminder that reality includes a mix of drafting behaviors. Some
> teams auto-draft, some strategize. … I need to restart the computer … hit resume … same results.

**Response notes:**
- **Full Yahoo pull:** `make fetch-yahoo` (8 leagues, 88 team-seasons, 29,597 player-weeks; one league
  throttled `Request denied` → resumed from cache) → `make compose-yahoo` (97% weeks-weighted match,
  mean cat-win-rate 0.500). Manifest committed (`8bf6b93`).
- **Composition→success model** (`eval/phase2_model.py` + `make model-phase2`): leave-one-league-out CV.
  **Four-way NULL** — archetype *shares* don't predict success for **any** target (`cat_win_rate`,
  `reg_win_pct`, `rank_score`): Ridge/Lasso oof R²≈0, GBM overfits, within-league ρ≈0/neg. Added
  `derive_targets` + within-league Spearman. Commits `a1dab9d`, `63eacc1`.
- **Draft simulator** (`eval/simulate.py`, validated, UNCOMMITTED): leak-safe prior-season 9-cat value
  draft + punt strategies + **auto-draft(~30%)/manager mix** (per-team softmax temperature), round-robin
  H2H scoring from players' *actual* stats, archetypes joined by `athlete_id`. **Representation shootout**
  (leave-one-season-out oof R² on `sim_cat_win_rate`): archetype shares **−0.007**, prior coverage
  **+0.018**, **actual coverage +0.358**; `punt_ft` best build (0.544, ~8σ). ⇒ (1) signal is real,
  (2) **archetype shares are the wrong success representation** (model coverage instead), (3) the real
  ceiling is **draft-time projection**.
- **Paused at a 3-option direction decision** (coverage model+optimizer / attack projection / finalize
  sim+contribution matrix) — see memory `basketball-phase2-RESUME`; user restarting, wants to resume here.

---

## Entry 051 — Phase-2 coverage model + roster optimizer (resume the paused decision)

**Date:** 2026-07-01

**Prompt (full text):**

> resume [then, at the direction question:] Coverage model + optimizer

**Response notes:**
- Resumed from `basketball-phase2-RESUME`; re-presented the sim shootout and re-asked the 3-way
  direction question. User chose **coverage model + optimizer**. Two commits on `basketball-phase2`.
- **Commit A — sim finalized + shootout committed** (`2a99a69`): the previously-uncommitted
  `eval/simulate.py` was finalized. Added `team_coverage()` (per-team 9-cat z-profile, ACTUAL season-N
  and PRIOR season-N-1/leak-safe) → sim table now carries `cov_act_*`/`cov_pri_*` beside archetype
  shares. Added `representation_shootout()` to `phase2_model.py` (leave-one-season-out Ridge): archetype
  shares **+0.009** (null), prior coverage **+0.044**, actual coverage **+0.370** — reproduces the
  diagnosis (shares are the wrong representation; coverage is the mechanism; draft-time projection is
  the binding constraint). `strategy_leaderboard()` = punt positive control (`punt_ft` 0.541, z=+10).
  `scripts/simulate_phase2.py` + `make simulate` + `REPORT_phase2_simulation.md`. 9 tests.
- **Commit B — roster optimizer** (`eval/optimize.py`): objective = maximize projected coverage via a
  field-calibrated win map, `mean_c Phi((cov_c-mu_c)/sigma_c)` over *contested* cats (Gaussian-CDF ⇒
  punt-aware: stops over-investing locked cats). `coverage_picker` plugs into `simulate_draft` via a new
  `pickers` hook; `optimize_roster` = standalone greedy best build; `evaluate_in_sim` = seat an
  optimizer team vs the manager/auto field. **Validated in-sim:** optimizer beats field, most on punt
  builds — `punt_ft` +0.040 lift, top-of-field 22% (≈2.6× the 1/12 random). Balanced lift small by
  design (projection ceiling). `scripts/optimize_phase2.py` + `make optimize` +
  `REPORT_phase2_optimizer.md`. 6 tests. 46 basketball tests pass, ruff clean.

---

## Entry 052 — Phase 3: next-season archetype predictor (YOE vs true-age, two models)

**Date:** 2026-07-01

**Prompt (full text):**

> Let's start phase 3 [branch base: "I merged phase 2, go off primary"; age sourcing: "Do two separate
> models. One with YOE instead of age, one where ages are fetched with nba_api"] … One thing to consider
> for archetypes. Are we giving more weight to players taken at the top of the draft? [decision: "Leave
> Phase 3 as-is"]

**Response notes:**
- Branched `basketball-phase3` off freshly-merged primary (Phase 2 = PR #20). Phase 3 = leak-safe
  **N→N+1** archetype prediction, returning players only; deliverable feeds projected archetypes into
  the Phase-2 optimizer.
- **Commit A (`123bca5`) — Model A (YOE):** `predict/build.py` (N→N+1 table: soft membership `p0..p11`,
  `top_prob`/`entropy`, 19 style z, one-year trajectory deltas, archetype tenure, YOE proxy =
  season−first-seen; target = arch + soft vector in N+1) + `predict/model.py` (persistence + marginal
  baselines, walk-forward multiclass GBM, top-1/macroF1/log-loss/Brier). `make predict`, 4 tests.
- **Age data:** `data/bio.py` + `make fetch-bio` — map ESPN athlete_id→nba_api id by normalized name
  (99.4% unambiguous), fetch birthdates (cached; 967 matched), derive per-season age. **Coverage 99.6%
  across all seasons** (vs ~50% from ESPN DOB alone). 4 monkeypatched tests.
- **Finding (both models):** persistence is a very strong argmax baseline (0.638) that **neither model
  beats on top-1** (YOE 0.605, age 0.608) — archetypes are sticky. But both **halve log-loss** (~1.32 vs
  2.28) → far better-calibrated soft membership, which is what Phase 2 consumes. **True age ≈ YOE
  proxy** (acc +0.003), and neither age nor trajectory cracks the top features ⇒ the plan's "age is the
  key signal" hypothesis is **not supported**; current membership+style dominate.
- **Draft-weighting question:** confirmed the whole pipeline is **equal-weighted** per player-season
  (only a 15-mpg/20-gp gate). Per user, left Phase 3 equal-weighted but **noted** the top-of-draft
  consideration in the report (Phase 2 is already value-centric via the draft prior).
- Commit B: unified `REPORT_phase3_predict.md` (A vs B vs baselines), `data/bio.py`, `scripts/fetch_bio.py`.
  54 basketball tests pass, ruff clean.

---

## Entry 053 — Phase-1 archetype refit: fg3_pct gating + k=13 (fix artifact + cluster 2)

**Date:** 2026-07-01

**Prompt (full text):**

> Let's go back to the archetype model, I'm not interested in predicting 9-cat production … [validate]
> Re-examine / validate … yes, prototype the fg3_pct attempt-gating and re-fit … Let's try tuning the
> threshold … including 0.5 3PA/G … fix cluster 2 … k=13 + gate (real split) … I merged phase 3, can
> you fix up the branching properly?

**Response notes:**
- **Validated the current k=12 taxonomy** (read-only): 4 coherent archetypes (Rim-Running Center, 3-and-D
  Wing, Foul-Drawing Iso, Slashing Non-Shooter) but two real defects — (1) a degenerate n≈15
  "Non-Shooting Center" cluster driven by `fg3_pct` noise (bigs who hit 1 of ~1 threes → `fg3_pct_z`
  +4.3, stay-rate 0.00), and (2) "High-Usage Primary" conflating Jokić/LeBron with Ish Smith/Elfrid
  Payton via ast+tov. Plus over-splitting in guard/wing space.
- **fg3_pct attempt-gating:** neutralize `fg3_pct` (→ NaN → 0 z) below a season-3PA floor; volume is
  still captured by `fg3a_rate`/`fg3a36`. Threshold tuning: **30 total 3PA** beat 25/50 and both
  per-game options (0.5 3PA/G was *worst* on stability — reliability is about sample size, not rate).
  Kills the artifact (max |fg3_pct_z| 4.3→1.1), stability held ~0.62.
- **Cluster 2 fix:** adding a "usage" feature is futile (it's a linear combo already in the PCA span);
  the lever is **k**. k=13 cleanly separates ball-dominant engines (Jokić/SGA/Giannis/Luka/Embiid) from
  low-usage distributors (Simmons/Payton/Draymond) into disjoint clusters; k=14 over-splits. Cost:
  YoY stability 0.63→0.57 (accepted — genuine granularity).
- **Productionized:** `features/build.py` gate + `features.min_3pa_for_fg3_pct: 30` and `archetypes.k: 13`
  in config; re-ran features→discover; **re-derived all 13 names** (verified mapping: Jokić→High-Usage
  Engine, Simmons→Non-Scoring Playmaker, Gobert/DeAndre Jordan→Rim-Running Center, no artifact). Report
  regenerated. 54 tests pass, ruff clean. Branch `basketball-archetype-refit`.
- **Branching:** branched off primary before Phase 3 merged; after the user merged Phase 3 (PR #21),
  re-based the branch onto updated primary (stash → reset --hard origin/primary → pop; no own-commits so
  clean). Branch now carries the full Phase 1-3 stack as base.
- **Cascade (follow-up):** the new taxonomy changes membership → Phase 2 (`compose`/`simulate`) and
  Phase 3 (`predict`) artifacts should be regenerated against it.

---

## Entry 054 — Round-by-round archetype weights: negative result → ship β as draft intel

**Date:** 2026-07-01

**Prompt (full text):**

> Where are we with soft vs hard archetypes … close the consolidation item with the rationale … Now
> let's move on to the round by round archetype selection weights … [prescriptive] … but what if those
> archetype shares are weighted depending on how good the player is? … show me the learned per-archetype
> exposure coefficients first … yes, build the round-by-round weights … [result: they lose] Ship β as
> draft intel only

**Response notes:**
- **Soft vs hard:** documented that soft membership is the representation of record and the planned
  soft→hard consolidation is decided-against (continuous space, ~half blends); closed the item in
  discover.py + PROJECT_PLAN §3/§8 (Entry 053 commit + a docs commit).
- **User's key insight tested:** equal-weighted archetype shares were the Phase-2 null; weighting by
  player value helps **only if you keep magnitude** — value-weighted *shares* (normalized) stayed ~0.05,
  but value-weighted **exposure** (`Σ value·p`, un-normalized) hit oof R² **0.30** (actual), near coverage
  (0.37). So archetype composition *does* carry success signal once value-weighted and un-normalized.
- **Learned per-archetype exposure coefficients** (draft-time/prior): PRIORITIZE High-Usage Engine
  (+0.020) & Wing Shot-Creator (+0.016); solid mid-tier; **trap** = Lead Scoring Guard (draft coef ~0,
  actual +0.023, top-bot Δexp −2.45 — empty scoring); **avoid** = Low-Usage Wing (−0.005, stable).
- **Round-by-round weights built and evaluated — NEGATIVE RESULT.** A picker scoring
  `projected_value × (dynamic_weight·archetype)` with diminishing returns **lost to the field** (lift
  −0.02, worst of three); static weighting only broke even; even pure-value lost (field is 70% punters).
  Balancing across archetypes sacrifices the value/coverage **concentration** (punt builds) that wins
  9-cat. The draft **engine** stays the coverage optimizer.
- **Shipped (per user): β as draft intel only.** `eval/archetype_value.py` (value-weighted exposure +
  bootstrap-stable coefficients + PRIORITIZE/trap/avoid labels), `scripts/archetype_value.py` +
  `make archetype-value` + `REPORT_archetype_value.md`. 4 tests, 58 basketball tests pass, ruff clean.
  Branch `basketball-archetype-refit` (stacked on the k=13 refit).

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

---

<!-- Template for new entries:

## Entry NNN — <short title>

**Date:** YYYY-MM-DD

**Prompt (full text):**

> ...

**Response notes:**
- ...

-->
