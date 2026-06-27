# Prompt Log — Basketball

A chronological record of every prompt related to the **basketball** project, with notes on the
response. Maintained for **reproducibility and decision provenance**.

> **Scope.** This is the basketball-domain view. The repo-wide superset (all sports + repo-level
> infrastructure prompts) lives in the root [`docs/PROMPT_LOG.md`](../../../docs/PROMPT_LOG.md).
> Entry numbers are **global** and shared with the primary, so they are stable for cross-referencing
> but **non-contiguous here** (basketball starts at Entry 039; lower numbers are football/repo-level).
>
> Convention: append every basketball prompt to **both** this log and the primary, quoted **in full**,
> primary first; keep response notes concise but specific.

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
- **3 phases.** (1) Archetype discovery — unsupervised, **soft GMM** membership → consolidate to a
  hard taxonomy after EDA, ~11–14 archetypes ("13 positions" inspiration). (2) Archetype composition
  → **9-cat** fantasy success, with real labels from the two Fantrax leagues (category-portfolio /
  punt-aware roster construction). (3) Next-season archetype predictor — supervised, **leak-safe
  N→N+1**, persistence baseline, age/trajectory the key signal. Production flow: predict archetypes →
  score compositions → draft.
- **Eras / scope:** floor **2013** (tracking era → rich features available throughout). E1 2013–17 /
  E2 2017–20 / E3 2020–present (provisional, reevaluate). **Phase 1 = E2+E3** (modern only); **Phase
  2/3 = all 2013+** (earlier seasons get archetypes assigned by the Phase-1 model). Returning players
  only; rookies deferred to a separate college/HS + draft-coverage model.
- **Data:** layered hoopR → Basketball-Reference → nba_api. Fantrax leagues `blk3bn3clw9njuhc` /
  `wserh14rmbbpqtcg` are **public** → no auth/cookie needed.
- Branch `basketball-plan`. Planning only — no pipeline code yet (next step: confirm/scaffold the
  stage layout + start data fetch).

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

<!-- Template for new entries:

## Entry NNN — <short title>

**Date:** YYYY-MM-DD

**Prompt (full text):**

> ...

**Response notes:**
- ...

-->
