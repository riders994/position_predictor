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

<!-- Template for new entries:

## Entry NNN — <short title>

**Date:** YYYY-MM-DD

**Prompt (full text):**

> ...

**Response notes:**
- ...

-->
