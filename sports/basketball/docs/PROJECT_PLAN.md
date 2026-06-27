# Basketball (NBA) — fantasy archetype project · PROJECT_PLAN

A standalone project under `sports/basketball/` (own code, data, pipeline, reports), reusing the
repo's reproducibility / config-driven / prompt-log conventions but **not** the football modeling
pipeline — this is a different problem shape (clustering + composition + a predictor), not
next-season rank regression.

Status: **planning** (2026-06-27). Nothing is built yet; this doc is the design of record. Full
prompt provenance in `docs/PROMPT_LOG.md` (per-sport) + the repo-wide `../../docs/PROMPT_LOG.md`.

---

## 1. Goal & scope

Help build a **winning fantasy-basketball roster** by thinking in *play-style archetypes* rather
than the traditional five positions (inspired by the "11–14 real positions" idea — Alagappan's
topological "13 positions", and the broader role-taxonomy literature). Three phases:

1. **Archetype discovery** (unsupervised) — define ~11–14 style archetypes from how players actually
   play.
2. **Archetype composition → fantasy success** — learn which *groupings of archetypes* on a fantasy
   roster produce successful **9-category** seasons.
3. **Next-season archetype predictor** (supervised) — from a player's past data, predict their
   archetype(s) for next season.

**Production flow:** Phase 3 predicts each player's *upcoming* archetype(s) → Phase 2 scores roster
compositions → you draft players whose **predicted** archetypes form a winning category portfolio.

**Fantasy format:** 9-cat (FG%, FT%, 3PM, PTS, REB, AST, STL, BLK, TO) — roto/H2H. The exact league
settings come from the two sample leagues (§2.3).

**Scope guards:** returning players only in v1 (rookies are a separate future model on
college/HS + draft-coverage soft data); modern game only for *defining* archetypes (§2.2).

---

## 2. Data

### 2.1 Sources (layered, cheapest-broadest first)
- **Base — hoopR / sportsdataverse:** player-season box + advanced totals, broad coverage; the
  backbone (analogous to nflreadpy's role in football).
- **Enrich — Basketball-Reference:** advanced *rate* stats (USG%, TS%, OREB%/DREB%, AST%, STL%,
  BLK%, TOV%) + shot-location splits — high signal for *style*.
- **Granular — nba_api (stats.nba.com):** play-type frequencies (Synergy), tracking (passing,
  contested rebounds, defensive matchups), shot zones — the highest-signal style features. Shallower
  history (tracking ~2013–14+, Synergy play-type ~2015–16+) and a flakier/rate-limited API, so it's
  the last layer.
- **Fantasy — Fantrax** (Phase 2): two sample leagues, IDs `blk3bn3clw9njuhc` and
  `wserh14rmbbpqtcg`. Source of real team-configuration examples + the target league settings.
  **Both leagues are public**, so reads via the unofficial API / `fantraxapi` package need **no
  authentication** (no login cookie). Settings (teams, roto-vs-H2H, roster slots, cats, history
  depth) to be pulled from these leagues.

### 2.2 Unit, features, eras
- **Unit:** player-season. Features are **rate / per-100-possession / profile-share** (style, not
  volume — a star and a role player can share an archetype). Eligibility floor on minutes/games
  (TBD in EDA).
- **Eras** (style is era-dependent; the 2013 floor keeps tracking features available across the
  whole span):
  - **E1: 2013–14 → 2016–17** (SportVU begins; pace-and-space ramp)
  - **E2: 2017–18 → 2019–20** (3-pt explosion; switch-everything defense)
  - **E3: 2020–21 → present** (post-bubble modern)
  - Boundaries are provisional — **revisit if results are weak.**
- **Era usage by phase:**
  - **Phase 1 (define archetypes): E2 + E3 only** (the modern game).
  - **Phase 2 & 3 (depth): E1 + E2 + E3** (all 2013+). Earlier-era player-seasons are *assigned*
    archetypes by the Phase-1 model (clean, because the feature set exists across 2013+;
    era-relative normalization expresses an E1 player in the modern archetype vocabulary). Minor
    tier: Synergy play-type starts ~2015–16, so 2013–15 seasons are slightly thinner.
- **Floor: 2013.** No data before the 2013–14 season in any phase.
- **Normalization:** era-relative z-scores so archetypes are comparable across the 2013+ window while
  reflecting each era's baseline.

### 2.3 Reproducibility
Cache + manifests, config-driven stages, committed reports/docs — same discipline as football.

---

## 3. Phase 1 — Archetype discovery

- **Feature families (all rate-based, style not volume):**
  - *Shot profile* — rim / mid / 3-pt attempt rates, FT rate, shot-zone distribution.
  - *Playmaking* — AST%, USG%, TOV%, passing-tracking.
  - *Rebounding* — OREB% / DREB%.
  - *Defense* — STL%, BLK%, defensive matchup / position proxies.
  - *Finishing / efficiency* — TS%, rim FG%.
- **Method:** **soft clustering (Gaussian mixture)** → per-player-season **membership vector**
  ("70% stretch-big / 30% rim-runner"). Soft first because players are blends and it feeds Phase 3's
  target naturally. `k` chosen empirically (BIC + silhouette + interpretability) targeting **~11–14**.
- **Soft → hard:** after EDA, **consolidate the soft clusters into a hard archetype set** (the
  human-named taxonomy). Keep soft membership available for Phase 2 blends / Phase 3 target.
- **Validation:** cross-season **stability** (consistent definitions year-to-year — a hard
  requirement because Phase 3's target is derived from these labels), and basketball sensibility
  (named, interpretable archetypes: e.g. 3-and-D wing, stretch big, rim-runner, primary initiator,
  off-ball combo guard, …).

**Deliverable:** named archetype taxonomy + per-player-season membership (soft vector + hard label).

---

## 4. Phase 2 — Archetype composition → 9-cat success

- **Framing:** 9-cat is **category-portfolio construction** — each archetype has a distinct category
  *signature* (rim-runner: REB/BLK/FG%↑, FT%↓, 3PM↓; primary initiator: AST/3PM/STL/FT%↑, TOV
  risk↑; 3-and-D wing: 3PM/STL↑, low TOV). "Which archetype groupings win" ≈ which mixes give
  balanced category coverage — explicitly including **punt strategies** (the 9-cat meta).
- **Success labels (real, not pure simulation):** from the **two Fantrax leagues** — extract each
  fantasy team's roster → its **archetype composition** (via Phase-1 assignment) and its **outcome**
  (roto standings / category z-totals / H2H record, per the leagues' actual settings). This grounds
  "success" in real rosters rather than synthetic drafts.
- **Models / deliverables:**
  - Archetype × category **contribution matrix** (expected per-cat value by archetype).
  - A model of **composition → standing** (which archetype mixes win, incl. punt builds).
  - A **roster optimizer** that targets category coverage (or a chosen punt build).

**Open design:** how to define "success" precisely from league data (final standing vs season-long
category z-totals vs H2H win%); small-sample handling (only 2 leagues × N seasons — may augment with
simulated fields seeded by real settings).

---

## 5. Phase 3 — Next-season archetype predictor

- **Supervised, leak-safe N→N+1** (reuses football's discipline): features from history *through
  season N*; target = archetype membership in *N+1*.
- **Target type:** follows Phase-1's soft→hard outcome — predict the **soft membership vector**
  (compositional / calibrated multiclass) and/or the **hard label**. Soft handles "archetype**(s)**".
- **Baseline to beat: persistence** ("same archetype as last season") — archetypes are sticky, so
  this is the must-beat floor (cf. football's persistence baseline).
- **Signal:** **age + multi-year style trajectory** is expected to carry most of the lift over
  persistence (archetypes drift predictably: athletic slasher → spot-up shooter; rim-runner →
  stretch big; high-usage initiator → off-ball). Walk-forward validation across the 2013+ span.
- **Scope:** returning players only (rookies deferred, §1).
- **Dependency:** prediction error compounds Phase-1 clustering noise → raises the bar on archetype
  **stability** (§3).

**Deliverable:** per-player projected next-season archetype(s) → consumed by Phase 2 at draft time.

---

## 6. Conventions
- Standalone project; config-driven, reproducible stages; committed reports + version snapshots.
- Prompt log maintained in **both** this per-sport log and the repo-wide primary (basketball-domain
  prompts in both; entry numbers are global/shared).
- Tests + lint as in football.

---

## 7. Open questions / parking lot
- **Fantrax access** — leagues are public (no auth); confirm the read path (`fantraxapi` /
  endpoints) and extract settings + rosters/standings.
- **Era boundaries** — provisional (§2.2); reevaluate if archetypes/results are weak.
- **`k`** and the **soft→hard consolidation** — resolved in Phase-1 EDA.
- **Phase-2 "success" definition** — standings vs category z-totals vs H2H; 2-league small-sample
  augmentation.
- **Eligibility floor** (minutes/games) — set in EDA.
- **Rookie model** — deferred (separate college/HS + draft-coverage project).
