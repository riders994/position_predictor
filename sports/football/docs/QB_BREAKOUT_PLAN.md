# Late-Breakout QB — Project Plan

> Identify quarterbacks who will break out **after the league has written them off**, using
> evidence available **before they ever played an NFL snap**.
> Sibling project to the fantasy-rank predictor in the same `sports/football` tree.
> Decision trail: [`PROMPT_LOG.md`](./PROMPT_LOG.md).

---

## 1. Problem statement

We are in the era of the late-breakout QB — Sam Darnold, Baker Mayfield, Geno Smith, Ryan
Tannehill. Each spent years as a punchline before becoming a genuinely valuable fantasy asset,
usually somewhere other than the team that drafted them. The question this project asks is
whether that outcome was visible in advance.

**Hard constraint on the modelling.** Final models may use only **pre-NFL** features: high-school
recruiting profile and college production. NFL data is used to *build the label* and to describe
what the cohort looked like, never as a model input.

This is a deliberate handicap, and it is the point. The sibling `position_predictor` project
already predicts next-season fantasy rank from NFL history and explicitly excludes rookies. A
model that waits for NFL evidence cannot answer "who is worth acquiring before anyone else
notices", and — as §3.4 shows — early NFL production barely separates a future late breakout
from a bust anyway. If the signal exists at all, it is upstream.

### 1.1 Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Breakout tier | **Top-20 PPR PPG** among eligible QBs | Superflex is the relevant format; QB15 is where draft-day value actually lives. |
| Breakout event | **First season the tier holds** (top-20 in ≥2 of 3 seasons from it) | A single top-20 season is rank 20 of ~32 QBs who play — too weak to mean anything. See §3.2. |
| "Late" | Breakout in **NFL year 4+** | The rookie contract's cheap years expired before the player was any good. |
| Second axis | **`relocated`** — breakout with a franchise other than the drafting one | Separates "developed late" from "needed a new building". |
| Eligibility | **≥7 games** for a season to be rankable | Reuses the QB cutoff the sibling project derived analytically. |
| Cohort | QBs entering the NFL **1999+** | nflverse weekly stats start in 1999; earlier careers cannot be indexed. |
| Model features | **Pre-NFL only** (high school + college) | The project's defining constraint (§1). |
| 2020 | **Kept** | A breakout is a career milestone; dropping COVID would erase real first seasons. |

### 1.2 Non-goals

- Predicting *when* a breakout lands (year 4 vs year 8). N is far too small.
- Ranking QBs for a current-season fantasy draft — that is `position_predictor`'s job.
- Evaluating NFL team/coaching quality as a cause. `relocated` records that a change happened,
  not that a particular building caused it.

---

## 2. Data

### 2.1 NFL side — label only

Reuses the caches the sibling pipeline already maintains (`make -C sports/football fetch`):
`weekly` (1999–), `draft_picks`, `rosters`. No new fetching. `draft_picks.cfb_player_id` is the
bridge to college data.

### 2.2 Pre-NFL side — the model's actual inputs

| Layer | Source | Key | Coverage | Notes |
|---|---|---|---|---|
| High school — recruiting | **ESPN recruiting API** (`sports.core.api.espn.com`) | name + class year | ~2006– | No API key. Grade, national/position/state/region rank, camp measurables (40, 3-cone, shuttle, vertical), HS name + city/state. |
| High school — recruiting (alt) | 247Sports composite via **CFBD** | name + class | 2000– | Needs a free API key. Adds stars and a consensus composite. |
| High school — box scores | **best-effort scrape** | name + HS | partial, ~2012+ | User-accepted: partial data beats none. Coverage is biased toward larger programs, so missingness must be modelled explicitly, not imputed away. |
| College — production | **cfbfastR-data** parquet (public GitHub), `sportsdataverse.cfb` | `cfb_player_id` / name+school | 2003– | Play-by-play → per-season passing/rushing, efficiency, competition level. |
| College — roster/bio | `sportsdataverse.cfb.load_cfb_rosters` | `athlete_id` | 2003– | Height/weight, hometown, `recruit_ids` linking to the recruiting layer. |

**Coverage is the binding constraint.** Recruiting data effectively begins around 2006, which
means QBs entering the NFL from roughly 2010. That is the era where the *feature* side is
complete — but the *label* side needs 4+ elapsed seasons, so the usable modelling window is
narrow and the analysis window (1999+) is deliberately wider than it.

### 2.3 Identity joins

There is no single key spanning high school → college → NFL. The chain is:

`draft_picks.cfb_player_id` → college roster → `recruit_ids` → recruiting profile, with
name + college + class-year fuzzy matching as the fallback and undrafted QBs (Romo, Keenum,
Hill) reachable only by the fallback. **Join quality is reported, not assumed** — an unmatched
QB is recorded as unmatched rather than silently dropped from the cohort.

---

## 3. Label construction

### 3.1 The four cells

Crossing `late` with `relocated` gives the structure the analysis is built around:

|  | same franchise | relocated |
|---|---|---|
| **on-time** | on-time franchise QB | early bloomer, moved |
| **late** | slow burn, same building (Rodgers, Cousins) | **the Darnold/Geno/Baker cell** |

### 3.2 Why "first top-20 season" is the wrong event

Three definitions were built and scored against QBs whose careers are not in dispute:

| Definition | Calls Mayfield late? | Tannehill? | Brady? | Allen/Burrow/Purdy? |
|---|---|---|---|---|
| First top-20 (`late`) | ✗ (year 1) | ✗ (year 3) | ✓ on-time | ✓ on-time |
| First top-12 (`late_qb1`) | ✓ | ✗ | ✗ — calls Brady **late** | ✓ on-time |
| **First sustained top-20 (`late_sustained`)** | ✓ year 6 | ✓ year 8 | ✓ on-time | ✓ on-time |

`late` fails because the bar is rank 20 of ~32 QBs who play: **Mayfield's 2018 rookie season ranks
exactly 20th**, so the archetypal late bloomer reads as on-time. `late_qb1` over-corrects and
calls Tom Brady a late breakout on a fantasy technicality. Only the sustained definition —
top-20 in ≥2 of the 3 seasons starting with the breakout — matches every unambiguous case.
**`late_sustained` is the modelling target**; the other two ship alongside as diagnostics.

### 3.3 Censoring, in both directions

- **Left:** careers starting before 1999 are dropped, not imputed. Entry season is resolved from
  `draft_picks.season`, then `rosters.entry_year`, then first observed season — and only when
  that season is inside the window. Without this, Vinny Testaverde (1987 #1 overall) reads as a
  1999 rookie.
- **Right:** a QB too recent to have reached year 4, or holding one breakout season whose
  3-season window is still open, is **censored** — excluded from base rates rather than counted
  as a failure. Sam Darnold is the live example.

### 3.4 The finding that justifies the project

Through three NFL seasons, future late breakouts look much more like busts than like on-time
breakouts (median PPG 5.97 vs 4.18 for busts and 15.68 for on-time; median early rank at the
unranked floor for both the late and never groups). Early NFL production does not separate them.
Whatever distinguishes a future late breakout is either upstream of the NFL or not in the box
score at all.

---

## 4. Stages

| # | Stage | Output | Status |
|---|---|---|---|
| 1 | Cohort + labels + descriptive analysis | `qb_breakout_careers.parquet`, `REPORT_qb_breakout_cohort.md` | **done** |
| 2 | High-school recruiting layer | `recruiting.parquet` + join-quality report | next |
| 3 | Best-effort HS box-score scrape | `hs_stats.parquet` + coverage report | planned |
| 4 | College production layer | `college_qb_seasons.parquet` | planned |
| 5 | Archetypes (clustering on pre-NFL features) | archetype assignments + profiles | planned |
| 6 | Pre-NFL-only model of P(late breakout) | model + honest validation | planned |

---

## 5. Modelling, and the small-N problem

**17 late breakouts** across 1999–2025 is the honest ceiling, and the recruiting-data era cuts it
further. This governs every methodological choice:

- **Evidence-weighting, not prediction at scale.** The deliverable is calibrated priors and
  interpretable archetype effects, not a leaderboard implying precision that N cannot support.
- **Leave-one-out / repeated stratified CV**, never a single split — one held-out fold would
  contain one or two positives.
- **Nested outcome, not one binary.** `ever_breakout` (n≈69) is far better powered than
  `late | breakout` (n=17). Model the well-powered part first and treat lateness as a
  conditional second stage.
- **Bootstrap every coefficient.** Report intervals; a feature that flips sign across resamples
  is noise, and at this N many will.
- **A negative result is a real result.** If pre-NFL features carry no signal for lateness, that
  is worth knowing and will be reported as such rather than tuned around.
- **Draft capital is a confound, not a feature to celebrate.** It encodes the league's own
  scouting opinion of the same college tape. Models are reported with and without it so the
  incremental value of high-school and college evidence is visible on its own.

---

## 6. Reproducibility

Same conventions as the sibling project: parquet caches with JSON manifests, deterministic seeds,
committed reports under `reports/`, tests under `tests/`. Scraped HS data is cached with its
retrieval date and source URL per row, since it is the one layer that cannot be re-derived
identically later.
