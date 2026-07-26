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

**Hard constraint on the modelling.** Final models may use only **pre-NFL** features. In practice
that means **college production**: the high-school layer was built, measured, and set aside
(§2.3). NFL data is used to *build the label* and to describe what the cohort looked like, never
as a model input.

This is a deliberate handicap, and it is the point. The sibling `position_predictor` project
already predicts next-season fantasy rank from NFL history and explicitly excludes rookies. A
model that waits for NFL evidence cannot answer "who is worth acquiring before anyone else
notices", and — as §3.4 shows — early NFL production barely separates a future late breakout
from a bust anyway. If the signal exists at all, it is upstream.

### 1.1 Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Breakout tier | **Top-15 PPR PPG** (quality bar) **held at top-20** (relevance bar) | QB15 is where genuine draft-day value starts; QB20 is still a startable superflex asset. One number cannot carry both claims — see §3.2. |
| Breakout event | **First top-15 season with top-20 in ≥2 of the 3 seasons from it** | Trigger high, confirm lower. A single top-20 season is rank 20 of ~32 QBs who play; requiring two top-15 seasons deletes the archetypes. |
| "Late" | Breakout in **NFL year 4+** | The rookie contract's cheap years expired before the player was any good. |
| Second axis | **`relocated`** — breakout with a franchise other than the drafting one | Separates "developed late" from "needed a new building". |
| Eligibility | **≥7 games** for a season to be rankable | Reuses the QB cutoff the sibling project derived analytically. |
| Cohort | QBs entering the NFL **1999+** | nflverse weekly stats start in 1999; earlier careers cannot be indexed. |
| Model features | **Pre-NFL only — college production** | The project's defining constraint (§1). High school dropped after measurement (§2.3). |
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

| Layer | Status | Source | Key | Coverage |
|---|---|---|---|---|
| **College — production** | **primary (fitting)** | **cfbfastR-data** parquet (public GitHub) | name + school | **2004–2021** |
| College — current seasons | **scoring extension** | **CFBD** `/stats/player/season` + `/ppa/players/season` (free key) | name + school | **2013–current** |
| College — roster/bio | available | `sportsdataverse.cfb.load_cfb_rosters` | `athlete_id` | 2003– |
| High school — recruiting | built, **demoted** (§2.3) | ESPN recruiting API (no key) | name + class year | 2006– |
| High school — box scores | **cancelled** (§2.3) | — | — | — |
| High school — 247 composite | not pursued | CFBD (needs a free key) | name + class | 2000– |

**Coverage is the binding constraint, and it is what decided the college-only design.** College
play-by-play reaches QBs entering the NFL from about 2005; recruiting starts with the 2006 class
and cannot reach earlier than about 2010. Since the label additionally needs 4+ elapsed NFL
seasons, the earlier floor is worth a great deal — see §2.3 for the count.

Two boundaries on the college source, both found by building it and both real:

- **Floor 2004, not 2002.** The 2002 and 2003 files exist but ship an older, thinner schema (366
  columns vs 405) with no `completion`, `pass_td`, `rush_td` or `EPA_success`, so no QB-season
  can be assembled from them. Careers straddling 2004 are **clipped** — Aaron Rodgers reads as one
  college season because only 2004 is in range — so `college_career_truncated` flags them (30 of
  the matched cohort, 2 late breakouts). Their `career_*` totals are unusable; their `final_*`
  block is fine, which is the argument for leaning the feature set on final-season form.
- **Ceiling 2021.** cfbfastR-data publishes no further, and `sportsdataverse` reads the same repo,
  so this is the source's limit rather than a fetch bug. It costs **nothing for fitting** (a QB
  whose last college season is 2022+ enters the NFL in 2023+ and is right-censored anyway), but it
  did mean the layer could not score today's prospects. **Resolved in stage 4** by adding CFBD
  (§2.5), which covers the current season.

### 2.5 CFBD: the current-season extension, and what may not be spliced

CFBD covers through the present season and closes the scoring gap. It is *not* interchangeable
with cfbfastR, and that was measured on the 2013–2021 overlap (1,774 joined QB-seasons) rather
than assumed — full evidence in `REPORT_qb_breakout_cfbd.md`.

**Volume agrees** (attempts/yards/TD correlate 0.99). Two systematic offsets both have
explanations: CFBD charges sacks as rushing attempts per NCAA convention (the ~16-attempt gap
correlates 0.78 with cfbfastR's sack count), and **cfbfastR runs low on totals** because its
play-by-play has game gaps — it averages 8.8 games per QB-season. Where both exist, CFBD's
official season totals are the more accurate; cfbfastR's per-play *rates* are unaffected.

**Efficiency does not agree.** Regressing each CFBD feature onto its cfbfastR counterpart and
reading `noise_ratio` — the share of between-player spread the mapping fails to reproduce:

| Feature | R² | noise ratio | Portable? |
|---|---|---|---|
| `rush_share` | 0.95 | 0.22 | yes — the −0.043 intercept *is* the sack correction |
| `td_rate` | 0.93 | 0.27 | yes |
| `yards_per_attempt` | 0.92 | 0.28 | yes |
| `completion_pct` | 0.82 | 0.43 | yes, lossy |
| `int_rate` | 0.58 | 0.65 | **no** — rare events, too noisy |
| **PPA → EPA per dropback** | **0.54** | **0.68** | **no** |

This forces a **two-tier feature set**, which is a real design constraint rather than bookkeeping:

- **Portable tier** — completion %, yards per attempt, TD rate, rush share, volume. Computed
  natively from both sources, so a model built on these can be **fit on history and used to score
  current prospects**.
- **cfbfastR-only tier** — EPA per dropback, success rate, adjusted yards per dropback. Better
  features, 2004–2021 only. A model using them is a **historical instrument**: it can explain what
  late breakouts looked like but cannot be pointed at this year's class.

Stage 6 fits both and reports what the portable model gives up. If the gap is small the project
gains a forward-looking tool; if it is large, that is itself a finding — the signal lives in
precisely the measure that cannot be carried forward. Calibrated values are written `*_est` with
`efficiency_is_estimated=1` so an estimate is never mistaken for a measurement.

**Credentials.** `CFBD_API_KEY`, else `~/.config/cfbd/api_key`. Never stored in the repo.

### 2.3 High school: built, measured, set aside

The recruiting layer was built and works — 22,175 QB prospects, 87–93% match from 2010 on, and
unbiased across outcomes. It is **not** the modelling evidence, because measuring it showed the
reach is too short:

| Earliest usable NFL entry | Cohort | Ever broke out | **Late breakouts** |
|---|---|---|---|
| 2010 (recruiting layer's reach) | 190 | 33 | **7** — of which **6** actually matched |
| 2006 | 232 | 37 | 7 |
| 2005 (college PBP reach) | 244 | 42 | **11** |
| 2004 | 257 | 46 | **12** |

Recruiting starts with the 2006 class, which puts its floor at roughly NFL entry 2010. Usable
college play-by-play starts in 2004, so it reaches entry ~2005 — and the late-breakout cohort
be **bimodal**, clustering in 2001–2005 (Brees, Garrard, Romo, Schaub, Rodgers, Alex Smith,
Cassel, Fitzpatrick) and again in 2011–2020, with a dead zone between. The earlier cluster is
exactly what the recruiting floor cuts off. Going college-only therefore **nearly doubles the
positives**, from 6 to 11–12.

The already-built recruiting artifacts (`espn_recruits_qb.parquet`, `data/recruiting.py`,
`data/link.py`, `REPORT_qb_breakout_recruiting.md`) are **retained but demoted**: usable as a
descriptive covariate or a pedigree control on the subset that has them, never a required input.
The planned high-school box-score scrape is **cancelled** — it would have been sparser still, on
the era that matters least.

### 2.4 Identity joins

There is no single key spanning college and the NFL. The chain is:

cfbfastR play-by-play names players but carries **no player ID**, so `draft_picks.cfb_player_id`
has nothing on the college side to join to. The match is therefore **normalised name + plausible
timing** (last college season 1-5 years before NFL entry), with **school as a tiebreaker, not a
filter** — requiring the school to match would discard real players over spelling ("Michigan St."
vs "Michigan State"), and spelling variance is worse for smaller programs, which is exactly where
late-round and undrafted QBs come from. Undrafted QBs (Romo, Keenum, Hill) have no
`draft_college` at all and are reachable only on name and timing.

Ambiguous candidates are **refused, not guessed**. **Join quality is reported, not assumed** — an
unmatched QB is recorded as unmatched rather than silently dropped, because a match rate that
varies by outcome would reshape the cohort invisibly. Both joins were checked for exactly that and
both came back essentially flat across outcomes (§2.3, §4.2).

---

## 3. Label construction

### 3.1 The four cells

Crossing `late` with `relocated` gives the structure the analysis is built around:

|  | same franchise | relocated |
|---|---|---|
| **on-time** | on-time franchise QB | early bloomer, moved |
| **late** | slow burn, same building (Rodgers, Cousins) | **the Darnold/Geno/Baker cell** |

### 3.2 One bar cannot define a breakout

Definitions were built and scored against 14 QBs whose careers are not in dispute:

| Definition | Mayfield | Tannehill | Geno | Brady | Allen/Burrow/Purdy | Garoppolo |
|---|---|---|---|---|---|---|
| First top-20 (`late`) | ✗ year 1 | ✗ year 3 | ✓ | ✓ on-time | ✓ on-time | late |
| First top-12 (`late_qb1`) | ✓ | ✗ year 3 | ✓ | ✗ calls Brady **late** | ✓ on-time | — |
| Sustained top-20 only | ✓ year 6 | ✓ year 8 | ✓ | ✓ on-time | ✓ on-time | late |
| Sustained top-15 only | ✗ **vanishes** | ✓ | ✗ **vanishes** | ✓ on-time | ✓ on-time | — |
| **top-15 trigger, top-20 hold** | ✓ year 7 | ✓ year 8 | ✓ year 10 | ✓ on-time | ✓ on-time | never |

A single number fails in **both** directions. At top-20, one ordinary season counts —
**Mayfield's 2018 ranks exactly 20th** — so the archetypal late bloomer reads as on-time. At
top-15, the archetypes disappear instead: Mayfield's real run is 17/4/19 and Geno Smith's is
9/21/16, and neither holds two top-15 seasons in any three-year window despite both plainly being
valuable. `late_qb1` fails differently again, calling Brady late on a fantasy technicality.

The fix is two bars for two different claims. **A top-15 season triggers** the breakout — QB15 is
where genuine draft-day value starts. **Top-20 in ≥2 of the 3 seasons from it confirms** the tier
held — QB20 is still a startable superflex asset. Trigger high, confirm lower.
**`late_sustained` is the modelling target**; the single-bar variants ship as diagnostics.

One consequence to keep visible: at a QB15 trigger, **Jimmy Garoppolo never breaks out** (career
best rank 18), where the old top-20 bar counted him late. That is the tier doing its job, but it
is the kind of borderline case the 15-vs-20 choice decides.

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
| 2 | High-school recruiting layer | `espn_recruits_qb.parquet`, `REPORT_qb_breakout_recruiting.md` | **done** |
| 3 | College production layer | `cfb_qb_seasons.parquet`, `REPORT_qb_breakout_college.md` | **done** |
| ~~3b~~ | ~~Best-effort HS box-score scrape~~ | — | **cancelled** (§2.3) |
| 4 | CFBD current-season extension + comparability test | `cfbd_qb_seasons.parquet`, `REPORT_qb_breakout_cfbd.md` | **done** |
| 5 | Archetypes (clustering on college features) | archetype assignments + profiles | **next** |
| 6 | Pre-NFL-only model — `ever_breakout` primary, lateness descriptive | model + honest validation | planned |

**The HS box-score scrape (stage 3b) was cancelled and the project is now college-only** (§2.4). The recruiting layer's
floor at NFL entry ~2010 cuts off the entire 2001–2005 cluster of late breakouts; college
play-by-play reaches back to entry ~2005 and nearly doubles the positives. A high-school scrape
would have been sparser still on precisely the era it could least afford to lose.

### 4.1 What stage 2 established

The recruiting join works — 87–93% match from 2010 on, and **essentially unbiased across
outcomes**, so the surviving sample is not skewed against the cohort of interest. ESPN also
supplies a scout-assigned high-school archetype (`QB-PP` pocket passer vs `QB-DT` dual threat)
with no hindsight, which remains available as a covariate on the players who have it.

What it could not fix is reach. That is why the layer is retained but demoted, and why stage 3
is college.

### 4.2 What stage 3 established

3,419 QB-seasons (≥50 dropbacks) across 1,684 college careers, 2004–2021. The cohort matches at
**88% for QBs entering 2005+**, and again **essentially unbiased across outcomes** (late 82% /
never 88% / on-time 94%).

**9 late breakouts carry a college profile against 6 for high school** — Rodgers, Alex Smith,
Tyrod Taylor, Cousins, Tannehill, Geno Smith, Winston, Mayfield, Love — and `ever_breakout` has
**38** matched QBs with a resolved outcome. That is the modelling sample stage 6 works with.

Building from play-by-play rather than a season-stats table paid for itself twice: efficiency
(EPA per dropback, success rate) exists at all, and **sacks separate from rushing**, which NCAA
box scores conflate. Mayfield's 2015 reads as 405 official rushing yards but 604 on actual rush
plays with the difference lost on 39 sacks — the distinction between measuring mobility and
measuring pass protection.

---

## 5. Modelling, and the small-N problem

### 5.1 The N, precisely

| Population | N |
|---|---|
| QBs entering 1999+ | 331 |
| Ever broke out (top-15 held at top-20) | 59 |
| Broke out **late** | 16 |
| Late, within college-data reach (entry 2005+) | 11 |
| **Late, with a college profile actually matched** | **9** |
| **Ever broke out, entry 2005+, matched** | **38** |
| *(for contrast)* Late with a high-school profile | 6 |

**Nine positives cannot support a late/not-late classifier.** The primary modelled
outcome is therefore `ever_breakout` (n=38 matched within college reach), with lateness handled
descriptively and as a conditional second stage. This is a change of emphasis forced by the data,
not a change of question — §1 still asks whether late breakouts were visible in advance; §5.2 is
how much of that the evidence can actually carry.

### 5.2 Methodological consequences

**16 late breakouts** across 1999–2025 is the honest ceiling, and college-data reach plus join
attrition cuts it to 9. This governs every methodological choice:

- **Evidence-weighting, not prediction at scale.** The deliverable is calibrated priors and
  interpretable archetype effects, not a leaderboard implying precision that N cannot support.
- **Leave-one-out / repeated stratified CV**, never a single split — one held-out fold would
  contain one or two positives.
- **Nested outcome, not one binary.** `ever_breakout` (n=38) is far better powered than
  `late | breakout` (n=9). Model the well-powered part first and treat lateness as a conditional
  second stage.
- **Prefer `final_*` features to `career_*`.** Career totals are clipped for anyone whose college
  years straddle 2004 (§2.2), while final-season form is intact for everyone.
- **Bootstrap every coefficient.** Report intervals; a feature that flips sign across resamples
  is noise, and at this N many will.
- **A negative result is a real result.** If pre-NFL features carry no signal for lateness, that
  is worth knowing and will be reported as such rather than tuned around.
- **Draft capital is a confound, not a feature to celebrate.** It encodes the league's own
  scouting opinion of the same college tape. Models are reported with and without it so the
  incremental value of college evidence is visible on its own.

---

## 6. Reproducibility

Same conventions as the sibling project: parquet caches with JSON manifests, deterministic seeds,
committed reports under `reports/`, tests under `tests/`. Scraped HS data is cached with its
retrieval date and source URL per row, since it is the one layer that cannot be re-derived
identically later.
