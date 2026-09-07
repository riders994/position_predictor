# QB Benching — Project Plan

Which quarterbacks who open a season as their club's starter lose the job to a **decision**?

Fourth standalone effort in the repo, living at `sports/football/src/qb_benching/` alongside
`medstaff` and `qb_breakout`, reusing the `position_predictor` nflverse caches. Branch
`qb-benching`.

---

## 1. Problem statement

A club's Week-1 quarterback stops starting for three quite different reasons — he gets hurt, he
gets benched, or he is traded or released — and only the second is a judgement about his play.
Lumping them together produces a model of injury attrition, which this repo has already measured
twice and found unpredictable (PROMPT_LOG entries 096–097). So the label separates all three and
the model targets benching alone.

Two prior results point here. `qb_breakout` §4.5 concluded that the right follow-up to its null
was *"a different outcome, not more factors — starts give a data point per season instead of one
per career."* And the fantasy pipeline has no depth-chart or role knowledge whatsoever (entry
090): its largest disagreements with the market are bench bodies it ranks as though they will
play. Benching risk is the missing dimension.

### 1.1 Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Population | the club's **opening starter** — the QB who started its first game | Keyed on the first game played, not `week == 1`, so a disturbed opening week (2001) still resolves |
| Outcome | three-way split; **benching** is modelled, injury carried as comparison | The user's decision; the injury arm is the declared control given entries 096–097 |
| Bar | benched in **≥3** team-games | One week is churn, not a lost job. See §3.2 |
| Horizon | **preseason** — one row per QB-season, features known by Sept 1 | Matches every other model in the repo and is what a drafter can act on |
| Starter source | `schedules.home_qb_id` / `away_qb_id` | The *announced* starter; 100% populated 1999–2025 |
| Role source | **weekly depth charts** | The only weekly role signal that survives the 2021 roster break. See §2.1 |
| Sample | 2009–2025 | The weekly injury report begins in 2009 |

### 1.2 Non-goals

* Nothing here feeds the fantasy ranking models, as with both prior efforts.
* No blending with the market. Preseason ECR/ADP is a benchmark — does the market already price
  benching risk? — and never an input.
* Not an injury model.
* No in-season weekly hazard in this pass. The preseason model is designed to be the baseline a
  later hazard stage would have to beat.
* `pbp` is not needed: `weekly` already carries `passing_epa` and `passing_cpoe`, so this effort
  never touches the standing "pbp is out of scope" decision in `PROJECT_PLAN.md` §13.

---

## 2. Data

| Source | Seasons | Role |
|---|---|---|
| `schedules` | 1999–2025 | announced starter per team-game; the week grid; head coach |
| `depth_charts` | 2001–2025 | the club's declared QB ordering — **the primary role signal** |
| `injuries` | 2009–2025 | weekly injury report; the sample floor |
| `rosters_weekly` | 2002–2025 | roster presence and status; fallback ladder |
| `weekly` | 1999–2025 | appearances, and prior-season production for features |

`depth_charts` was **registered by this project** — it is the one obvious nflverse endpoint the
repo had never fetched. One `Dataset(...)` line in `data/fetch.py`; caching and manifests come
free.

### 2.1 ⚠️ Before 2021 `rosters_weekly.status` is not a weekly value (found in stage 1)

It is the player's **season-final status stamped on every one of his weeks**. Only 1–4% of
pre-2021 QB-seasons have a status that varies by week, against 50–72% from 2021 on.

This is a different and worse defect than the one `medstaff` records as
`FIRST_COMPARABLE_SEASON = 2021` (there, that the ACT/INA split is unpopulated). It was found
here by a first version of the label that reported **zero benchings in 2015** — a season in which
Colin Kaepernick visibly lost the San Francisco job to Blaine Gabbert. Kaepernick started the
first eight games and finished the year on injured reserve, so all ten of his rows read `RES`,
including the weeks he was starting and the weeks he was benched and healthy.

What *is* informative before 2021 is whether a row exists at all: **74.6%** of pre-2021
QB-seasons ending in `RES` are truncated before the end of the season, against **0%** from 2021
on, where reserve players keep weekly `RES` rows all year.

### 2.2 ⚠️ Injured reserve is invisible to the injury report

A player placed on IR drops off the weekly report entirely, so "not listed hurt" does not mean
healthy. Measured against the trustworthy 2021+ status, displaced weeks with no report listing
and no appearance split **195 reserve / 104 active / 20 inactive** — roughly 60% genuinely hurt.
A label keyed on the report alone calls all of them benchings.

### 2.3 The depth chart has neither problem — and two schemas

It is published weekly from 2001 and records both facts directly. Kaepernick's 2015 reads
straight off it: `depth_team` 1 through week 9, **2 in week 10 with Gabbert at 1** — the
benching, as the club declared it — then absent from week 11, when he went on reserve.

nflverse changed the feed for 2025. Through 2024 it is weekly, keyed on
`(season, week, club_code)` with a string `depth_team`. From 2025 it is a **dated snapshot**
feed — `dt`, `team`, `pos_abb`, integer `pos_rank` — with no season or week column and ~15× the
rows. `labels/depth.py::qb_depth` normalises both, resolving each team-game in the snapshot era
to the latest chart **strictly before kickoff**. A useful side effect: the snapshot feed starts in
early August, so it also carries the preseason chart a live board needs.

### 2.4 ⚠️ `schedules` and `rosters_weekly` do not spell St. Louis the same way

The roster feed uses `SL` for all 14 St. Louis seasons (2002–2015) and never `STL`, which is what
`schedules` uses. Uncanonicalised, the club comparison fires and every Rams quarterback reads as
playing for another club — which classified Nick Foles's 2015 benching as a departure. `SL` was
missing from `medstaff.data.teams.TEAM_ALIASES` and has been added there; this project reuses
that map rather than adding a third copy. Every other code in all three tables already
canonicalises to a code `schedules` uses.

### 2.5 Starter definition, and what it deliberately excludes

`schedules` records the announced starter. Against the alternative — the QB with the most
dropbacks in the game — it agrees on **96.9%** of 13,183 team-games. The residual is in-game
changes, a hook or an injury, and counting those as displacement would fold within-game events
into a label meant to capture a week-to-week decision.

---

## 3. Label construction

### 3.1 The ladder

For each team-game after the club's opener, in order:

1. he started — `held`
2. listed on the injury report — `injured`
3. absent from a chart that covers his club that week — unavailable; the roster says whether that
   is reserve or a departure
4. on the chart with another quarterback at rank 1 — **`benched`**
5. not on the roster table that week — his season-final status decides
6. rostered by another club — `gone`
7. from 2021 only, a reserve or departure status on the week's own row
8. otherwise — `benched` (residual)

Step 7 is restricted to the era where `status` is a weekly value; applying it earlier is what
produced the zero-benching 2015. Step 2 sits above the chart deliberately: a quarterback both
listed hurt and demoted is counted hurt, which keeps benching a **lower bound**.

The chart resolves most of it directly — 526 of 672 benched team-games, against 146 falling
through to the residual.

### 3.2 The bar

Three weeks. One is churn — a coach can sit a quarterback for a week and hand the job back — and
the one-week bar catches roughly a third of all openers, which measures week-to-week noise rather
than losing a job. Three is also where the rate stops moving quickly with the bar. Every report
prints the sensitivity table beside the headline.

### 3.3 The sample

**544 opening starters, 2009–2025. 90 benched (16.5%)**; 142 lost ≥3 weeks to injury and 213 for
any reason at all.

### 3.4 The regime check that has to keep passing

The label reads different evidence either side of 2021, so its rate had better not move. It does
not: **0.159 before 2021, 0.181 from 2021**, a 0.022 gap inside season-to-season variation.
`cohort.regime_check` recomputes it so a refresh that reintroduced the dependency shows up as a
moved rate rather than a quiet bias. Before the depth chart entered the ladder this gap was
0.206 vs 0.150 the other way, which is what exposed §2.1.

### 3.5 Known limitation — the opener is not always the intended starter

Some openers are fill-ins for a quarterback hurt in preseason (Derek Anderson, Carolina 2014).
They lose the job when the incumbent returns, which is not a benching on the merits. The model
should see this as predictable rather than as noise, but the label does not distinguish it and
the modelling stage must carry a flag for it.

---

## 4. Stages

1. **Cohort and label** — done. `labels/`, `scripts/qb_benching_cohort.py`,
   `reports/REPORT_qb_benching_cohort.md`.
2. **Preseason features** — prior play, tenure and standing, QB room, team and coaching.
   ⚠️ Do **not** reuse `features/build.py::add_offseason`: it merges on `team_next` from the N+1
   full-season roster, which entry 096 established is a deterministic label leak for any
   absence-shaped target. Build the room snapshot from the week-1 roster and depth chart instead,
   which is a genuine Sept-1 view.
3. **Model** — binary classification on the `qb_breakout/model/fit.py` template (logistic +
   depth-2 boosting, repeated stratified CV kept per repeat, permutation null, temporal split).
   The **within-stratum AUC is the headline**, not the pooled one: "journeyman opening as QB1" is
   an easy positive, and `qb_breakout` already showed how a variable that allocates opportunity
   (draft capital, AUC 0.890) collapses to chance within a band (0.496).
4. **Report and live board** — bench-risk tiers for the coming season, taking the projected
   Week-1 starter from the preseason depth chart.

---

## 5. Reproducibility

```bash
export PATH="$HOME/.local/bin:$PATH"
cd sports/football
uv run python scripts/qb_benching_cohort.py
uv run pytest -q tests/test_qb_benching_labels.py
```
