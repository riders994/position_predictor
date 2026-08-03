# Medical-Staff Injury Grades — Project Plan

> **Question:** can NFL clubs be told apart by how well they keep players available — and in
> particular by whether injuries *come back*?
> Sibling of `position_predictor` (see `PROJECT_PLAN.md`) and `qb_breakout`
> (`QB_BREAKOUT_PLAN.md`); shares their raw caches and conventions.
> Decision trail in `PROMPT_LOG.md`, entries 062+.

---

## 1. Problem statement

Grade each team's availability system over trailing **3-year and 5-year** windows, with
**reinjury risk** as a core component and **position-group** breakouts.

**Why this is a research project and not a leaderboard query.** Raw injury burden is mostly not
the training room's doing: roster age, prior injury history, position mix, snap exposure, pace,
stadium surface — and above all luck. The spread is real and large (measured: "Out" designations
per club 2021–25 range **98 → 277**, sd 46 on mean 161, versus Poisson noise of ~13), but
attributing it requires stripping out everything the staff does not control.

### 1.1 What this actually grades — a non-goal, not a caveat

**"Medical staff" is not identified by this data.** The residual bundles the athletic training
staff, strength and conditioning, sports science, the head coach's practice-intensity choices,
the general manager's taste for durable players, and the scheme. What is measurable is a **team
availability system**; the training room is one input. No result here may be described as
grading a training staff. Head-coach change is the only turnover instrument available;
`--staff-table` is the hook for better.

### 1.2 Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Roster scope | Whole roster, all positions, 8 groups | The system serves everyone; ~3× the sample, which the tiny per-club cells need |
| Headline | **Return-speed × recurrence 2×2** | A single availability number conflates good rehab with rushing players back |
| Reliability posture | Grades **always publish**, verdict leads | User decision; the defence is presentational, not suppression |
| Staff identity | Team-window grade + `--staff-table` CSV hook | No free structured head-athletic-trainer source exists (same problem as GM in `qb_breakout`) |
| Estimator | Penalised discrete-time logistic hazard | A flexible learner eats the residual variance being measured (§5.1) |
| Spine | `rosters_weekly`, not `snap_counts` | Transaction record, not a disclosure; all positions; gsis-keyed; joins 99.2% |

### 1.3 Non-goals

- Not a fantasy tool, and nothing here feeds the ranking models. The standing decision at
  `PROJECT_PLAN.md:151` ("no injury-report text or qualitative tags") is **unchanged**.
- No per-player injury prediction. The unit of analysis is the club-window.
- No causal claim about any individual club's medical practice.

---

## 2. Data

| Dataset | Loader | Seasons | Role |
|---|---|---|---|
| `injuries` | `load_injuries` | 2009–2025 | Body part, game designation, practice participation |
| `rosters_weekly` | `load_rosters_weekly` | 2002–2025 | **The availability spine** — ACT/INA/RES-IR/PUP/PS |
| `schedules` | `load_schedules` | all | Surface & roof (confounders), rest days, head coach |
| `players` | `load_players` | all | The **only** unbiased `gsis_id`↔`pfr_id` crosswalk (§2.6) |

Registered as three `Dataset(...)` lines in `position_predictor/data/fetch.py`; the existing
`L()` / `_clip_seasons` / `_load_resilient` / `_write_manifest` machinery is unchanged.

### 2.1 Verified joins

- **Injury rows join 99.2% to `rosters_weekly`** on `(season, week, gsis_id)` — 97.4–98.2% for
  2009–2015, **99.9–100% from 2016**.
- **`snap_counts` is not the spine.** It keys on `pfr_player_id`; a combined
  `ff_playerids` + `rosters_weekly` crosswalk reaches only **81.7%**, and it starts 2012. Snaps
  are an exposure/intensity covariate and the episode-termination signal, never the spine.
- The caches disagree on join-key dtype (`rosters_weekly` lands `season`/`week` as f64 because
  nflreadpy concatenates multi-season pulls with `diagonal_relaxed`); `_normalize_keys` pins
  both to Int32 at load.

### 2.2 The 2016 regime break

`report_status` is **3.0–6.0%** null for 2009–2015 and **39.2%** in 2016, then 48.7–54.7% — the
league dropped the "Probable" designation. Any measure built on that column is not comparable
across the boundary.

Over the same seasons the coalesced body part is **≤0.1%** null and `practice_status` **≤0.7%**.
**Every outcome is therefore built on those two**; `report_status` is used only as a severity
refinement in a 2016+ sensitivity run, with a season-level `report_regime` control.

### 2.6 ⚠️ The ID crosswalk is position-biased (found in stage 3)

`snap_counts` keys on `pfr_player_id`, so using it needs a crosswalk to `gsis_id` — and the two
obvious sources are biased in the one direction that would have corrupted this project:

- **`rosters_weekly.pfr_id` is null for 99.8% of offensive-line rows**, against 9–24% elsewhere
- **`ids` is a fantasy table**: of 352 distinct linemen in one season's snap counts it resolves
  **two**

Either would have left snap-workload covariates present for skill players and absent for
linemen. Position correlates with body part, and body part is exactly what the stage-5 signature
analysis compares — so position-biased missingness would have arrived looking like a finding.

**`load_players()` is the fix**: ~12% null for linemen and ~11% for skill players, i.e. missing
at roughly the same rate everywhere. OL snap coverage goes from **2.4% → 83.2%** with a sensible
mean snap share (0.558). Snaps remain an *intensity* covariate and never the availability signal.

### 2.5 ⚠️ The 2021 roster-status break (found in stage 2)

Two `rosters_weekly` fields silently change meaning at 2021, and both would have corrupted every
absence measure in the project:

- **`status_description_abbr` carries no reserve codes before 2021** — R-codes are 0.0% of rows
  for 2012–2015 and the field is 51% null in 2016. Keying IR off it made injured reserve look
  like something clubs invented in 2021 (2 episodes across 2012–2019, then 1,059 per season).
- **`status == "INA"` is barely populated before 2021** (2.8k rows across 2012–2019 against
  16.8k across 2021–2025) while `ACT` falls 0.86 → 0.59. The gameday active/inactive split is
  simply not in the earlier data, so games-missed is not comparable across the break.

`status == "RES"` is the one stable signal (6–14% of rows in every season since 2012), so
**reserve is read from `status`, never from the abbr codes**.

This is the same shape as the sibling project's cfbfastR flag defect: an unpopulated field
aggregates to a clean zero rather than a null, so nothing errors and the series just quietly
means something different either side of the boundary. `roster_status_regime_table()` measures
it per season, and there are tests pinning both halves.

**Reserve/COVID-19 (`R59`) is excluded.** It occurs in **2021 only** (725 player-weeks, exactly
zero in every other season) and carries `RES` status without being an injury. Left in, it pushed
2021's reserve share to 0.46 against ~0.31 for 2022–2025 and made every club look worse at
medicine in the first year of the five-year window.

### 2.3 Body-part taxonomy

292 distinct raw strings → **16 groups**, mapping **99.8%** of rows (200 rows, 0.2%, fall to
`other`; no single unmapped string exceeds 0.1%). Two rules do the work:

- **Non-injury is checked first, on the raw string** — so
  `Ankle [Not Injury Related - Personal, Thursday Only]` is excluded rather than counted.
  Exclusions run wider than they look: `Illness`, `Appendicitis`, `Jury Duty`, `Travel`,
  `Inactive`, `Returning from Suspension`.
- **Earliest match wins; longest match breaks ties.** Earliest encodes "the primary injury is
  listed first" (`Foot/Wrist/Hip` → foot); longest keeps `hip flexor` in the soft-tissue group
  instead of collapsing to `hip` — **structurally, rather than depending on rule order**.

Handles laterality prefixes (`right Shoulder`), case variants, plurals (`Rib`/`Ribs`), bracketed
clauses, and whole free-text sentences.

### 2.4 Sample and windows

- **Fitting sample 2021–2025** for every absence-based outcome — see §2.5. The injury *report*
  is comparable from 2009 and supplies the prior-injury lookback at any depth, but anything
  counting missed games is restricted to the comparable window.
- **5-year window 2021–2025; 3-year window 2023–2025.** The comparable window *is* the 5-year
  grading window, and it sits entirely inside the post-COVID 17-game era and the post-2016
  reporting regime — one homogeneous regime rather than a compromise.
- Relocations (SD→LAC, STL→LA, OAK→LV, JAC/JAX) fall inside the sample, so `canonical_team` is
  applied on both sides of every join.

---

## 3. Episodes

An **episode** is one continuous injury spell. Everything downstream is computed on episodes.

**Snaps subordinate the report everywhere.** A gap of up to one rostered **zero-snap** week is
bridged (a single unlisted week between two listed weeks is not a return); a week with **any**
snaps terminates the episode regardless of what the report says.

Key edge cases: bye weeks continue the episode but do not count as games missed · a midseason
trade **censors** for the onset club and opens a new episode for the acquiring club (rehab
credit belongs to whoever did the rehab) · playoffs count as return opportunities but are
**excluded from the incidence risk set** (playoff participation is endogenous to club quality) ·
season end censors · IR weeks with no report row continue the episode · practice-squad weeks are
excluded from the risk set.

**Recurrence** counts **games the player was available for**, not calendar weeks, so it pauses
across byes and the offseason. K = 6 primary (3/12/same-season as sensitivity). Attributed to
the club of the **return** — the staff that cleared him.

---

## 4. Components

| Component | Outcome | Attributable? |
|---|---|---|
| Incidence | Episode onset per exposed player-week | **Weakly** — conditioning, scheme, surface, luck |
| Duration | Games missed vs expected for body part/severity/position/age | **Moderately**, but ambiguous alone |
| Recurrence | Same body group returning after a return | **Most** — least contaminated by luck |

Duration × recurrence forms the headline 2×2: fast/low = efficient rehab, **fast/high = rushing
them back**, slow/low = conservative, slow/high = poor outcomes. The reportable claim is
`corr(z_dur, z_rec)` across clubs with a permutation CI — if it is ≈0 league-wide, "rushing
players back costs you" is not visible in this data, and that is the most defensible sentence
the project can produce.

---

## 5. Method

### 5.1 One estimator, three uses

A **penalised discrete-time logistic hazard** on a person-period panel. Censoring is pervasive
and discrete-time handles it exactly by construction, with no `lifelines` dependency. **A GBM is
deliberately avoided**: the estimand is a *residual*, and a flexible learner adjusting on
roster-composition proxies absorbs precisely the between-club variation being measured. A
shallow `HistGradientBoostingClassifier` runs as a robustness check only.

### 5.2 Leave-one-team-out, not random k-fold

Score club T with a model fit on the other 31. A random fold containing some of T's rows lets
the expectation absorb T's own effect and understates it — the same concern that made
`expected_from_draft` out-of-fold in `qb_breakout`. Tested by injecting a synthetic club effect
and asserting LOTO recovers it while random-fold shrinks it.

### 5.3 The bound that adjustment forces

Incidence is fit **twice, with and without prior-injury history**. A poor system manufactures
players who look fragile, so adjusting for prior injuries adjusts away part of its own effect.
The no-history fit is the **upper bound**, with-history the **lower bound**; both are reported.

### 5.4 Informative censoring is gameable

A club scores well on duration by having unrecovered cases silently censored, or by releasing
injured players. So `P(returns at all this season)`, `never_returned` and
`released_while_injured` are modelled as **outcomes**, never swept into the censor.

### 5.5 Null and power

1. **Whole-factor permutation — the primary inference.** Does club identity explain any variance
   at all? Reassign whole players to clubs. One test, no multiplicity.
2. **Poisson-binomial per club-window**, reported uncorrected with the comparison count and
   expected chance hits stated inline.
3. **Player-block bootstrap** — a fragile player is fragile all season, so independent Bernoulli
   understates variance. The more conservative of (2) and (3) is used.

Plus over-dispersion decomposition (`τ̂² = max(0, var(d) − mean(s²))`, reported as an intraclass
share) and **minimum detectable effect** in medical units for every component and window.

### 5.6 Reporting culture

The report is a strategic artifact. Mitigations: outcomes anchored in snaps and IR rather than
designations; three per-club disclosure indices (`listing_rate`, `questionable_share`,
**`questionable_play_rate`**); raw and adjusted grades side by side with their rank correlation;
and an IR-only sensitivity run as independent replication.

### 5.7 Cross-position-group signature — the scheme instrument

If a club's excess sits in **one body part** and appears across position groups sharing nothing
except the building, that implicates a common cause. Focal parts: **knee, ankle, back, hip,
concussion**. Body-part *composition* is **disclosure-robust** — a club that lists everyone
inflates numerator and denominator alike.

The statistic is a **variance decomposition** (club main effect vs club×group interaction), not
a grade: club × body-part × position-group cells hold a median of **4 episodes over 5 years**, so
cells are never graded and concordance aggregates across the 8 groups. Tested across
offense/defense, random split-half, and leave-one-group-out, in **both** a rate spec (level +
signature) and a composition spec (signature net of level) — they disagree, and the disagreement
is the finding.

**Preregistered from the exploratory probe** (2021–25, crude exposure denominator):

| part | share-based | rate-based | reading |
|---|---|---|---|
| concussion | 0.405 (p=.022) | 0.404 (p=.022) | **stable in both — the candidate signature** |
| knee | −0.027 | 0.335 (p=.061) | rate-only ⇒ level artifact |
| back | 0.433 (p=.013) | 0.172 | share-only ⇒ compositional |
| ankle | 0.258 | 0.174 | weak |
| hip | 0.209 | 0.106 | weak |

**Multiplicity preregistered:** 5 parts ⇒ Bonferroni 0.01, and **no probe result clears it**.
**Concussion is the primary a-priori hypothesis** — practice contact policy and tackling
technique are documented coach decisions, and concussion reporting is protocol-mandated hence the
least disclosure-contaminated outcome in the project. The other four are exploratory. At ~22
episodes per club over 5 years concussion stays **team-level, never split by position group**.

**The strongest causal handle: does the signature follow the head coach across franchises?** It
is the only design element that breaks the shared-roster confound, since offense and defense are
not fully independent (an old roster is old on both sides).

---

## 6. Stages

| # | Stage | Script | Status |
|---|---|---|---|
| 1 | Data: registration, taxonomy, diagnostics | `medstaff_ingest.py` | **done** |
| 2 | Episodes | `medstaff_episodes.py` | **done** |
| 3 | Exposure & confounders | `medstaff_exposure.py` | **done** |
| 4 | Expected-value models | `medstaff_expected.py` | **done** |
| 5 | Cross-group signature (§5.7) | `medstaff_signature.py` | |
| 6 | Reliability & power | `medstaff_reliability.py` | |
| 7 | Grades | `medstaff_grades.py` | |

**Stage 6 before stage 7 is deliberate.** The reliability verdict must exist before anything
resembling a leaderboard does; stage 7 reads it and interpolates it into its own header, so the
verdict cannot drift from the evidence. A reader who stops at stage 6 has the honest answer.

### 6.3 What stage 4 established

Fitted out-of-fold with leave-one-team-out; **club identity is never a feature** (guarded).

| component | rows | base | permutation p | intraclass |
|---|---|---|---|---|
| incidence (no history) | 144,552 | 0.100 | **0.025** | 0.867 |
| incidence (with history) | 144,552 | 0.100 | **0.039** | 0.859 |
| duration | 48,267 | 0.163 | **0.0005** | 0.882 |
| recurrence | 37,859 | 0.018 | 0.058 | 0.440 |
| returns_at_all | 14,420 | 0.545 | **0.0045** | 0.625 |

- **The evidence runs opposite to attributability.** Recurrence — the component most plausibly
  owned by a medical staff — is the **only one that does not clear** its permutation test.
  Incidence, the component least attributable to a training room, separates clubs most strongly.
  That ordering is a caution, not a finding: it is consistent with clubs differing mainly in
  exposure and IR usage rather than in medicine.
- **The with/without-history bound is tight** (0.859–0.867), so the §5.3 ambiguity turns out not
  to be load-bearing in practice — worth knowing, and it could not have been known in advance.
- **⚠️ `intraclass` is not "share attributable to the medical staff."** It says the spread is not
  Poisson noise. The surviving variance can still be roster construction, disclosure, scheme or
  stadium — and because the model is deliberately restrained (§5.1), composition the covariates
  miss stays *in* the residual by design. This is the most likely misreading of the table.
- Calibration tracks the diagonal for both incidence and recurrence.

### 6.2 What stage 3 established

- The risk set is **130,139 player-weeks**: weeks inside an open spell are excluded (counting
  them would turn one long absence into weeks of injury-free exposure), as are practice-squad
  weeks and byes.
- Covariate coverage is near-complete for everything the club does not choose — surface 96.3%,
  home surface / rest days / indoor 100%, age 99.3%, BMI 99.9%.
- The crosswalk defect in §2.6 was found and fixed here.
- Prior injury history reaches back to 2009 from the *report*, strictly prior seasons only, and
  ships as its own table so stage 4 can fit incidence with and without it (§5.3).

### 6.1 What stage 1 established

- The 2016 regime break is real and measured, and the regime-invariant columns are clean
  (body part ≤0.1% null, `practice_status` ≤0.7%, in every season).
- The spine joins at 99.2%, and at 99.9–100% across the whole grading window.
- The taxonomy maps 99.8% of rows with no unmapped string above 0.1%.
- **Sanity checks pass** (the `qb_breakout` flag-defect lesson): distinct concussed players per
  season runs **90–183**, matching the known 113–179 range for 2015–25; and there are **zero
  clean-zero season × focal-group cells** — the exact shape that defect took.

---

## 7. Reproducibility

Same conventions as the sibling projects: raw caches shared via `position_predictor.utils.io`,
derived tables in `data/processed/medstaff_*.parquet`, reports committed and unversioned at
`reports/REPORT_medstaff_*.md`, tests flat in `tests/test_medstaff_*.py` on synthetic fixtures
with no network. Every report is generated as f-string markdown with numbers interpolated from
the frames, so the prose cannot drift from the tables.
