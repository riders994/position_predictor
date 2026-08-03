# Medical-staff grades — Stage 3: exposure and confounders

The risk set is **130,139 player-weeks** across 2021–2025, drawn from
221,313 panel rows. Nothing here is an outcome and nothing here is a grade — this is the
input side of the observed-minus-expected the grades are built from.

## 1. What the risk set excludes, and why

A player-week only counts as exposure if a new injury could have *started* in it:

- **Weeks already inside an open spell are dropped.** Counting them would turn one long absence
  into many weeks of injury-free exposure, which is exactly backwards.
- **Practice-squad weeks are dropped** — not exposure to an NFL game.
- **Byes are dropped** — the club did not play.

## 2. Covariate coverage

| covariate | non_null_share |
|---|---|
| surface (game) | 0.963 |
| home_surface (club) | 1.000 |
| rest_days | 1.000 |
| indoor | 1.000 |
| age | 0.993 |
| years_exp | 1.000 |
| bmi | 0.999 |
| snaps | 0.857 |
| snap_share_3wk | 0.857 |
| prior_designated_weeks | 1.000 |
| weeks_since_return | 0.298 |

**Snap coverage is 85.7%.** `snap_counts` keys on `pfr_player_id`, so it needs a
crosswalk to `gsis_id`, and the obvious sources are biased in the one direction that would have
corrupted this project:

- `rosters_weekly.pfr_id` is null for **99.8% of offensive-line rows**, against 9–24% elsewhere
- `ids` is a *fantasy* table — of 352 distinct linemen in one season's snap counts it resolves
  **two**

Either would have left snap-workload covariates present for skill players and absent for
linemen. Position correlates with body part, and body part is exactly what the stage-5 signature
analysis compares, so position-biased missingness would have looked like a finding. The
crosswalk therefore comes from `players`, which is ~12% null for linemen and ~11% for skill
players — missing at roughly the same rate everywhere.

| position_group | snap_coverage | mean_snap_share |
|---|---|---|
| ST | 0.962 | 0.000 |
| LB | 0.910 | 0.416 |
| DB | 0.892 | 0.518 |
| WR_TE | 0.880 | 0.457 |
| RB | 0.863 | 0.308 |
| DL | 0.857 | 0.453 |
| OL | 0.832 | 0.558 |
| QB | 0.470 | 0.780 |

Snaps remain an *intensity* covariate and never the availability signal — the roster does that
job at full coverage — and every snap-derived column is null-safe.

## 3. Conditions the club does not choose

Surface is free text and dirty — `"grass "` with a trailing space is a distinct value from
`"grass"`, and `""` means missing — so it is normalised to grass/turf, with turf brands
collapsed because the brand distinction is not an injury-risk distinction.

Per player-week, the surface actually played on:

| surface | player_weeks |
|---|---|
| grass | 66804 |
| turf | 58469 |
| — | 4866 |

Club home surfaces, which are a stadium property rather than a staff choice and the canonical
mechanism behind knee and ankle risk:

| home_surface | clubs |
|---|---|
| grass | 17 |
| turf | 15 |

Short weeks (four or five days' rest — Thursday games):

| short_week | player_weeks |
|---|---|
| no | 121781 |
| yes | 8358 |

## 4. Who is exposed

| position_group | risk_weeks | mean_age | mean_snap_share |
|---|---|---|---|
| OL | 24846 | 27.152 | 0.558 |
| DB | 24402 | 26.089 | 0.518 |
| WR_TE | 22095 | 26.259 | 0.457 |
| DL | 19326 | 26.812 | 0.453 |
| LB | 18348 | 26.222 | 0.416 |
| RB | 9484 | 25.891 | 0.308 |
| QB | 6154 | 28.249 | 0.780 |
| ST | 5484 | 28.985 | 0.000 |

## 5. Prior injury history — the covariate that cuts both ways

Mean prior designated weeks per risk row: **7.4**
(median 3,
max 121).

History is counted **strictly from prior seasons**, so a season never contributes to its own
covariate, and from the **injury report back to 2009** rather than from
episodes — the report is comparable across that whole span even though the roster fields that
define episodes only become comparable at 2021.

**This covariate is not neutral, and stage 4 must not treat it as such.** A poor availability
system manufactures players who look fragile, so prior injuries are partly its own output.
Adjusting for them therefore adjusts away part of the effect being measured. Stage 4 fits
incidence **with and without** history and reports the pair as a **bound** — no-history as the
upper bound on the club effect, with-history as the lower — rather than picking one and calling
it the answer. That is why history ships as its own table instead of being folded in here.
