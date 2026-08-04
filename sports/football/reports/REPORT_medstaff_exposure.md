# Medical-staff grades — Stage 3: exposure and confounders

The risk set is **150,057 player-weeks** across 2021–2025, drawn from
242,407 panel rows. Nothing here is an outcome and nothing here is a grade — this is the
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
| surface (game) | 0.962 |
| home_surface (club) | 1.000 |
| rest_days | 1.000 |
| indoor | 1.000 |
| age | 0.993 |
| years_exp | 0.948 |
| bmi | 0.998 |
| snaps | 0.786 |
| snap_share_3wk | 0.786 |
| prior_designated_weeks | 1.000 |
| weeks_since_return | 0.307 |

**Snap coverage is 78.6%.** `snap_counts` keys on `pfr_player_id`, so it needs a
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
| ST | 0.950 | 0.000 |
| LB | 0.879 | 0.424 |
| DB | 0.854 | 0.525 |
| WR_TE | 0.847 | 0.463 |
| DL | 0.835 | 0.457 |
| RB | 0.833 | 0.314 |
| OL | 0.810 | 0.570 |
| QB | 0.475 | 0.789 |
| — | 0.000 | — |

Snaps remain an *intensity* covariate and never the availability signal — the roster does that
job at full coverage — and every snap-derived column is null-safe.

## 3. Conditions the club does not choose

Surface is free text and dirty — `"grass "` with a trailing space is a distinct value from
`"grass"`, and `""` means missing — so it is normalised to grass/turf, with turf brands
collapsed because the brand distinction is not an injury-risk distinction.

Per player-week, the surface actually played on:

| surface | player_weeks |
|---|---|
| grass | 77119 |
| turf | 67192 |
| — | 5746 |

Club home surfaces, which are a stadium property rather than a staff choice and the canonical
mechanism behind knee and ankle risk:

| home_surface | clubs |
|---|---|
| grass | 17 |
| turf | 15 |

Short weeks (four or five days' rest — Thursday games):

| short_week | player_weeks |
|---|---|
| no | 140413 |
| yes | 9644 |

## 4. Who is exposed

| position_group | risk_weeks | mean_age | mean_snap_share |
|---|---|---|---|
| DB | 27042 | 26.096 | 0.525 |
| OL | 26927 | 27.157 | 0.570 |
| WR_TE | 24452 | 26.263 | 0.463 |
| DL | 21036 | 26.839 | 0.457 |
| LB | 20124 | 26.239 | 0.424 |
| RB | 10436 | 25.888 | 0.314 |
| — | 7867 | 25.971 | — |
| QB | 6538 | 28.246 | 0.789 |
| ST | 5635 | 28.994 | 0.000 |

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
