# Medical-staff grades — Stage 1: data

Injury reports, weekly rosters and schedules for **2009–2025**. Nothing is
modelled here; this stage measures the three data properties that decide the rest of the design.

## 1. The 2016 regime break

`report_status` is **3%–6%** null before 2016 and
**39%–55%** null from 2016, when the league dropped the "Probable"
designation. Any measure built on that column is not comparable across the boundary.

The contrast is what matters: over the same seasons the coalesced body part is at most
**0.1%** null and `practice_status` at most **0.7%**. Those two are
regime-invariant, so every outcome in this project is built on them, and `report_status` is used
only as a severity refinement in a 2016+ sensitivity run.

| season | rows | report_status_null | practice_status_null | body_part_null | post_probable_drop |
|---|---|---|---|---|---|
| 2009 | 4596 | 0.059 | 0.000 | 0.000 | no |
| 2010 | 4324 | 0.060 | 0.000 | 0.000 | no |
| 2011 | 4748 | 0.051 | 0.000 | 0.000 | no |
| 2012 | 5266 | 0.033 | 0.000 | 0.000 | no |
| 2013 | 4913 | 0.027 | 0.000 | 0.000 | no |
| 2014 | 4912 | 0.044 | 0.000 | 0.000 | no |
| 2015 | 5009 | 0.030 | 0.000 | 0.000 | no |
| 2016 | 4928 | 0.392 | 0.000 | 0.000 | yes |
| 2017 | 4949 | 0.487 | 0.000 | 0.000 | yes |
| 2018 | 4961 | 0.523 | 0.000 | 0.000 | yes |
| 2019 | 5202 | 0.527 | 0.000 | 0.000 | yes |
| 2020 | 5414 | 0.547 | 0.000 | 0.000 | yes |
| 2021 | 5348 | 0.534 | 0.000 | 0.000 | yes |
| 2022 | 5450 | 0.510 | 0.000 | 0.000 | yes |
| 2023 | 5451 | 0.510 | 0.000 | 0.000 | yes |
| 2024 | 5954 | 0.538 | 0.000 | 0.001 | yes |
| 2025 | 5783 | 0.537 | 0.007 | 0.000 | yes |

## 2. Does the spine join?

The availability spine is `rosters_weekly`, not the injury report — roster status is a
transaction record rather than a disclosure, and it covers every position. So the join has to
hold: **99.2%** of injury rows (86,489 of 87,208) find a
weekly roster row on `(season, week, gsis_id)`.

A drop here would be the most dangerous silent failure in the project — an injury row with no
roster row has no availability spine, so the absence becomes invisible rather than missing.

| season | rows | matched | match_rate |
|---|---|---|---|
| 2009 | 4596 | 4509 | 0.981 |
| 2010 | 4324 | 4210 | 0.974 |
| 2011 | 4748 | 4663 | 0.982 |
| 2012 | 5266 | 5167 | 0.981 |
| 2013 | 4913 | 4819 | 0.981 |
| 2014 | 4912 | 4788 | 0.975 |
| 2015 | 5009 | 4896 | 0.977 |
| 2016 | 4928 | 4925 | 0.999 |
| 2017 | 4949 | 4949 | 1.000 |
| 2018 | 4961 | 4961 | 1.000 |
| 2019 | 5202 | 5202 | 1.000 |
| 2020 | 5414 | 5414 | 1.000 |
| 2021 | 5348 | 5348 | 1.000 |
| 2022 | 5450 | 5450 | 1.000 |
| 2023 | 5451 | 5451 | 1.000 |
| 2024 | 5954 | 5954 | 1.000 |
| 2025 | 5783 | 5783 | 1.000 |

## 3. Disclosure behaviour varies by club

The injury report is a strategic artifact and clubs differ in how much they put on it. These are
**covariates, not grades**: a club that discloses more is not a club that injures more, and
keeping those apart is the point of measuring it.

Fewest report rows: **LA** (1,973), **ATL** (2,292), **CHI** (2,353).
Most: **CLE** (3,380), **NYJ** (3,414), **HOU** (3,543).

`questionable_play_rate` — the share of Questionable-listed players who were not inactive — is
the cleanest behavioural index, because a club that lists everybody has a high play-rate among
its Questionables.

| team | listed | q_play_rate |
|---|---|---|
| WAS | 2811 | 0.790 |
| ARI | 2887 | 0.848 |
| MIA | 2918 | 0.811 |
| GB | 3124 | 0.836 |
| IND | 3131 | 0.848 |
| SEA | 3209 | 0.798 |
| NE | 3246 | 0.863 |
| CLE | 3380 | 0.824 |
| NYJ | 3414 | 0.878 |
| HOU | 3543 | 0.751 |

## 4. Body-part taxonomy

17 groups populated from 292 distinct raw
strings. The five **focal** groups are those with a plausible common-cause mechanism — surface,
practice contact policy, S&C programme, medical protocol — and are the subject of the
cross-position-group signature analysis. Hamstring and shoulder are deliberately not focal:
soft-tissue and contact-incidental injuries are individually driven, so a team-wide signature in
them would have no mechanism to point at.

| body_part_group | rows | share | focal |
|---|---|---|---|
| knee | 15117 | 0.173 | yes |
| ankle | 11422 | 0.131 | yes |
| concussion | 4103 | 0.047 | yes |
| back | 3199 | 0.037 | yes |
| hip | 2444 | 0.028 | yes |

All groups:

| body_part_group | rows | share | focal |
|---|---|---|---|
| knee | 15117 | 0.173 | yes |
| ankle | 11422 | 0.131 | yes |
| hamstring | 7771 | 0.089 | no |
| shoulder | 7026 | 0.081 | no |
| non_injury | 6790 | 0.078 | no |
| soft_tissue_lower | 6171 | 0.071 | no |
| foot | 5631 | 0.065 | no |
| arm_hand | 5547 | 0.064 | no |
| concussion | 4103 | 0.047 | yes |
| lower_leg | 3404 | 0.039 | no |
| illness | 3225 | 0.037 | no |
| back | 3199 | 0.037 | yes |
| torso | 3178 | 0.036 | no |
| hip | 2444 | 0.028 | yes |
| neck | 1812 | 0.021 | no |
| other | 200 | 0.002 | no |
| head_face | 159 | 0.002 | no |

### Taxonomy audit

**No raw string above 0.1% of rows falls through to `other`.**



## 5. Sample decisions

- **Injuries exist from 2009** and the injury *report* is comparable across
  that whole span, so it supplies the prior-injury lookback at any depth.
- **⚠️ But absence measures are only comparable from 2021**, which stage 2 established: the
  gameday active/inactive split is absent from `rosters_weekly` before then (`INA` is 2.8k rows
  across 2012–2019 vs 16.8k across 2021–2025, and `ACT` falls 0.86 → 0.59), and the reserve
  codes in `status_description_abbr` carry **no** R-codes before 2021. Anything counting missed
  games is restricted to 2021+.
- **2020 excluded** — the practice and roster regime was
  unlike any other season, and the sibling pipeline already excludes it.
- Grading windows are therefore **2021–2025** (5yr) and **2023–2025** (3yr), which places both
  entirely inside the post-COVID 17-game era and the post-2016 reporting regime.
