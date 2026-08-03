# Medical-staff grades — Stage 2: episodes

**14,420 injury episodes** across 2021–2025
(excluding 2020), built from
221,313 player-weeks.

## 1. How a spell is bounded

**Roster status is the availability signal, not snaps.** Players on `RES`, `INA`, `DEV` or
`CUT` take a snap in about 0.02% of player-weeks, so weekly status separates played from
did-not-play almost perfectly — while keying on `gsis_id` with no crosswalk loss and reaching
back to 2002. `snap_counts` keys on `pfr_player_id`, crosswalks at only 81.7% from 2012, and
would misread the ~21% of active player-weeks with zero snaps: a healthy scratch or a deep
backup is **available**, not injured.

Only an `ACT` week with no designation ends a spell. Inactive, reserve and bye weeks all
continue it, which absorbs report noise — a player listed in weeks 5 and 7 but merely inactive
in week 6 never "returned" in week 6.

| week_state | player_weeks |
|---|---|
| available | 115308 |
| impaired | 47320 |
| practice_squad | 42825 |
| out_other | 15767 |
| bye | 93 |

## 2. ⚠️ Why the sample starts at 2021

Two `rosters_weekly` fields change meaning at 2021, and both would have corrupted every absence
measure here. `status_description_abbr` carries **no** reserve codes before 2021 — so reading IR
off it produced 2 episodes across 2012–2019 against ~1,000 per season after. And `status ==
"INA"` is barely populated earlier (2.8k rows across 2012–2019 vs 16.8k across 2021–2025) while
`ACT` falls 0.86 → 0.59: the gameday active/inactive split is simply absent from the earlier
data.

`status == "RES"` is the one stable signal, so **reserve is read from `status`, never from the
abbr codes**. This is the same shape as the sibling project's cfbfastR flag defect — an
unpopulated field aggregates to a clean zero rather than a null, so nothing errors and the
series quietly means something different on each side of the boundary.

| season | rows | reserve_share | abbr_R_share | active_share | inactive_share | comparable |
|---|---|---|---|---|---|---|
| 2012 | 30005 | 0.080 | 0.000 | 0.861 | 0.000 | no |
| 2013 | 30437 | 0.062 | 0.000 | 0.860 | 0.000 | no |
| 2014 | 30492 | 0.074 | 0.000 | 0.850 | 0.000 | no |
| 2015 | 30636 | 0.082 | 0.000 | 0.834 | 0.000 | no |
| 2016 | 32944 | 0.134 | 0.000 | 0.857 | 0.000 | no |
| 2017 | 49210 | 0.094 | 0.000 | 0.553 | 0.000 | no |
| 2018 | 50113 | 0.086 | 0.000 | 0.542 | 0.000 | no |
| 2019 | 49561 | 0.095 | 0.000 | 0.491 | 0.057 | no |
| 2020 | 41972 | 0.109 | 0.088 | 0.591 | 0.072 | no |
| 2021 | 44539 | 0.120 | 0.149 | 0.586 | 0.071 | yes |
| 2022 | 44059 | 0.113 | 0.124 | 0.593 | 0.078 | yes |
| 2023 | 43545 | 0.108 | 0.113 | 0.600 | 0.078 | yes |
| 2024 | 44473 | 0.117 | 0.126 | 0.587 | 0.077 | yes |
| 2025 | 44697 | 0.123 | 0.131 | 0.585 | 0.076 | yes |

**Reserve/COVID-19 (`R59`) is excluded.** It occurs in 2021 only — 725 player-weeks, exactly
zero in every other season — and carries `RES` status without being an injury. Left in, it
pushed 2021's reserve share to 0.46 against ~0.31 for 2022–2025, making every club look worse
at medicine in the first year of the five-year window.

## 3. Episodes

| season | episodes | mean_games_missed | censored_share |
|---|---|---|---|
| 2021 | 2936 | 3.369 | 0.446 |
| 2022 | 2791 | 3.314 | 0.461 |
| 2023 | 2826 | 3.226 | 0.449 |
| 2024 | 2919 | 3.401 | 0.455 |
| 2025 | 2948 | 3.408 | 0.464 |

### By body part

| body_group | episodes | mean_games_missed | median_games_missed | ir_share |
|---|---|---|---|---|
| unknown | 2813 | 6.416 | 6.000 | 1.000 |
| knee | 2057 | 2.982 | 2.000 | 0.214 |
| soft_tissue_lower | 1950 | 2.650 | 2.000 | 0.154 |
| ankle | 1684 | 2.527 | 2.000 | 0.129 |
| shoulder | 1055 | 2.501 | 2.000 | 0.120 |
| arm_hand | 886 | 2.558 | 2.000 | 0.139 |
| foot | 731 | 2.763 | 2.000 | 0.142 |
| concussion | 674 | 2.083 | 2.000 | 0.093 |
| torso | 575 | 2.539 | 2.000 | 0.141 |
| lower_leg | 562 | 2.899 | 2.000 | 0.210 |
| back | 555 | 2.263 | 1.000 | 0.114 |
| hip | 477 | 2.220 | 1.000 | 0.103 |
| neck | 336 | 2.265 | 1.000 | 0.152 |
| head_face | 37 | 1.784 | 1.000 | 0.081 |
| other | 28 | 2.429 | 1.000 | 0.214 |

**1 episodes (0.0%) cost zero games** — knocks that were listed but
played through. They are kept, because incidence and recurrence both want them; the duration
model is the one that conditions on missed time.

**2,813 episodes (19.5%) have an `unknown` body part** — spells that opened on
a bare reserve week and never picked up a report row. Body part carries forward *within* a
spell but never across one, so an unrelated later stint cannot inherit an earlier injury.

### By position group

| position_group | episodes |
|---|---|
| DB | 3074 |
| WR_TE | 2822 |
| OL | 2560 |
| LB | 2114 |
| DL | 2045 |
| RB | 1157 |
| QB | 466 |
| ST | 182 |

## 4. Recurrence

Risk set is **returns, not episodes** — a spell that never resolved cannot recur, and counting
it would score an unresolved injury as a clean outcome. **7,860 of 14,420 episodes
(54.5%) resolved** and are at risk.

The clock counts **games the player was available for**, not calendar weeks. A player returning
in week 17 has two games of exposure, not six weeks of it; counting calendar time would score a
late-season return as clean purely because the season ran out.

Overall, at the primary `k6` horizon:

| returns | recurrences | rate |
|---|---|---|
| 7860 | 728 | 0.093 |

Focal body parts — the five with a plausible common-cause mechanism, and the subject of the
stage-5 signature analysis:

| body_group | returns | recurrences | rate |
|---|---|---|---|
| knee | 1250 | 167 | 0.134 |
| ankle | 1075 | 106 | 0.099 |
| concussion | 445 | 23 | 0.052 |
| back | 374 | 30 | 0.080 |
| hip | 338 | 17 | 0.050 |

## 5. Censoring

| censor_reason | episodes |
|---|---|
| season_end | 3378 |
| off_roster | 3051 |
| team_change | 82 |
| practice_squad | 49 |

Censoring is not neutral here: a club can look good on return-to-play by having unrecovered
cases quietly censored, or by releasing injured players. Stage 4 therefore models
"returns at all" as its own outcome rather than folding it into duration.

## 6. Per-club dispersion

Episodes per club range **336 → 564** (mean 451,
sd 58). Poisson noise alone at this mean would give sd ≈ 21, so the
spread is **2.7× wider than chance**.

That is the number this project exists to decompose — and most of it is not medicine. Roster
age, position mix, snap exposure, surface, and disclosure behaviour all sit inside it. Nothing
here is a grade; stage 4 builds the expectation that has to be subtracted first.
