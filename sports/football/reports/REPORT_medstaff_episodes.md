# Medical-staff grades — Stage 2: episodes

**12,280 injury episodes** across 2021–2025
(excluding 2020), built from
242,407 player-weeks.

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
| bye | 13317 |
| off_roster | 7870 |

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

## 3. Sanity checks against the world

Distributions can look healthy while the data is wrong. These check *extremes and impossibilities*
on real data, because the bye-week defect (§2) survived four stages behind a perfectly reasonable
mean of 3.3 games missed per spell.

- longest spell: **17 games missed** (must approach season length)
- `off_roster` censoring: **0.8%** (genuine in-season releases of
  injured players are uncommon)
- player-seasons with interior missing weeks: **0**

**All clear.**

## 4. Episodes

| season | episodes | mean_games_missed | censored_share |
|---|---|---|---|
| 2021 | 2503 | 3.972 | 0.281 |
| 2022 | 2387 | 3.893 | 0.302 |
| 2023 | 2436 | 3.760 | 0.301 |
| 2024 | 2467 | 4.045 | 0.293 |
| 2025 | 2487 | 4.054 | 0.304 |

### By body part

| body_group | episodes | mean_games_missed | median_games_missed | ir_share |
|---|---|---|---|---|
| knee | 1879 | 3.692 | 2.000 | 0.221 |
| ankle | 1562 | 2.955 | 2.000 | 0.131 |
| unknown | 1524 | 9.961 | 10.000 | 1.000 |
| shoulder | 998 | 2.949 | 2.000 | 0.123 |
| hamstring | 911 | 3.392 | 2.000 | 0.189 |
| soft_tissue_lower | 859 | 2.786 | 2.000 | 0.120 |
| arm_hand | 816 | 3.012 | 2.000 | 0.137 |
| foot | 676 | 3.209 | 2.000 | 0.149 |
| concussion | 635 | 2.387 | 2.000 | 0.101 |
| back | 538 | 2.665 | 1.000 | 0.113 |
| torso | 535 | 3.071 | 2.000 | 0.142 |
| lower_leg | 514 | 3.644 | 2.000 | 0.220 |
| hip | 453 | 2.530 | 1.000 | 0.099 |
| neck | 317 | 2.830 | 2.000 | 0.158 |
| head_face | 36 | 2.639 | 1.000 | 0.111 |
| other | 27 | 2.370 | 1.000 | 0.185 |

**1 episodes (0.0%) cost zero games** — knocks that were listed but
played through. They are kept, because incidence and recurrence both want them; the duration
model is the one that conditions on missed time.

**1,524 episodes (12.4%) have an `unknown` body part** — spells that opened on
a bare reserve week and never picked up a report row. Body part carries forward *within* a
spell but never across one, so an unrelated later stint cannot inherit an earlier injury.

### By position group

| position_group | episodes |
|---|---|
| DB | 2667 |
| WR_TE | 2399 |
| OL | 2118 |
| LB | 1807 |
| DL | 1754 |
| RB | 981 |
| QB | 403 |
| ST | 151 |

## 5. Recurrence

Risk set is **returns, not episodes** — a spell that never resolved cannot recur, and counting
it would score an unresolved injury as a clean outcome. **8,646 of 12,280 episodes
(70.4%) resolved** and are at risk.

The clock counts **games the player was available for**, not calendar weeks. A player returning
in week 17 has two games of exposure, not six weeks of it; counting calendar time would score a
late-season return as clean purely because the season ran out.

Overall, at the primary `k6` horizon:

| returns | recurrences | rate |
|---|---|---|
| 8646 | 763 | 0.088 |

Focal body parts — the five with a plausible common-cause mechanism, and the subject of the
stage-5 signature analysis:

| body_group | returns | recurrences | rate |
|---|---|---|---|
| knee | 1398 | 201 | 0.144 |
| ankle | 1209 | 123 | 0.102 |
| concussion | 497 | 28 | 0.056 |
| back | 409 | 32 | 0.078 |
| hip | 370 | 20 | 0.054 |

## 6. Censoring

| censor_reason | episodes |
|---|---|
| season_end | 3391 |
| off_roster | 96 |
| team_change | 88 |
| practice_squad | 59 |

Censoring is not neutral here: a club can look good on return-to-play by having unrecovered
cases quietly censored, or by releasing injured players. Stage 4 therefore models
"returns at all" as its own outcome rather than folding it into duration.

## 7. Per-club dispersion

Episodes per club range **286 → 489** (mean 384,
sd 52). Poisson noise alone at this mean would give sd ≈ 20, so the
spread is **2.6× wider than chance**.

That is the number this project exists to decompose — and most of it is not medicine. Roster
age, position mix, snap exposure, surface, and disclosure behaviour all sit inside it. Nothing
here is a grade; stage 4 builds the expectation that has to be subtracted first.
