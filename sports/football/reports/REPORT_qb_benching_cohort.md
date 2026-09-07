# QB Benching — Stage 1: the cohort and the label

_2009–2025 · 544 opening starters · benched at ≥3 weeks_

## What this counts

A club's **opening starter** is the quarterback who started its first game of the season. Every later game the club played is then one row: either he started it, or he did not and the reason is resolved three ways.

| outcome | team_games | share |
|---|---|---|
| held | 6467 | 0.777 |
| injured | 1103 | 0.133 |
| benched | 672 | 0.081 |
| gone | 76 | 0.009 |

## Base rate

**90 of 544 opening starters (16.5%) were benched.** For comparison, 142 (26.1%) lost the same number of weeks to injury and 213 (39.2%) lost them for any reason at all.

| season | openers | positives | rate |
|---|---|---|---|
| 2009 | 32 | 7 | 0.219 |
| 2010 | 32 | 6 | 0.188 |
| 2011 | 32 | 3 | 0.094 |
| 2012 | 32 | 3 | 0.094 |
| 2013 | 32 | 6 | 0.188 |
| 2014 | 32 | 6 | 0.188 |
| 2015 | 32 | 2 | 0.062 |
| 2016 | 32 | 5 | 0.156 |
| 2017 | 32 | 5 | 0.156 |
| 2018 | 32 | 6 | 0.188 |
| 2019 | 32 | 6 | 0.188 |
| 2020 | 32 | 6 | 0.188 |
| 2021 | 32 | 2 | 0.062 |
| 2022 | 32 | 7 | 0.219 |
| 2023 | 32 | 4 | 0.125 |
| 2024 | 32 | 7 | 0.219 |
| 2025 | 32 | 9 | 0.281 |

## The bar is a choice, so here are the others

| bar | openers | benched | rate |
|---|---|---|---|
| 2 | 544 | 99 | 0.182 |
| 3 | 544 | 90 | 0.165 |
| 4 | 544 | 74 | 0.136 |
| 6 | 544 | 46 | 0.085 |

## How each week was resolved

Benching is read off the club's **own depth chart** wherever one was published: a quarterback listed behind someone else has been demoted, and one missing from his club's chart is unavailable. That is a direct observation rather than an inference, and it resolves **526 of the 672** benched weeks; the remaining 146 fall through to the roster ladder as a residual.

| era | evidence | outcome | team_games |
|---|---|---|---|
| 2021+ | injury_report | injured | 225 |
| 2021+ | depth_chart | benched | 153 |
| 2021+ | off_chart | injured | 151 |
| 2021+ | residual | benched | 45 |
| 2021+ | off_chart | gone | 26 |
| 2021+ | weekly_status | injured | 16 |
| 2021+ | other_club | gone | 1 |
| pre-2021 | injury_report | injured | 430 |
| pre-2021 | depth_chart | benched | 373 |
| pre-2021 | off_chart | injured | 272 |
| pre-2021 | residual | benched | 101 |
| pre-2021 | off_chart | gone | 49 |
| pre-2021 | off_roster | injured | 9 |

## Two roster-table defects this label had to route around

**Before 2021, `rosters_weekly.status` is not a weekly value.** It is the player's season-final status stamped on every one of his weeks: only 1–4% of pre-2021 QB-seasons have a status that varies by week, against 50–72% from 2021 on. Colin Kaepernick started the first eight games of 2015 and finished on injured reserve, so all ten of his rows read `RES` — including the weeks he was starting. A first version of this label used that column and reported **zero benchings in 2015**, a season in which he visibly lost the job to Blaine Gabbert.

**Injured reserve is invisible to the injury report** — a player placed on it drops off the report entirely. Measured against the trustworthy 2021+ status, weeks with no report listing and no appearance split 195 reserve / 104 active / 20 inactive, so a label keyed on the report alone would call roughly 60% of them benchings.

The depth chart has neither problem, which is why it leads the ladder. The check below is what says so: the label reads different evidence either side of 2021, so its rate had better not move.

| regime | openers | positives | rate |
|---|---|---|---|
| pre-2021 | 384 | 61 | 0.159 |
| 2021+ | 160 | 29 | 0.181 |

The rates differ by 0.022, inside season-to-season variation — the two eras are reading the same thing even though the roster column beneath them changed.

## How far to trust the starter column

`schedules` records the announced starter. Against the obvious alternative — the QB with the most dropbacks in the game — it agrees on **96.9%** of team-games. The disagreements are in-game changes, a hook or an injury, and counting them as displacement would fold within-game events into a week-to-week decision. A sample:

| season | week | team | starter_name | dropbacks |
|---|---|---|---|---|
| 2025 | 14 | WAS | Marcus Mariota | 21 |
| 2025 | 16 | KC | Gardner Minshew | 20 |
| 2025 | 16 | NYJ | Tyrod Taylor | 43 |
| 2025 | 16 | BAL | Lamar Jackson | 11 |
| 2025 | 17 | NYJ | Tyrod Taylor | 34 |
| 2025 | 18 | LV | Kenny Pickett | 23 |
| 2025 | 18 | DAL | Dak Prescott | 13 |
| 2025 | 18 | TEN | Cam Ward | 31 |

## Reproduce

```bash
uv run python sports/football/scripts/qb_benching_cohort.py
```
