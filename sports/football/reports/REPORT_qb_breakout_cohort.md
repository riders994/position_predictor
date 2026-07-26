# Late-breakout QBs — cohort and descriptive analysis

Which quarterbacks became fantasy-relevant *after* the league had moved on, and what they had in
common before they got there. This report covers the **NFL-side ground truth only**; the
pre-NFL (high-school and college) evidence that the models are restricted to is built in later
stages. Design rationale: [`docs/QB_BREAKOUT_PLAN.md`](../docs/QB_BREAKOUT_PLAN.md).

- **Seasons covered:** 1999–2025 (nflverse weekly stats)
- **Cohort:** 331 QBs entering the NFL in 1999 or later
- **Breakout tier:** top-20 PPR points-per-game among QBs clearing 7
  games — superflex-relevant, since QB15 is where draft-day value lives
- **Late:** the breakout arrived in NFL year 4 or later
- **Late breakouts found:** **17** of 69 QBs who ever broke out

---

## 1. Defining "breakout" is the whole problem

Three candidate definitions were built and scored against QBs whose careers are not in dispute.
They disagree sharply:

| definition | n_defined | n_late | pct_late |
|---|---|---|---|
| late | 93 | 25 | 26.9 |
| late_qb1 | 73 | 22 | 30.1 |
| late_sustained | 69 | 17 | 24.6 |

`late` (first top-20 season) is too loose. The bar is rank 20 of roughly 32 QBs who
play, so one ordinary season clears it — **Baker Mayfield's 2018 rookie year ranks exactly 20th**,
and Ryan Tannehill's 2014 ranks 10th. Both therefore read as *on-time* breakouts, which is the
opposite of what their careers show.

`late_qb1` (first top-12 season) over-corrects: it labels **Tom Brady** a late breakout because
his first top-12 fantasy season came in year 6, even though he was a quality starter from year 2.

`late_sustained` requires the tier to **hold** — top-20 in at least 2 of the 3
seasons starting with the breakout. This is the definition the project models. It is the only one
of the three that puts Mayfield (year 6), Tannehill (year 8) and Geno Smith (year 10) in the late
cell while leaving Brady, Josh Allen, Joe Burrow and Brock Purdy on time.

## 2. Developed late, or needed a new building?

| breakout timing | relocated | same franchise | undrafted |
|---|---|---|---|
| late | 7 | 8 | 2 |
| on-time | 4 | 47 | 1 |

## 3. Which draft situations produce late breakouts

Censored careers — QBs who entered too recently to have had a year-4 season yet —
are excluded so recent draft classes do not inflate the never-broke-out rate.

| bucket | never | on_time | late | n | pct_ever_broke_out | pct_late_given_breakout |
|---|---|---|---|---|---|---|
| R1 top-10 | 17 | 27 | 3 | 47 | 63.8 | 10.0 |
| R1 11-32 | 16 | 12 | 2 | 30 | 46.7 | 14.3 |
| R2 | 16 | 4 | 3 | 23 | 30.4 | 42.9 |
| R3 | 25 | 2 | 1 | 28 | 10.7 | 33.3 |
| R4-7 | 97 | 6 | 6 | 109 | 11.0 | 50.0 |
| Undrafted | 63 | 1 | 2 | 66 | 4.5 | 66.7 |

Read `pct_late_given_breakout` rather than `pct_ever_broke_out`: the first says a bucket produces
good quarterbacks, the second says it produces them *slowly*.

## 4. How deep was the trough?

Median profile over each QB's first three NFL seasons — the rookie-contract window the league
judges them on. Seasons too short to be ranked are counted as rank 40, worse than the worst real
rank, so "did not play enough to rank" registers as the bad outcome it is.

| outcome | n | early_games | mean_early_ppg | mean_early_rank | best_early_rank |
|---|---|---|---|---|---|
| never | 218 | 7.0 | 4.18 | 40.0 | 40.0 |
| on_time | 52 | 40.0 | 15.68 | 15.83 | 7.0 |
| late | 16 | 14.5 | 5.97 | 40.0 | 40.0 |

**This is the central result, and it justifies the whole project.** Through three NFL seasons,
future late breakouts look far more like QBs who never broke out than like QBs who broke out on
time: median early PPG of 5.97 against 4.18 for the busts and 15.68 for the on-time group, and a
median early rank pinned at the unranked floor for both the late and the never groups. They do get
more early playing time than the busts (median 14.5 games vs 7), so there is *some* separation —
but on production, early NFL evidence barely distinguishes a future late breakout from a bust.

If early NFL performance cannot separate them, then a model that waits for NFL evidence is waiting
for information that does not arrive. That is the case for going back to what was knowable before
the player was drafted at all — which is what the remaining stages build.

## 5. When the breakout arrives

| nfl_year | n_qbs | cumulative_pct |
|---|---|---|
| 1 | 21 | 30.4 |
| 2 | 23 | 63.8 |
| 3 | 8 | 75.4 |
| 4 | 8 | 87.0 |
| 5 | 3 | 91.3 |
| 6 | 2 | 94.2 |
| 7 | 1 | 95.7 |
| 8 | 2 | 98.6 |
| 10 | 1 | 100.0 |

## 6. The late-breakout roster

The cohort every later stage is built to predict from pre-NFL evidence alone.

| player_name | entry_season | draft_round | draft_pick | draft_franchise | draft_college | sustained_season | sustained_nfl_year | sustained_rank | peak_season | peak_nfl_year | peak_rank | sustained_relocated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jordan Love | 2020 | 1.0 | 26.0 | GB | Utah St. | 2023.0 | 4.0 | 5.0 | 2023.0 | 4.0 | 5.0 | 0.0 |
| Baker Mayfield | 2018 | 1.0 | 1.0 | CLE | Oklahoma | 2023.0 | 6.0 | 17.0 | 2024.0 | 7.0 | 4.0 | 1.0 |
| Geno Smith | 2013 | 2.0 | 39.0 | NYJ | West Virginia | 2022.0 | 10.0 | 9.0 | 2022.0 | 10.0 | 9.0 | 1.0 |
| Jimmy Garoppolo | 2014 | 2.0 | 62.0 | NE | Eastern Illinois | 2021.0 | 8.0 | 18.0 | 2021.0 | 8.0 | 18.0 | 1.0 |
| Ryan Tannehill | 2012 | 1.0 | 8.0 | MIA | Texas A&M | 2019.0 | 8.0 | 9.0 | 2019.0 | 8.0 | 9.0 | 1.0 |
| Kirk Cousins | 2012 | 4.0 | 102.0 | WAS | Michigan St. | 2015.0 | 4.0 | 12.0 | 2016.0 | 5.0 | 6.0 | 0.0 |
| Tyrod Taylor | 2011 | 6.0 | 180.0 | BAL | Virginia Tech | 2015.0 | 5.0 | 6.0 | 2015.0 | 5.0 | 6.0 | 1.0 |
| Ryan Fitzpatrick | 2005 | 7.0 | 250.0 | LA | Harvard | 2010.0 | 6.0 | 15.0 | 2018.0 | 14.0 | 5.0 | 1.0 |
| Alex Smith | 2005 | 1.0 | 1.0 | SF | Utah | 2009.0 | 5.0 | 17.0 | 2017.0 | 13.0 | 4.0 | 0.0 |
| Shaun Hill | 2002 |  |  |  |  | 2008.0 | 7.0 | 10.0 | 2008.0 | 7.0 | 10.0 |  |
| Matt Schaub | 2004 | 3.0 | 90.0 | ATL | Virginia | 2008.0 | 5.0 | 7.0 | 2009.0 | 6.0 | 5.0 | 1.0 |
| Matt Cassel | 2005 | 7.0 | 230.0 | NE | USC | 2008.0 | 4.0 | 11.0 | 2008.0 | 4.0 | 11.0 | 0.0 |
| Aaron Rodgers | 2005 | 1.0 | 24.0 | GB | California | 2008.0 | 4.0 | 2.0 | 2009.0 | 5.0 | 1.0 | 0.0 |
| Kyle Orton | 2005 | 4.0 | 106.0 | CHI | Purdue | 2008.0 | 4.0 | 19.0 | 2010.0 | 6.0 | 9.0 | 0.0 |
| Tony Romo | 2003 |  |  |  |  | 2006.0 | 4.0 | 10.0 | 2007.0 | 5.0 | 2.0 |  |
| David Garrard | 2002 | 4.0 | 108.0 | JAX | East Carolina | 2005.0 | 4.0 | 12.0 | 2007.0 | 6.0 | 7.0 | 0.0 |
| Drew Brees | 2001 | 2.0 | 32.0 | LAC | Purdue | 2004.0 | 4.0 | 6.0 | 2008.0 | 8.0 | 1.0 | 0.0 |

## 7. Still open

QBs whose window has not closed — either too recent to have reached year 4, or
holding one breakout season that has not yet had time to prove it sustains. **Sam Darnold is the
live case**: his 2024 with Minnesota ranked 9th, but 2025 came in at 25th, so the sustained label
is still pending rather than earned. These rows are excluded from every base rate above, and they
are precisely who a working model would be scoring today.

| player_name | pending_kind | entry_season | draft_round | draft_pick | draft_franchise | draft_college | seasons_elapsed | breakout_season | breakout_nfl_year | best_ppg_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| Drake Maye | breakout pending sustain | 2024 | 1.0 | 3.0 | NE | North Carolina | 2 | 2025.0 | 2.0 | 2.0 |
| Jacoby Brissett | breakout pending sustain | 2016 | 3.0 | 91.0 | NE | North Carolina St. | 10 | 2025.0 | 10.0 | 18.0 |
| Sam Darnold | breakout pending sustain | 2018 | 1.0 | 3.0 | NYJ | USC | 8 | 2024.0 | 7.0 | 9.0 |
| Anthony Richardson | window still open | 2023 | 1.0 | 4.0 | IND | Florida | 3 |  |  | 21.0 |
| Bryce Young | window still open | 2023 | 1.0 | 1.0 | CAR | Alabama | 3 |  |  | 25.0 |
| Aidan O'Connell | window still open | 2023 | 4.0 | 135.0 | LV | Purdue | 3 |  |  | 27.0 |
| Michael Penix Jr. | window still open | 2024 | 1.0 | 8.0 | ATL | Washington | 2 |  |  | 27.0 |
| Will Levis | window still open | 2023 | 2.0 | 33.0 | TEN | Kentucky | 3 |  |  | 28.0 |
| Tommy DeVito | window still open | 2023 |  |  |  |  | 3 |  |  | 29.0 |
| J.J. McCarthy | window still open | 2024 | 1.0 | 10.0 | MIN | Michigan | 2 |  |  | 29.0 |
| Spencer Rattler | window still open | 2024 | 5.0 | 150.0 | NO | South Carolina | 2 |  |  | 34.0 |
| Dorian Thompson-Robinson | window still open | 2023 | 5.0 | 140.0 | CLE | UCLA | 3 |  |  | 38.0 |
| Jake Haener | window still open | 2023 | 4.0 | 127.0 | NO | Fresno St. | 3 |  |  | 44.0 |
| Adrian Martinez | window still open | 2023 |  |  |  |  | 3 |  |  |  |
| Clayton Tune | window still open | 2023 | 5.0 | 139.0 | ARI | Houston | 3 |  |  |  |
| Hendon Hooker | window still open | 2023 | 3.0 | 68.0 | DET | Tennessee | 3 |  |  |  |
| Jaren Hall | window still open | 2023 | 5.0 | 164.0 | MIN | BYU | 3 |  |  |  |
| Sean Clifford | window still open | 2023 | 5.0 | 149.0 | GB | Penn St. | 3 |  |  |  |
| Tanner McKee | window still open | 2023 | 6.0 | 188.0 | PHI | Stanford | 3 |  |  |  |
| Tyson Bagent | window still open | 2023 |  |  |  |  | 3 |  |  |  |

---

## Caveats

- **Small N.** 17 late breakouts across 1999–2025 is the honest ceiling on this
  question. Any model here is evidence-weighting, not prediction at scale, and the validation
  design has to reflect that.
- **Right-censoring is real.** 28 QBs are too recent to label. They
  are excluded from base rates rather than counted as failures.
- **2020 is kept.** The sibling project drops the COVID season from supervised use, but a
  breakout is a career milestone — dropping 2020 would erase the first good season of anyone who
  broke out that year.
- **Fantasy PPG is the tier, not football quality.** Rushing production inflates QB fantasy
  scoring relative to passing quality, so a mobile QB clears the tier on less passing skill.
