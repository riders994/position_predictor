# College QB archetypes

The question this stage answers is *what kind of quarterback* a prospect was, not how good he was.
That distinction is the whole design: clustering on efficiency would return a leaderboard with
four bins, and stage 6 already has a model for quality.

- **Clustered on:** `rush_share`, `rush_yds_per_att`, `rush_td_share`, `yards_per_completion`
- **Seasons defining the centroids:** 2078 (≥150 dropbacks)
- **Seasons assigned:** 3351 of 3419
- **Cohort quarterbacks with a career archetype:** 214 of 331

Features are z-scored **within season**. College offense moved far enough across 2004–2021 that
raw features cluster on date — the first split found without this is simply a decade.

## Choosing k

| k | silhouette | inertia | smallest_share |
|---|---|---|---|
| 2 | 0.311 | 3432 | 0.419 |
| 3 | 0.238 | 2829 | 0.249 |
| 4 | 0.242 | 2422 | 0.141 |
| 5 | 0.216 | 2175 | 0.102 |
| 6 | 0.214 | 1974 | 0.093 |
| 7 | 0.205 | 1835 | 0.065 |
| 8 | 0.203 | 1714 | 0.058 |
| 9 | 0.202 | 1607 | 0.057 |
| 10 | 0.208 | 1512 | 0.041 |

Silhouette peaks at k=2, which is the usual result for a continuous space: the honest reading is
that **style is a continuum with no natural gaps**, and any k is a partition of it rather than a
discovery of natural kinds. k=4 is chosen as a local maximum with no cluster smaller than a
seventh of the data, and — the real reason — because every cluster at k=4 has an obvious name.

## The taxonomy

| archetype_name | n | rush_share | rush_yds_per_att | rush_td_share | yards_per_completion | completion_pct | yards_per_attempt | pass_epa_per_db | pass_success_rate | td_rate | int_rate | sack_rate | dropbacks | games |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pocket_downfield | 912 | 0.131 | 5.28 | 0.071 | 13.042 | 0.664 | 8.622 | 0.183 | 0.479 | 0.066 | 0.025 | 0.064 | 257 | 10 |
| pocket_quick | 970 | 0.075 | 2.929 | 0 | 11.38 | 0.659 | 7.504 | 0.079 | 0.458 | 0.05 | 0.029 | 0.06 | 207 | 9 |
| runner_quick | 860 | 0.206 | 5 | 0.25 | 10.833 | 0.654 | 7.05 | 0.013 | 0.44 | 0.044 | 0.028 | 0.069 | 199 | 9 |
| runner_downfield | 609 | 0.34 | 5.853 | 0.412 | 13.143 | 0.631 | 8.263 | 0.104 | 0.446 | 0.058 | 0.029 | 0.074 | 144 | 10 |

**The algorithm recovered a 2×2 it was never told to look for.** The four centroids fall one to a
quadrant of mobility × depth-of-target, which is the strongest available evidence that these two
axes are real rather than imposed. Names are therefore derived from *centroid position*, not from
the k-means label integer — refit with a different seed and 99.1% of seasons keep the
same name, where label integers would have been reshuffled entirely.

**Highest-EPA seasons in each archetype** — the name has to survive these:

- **pocket_downfield** — Sam Bradford 2008, Jameis Winston 2013, Johnny Manziel 2013, Chase Daniel 2008, Joe Burrow 2019, Bryce Petty 2013
- **pocket_quick** — C.J. Stroud 2021, Bailey Zappe 2021, Nick Foles 2011, Sonny Cumbie 2004, Kellen Moore 2011, Dwayne Haskins 2018
- **runner_quick** — Colt McCoy 2008, Brett Basanez 2004, Alex Carder 2011, Chase Daniel 2007, Taylor Kelly 2013, Jake Locker 2009
- **runner_downfield** — Kyler Murray 2018, Nick Florence 2012, Robert Griffin III 2011, Marcus Mariota 2013, Brett Hundley 2013, Marcus Mariota 2014

## Is this secretly a quality ranking?

The clustering features exclude every efficiency measure, and the held-out ones are used to check
whether that worked. The share of within-season EPA variance falling **between** archetypes:

| held_out_column | n | eta_squared |
|---|---|---|
| pass_epa_per_db | 2078 | 0.135 |
| pass_success_rate | 2078 | 0.082 |
| td_rate | 2078 | 0.171 |

At 0.14 the taxonomy explains a modest amount of quality — not
zero, because real styles do differ in average quality, but small enough that the clusters are
kinds rather than ranks. For contrast, clustering on quality stats gives 0.56, and this is exactly
why `completion_pct` was cut from the feature set: it is an accuracy measure, but it is also the
most quality-loaded rate a quarterback has, and including it doubled the leakage.

## Archetype against the breakout label

This is the part the project exists for, and it is also the part the data cannot yet carry.

### Did a sustained breakout ever happen?

`ever_sustained`, over matched quarterbacks whose outcome is settled (right-censored careers dropped — a 2022 entrant has not had time to fail).

| archetype | n | ever_sustained | rate | 95% CI |
|---|---|---|---|---|
| pocket_downfield | 91 | 20 | 0.22 | 0.15–0.32 |
| pocket_quick | 60 | 3 | 0.05 | 0.02–0.14 |
| runner_downfield | 32 | 9 | 0.281 | 0.16–0.45 |
| runner_quick | 28 | 6 | 0.214 | 0.10–0.40 |
| **all** | 211 | 38 | 0.18 | 0.13–0.24 |

### Among quarterbacks who broke out, which did it late?

`late_sustained`, over matched quarterbacks with a sustained breakout.

| archetype | n | late_sustained | rate | 95% CI |
|---|---|---|---|---|
| pocket_downfield | 20 | 5 | 0.25 | 0.11–0.47 |
| pocket_quick | 3 | 2 | 0.667 | 0.21–0.94 |
| runner_downfield | 9 | 2 | 0.222 | 0.06–0.55 |
| runner_quick | 6 | 0 | 0 | 0.00–0.39 |
| **all** | 38 | 9 | 0.237 | 0.13–0.39 |

- **Did a sustained breakout ever happen?** widest gap between archetypes 0.231; label shuffles reproduce a gap that wide **p = 0.034** (n = 211, 38 positive)
- **Among quarterbacks who broke out, which did it late?** widest gap between archetypes 0.667; label shuffles reproduce a gap that wide **p = 0.064** (n = 38, 9 positive)

**What the numbers support.** Read the intervals, not the point estimates —
- *Did a sustained breakout ever happen?* — the 0.23 spread survives label shuffling (p = 0.034), so archetype and this outcome are **not independent**. It is driven by one cell: **pocket_quick** at 0.05 (3/60) against **runner_downfield** at 0.28 (9/32), and those two intervals do not overlap. Still only 38 positives in 211 split four ways: carry the direction into stage 6 as a prior, not as an effect size.
- *Among quarterbacks who broke out, which did it late?* — a 0.67 spread arises from shuffled labels 6% of the time, so at 9 positives in 38 this is **consistent with noise**. The base rate 0.24 is the honest summary of every cell.

Neither result licenses using archetype as a classifier on its own. What both say is that the cells are too small for a four-way rate comparison to settle anything — a finding about power as much as about football.

What the taxonomy is genuinely for is stage 6: as a categorical feature alongside the continuous
ones, as a stratification variable for validation folds, and as a way to ask whether the *same*
pre-NFL evidence means different things for different kinds of quarterback — a question with more
statistical room in it than a four-way rate comparison.

## Career-level columns produced

| Column | Meaning |
|---|---|
| `archetype` / `archetype_name` / `archetype_label` | the **final** college season's cluster |
| `archetype_first` | the first season's cluster |
| `archetype_modal` | the most common cluster across the career |
| `archetype_stability` | share of college seasons spent in the final archetype |
| `archetype_changed` | whether the first and last archetypes differ |
| `n_archetype_seasons` | seasons contributing to the above |

The final season is the anchor, matching the project's established preference for `final_*`
features: a career clipped by the 2004 coverage floor still has a genuine last season, and it is
the one NFL evaluators weighted most. `archetype_changed` is a candidate feature in its own right
— 47% of cohort quarterbacks changed style during college,
and a quarterback who did is not obviously the same prospect as one who never did.
