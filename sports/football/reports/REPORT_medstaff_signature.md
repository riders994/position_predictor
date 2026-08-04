# Medical-staff grades — Stage 5: cross-position-group injury signature

If a club's excess concentrates in **one body part** and appears across position groups that
share nothing except the building, that implicates a common cause — practice contact policy,
tackling technique, strength and conditioning, field surface, medical protocol. If the excess is
confined to one group, it is roster, scheme or luck specific to that group.

**This is a variance decomposition, not a grade.** A club × body-part × position-group cell holds
a median of four episodes over five years, so cells are never graded; the concordance statistic
aggregates across the eight groups instead.

## 1. Multiplicity, preregistered

Five parts are tested, so the family-wise threshold is **0.010**.
**Concussion is the single primary hypothesis**, on an a-priori mechanism:
practice contact policy and tackling technique are documented coach decisions, and concussion
reporting is protocol-mandated rather than discretionary, which makes it the least
disclosure-contaminated outcome in the project. The other four are **exploratory** and are
labelled as such wherever they appear.

**The primary hypothesis, concussion, holds.** Composition **+0.437**
(p 0.012), rate **+0.445** (p 0.011) — it
clears α = 0.05 in
**both** specifications, and it is the **strongest part in both**. Because it was named in advance
it is tested at 0.05 rather than the corrected level; applying the family correction to it would
discard the point of preregistering one.

It also replicates the exploratory probe closely — concussion was 0.405 / 0.404 there against
**0.437 / 0.445** here, now under a properly
exposure-adjusted model instead of the probe's crude denominator. That the number barely moved
when the denominator was fixed is the strongest thing that can be said for it.

2 parts clear an uncorrected 0.05 in **both** specifications
(back, concussion) — but only concussion was named in advance, so the other
is exploratory.

**Of the four exploratory parts, 0 clear the corrected threshold
(0.010) in either specification.** Back is the near miss — it is the only other part
positive in both specs — and it is labelled exploratory, not a finding.

Spells with an unlabelled body part are **excluded, not pooled** (unknown, other, non_injury, illness).
The `unknown` group alone is 21.6% of episodes — spells that opened on a bare reserve week and
never picked up a report row — and it is concentrated in long absences, so pooling it would swamp
the parts being compared.

## 2. The two specifications

`composition` is B's share of that side's own episodes, additive-log-ratio transformed off
`soft_tissue_lower`. It captures **signature net of level** and is **disclosure-robust** — a club
that lists everyone inflates numerator and denominator alike. That is a real advantage over every
level-based measure elsewhere in this project.

`rate` is episodes of B per unit exposure against a leave-one-team-out expectation. It captures
**level plus signature**: a club with more injuries overall has more of everything.

An effect in `rate` but not `composition` is a **level artifact** — the club's overall burden
travelling, not a part-specific tendency. An effect in `composition` but not `rate` is a
compositional shift with no absolute excess behind it. Only agreement in both is a candidate
signature.

The `prereg_*` columns are the exploratory probe recorded **before** this run, so it is
confirmatory rather than fishing.

| body_group | comp_off_def | comp_p | rate_off_def | rate_p | comp_random_half | prereg_share | prereg_rate |
|---|---|---|---|---|---|---|---|
| knee | 0.197 | 0.279 | 0.224 | 0.218 | 0.357 | -0.027 | 0.335 |
| ankle | 0.361 | 0.042 | 0.216 | 0.234 | 0.171 | 0.258 | 0.174 |
| back | 0.368 | 0.038 | 0.402 | 0.023 | 0.517 | 0.433 | 0.172 |
| hip | 0.106 | 0.562 | 0.150 | 0.412 | 0.319 | 0.209 | 0.106 |
| concussion | 0.437 | 0.012 | 0.445 | 0.011 | 0.529 | 0.405 | 0.404 |

## 3. Offense against defense, and why

Different position coaches and different drills; the same building, S&C programme, medical staff
and surface. It is the sharpest split available — but **it is not fully independent**: an old or
badly-conditioned roster is old on both sides, which can manufacture concordance. The composition
specification removes the level effect by construction, and the random split-half column is run
alongside as a pure-consistency check.

### Leave-one-position-group-out — concussion

Guards against a single group, usually the offensive line, carrying a result that then reads as
club-wide.

| dropped_group | spearman | n |
|---|---|---|
| DB | 0.142 | 32 |
| DL | 0.368 | 32 |
| LB | 0.469 | 32 |
| OL | 0.288 | 32 |
| QB | 0.443 | 32 |
| RB | 0.398 | 32 |
| ST | 0.437 | 32 |
| WR_TE | 0.299 | 32 |

## 4. Club main effect against club × group interaction

A large `share_common` means the part's excess is spread across the club's position groups, which
is what a shared cause looks like. A large interaction means it is specific to certain groups,
which points at roster or scheme.

| body_group | cells | club_var | interaction_var | share_common |
|---|---|---|---|---|
| knee | 201 | 0.001 | 0.006 | 0.089 |
| ankle | 201 | 0.001 | 0.004 | 0.201 |
| back | 201 | 0.000 | 0.001 | 0.237 |
| hip | 201 | 0.000 | 0.001 | 0.207 |
| concussion | 201 | 0.001 | 0.001 | 0.299 |

## 5. Does the signature follow the head coach?

The only design element that breaks the shared-roster confound: a coach who carries an elevated
share of one body part across **different franchises**, with different rosters and different
buildings, is much harder to explain by roster construction.

**4 head coaches ran two or more clubs** with enough episodes
inside this window. That is a power bound, not a result — five seasons is simply not long enough
for many coaches to change jobs and accumulate evidence at both.

| coach | team | episodes | share_concussion |
|---|---|---|---|
| Frank Reich | IND | 122 | 0.098 |
| Frank Reich | CAR | 107 | 0.037 |
| Mike Vrabel | TEN | 273 | 0.073 |
| Mike Vrabel | NE | 81 | 0.074 |
| Pete Carroll | SEA | 305 | 0.033 |
| Pete Carroll | LV | 53 | 0.057 |
| Sean Payton | NO | 56 | 0.054 |
| Sean Payton | DEN | 158 | 0.032 |

## 6. Reading this against stage 6

Stage 6 found that what persists year over year is **duration and returns-at-all** — how a club
uses injured reserve and times a return — while **recurrence, the outcome most owned by a medical
staff, does not persist at all**. A body-part signature that survives here would be evidence for
a mechanism in a place where the persistence test says clubs otherwise do not differ, which is
why it was worth testing separately rather than folding into the grades.
