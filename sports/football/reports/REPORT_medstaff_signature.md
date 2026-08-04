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

The primary hypothesis, concussion: **does not hold**. Composition
**+0.238** (p 0.190), rate **+0.340**
(p 0.057). Because it was named in advance it is tested at 0.05 rather than the
corrected level; applying the family correction to a preregistered primary would discard the point
of naming one.

Against the exploratory probe, which recorded concussion at 0.405 / 0.404 before this run:
now **0.238 / 0.340** under a properly
exposure-adjusted model. It moved by 0.167 in composition once the denominator was fixed, so the probe result was substantially an artifact of the crude denominator.

Strongest part by composition is **ankle**; by rate, **back**.
0 part(s) clear an uncorrected 0.05 in **both** specifications. **Of the four
exploratory parts, 0 clear the corrected threshold (0.010)** in either
specification.

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
| knee | 0.003 | 0.986 | 0.264 | 0.145 | 0.346 | -0.027 | 0.335 |
| ankle | 0.389 | 0.028 | 0.243 | 0.179 | 0.432 | 0.258 | 0.174 |
| back | 0.249 | 0.170 | 0.346 | 0.053 | 0.512 | 0.433 | 0.172 |
| hip | -0.045 | 0.806 | 0.134 | 0.464 | 0.259 | 0.209 | 0.106 |
| concussion | 0.238 | 0.190 | 0.340 | 0.057 | 0.300 | 0.405 | 0.404 |

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
| DB | 0.166 | 32 |
| DL | 0.201 | 32 |
| LB | 0.234 | 32 |
| OL | 0.187 | 32 |
| QB | 0.216 | 32 |
| RB | 0.296 | 32 |
| ST | 0.238 | 32 |
| WR_TE | 0.229 | 32 |

## 4. Club main effect against club × group interaction

A large `share_common` means the part's excess is spread across the club's position groups, which
is what a shared cause looks like. A large interaction means it is specific to certain groups,
which points at roster or scheme.

| body_group | cells | club_var | interaction_var | share_common |
|---|---|---|---|---|
| knee | 200 | 0.001 | 0.006 | 0.085 |
| ankle | 200 | 0.001 | 0.004 | 0.220 |
| back | 200 | 0.000 | 0.001 | 0.229 |
| hip | 200 | 0.000 | 0.001 | 0.206 |
| concussion | 200 | 0.001 | 0.001 | 0.280 |

## 5. Does the signature follow the head coach?

The only design element that breaks the shared-roster confound: a coach who carries an elevated
share of one body part across **different franchises**, with different rosters and different
buildings, is much harder to explain by roster construction.

**4 head coaches ran two or more clubs** with enough episodes
inside this window. That is a power bound, not a result — five seasons is simply not long enough
for many coaches to change jobs and accumulate evidence at both.

| coach | team | episodes | share_concussion |
|---|---|---|---|
| Frank Reich | IND | 119 | 0.101 |
| Frank Reich | CAR | 97 | 0.041 |
| Mike Vrabel | TEN | 250 | 0.068 |
| Mike Vrabel | NE | 77 | 0.078 |
| Pete Carroll | LV | 50 | 0.060 |
| Pete Carroll | SEA | 292 | 0.034 |
| Sean Payton | DEN | 147 | 0.034 |
| Sean Payton | NO | 53 | 0.057 |

## 6. Reading this against stage 6

Stage 6 found that what persists year over year is **duration and returns-at-all** — how a club
uses injured reserve and times a return — while **recurrence, the outcome most owned by a medical
staff, does not persist at all**. A body-part signature that survives here would be evidence for
a mechanism in a place where the persistence test says clubs otherwise do not differ, which is
why it was worth testing separately rather than folding into the grades.
