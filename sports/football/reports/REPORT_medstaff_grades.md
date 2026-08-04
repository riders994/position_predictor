# Medical-staff grades — the board

## 0. Read this first

**These are not grades of medical staffs.** This data cannot identify one. The residual behind
every letter below bundles the athletic training staff, strength and conditioning, sports
science, the head coach's practice-intensity choices, the general manager's taste for durable
players, and scheme. What is measurable is a **team availability system**.

**Stage 6 narrowed it further.** Of five components, only **duration** and **returns-at-all**
persist year over year — and both are largely *how a club uses injured reserve and times a
return*. **Recurrence, the one outcome most plausibly owned by a medical staff, does not persist
at all.** So the board below separates clubs by **availability-management policy**, not by
quality of medicine.

| component | split_half_r | temporal_r | temporal_p | permutation_p |
|---|---|---|---|---|
| incidence_no_history | 0.708 | 0.293 | 0.104 | 0.020 |
| incidence_with_history | 0.701 | 0.192 | 0.293 | 0.033 |
| duration | 0.827 | 0.529 | 0.002 | 0.001 |
| recurrence | 0.414 | 0.078 | 0.670 | 0.057 |
| returns_at_all | 0.664 | 0.397 | 0.024 | 0.003 |

**The curve is a forced rank, by decision: 3×A · 5×B · 8×C · 8×D · 8×F.** It orders clubs; it does not test
them. Three A's and eight F's are assigned whether or not any club is distinguishable from
another. This is a deliberate provisional choice, to be replaced by an absolute score-to-grade
mapping once several seasons of reports exist — the code already accepts one, so that switch is
a parameter rather than a rewrite.

**0 of 32 clubs are separable from the club ranked immediately below them** at 80%
confidence, and **17 of 32 are separable from the league average**. Adjacent-pair
separation is a strict test in a 32-club field — neighbours are rarely distinguishable anywhere —
so the second number is the one that says whether the extremes mean anything. Read the letters as
an ordering with heavy overlap, not as five distinct tiers.

## 1. Weights

Weights are proportional to each component's **measured split-half reliability**, which is
self-limiting: a component carrying no signal gets a weight near zero without anyone deciding
that it should.

The attributability prior — what one would weight by if grading *medicine* — is shown beside it
and **orders the components almost exactly the other way round**. Recurrence is the most
attributable and the least reliable; duration the least attributable and the most reliable.
There is no way to satisfy both, and that tension is the honest content of this table.

| component | split_half_r | weight_reliability | weight_attributability_prior | gate |
|---|---|---|---|---|
| incidence_no_history | 0.708 | 0.271 | 0.200 | FAIL |
| duration | 0.827 | 0.317 | 0.350 | PASS |
| recurrence | 0.414 | 0.158 | 0.450 | FAIL |
| returns_at_all | 0.664 | 0.254 | — | PASS |

## 2. The board — 5-year window (2021–2025)

`separated_from_average` is the column that says whether a letter means anything: it marks the
clubs whose interval excludes the league mean. **No club is separable from the club immediately
below it**, so the ordering *within* a letter band — and between adjacent bands — carries no
information. `regime_changed` marks clubs that changed head coach inside the window, where the
grade averages over more than one regime.

| team | letter | score | score_sd | vs_average | separated_from_average | regime_changed | letter_3yr | score_3yr |
|---|---|---|---|---|---|---|---|---|
| LA | A | 2.140 | 0.449 | 2.114 | yes | no | A | 1.818 |
| CHI | A | 2.010 | 0.449 | 1.984 | yes | yes | A | 1.635 |
| CIN | A | 1.832 | 0.449 | 1.806 | yes | no | B | 1.324 |
| NO | B | 1.461 | 0.449 | 1.436 | yes | yes | C | 0.752 |
| PHI | B | 1.327 | 0.449 | 1.302 | yes | no | A | 1.419 |
| ATL | B | 1.094 | 0.449 | 1.068 | yes | yes | D | -0.042 |
| IND | B | 1.024 | 0.449 | 0.999 | yes | yes | C | 0.781 |
| CAR | B | 0.948 | 0.449 | 0.922 | yes | yes | C | 0.266 |
| JAX | C | 0.907 | 0.449 | 0.881 | yes | yes | B | 0.918 |
| DET | C | 0.585 | 0.449 | 0.560 | no | no | B | 1.207 |
| BAL | C | 0.581 | 0.449 | 0.556 | no | no | C | 0.509 |
| PIT | C | 0.532 | 0.449 | 0.507 | no | no | C | 0.675 |
| ARI | C | 0.494 | 0.449 | 0.469 | no | yes | C | 0.644 |
| NYJ | C | 0.410 | 0.449 | 0.384 | no | yes | D | -0.293 |
| MIN | C | 0.374 | 0.449 | 0.348 | no | yes | C | 0.297 |
| SF | C | 0.361 | 0.449 | 0.336 | no | no | D | 0.053 |
| DEN | D | 0.174 | 0.449 | 0.148 | no | yes | B | 1.200 |
| LV | D | 0.135 | 0.449 | 0.109 | no | yes | B | 0.891 |
| TB | D | 0.056 | 0.449 | 0.030 | no | yes | D | -0.159 |
| HOU | D | -0.039 | 0.449 | -0.065 | no | yes | F | -0.983 |
| WAS | D | -0.044 | 0.449 | -0.070 | no | yes | D | -0.106 |
| TEN | D | -0.211 | 0.449 | -0.236 | no | yes | C | 0.308 |
| NYG | D | -0.264 | 0.449 | -0.289 | no | yes | D | -0.430 |
| LAC | D | -0.311 | 0.449 | -0.337 | no | yes | D | -0.215 |
| BUF | F | -1.053 | 0.449 | -1.079 | yes | no | F | -1.222 |
| DAL | F | -1.148 | 0.449 | -1.173 | yes | yes | D | -0.889 |
| GB | F | -1.639 | 0.449 | -1.665 | yes | no | F | -1.548 |
| NE | F | -1.744 | 0.449 | -1.770 | yes | yes | F | -1.455 |
| MIA | F | -2.083 | 0.449 | -2.108 | yes | yes | F | -1.011 |
| SEA | F | -2.190 | 0.449 | -2.216 | yes | yes | F | -1.188 |
| KC | F | -2.380 | 0.449 | -2.406 | yes | no | F | -1.797 |
| CLE | F | -2.520 | 0.449 | -2.546 | yes | no | F | -2.311 |

**17 of 32 clubs receive the same letter on the 3-year window**, which is the
most direct stability check available on the board itself.

## 3. Position groups

Every cell is shrunk toward its own club's overall value, and cells are **never ranked**. A club
× position-group cell holds a median of four episodes over five years; at that size the shrinkage
factor does most of the work and the ordering within a club is not interpretable.

| component | cells | shrinkage_k |
|---|---|---|
| duration | 256 | 0.517 |
| incidence_no_history | 256 | 0.646 |
| recurrence | 253 | 0.633 |
| returns_at_all | 256 | 0.464 |

Full cell-level estimates with intervals ship in
`data/processed/medstaff_grades_positions.parquet` rather than as a table here, precisely so they
are not read as a ranking.

## 4. Staff identity

**No staff table was supplied, so these are franchise grades over the window.** nflverse publishes head coaches, not head athletic trainers, and no free structured trainer-by-season source exists. Pass `--staff-table` to attach one; without it the report grades the club, and says so rather than implying more.

## 5. What would change these

- An absolute score-to-grade mapping, once several seasons exist, replacing the forced curve.
- A trainer tenure table, which would let the grade attach to people rather than franchises —
  though tenures are shorter than franchise histories, so the cells would be thinner still.
- More seasons. The comparable window opens in 2021 because the roster fields change meaning
  there; every additional season widens it by 20%.
