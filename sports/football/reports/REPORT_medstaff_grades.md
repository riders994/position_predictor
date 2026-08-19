# Medical-staff grades — the board

## 0. Read this first

**These are not grades of medical staffs.** This data cannot identify one. The residual behind
every letter below bundles the athletic training staff, strength and conditioning, sports
science, the head coach's practice-intensity choices, the general manager's taste for durable
players, and scheme. What is measurable is a **team availability system**.

**Stage 6 narrowed it further.** Of five components, the ones that persist year over year are
**incidence_no_history, duration**; the ones that do not are **incidence_with_history, recurrence, returns_at_all**. **Recurrence, the one outcome most plausibly owned by a medical staff, is not among them.** Everything that persists is about exposure and how a club uses injured reserve and times a return, so the board below separates clubs by **availability-management policy**, not by quality of medicine.

| component | split_half_r | temporal_r | temporal_p | permutation_p |
|---|---|---|---|---|
| incidence_no_history | 0.723 | 0.343 | 0.054 | 0.004 |
| incidence_with_history | 0.716 | 0.272 | 0.133 | 0.007 |
| duration | 0.807 | 0.518 | 0.002 | 0.001 |
| recurrence | 0.186 | -0.194 | 0.288 | 0.110 |
| returns_at_all | 0.607 | 0.016 | 0.932 | 0.002 |

**The curve is a forced rank, by decision: 3×A · 5×B · 8×C · 8×D · 8×F.** It orders clubs; it does not test
them. Three A's and eight F's are assigned whether or not any club is distinguishable from
another. This is a deliberate provisional choice, to be replaced by an absolute score-to-grade
mapping once several seasons of reports exist — the code already accepts one, so that switch is
a parameter rather than a rewrite.

**0 of 32 clubs are separable from the club ranked immediately below them** at 80%
confidence, and **21 of 32 are separable from the league average**. Adjacent-pair
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
| incidence_no_history | 0.723 | 0.311 | 0.200 | PASS |
| duration | 0.807 | 0.348 | 0.350 | PASS |
| recurrence | 0.186 | 0.080 | 0.450 | FAIL |
| returns_at_all | 0.607 | 0.261 | — | FAIL |

## 2. The board — 5-year window (2021–2025)

`separated_from_average` is the column that says whether a letter means anything: it marks the
clubs whose interval excludes the league mean. **No club is separable from the club immediately
below it**, so the ordering *within* a letter band — and between adjacent bands — carries no
information. `regime_changed` marks clubs that changed head coach inside the window, where the
grade averages over more than one regime.

| team | letter | score | score_sd | vs_average | separated_from_average | regime_changed | letter_3yr | score_3yr |
|---|---|---|---|---|---|---|---|---|
| LA | A | 2.239 | 0.477 | 2.223 | yes | no | A | 1.770 |
| CHI | A | 1.767 | 0.477 | 1.751 | yes | yes | A | 1.352 |
| ATL | A | 1.483 | 0.477 | 1.467 | yes | yes | C | 0.286 |
| CIN | B | 1.344 | 0.477 | 1.328 | yes | no | A | 1.274 |
| NO | B | 1.174 | 0.477 | 1.158 | yes | yes | C | 0.458 |
| CAR | B | 1.132 | 0.477 | 1.116 | yes | yes | C | 0.486 |
| IND | B | 1.024 | 0.477 | 1.008 | yes | yes | B | 0.979 |
| PHI | B | 1.004 | 0.477 | 0.988 | yes | no | B | 1.022 |
| DET | C | 0.812 | 0.477 | 0.796 | yes | no | B | 1.268 |
| JAX | C | 0.802 | 0.477 | 0.786 | yes | yes | B | 0.947 |
| PIT | C | 0.687 | 0.477 | 0.671 | yes | no | C | 0.653 |
| MIN | C | 0.683 | 0.477 | 0.667 | yes | yes | C | 0.418 |
| NYJ | C | 0.646 | 0.477 | 0.630 | yes | yes | D | -0.078 |
| BAL | C | 0.422 | 0.477 | 0.406 | no | no | C | 0.427 |
| ARI | C | 0.335 | 0.477 | 0.319 | no | yes | C | 0.706 |
| HOU | C | 0.206 | 0.477 | 0.190 | no | yes | F | -0.852 |
| TB | D | 0.169 | 0.477 | 0.153 | no | yes | D | -0.010 |
| SF | D | 0.152 | 0.477 | 0.136 | no | no | D | -0.251 |
| DEN | D | 0.127 | 0.477 | 0.111 | no | yes | B | 1.215 |
| WAS | D | 0.108 | 0.477 | 0.092 | no | yes | D | -0.145 |
| NYG | D | -0.216 | 0.477 | -0.232 | no | yes | D | -0.325 |
| LAC | D | -0.318 | 0.477 | -0.334 | no | yes | D | -0.256 |
| LV | D | -0.342 | 0.477 | -0.358 | no | yes | C | 0.438 |
| TEN | D | -0.450 | 0.477 | -0.467 | no | yes | D | 0.134 |
| DAL | F | -0.885 | 0.477 | -0.901 | yes | yes | D | -0.781 |
| BUF | F | -0.955 | 0.477 | -0.971 | yes | no | F | -0.875 |
| GB | F | -1.706 | 0.477 | -1.722 | yes | no | F | -1.442 |
| NE | F | -1.937 | 0.477 | -1.953 | yes | yes | F | -1.626 |
| MIA | F | -2.147 | 0.477 | -2.163 | yes | yes | F | -1.084 |
| SEA | F | -2.182 | 0.477 | -2.198 | yes | yes | F | -1.137 |
| KC | F | -2.322 | 0.477 | -2.338 | yes | no | F | -1.983 |
| CLE | F | -2.342 | 0.477 | -2.358 | yes | no | F | -2.369 |

**21 of 32 clubs receive the same letter on the 3-year window**, which is the
most direct stability check available on the board itself.

## 3. Position groups

Every cell is shrunk toward its own club's overall value, and cells are **never ranked**. A club
× position-group cell holds a median of four episodes over five years; at that size the shrinkage
factor does most of the work and the ordering within a club is not interpretable.

| component | cells | shrinkage_k |
|---|---|---|
| duration | 256 | 0.474 |
| incidence_no_history | 256 | 0.572 |
| recurrence | 254 | 0.615 |
| returns_at_all | 256 | 0.448 |

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
