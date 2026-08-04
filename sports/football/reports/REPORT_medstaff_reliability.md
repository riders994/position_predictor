# Medical-staff grades — Stage 6: reliability and power

**Read this before any grade.** Stage 4 established that club residuals are larger than sampling
noise. That is a weaker claim than it sounds: a residual can be large and still be a one-off, and
a leaderboard built on one-offs is a leaderboard of luck. The question that decides whether the
grades mean anything is whether a club's residual in one part of the data predicts its residual
in another.

## 1. The verdict

**2 of 5 components pass the preregistered gate**
(split-half ≥ 0.3, temporal ≥ 0.3, permutation p < 0.05).

| component | split_half_r | split_half_95CI | temporal_r | temporal_p | permutation_p | gate | failed_on |
|---|---|---|---|---|---|---|---|
| incidence_no_history | 0.723 | [0.56, 0.85] | 0.343 | 0.054 | 0.004 | PASS | — |
| incidence_with_history | 0.716 | [0.56, 0.84] | 0.272 | 0.133 | 0.007 | FAIL | temporal |
| duration | 0.807 | [0.70, 0.90] | 0.518 | 0.002 | 0.001 | PASS | — |
| recurrence | 0.186 | [-0.28, 0.56] | -0.194 | 0.288 | 0.110 | FAIL | split_half, temporal, permutation |
| returns_at_all | 0.607 | [0.39, 0.76] | 0.016 | 0.932 | 0.002 | FAIL | temporal |

The gate **does not suppress anything** — grades publish either way, which was a deliberate
choice. It exists to *label* each component at every appearance. A number that fails here is
still printed; it is printed marked.

## 2. What the two tests mean, and why both

**Split-half** splits each club's *players* — never its rows — into halves and correlates the two
residuals, corrected by Spearman-Brown. Splitting rows would leak: a fragile player's weeks would
land on both sides and manufacture agreement out of one man's hamstring.

**Temporal** grades 2021–2023 and asks whether that predicts 2024–2025. This is the harder and
more honest test, and the one a reader actually cares about, because a grade is only useful if it
says something about the club going forward.

**Split-half can pass while temporal fails, and that gap is itself a finding**: it means the
residual is a property of a period rather than of a club — real, but not a thing you can carry
forward into next season.

## 3. The gap between the two tests is the headline

Split-half is high everywhere (0.19–0.81):
the residuals are internally consistent, so they are not measurement noise. **Temporal is much
lower across the board.** That gap is the finding — for most components the residual is a
property of *a period*, not a carry-forward property of the club. Real, but not something you
could put on next season's board.

- **Recurrence** — the outcome most plausibly owned by a medical staff — is **weakest on every
  test**: split-half 0.186, temporal **-0.194**
  (p 0.288), permutation 0.110. Whether injuries come back is
  the thing one would actually want to call medical quality, and it does not persist.
- **Duration** — split-half 0.807, temporal **0.518**
  (p 0.002).
- **Strongest on the temporal test: duration**; weakest: **recurrence**.
- **Incidence** — *least* attributable to a training room — split-half 0.723,
  temporal 0.343 (p 0.054).

### What is actually stable

The component(s) that pass: **incidence_no_history, duration**. What does *not* pass: **incidence_with_history, recurrence, returns_at_all**.

**Recurrence, the one outcome most plausibly owned by a medical staff, is not among them.** Everything that persists is about *exposure and how a club uses injured reserve and times a return* — an operational and roster-policy signature. So the defensible reading of the stage-7 board is that it separates clubs by **availability-management policy**, not by quality of medicine. That is a narrower claim than "medical staff grades", and it is the one the evidence supports.

## 4. Power — what would have been findable

Where a component or a cell fails, this is the honest headline. "Nothing this size or smaller was
findable" is a result; "no effect" is not.

Minimum detectable observed-minus-expected per club × position group, in returns:

| position_group | clubs | median_n | median_expected | median_min_detectable |
|---|---|---|---|---|
| DB | 32 | 320.000 | 56.830 | 10.668 |
| WR_TE | 32 | 304.000 | 51.176 | 10.153 |
| OL | 32 | 294.500 | 43.398 | 9.953 |
| DL | 32 | 230.500 | 36.313 | 8.675 |
| LB | 32 | 220.500 | 42.371 | 8.947 |
| RB | 32 | 133.000 | 20.550 | 6.589 |
| QB | 32 | 52.500 | 7.868 | 4.068 |
| ST | 32 | 18.000 | 2.720 | 2.237 |

## 5. What this means for the grades

Every component that fails the gate is labelled as not distinguishable from luck **at every
appearance** in stage 7 — not in a footnote. Position-group cells whose minimum detectable effect
exceeds any plausible club difference are reported as point estimates with intervals and never
ranked.
