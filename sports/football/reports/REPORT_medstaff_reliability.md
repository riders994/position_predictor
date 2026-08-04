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
| incidence_no_history | 0.708 | [0.53, 0.85] | 0.293 | 0.104 | 0.020 | FAIL | temporal |
| incidence_with_history | 0.701 | [0.55, 0.84] | 0.192 | 0.293 | 0.033 | FAIL | temporal |
| duration | 0.827 | [0.72, 0.90] | 0.529 | 0.002 | 0.001 | PASS | — |
| recurrence | 0.414 | [0.04, 0.69] | 0.078 | 0.670 | 0.057 | FAIL | temporal, permutation |
| returns_at_all | 0.664 | [0.46, 0.82] | 0.397 | 0.024 | 0.003 | PASS | — |

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

Split-half is high everywhere (0.41–0.83):
the residuals are internally consistent, so they are not measurement noise. **Temporal is much
lower across the board.** That gap is the finding — for most components the residual is a
property of *a period*, not a carry-forward property of the club. Real, but not something you
could put on next season's board.

- **Recurrence** — the outcome most plausibly owned by a medical staff — is **weakest on every
  test**: split-half 0.414, temporal **0.078**
  (p 0.670), permutation 0.057. Whether injuries come back is
  the thing one would actually want to call medical quality, and it does not persist.
- **Duration** — split-half 0.827, temporal **0.529**
  (p 0.002) — is the strongest, and **returns-at-all** is close behind.
- **Incidence** — *least* attributable to a training room — split-half 0.708,
  temporal 0.293 (p 0.104).

### What is actually stable

The two components that pass are **duration** and **returns-at-all** — both of which are largely
*how a club uses injured reserve and times a return*. That is an operational and roster-policy
signature, and it is genuinely persistent. The component that is not stable is the medical one.

So the defensible reading of the stage-7 board is: it separates clubs by **availability
management policy**, not by quality of medicine. That is a narrower claim than "medical staff
grades" and it is the one the evidence supports.

## 4. Power — what would have been findable

Where a component or a cell fails, this is the honest headline. "Nothing this size or smaller was
findable" is a result; "no effect" is not.

Minimum detectable observed-minus-expected per club × position group, in returns:

| position_group | clubs | median_n | median_expected | median_min_detectable |
|---|---|---|---|---|
| DB | 32 | 299.000 | 50.296 | 10.588 |
| WR_TE | 32 | 284.000 | 45.383 | 10.199 |
| OL | 32 | 276.500 | 40.662 | 9.858 |
| DL | 32 | 217.000 | 33.355 | 8.805 |
| LB | 32 | 206.000 | 38.473 | 8.936 |
| RB | 32 | 120.000 | 18.389 | 6.541 |
| QB | 32 | 48.500 | 6.997 | 4.036 |
| ST | 32 | 17.500 | 2.772 | 2.227 |

## 5. What this means for the grades

Every component that fails the gate is labelled as not distinguishable from luck **at every
appearance** in stage 7 — not in a footnote. Position-group cells whose minimum detectable effect
exceeds any plausible club difference are reported as point estimates with intervals and never
ranked.
