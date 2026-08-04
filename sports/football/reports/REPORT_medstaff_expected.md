# Medical-staff grades — Stage 4: expectation models

Every grade in this project is observed minus expected. This stage builds the expected side and
reports the residuals. **Nothing here is a grade** — stage 6 decides whether these residuals are
distinguishable from luck, and only then does stage 7 turn them into letters.

## 1. Why this is a penalised logistic hazard and not a boosted tree

The estimand is a **residual**. A flexible learner eats residual variance: a boosted model
adjusting on roster-composition proxies absorbs precisely the between-club variation being
measured, and the grades would then read as "how well does a GBM predict this club", not "how
did this club do". Penalised logistic regression with splines on age and week is the more
complex thing that is *less* dangerous here.

Censoring is pervasive — seasons end, players are traded and released — and discrete-time hazard
handles it exactly by construction. Duration and recurrence are person-period expansions, so a
spell that ran out of season contributes the weeks it actually had rather than being dropped
(which biases towards fast returns) or counted as a return (which is a lie).

## 2. Why leave-one-team-out

Club T is scored by a model fit on the other 31. A random fold contains some of T's own rows, so
the model learns T's elevated rate and predicts it back, shrinking the residual toward zero.
There is a test that injects a club effect of known size and asserts LOTO recovers it while
random k-fold shrinks it.

**Club identity is never a feature**, as a column or a proxy — there is a guard that raises if it
reaches the design matrix.

## 3. The components

| component | rows | base_rate | permutation_p | intraclass | signal_sd | min_detectable |
|---|---|---|---|---|---|---|
| incidence_no_history | 144552 | 0.100 | 0.024 | 0.867 | 48.101 | 36.882 |
| incidence_with_history | 144552 | 0.100 | 0.039 | 0.859 | 46.357 | 36.708 |
| duration | 48267 | 0.163 | 0.000 | 0.882 | 32.712 | 23.188 |
| recurrence | 37859 | 0.018 | 0.057 | 0.440 | 4.007 | 8.779 |
| returns_at_all | 14420 | 0.545 | 0.004 | 0.625 | 10.864 | 16.523 |

`permutation_p` is the **primary inference**: does club identity explain *any* variance in the
residual? One test rather than 32, so there is no multiplicity problem. The null reassigns whole
players to clubs, preserving each player's own correlated run of weeks.

`intraclass` is the share of between-club spread that survives after subtracting the sampling
variance the model already explains. **It is not the share attributable to a medical staff**, and
reading it that way is the single most likely misuse of this table. It says the spread is not
Poisson noise; the surviving variance can still be roster construction, disclosure behaviour,
scheme or stadium. The model here is deliberately restrained (§1), which means composition the
covariates do not capture stays *in* the residual rather than being absorbed — that is the right
trade for measuring a club effect, but it inflates this number relative to anything causal.

`min_detectable` is what a typical club would have had to differ by, in the outcome's own units,
to be distinguishable.

### The pattern worth reading first

**The components differ in how attributable they are, and they run in the opposite direction to
the evidence.** Recurrence is the outcome most plausibly owned by a medical staff — it is
downstream of the return-to-play decision and least contaminated by luck — and it is the **only
component that does not clear its permutation test**
(p 0.057, intraclass 0.440).
Incidence is the component *least* attributable to a training room — conditioning, scheme,
surface and luck — and it shows the strongest club separation.

That ordering is a caution, not a finding. It is consistent with clubs differing mainly in how
much football their rosters are exposed to and how they use injured reserve, rather than in
medicine. Stage 6 tests whether any of it is stable year over year, which is the question that
decides whether the stage-7 letters mean anything.

## 4. The bound that adjustment forces

Incidence is fitted **twice**, and both are reported, because prior injury history is not a
neutral covariate. A club with a poor availability system *manufactures* players who look
fragile, so history is partly its own output — adjusting for it adjusts away part of the effect
being measured.

- **without history** — intraclass **0.867** — the *upper* bound on the club effect
- **with history** — intraclass **0.859** — the *lower* bound

The truth is inside that interval and this data cannot say where. Reporting either number alone
would be a choice dressed as a measurement.

## 5. Calibration

Predicted against observed by decile of predicted risk. A model that is not on the diagonal here
produces residuals that are model error rather than club effect.

### Incidence (no history)

| decile | n | predicted | observed |
|---|---|---|---|
| 1 | 14456 | 0.017 | 0.021 |
| 2 | 14456 | 0.027 | 0.031 |
| 3 | 14455 | 0.035 | 0.043 |
| 4 | 14455 | 0.044 | 0.051 |
| 5 | 14455 | 0.054 | 0.066 |
| 6 | 14455 | 0.066 | 0.074 |
| 7 | 14455 | 0.082 | 0.081 |
| 8 | 14455 | 0.108 | 0.085 |
| 9 | 14455 | 0.196 | 0.135 |
| 10 | 14455 | 0.368 | 0.411 |

### Recurrence

| decile | n | predicted | observed |
|---|---|---|---|
| 1 | 3786 | 0.006 | 0.009 |
| 2 | 3786 | 0.009 | 0.008 |
| 3 | 3786 | 0.011 | 0.012 |
| 4 | 3786 | 0.014 | 0.012 |
| 5 | 3786 | 0.016 | 0.021 |
| 6 | 3786 | 0.019 | 0.020 |
| 7 | 3786 | 0.021 | 0.025 |
| 8 | 3786 | 0.023 | 0.024 |
| 9 | 3786 | 0.026 | 0.026 |
| 10 | 3785 | 0.031 | 0.019 |

## 6. Club residuals

Most extreme five in each direction. **These are not grades** — the p-values are uncorrected,
there are 32 of them per component, and roughly 1.6 clubs per component clear 0.05 by chance
alone. `p_used` is the more conservative of the independent-Bernoulli and player-block nulls,
because a fragile player is fragile all season and his weeks are correlated.

### Recurrence — the most attributable component

| team | n | observed | expected | diff | z_indep | p_used |
|---|---|---|---|---|---|---|
| LV | 1323 | 12 | 23.137 | -11.137 | -2.340 | 0.030 |
| HOU | 1252 | 12 | 21.199 | -9.199 | -2.019 | 0.066 |
| GB | 1126 | 12 | 21.196 | -9.196 | -2.020 | 0.066 |
| ARI | 1245 | 15 | 23.696 | -8.696 | -1.807 | 0.122 |
| SF | 1319 | 16 | 23.446 | -7.446 | -1.554 | 0.178 |
| BAL | 1136 | 28 | 20.744 | 7.256 | 1.610 | 0.132 |
| TB | 1189 | 28 | 19.741 | 8.259 | 1.878 | 0.108 |
| TEN | 1328 | 33 | 23.875 | 9.125 | 1.888 | 0.078 |
| LAC | 1154 | 31 | 21.464 | 9.536 | 2.081 | 0.070 |
| SEA | 1528 | 39 | 25.979 | 13.021 | 2.581 | 0.016 |

### Duration

| team | n | observed | expected | diff | z_indep | p_used |
|---|---|---|---|---|---|---|
| KC | 1387 | 250 | 345.150 | -95.150 | -7.488 | 0.008 |
| BUF | 1314 | 219 | 279.952 | -60.952 | -4.893 | 0.096 |
| MIA | 1844 | 298 | 348.332 | -50.332 | -3.627 | 0.170 |
| CLE | 1962 | 281 | 329.045 | -48.045 | -3.457 | 0.152 |
| NE | 1839 | 251 | 297.393 | -46.393 | -3.382 | 0.154 |
| CIN | 1206 | 245 | 206.660 | 38.340 | 3.556 | 0.086 |
| NYG | 1891 | 306 | 267.570 | 38.430 | 2.918 | 0.098 |
| ARI | 1700 | 254 | 214.998 | 39.002 | 3.315 | 0.058 |
| CAR | 1485 | 293 | 248.970 | 44.030 | 3.685 | 0.082 |
| CHI | 1253 | 272 | 219.359 | 52.641 | 4.707 | 0.014 |

## 7. The censoring guard

`returns_at_all` is fitted as its own outcome rather than folded into duration. Without it a club
scores well on return-to-play simply by having its unrecovered cases quietly censored — the
failure mode is gameable, so it is measured directly.

Base rate: **0.545** of episodes resolved within the
season; permutation p **0.004**.
