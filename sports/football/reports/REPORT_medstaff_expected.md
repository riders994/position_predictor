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
| incidence_no_history | 150059 | 0.082 | 0.002 | 0.856 | 43.550 | 34.994 |
| incidence_with_history | 150059 | 0.082 | 0.005 | 0.846 | 41.763 | 34.711 |
| duration | 51455 | 0.168 | 0.000 | 0.849 | 28.342 | 23.153 |
| recurrence | 42192 | 0.017 | 0.105 | 0.260 | 2.750 | 9.044 |
| returns_at_all | 12280 | 0.704 | 0.001 | 0.630 | 8.256 | 12.212 |

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
(p 0.105, intraclass 0.260).
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

- **without history** — intraclass **0.856** — the *upper* bound on the club effect
- **with history** — intraclass **0.846** — the *lower* bound

The truth is inside that interval and this data cannot say where. Reporting either number alone
would be a choice dressed as a measurement.

## 5. Calibration

Predicted against observed by decile of predicted risk. A model that is not on the diagonal here
produces residuals that are model error rather than club effect.

### Incidence (no history)

| decile | n | predicted | observed |
|---|---|---|---|
| 1 | 15006 | 0.006 | 0.009 |
| 2 | 15006 | 0.021 | 0.024 |
| 3 | 15006 | 0.030 | 0.033 |
| 4 | 15006 | 0.038 | 0.048 |
| 5 | 15006 | 0.048 | 0.053 |
| 6 | 15006 | 0.060 | 0.068 |
| 7 | 15006 | 0.074 | 0.075 |
| 8 | 15006 | 0.095 | 0.078 |
| 9 | 15006 | 0.145 | 0.101 |
| 10 | 15005 | 0.302 | 0.329 |

### Recurrence

| decile | n | predicted | observed |
|---|---|---|---|
| 1 | 4220 | 0.006 | 0.009 |
| 2 | 4220 | 0.009 | 0.008 |
| 3 | 4219 | 0.011 | 0.014 |
| 4 | 4219 | 0.013 | 0.016 |
| 5 | 4219 | 0.015 | 0.017 |
| 6 | 4219 | 0.016 | 0.018 |
| 7 | 4219 | 0.019 | 0.014 |
| 8 | 4219 | 0.021 | 0.021 |
| 9 | 4219 | 0.025 | 0.023 |
| 10 | 4219 | 0.032 | 0.028 |

## 6. Club residuals

Most extreme five in each direction. **These are not grades** — the p-values are uncorrected,
there are 32 of them per component, and roughly 1.6 clubs per component clear 0.05 by chance
alone. `p_used` is the more conservative of the independent-Bernoulli and player-block nulls,
because a fragile player is fragile all season and his weeks are correlated.

### Recurrence — the most attributable component

| team | n | observed | expected | diff | z_indep | p_used |
|---|---|---|---|---|---|---|
| LV | 1495 | 17 | 25.180 | -8.180 | -1.647 | 0.134 |
| KC | 1339 | 14 | 21.128 | -7.128 | -1.566 | 0.150 |
| GB | 1254 | 15 | 22.124 | -7.124 | -1.531 | 0.184 |
| HOU | 1398 | 15 | 21.880 | -6.880 | -1.485 | 0.168 |
| ARI | 1385 | 18 | 24.595 | -6.595 | -1.344 | 0.210 |
| PHI | 1354 | 26 | 20.982 | 5.018 | 1.106 | 0.373 |
| LAC | 1305 | 30 | 23.019 | 6.981 | 1.470 | 0.218 |
| NYJ | 1245 | 28 | 20.893 | 7.107 | 1.571 | 0.152 |
| CLE | 1593 | 33 | 25.103 | 7.897 | 1.592 | 0.178 |
| SEA | 1749 | 43 | 28.468 | 14.532 | 2.751 | 0.012 |

### Duration

| team | n | observed | expected | diff | z_indep | p_used |
|---|---|---|---|---|---|---|
| KC | 1484 | 275 | 354.052 | -79.052 | -6.205 | 0.032 |
| BUF | 1404 | 247 | 310.896 | -63.896 | -5.153 | 0.058 |
| MIA | 1973 | 331 | 376.724 | -45.724 | -3.284 | 0.188 |
| NE | 1955 | 268 | 307.688 | -39.688 | -2.968 | 0.218 |
| CLE | 2101 | 322 | 360.963 | -38.963 | -2.830 | 0.299 |
| LA | 1202 | 219 | 187.660 | 31.340 | 3.101 | 0.180 |
| ARI | 1817 | 281 | 246.996 | 34.004 | 2.885 | 0.102 |
| NYG | 2011 | 336 | 295.778 | 40.222 | 3.055 | 0.146 |
| CHI | 1319 | 283 | 239.699 | 43.301 | 3.830 | 0.056 |
| CAR | 1581 | 317 | 268.889 | 48.111 | 4.029 | 0.052 |

## 7. The censoring guard

`returns_at_all` is fitted as its own outcome rather than folded into duration. Without it a club
scores well on return-to-play simply by having its unrecovered cases quietly censored — the
failure mode is gameable, so it is measured directly.

Base rate: **0.704** of episodes resolved within the
season; permutation p **0.001**.
