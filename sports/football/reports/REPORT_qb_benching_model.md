# QB Benching — Stage 3: the model

_2009–2025 · 544 opening starters · 90 benched (16.5%) · 32 preseason features_

## Read the stratified number, not the pooled one

Benching rate runs from 47% for an opener with under 100 prior attempts to 12% for one over 300. A model that learns only *"this man was never really the starter"* posts a strong pooled AUC and tells a reader nothing the depth chart had not already told them. `qb_breakout` documented the same trap, where draft position scored 0.890 pooled and 0.496 within a band. So the question this report is built to answer is whether the model separates benchings **among similarly-established starters**.

| volume_band | n | positives | rate |
|---|---|---|---|
| none | 42 | 4 | 0.095 |
| <100 | 38 | 18 | 0.474 |
| 100-299 | 68 | 22 | 0.324 |
| 300+ | 396 | 46 | 0.116 |

## Headline

Pooled cross-validated AUC **0.788** (±0.012 across repeats), against a label-permutation null of 0.497 ± 0.047 — p = 0.000 over 200 shuffles. Brier 0.169. Of the 5 riskiest openers flagged in each season, **45.9%** were benched, against a 16.5% base rate.

**Within prior-volume bands — the number that decides whether this is a finding:**

| volume_band | n | positives | model_auc |
|---|---|---|---|
| none | 42 | 4 | 0.513 |
| <100 | 38 | 18 | 0.556 |
| 100-299 | 68 | 22 | 0.659 |
| 300+ | 396 | 46 | 0.802 |

## The control: the same features cannot predict injury

Fitting the identical feature set on the identical rows against `injured_out` (142 positives) reaches AUC **0.528** against a null of 0.505 — p = 0.300. It is at chance in every band:

| volume_band | n | positives | model_auc |
|---|---|---|---|
| none | 42 | 12 | 0.408 |
| <100 | 38 | 11 | 0.572 |
| 100-299 | 68 | 22 | 0.474 |
| 300+ | 396 | 97 | 0.543 |

This is the result that makes the benching number worth believing. PROMPT_LOG entries 096-097 already established that injury is not predictable from this repo's data; if these features had 'predicted' it too, the benching AUC would be an artefact of the evaluation rather than a fact about benching. They do not.

## Against the baselines a reader already has

| baseline | auc | precision_at_k |
|---|---|---|
| prior-season attempts (the 'was he really the starter' axis) | 0.701 | 0.294 |
| prior-season EPA per dropback | 0.661 | 0.282 |
| draft capital (later pick = more risk) | 0.588 | 0.294 |
| lost the job last season | 0.589 | 0.294 |

The best single column is **prior-season attempts (the 'was he really the starter' axis)** at AUC 0.701. The model beats it pooled.

## Deployment simulation

Trained on 288 openers through 2017 (43 positives), tested on the 256 after (47 positives): AUC **0.686**, top-5 precision 32.5%. Where this disagrees with the random-fold number, this is the one to believe — random folds let 2024 inform a prediction about 2009.

## Does a more flexible learner help?

Gradient boosting reaches AUC 0.780 against the linear model's 0.788. It is fitted only so this line can be written.

## What the model reads

Coefficients from a fit on everything, with how often each sign survives a bootstrap. Read the direction and the stability, not the magnitude.

| feature | coef | sign_stability |
|---|---|---|
| has_prior_season | 0.544 | 1.000 |
| is_rookie | -0.500 | 1.000 |
| rookie_qb_drafted | 0.433 | 1.000 |
| rookie_qb_pick | -0.383 | 1.000 |
| years_exp | -0.378 | 1.000 |
| prior_ypa | -0.366 | 1.000 |
| prior_int_rate | -0.318 | 1.000 |
| displaced_last_season | 0.310 | 1.000 |
| prior_attempts | -0.307 | 1.000 |
| prior_point_diff | -0.267 | 0.930 |
| prior_comp_pct | 0.261 | 0.880 |
| was_drafted | -0.254 | 0.980 |
| room_size | 0.249 | 0.980 |
| prior_dropbacks | -0.231 | 1.000 |

⚠️ **Several of these are collinear and must not be read as effects.** `has_prior_season` and `is_rookie` are near-mirrors of each other; `prior_attempts`, `prior_dropbacks` and `prior_starts` all measure the same thing; and the rate columns carry suppression signs as a result — `prior_int_rate` reads negative and `prior_comp_pct` positive, neither of which is a claim that throwing interceptions keeps a job. A stable sign here means the fit is reproducible, not that the mechanism is identified. The C=0.1 shrink is what keeps them from flipping fold to fold.

## Reproduce

```bash
uv run python sports/football/scripts/qb_benching_features.py
uv run python sports/football/scripts/qb_benching_model.py
```
