# CFBD extension — can the two college sources be spliced?

cfbfastR play-by-play stops at **2021**, so the college layer could describe history
but not score a current prospect. CFBD covers through the present season and closes that gap. The
question this report answers is whether its numbers can be used *as if* they were the cfbfastR
ones — measured on the 2013–2021 overlap rather than assumed.

- **CFBD QB seasons:** 3511 (2013–2025, ≥50 attempts)
- **Overlap rows joined:** 1774

## Do the two sources agree?

| metric | n | corr | mean_cfbd | mean_cfbfastr | mean_diff |
|---|---|---|---|---|---|
| attempts | 1774 | 0.99 | 233.957 | 210.152 | 23.804 |
| pass_yds | 1774 | 0.994 | 1733.983 | 1691.692 | 42.29 |
| pass_td | 1774 | 0.989 | 12.44 | 12.512 | -0.072 |
| rush_att | 1774 | 0.981 | 65.566 | 49.54 | 16.026 |
| rush_share | 1774 | 0.976 | 0.223 | 0.182 | 0.041 |
| efficiency (ppa_pass vs epa/db) | 1716 | 0.735 | 0.219 | 0.049 | 0.17 |

**Volume agrees; efficiency does not.** Attempts, yards and touchdowns correlate at 0.99 — these
are the same events counted twice. Two systematic offsets are explained rather than mysterious:

- **Rushing attempts differ by ~16**, and that difference correlates 0.78 with cfbfastR's sack
  count. CFBD follows the NCAA convention of charging sacks as rushing attempts; the cfbfastR
  layer deliberately does not (see `data/college.py`). The gap *is* the convention.
- **cfbfastR totals run low** — it averages 8.8 games per QB-season, so its play-by-play has game
  gaps. For per-play *rates* this is harmless; for career *totals* it means cfbfastR understates,
  and CFBD's official season totals are the more accurate of the two where both exist.

Efficiency is the real problem. `ppa_pass` and `pass_epa_per_db` correlate only 0.735 and sit on
different scales — the level shift is larger than the target's own mean.

## Calibration: what survives the mapping

Each CFBD feature is regressed onto its cfbfastR counterpart on the overlap. `noise_ratio` is the
share of real between-player spread the mapping **fails** to reproduce; low is good.

| feature | n | slope | intercept | r2 | resid_sd | target_sd | noise_ratio |
|---|---|---|---|---|---|---|---|
| pass_epa_per_db (from ppa_pass) | 1716 | 1.0256 | -0.1752 | 0.54 | 0.1599 | 0.2357 | 0.679 |
| rush_share | 1774 | 1.01 | -0.0429 | 0.953 | 0.0259 | 0.12 | 0.216 |
| completion_pct | 1774 | 0.9393 | 0.0873 | 0.816 | 0.0297 | 0.0693 | 0.429 |
| yards_per_attempt | 1774 | 1.0264 | 0.4187 | 0.922 | 0.4092 | 1.4683 | 0.279 |
| td_rate | 1774 | 1.0337 | 0.0045 | 0.925 | 0.0066 | 0.024 | 0.273 |
| int_rate | 1585 | 1.0263 | 0.0004 | 0.936 | 0.004 | 0.016 | 0.253 |

**Portable (noise ratio ≤ 0.5):** rush_share, completion_pct, yards_per_attempt, td_rate, int_rate
**Not portable:** pass_epa_per_db (from ppa_pass)

The `rush_share` calibration is the clean case: slope ≈ 1.01 and an intercept of −0.043 that is
simply the sack correction, recovering 95% of the variance. Efficiency is the opposite — a
calibrated EPA column would carry roughly two-thirds of the between-player spread as error, so
using it as a measured feature would be inventing precision.

## What this means for the model

The feature set splits in two, and the split is a real design constraint rather than bookkeeping:

**Portable tier** — computed natively and near-identically from both sources: completion %, yards
per attempt, TD rate, INT rate, rush share, volume. A model built on these can be **fit on history
and used to score current prospects**.

**cfbfastR-only tier** — EPA per dropback, success rate, adjusted yards per dropback. Better
features, available 2004–2021 only. A model using them is a **historical analysis
instrument**: it can explain what late breakouts looked like, but it cannot be pointed at this
year's draft class.

Stage 6 should fit both and report what the portable model gives up. If the gap is small, the
project gets a usable forward-looking tool; if it is large, that is itself the finding — the
signal lives in exactly the measure that cannot be carried forward.

Calibrated values, where used, are written as `*_est` with `efficiency_is_estimated=1` so an
estimate can never be mistaken for a measurement downstream.
