"""Stage 6 feature tiers, and the rule for who is allowed into the sample.

Two decisions live here, and both are about what may *not* be used.

The two tiers
-------------
§2.5 established that the two college sources agree on volume and rate stats but not on
efficiency. That splits the feature set in a way that decides what the model *is*:

- :data:`PORTABLE` — computed natively and near-identically by cfbfastR and CFBD. A model on
  these can be fitted on history and pointed at this year's draft class.
- :data:`CFBFASTR_ONLY` — EPA per dropback, success rate, adjusted yards. Better features,
  2004–2021 only. A model on these is a historical instrument: it can say what breakouts looked
  like, but it cannot score a current prospect.

Stage 6 fits both and reports the gap, because the gap is the finding. If the portable model gives
up little, the project has a usable forward-looking tool; if it gives up a lot, then the signal
lives precisely in the measure that cannot be carried forward.

What is deliberately excluded
-----------------------------
**Draft position is not a feature.** It is pre-NFL in the trivial sense — it happens before a snap
— but it is the league's own opinion about the player, and a model that reads it is partly just
reporting what scouts already decided. It is used here exactly as ECR is used elsewhere in this
repo: as an **independent benchmark to be compared against, never blended in**. If a model built
only on college production cannot beat the draft, that is worth knowing plainly.

**Interception rate is excluded** even though §2.5 rates it portable across sources: cfbfastR
records interceptions only in 2005 and 2014+ (§2.6), so it is missing for most pre-2014 careers
and imputing it would fabricate the exact quantity the label is sensitive to.

**Anything ``career_*``** is avoided in favour of ``final_*``. Careers straddling 2004 are clipped
by the coverage floor — Rodgers reads as a one-season starter — so career totals mean different
things for different players. A final season is a final season regardless.
"""

from __future__ import annotations

# Style and production from the last college season, plus career shape. Every column here is
# computable from CFBD, so this model can score a prospect whose college career ended last autumn.
PORTABLE = (
    "final_completion_pct",
    "final_yards_per_attempt",
    "final_yards_per_completion",
    "final_td_rate",
    "final_rush_share",
    "final_rush_td_share",
    "final_attempts",
    "n_college_seasons",
    "transferred",
    "archetype_stability",
    "archetype_changed",
)

# Efficiency, and the career-shape features derived from it. cfbfastR 2004-2021 only.
CFBFASTR_ONLY = (
    "final_epa_per_db",
    "final_success_rate",
    "best_epa_per_db",
    "mean_epa_per_db",
    "epa_trend",
    "peaked_in_final_season",
    "breakout_season_idx",
)

# One-hot expanded at fit time; the taxonomy is a categorical, not an ordinal.
CATEGORICAL = ("archetype_name",)

# Never a feature. The league's own opinion, kept as an independent benchmark (see module docs).
BENCHMARK = ("draft_round", "draft_pick", "undrafted")

# A quarterback whose outcome is not yet settled must not be taught to the model as a negative.
# 90% of sustained breakouts happen by NFL year 5 and 95% by year 7, so five seasons of NFL
# opportunity is where "has not broken out" starts to mean "did not break out". The threshold is
# reported with a sensitivity table rather than asserted — the observed base rate is flat at ~0.18
# from three seasons to eleven, which is the evidence that this cut is not doing much work.
MIN_SETTLED_SEASONS = 5


def modelling_frame(cohort, *, min_settled_seasons=MIN_SETTLED_SEASONS):
    """The rows stage 6 is entitled to fit on, and the outcome column.

    Three filters, each for a different reason:

    - **has a college profile** — no features otherwise;
    - **outcome settled** — a 2024 entrant labelled "never broke out" is a false negative wearing
      a label, and there are enough of them to matter;
    - **not right-censored mid-window** — a trigger whose three-season confirmation window has not
      finished is genuinely unknown, not negative.
    """
    frame = cohort.to_pandas() if hasattr(cohort, "to_pandas") else cohort.copy()
    frame = frame[frame["archetype_name"].notna()]
    frame = frame[frame["seasons_elapsed"] >= min_settled_seasons]
    if "sustained_censored" in frame.columns:
        frame = frame[~frame["sustained_censored"].astype(bool)]
    frame = frame.copy()
    frame["ever_sustained"] = frame["sustained_season"].notna().astype(int)
    return frame


def feature_columns(tier: str):
    """Columns for a tier. ``portable`` is a strict subset of ``full``."""
    if tier == "portable":
        return list(PORTABLE), list(CATEGORICAL)
    if tier == "full":
        return list(PORTABLE) + list(CFBFASTR_ONLY), list(CATEGORICAL)
    raise ValueError(f"unknown tier {tier!r}")
