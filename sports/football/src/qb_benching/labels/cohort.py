"""One row per opening starter per season, and the label.

The modelling row is a club's Week-1 quarterback, and the label is whether he was benched during
that season. Everything here is a count over :func:`displacement.classify_displacement`'s panel.
"""

from __future__ import annotations

#: How many benched weeks make a benching. Three, for two reasons. One start missed is noise —
#: a coach can sit a quarterback for a week and hand him back the job — and the one-week bar
#: catches 31.8% of all openers, which is a measure of week-to-week churn rather than of losing
#: a job. Three is also where the rate stops moving quickly with the bar (17.0% at two, 14.6% at
#: three, 12.0% at four), so the label is not balanced on a knife edge. ``SENSITIVITY_BARS`` is
#: reported beside the headline every time so the choice stays visible.
BENCHED_WEEKS = 3

#: Bars re-reported alongside the primary one. If a result only exists at ``BENCHED_WEEKS`` it
#: is not a result.
SENSITIVITY_BARS = (2, 3, 4, 6)

#: The season the weekly injury report begins. Before this the three-way split cannot be made,
#: so the benching model's sample starts here even though ``schedules`` reaches 1999. The
#: undifferentiated "displaced" arm can still use the full history.
FIRST_REPORT_SEASON = 2009

#: The boundary the regime check is computed across. It is where ``rosters_weekly.status`` stops
#: being a season-final stamp and becomes a genuine weekly value; see
#: :mod:`qb_benching.labels.displacement`. The label reads different evidence either side of it,
#: so the rate had better not move.
from .displacement import ROSTER_REGIME_SEASON  # noqa: E402


def build_cohort(classified, *, benched_weeks: int = BENCHED_WEEKS):
    """One row per ``(season, team, opener)``, with counts and labels.

    The counts close by construction: ``held + benched + injured + gone == games_after_opener``.
    :func:`reconciles` asserts it, and the stage-1 report states it.
    """
    import polars as pl

    counts = classified.group_by(["season", "team", "opener_id", "opener_name"]).agg([
        pl.len().alias("games_after_opener"),
        (pl.col("outcome") == "held").sum().alias("n_held"),
        (pl.col("outcome") == "benched").sum().alias("n_benched"),
        (pl.col("outcome") == "injured").sum().alias("n_injured"),
        (pl.col("outcome") == "gone").sum().alias("n_gone"),
        pl.col("week").filter(pl.col("outcome") == "benched").min().alias("first_benched_week"),
        pl.col("week").filter(pl.col("outcome") != "held").min().alias("first_displaced_week"),
    ])
    counts = counts.with_columns(
        (pl.col("n_benched") + pl.col("n_injured") + pl.col("n_gone")).alias("n_displaced"),
    )
    return counts.with_columns([
        (pl.col("n_benched") >= benched_weeks).alias("benched"),
        (pl.col("n_injured") >= benched_weeks).alias("injured_out"),
        (pl.col("n_displaced") >= benched_weeks).alias("displaced"),
    ]).sort(["season", "team"])


def reconciles(cohort) -> bool:
    """Every displaced week is classified exactly once."""
    import polars as pl

    total = cohort.select(
        (pl.col("n_held") + pl.col("n_benched") + pl.col("n_injured") + pl.col("n_gone"))
        == pl.col("games_after_opener")
    )
    return bool(total.to_series().all())


def label_rate_by_season(cohort, *, label: str = "benched"):
    """Per-season base rate of a label — the first table in the stage-1 report."""
    import polars as pl

    return (
        cohort.group_by("season")
        .agg([
            pl.len().alias("openers"),
            pl.col(label).sum().alias("positives"),
        ])
        .with_columns((pl.col("positives") / pl.col("openers")).alias("rate"))
        .sort("season")
    )


def regime_check(cohort, *, label: str = "benched", boundary: int = ROSTER_REGIME_SEASON):
    """Base rate either side of the 2021 roster-status break.

    The label is built to be immune to that break — it keys on the injury report, not on the
    ACT/INA split that changes there. This recomputes the comparison so a data refresh that
    reintroduced the dependency would show up as a moved rate rather than as a quiet bias.
    """
    import polars as pl

    return (
        cohort.with_columns(
            pl.when(pl.col("season") < boundary)
            .then(pl.lit(f"pre-{boundary}"))
            .otherwise(pl.lit(f"{boundary}+"))
            .alias("regime")
        )
        .group_by("regime")
        .agg([pl.len().alias("openers"), pl.col(label).sum().alias("positives")])
        .with_columns((pl.col("positives") / pl.col("openers")).alias("rate"))
        .sort("regime", descending=True)
    )


def sensitivity(classified, *, bars=SENSITIVITY_BARS):
    """Positive count and rate at each candidate bar, so the chosen one is never alone."""
    import polars as pl

    rows = []
    for bar in bars:
        c = build_cohort(classified, benched_weeks=bar)
        rows.append({
            "bar": bar,
            "openers": c.height,
            "benched": int(c["benched"].sum()),
            "rate": float(c["benched"].mean()),
        })
    return pl.DataFrame(rows)
