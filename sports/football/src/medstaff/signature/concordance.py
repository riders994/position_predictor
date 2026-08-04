"""Cross-position-group injury signature: does a club's excess in one body part travel?

The idea this implements: if a club's excess concentrates in **one body part** and shows up
across position groups that share nothing except the building, that implicates a common cause —
practice contact policy, tackling technique, strength and conditioning, field surface, medical
protocol. If the excess is confined to one group it is roster, scheme or luck specific to that
group.

**The statistic is a variance decomposition, not a grade.** A club x body-part x position-group
cell holds a median of four episodes over five years, so cells are never graded. Concordance
aggregates across groups instead.

**Two specifications, always reported together, because they disagree and the disagreement is
the finding.**

``rate``          episodes of part B per unit exposure, against a leave-one-team-out expectation.
                  Captures *level plus signature* — a club with more injuries overall has more of
                  everything.
``composition``   B's share of that side's own episodes, additive-log-ratio transformed off a
                  reference part. Captures *signature net of level*, and is **disclosure-robust**:
                  a club that lists everyone inflates numerator and denominator alike. This is a
                  real advantage over every level-based measure in the project.

An effect in ``rate`` but not ``composition`` is a level artifact — the club's overall burden
travelling, not a part-specific tendency. An effect in ``composition`` but not ``rate`` is a
compositional shift with no absolute excess behind it. Only agreement in both is a candidate
signature.
"""

from __future__ import annotations

import numpy as np

# The reference part for the additive log-ratio. Shares live on a simplex, so they cannot all be
# compared independently; ALR fixes one part as the denominator. Soft-tissue-lower is the choice
# because it is large (so the ratio is stable) and is deliberately *not* focal — a common cause
# acting on it would be a different mechanism from the five being tested.
ALR_REFERENCE = "soft_tissue_lower"

# Groups that are not a real body part, or that carry no mechanism worth testing.
EXCLUDED_GROUPS = ("unknown", "other", "non_injury", "illness")

MIN_EPISODES_PER_CELL = 15


def club_side_counts(episodes, *, side_col="side", team_col="team",
                     group_col="body_group"):
    """Episode counts per club x side x body part, dropping unlabelled spells.

    The ``unknown`` group — spells that opened on a bare reserve week and never picked up a
    report row — is **excluded**, not pooled. It is 21.6% of episodes and is concentrated in long
    absences, so pooling it would swamp the parts being compared.
    """
    import polars as pl

    return (
        episodes.filter(~pl.col(group_col).is_in(list(EXCLUDED_GROUPS)))
        .group_by([team_col, side_col, group_col]).len()
        .rename({"len": "episodes"})
    )


def composition_alr(counts, *, focal, team_col="team", side_col="side",
                    group_col="body_group", reference=ALR_REFERENCE, alpha=0.5):
    """Additive log-ratio of each focal part's share, per club x side.

    ``alpha`` is a Haldane-Anscombe style continuity correction: cells hold single-digit counts,
    and a zero would otherwise send the log to negative infinity.
    """
    import polars as pl

    totals = counts.group_by([team_col, side_col]).agg(
        pl.col("episodes").sum().alias("total"))
    wide = counts.pivot(values="episodes", index=[team_col, side_col], on=group_col)
    wide = wide.join(totals, on=[team_col, side_col], how="left")

    for col in (*focal, reference):
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(0).alias(col))
    wide = wide.with_columns([pl.col(c).fill_null(0) for c in (*focal, reference)])

    out = wide.with_columns([
        ((pl.col(c) + alpha) / (pl.col(reference) + alpha)).log().alias(f"alr_{c}")
        for c in focal
    ])
    return out


def concordance(values_a, values_b):
    """Spearman + Pearson between two halves' club-level values, with n."""
    from scipy import stats

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 4:
        return {"n": int(len(a)), "spearman": float("nan"), "pearson": float("nan"),
                "p_value": float("nan")}
    sp = stats.spearmanr(a, b)
    pe = stats.pearsonr(a, b)
    return {"n": int(len(a)), "spearman": float(sp.statistic), "pearson": float(pe.statistic),
            "p_value": float(sp.pvalue)}


def composition_concordance(episodes, *, focal, split="side", team_col="team",
                            group_col="body_group", seed=17):
    """Concordance of the composition signature between two halves of each club's roster.

    ``split="side"`` is the primary test — offense against defense. Different position coaches and
    different drills, but the same building, the same S&C programme, the same medical staff and
    the same surface.
    """
    import polars as pl

    eps = episodes.filter(~pl.col(group_col).is_in(list(EXCLUDED_GROUPS)))
    if split == "side":
        eps = eps.filter(pl.col("side").is_in(["OFF", "DEF"]))
        side_col = "side"
    else:
        rng = np.random.default_rng(seed)
        halves = rng.integers(0, 2, size=eps.height)
        eps = eps.with_columns(
            pl.Series("rand_half", ["A" if h == 0 else "B" for h in halves]))
        side_col = "rand_half"

    counts = club_side_counts(eps, side_col=side_col, team_col=team_col, group_col=group_col)
    alr = composition_alr(counts, focal=focal, team_col=team_col, side_col=side_col,
                          group_col=group_col)

    sides = sorted(alr[side_col].unique().to_list())
    if len(sides) != 2:
        return {}
    a = alr.filter(pl.col(side_col) == sides[0])
    b = alr.filter(pl.col(side_col) == sides[1])
    joined = a.join(b, on=team_col, how="inner", suffix="_b")

    out = {}
    for part in focal:
        out[part] = concordance(joined[f"alr_{part}"], joined[f"alr_{part}_b"])
    return out


def rate_concordance(residuals, *, focal, team_col="team", side_col="side"):
    """Concordance of the *rate* residual between the two sides, per focal part.

    ``residuals`` is a frame of club x side x part observed-minus-expected, produced against a
    leave-one-team-out expectation so a club never informs its own baseline.
    """
    import polars as pl

    out = {}
    for part in focal:
        sub = residuals.filter(pl.col("body_group") == part)
        a = sub.filter(pl.col(side_col) == "OFF").select([team_col, "z"])
        b = sub.filter(pl.col(side_col) == "DEF").select([team_col, "z"])
        joined = a.join(b, on=team_col, how="inner", suffix="_def")
        out[part] = concordance(joined["z"], joined["z_def"])
    return out


def leave_one_group_out(episodes, *, focal, group_col="body_group",
                        position_col="position_group"):
    """Composition concordance recomputed with each position group dropped in turn.

    Guards against a single group — usually the offensive line, which is large and injury-dense —
    carrying a result that then reads as a club-wide signature.
    """
    import polars as pl

    groups = sorted(episodes[position_col].drop_nulls().unique().to_list())
    rows = []
    for dropped in groups:
        sub = episodes.filter(pl.col(position_col) != dropped)
        res = composition_concordance(sub, focal=focal, group_col=group_col)
        for part, stats_ in res.items():
            rows.append({"dropped_group": dropped, "body_group": part,
                         "spearman": stats_["spearman"], "n": stats_["n"]})
    return rows


def variance_decomposition(episodes, *, focal, team_col="team",
                           position_col="position_group", group_col="body_group"):
    """Club main effect against club x position-group interaction, per focal part.

    A large main effect means the part's excess is common across the club's position groups —
    which is what a shared cause looks like. A large interaction means it is specific to certain
    groups, which points at roster or scheme instead.
    """
    import polars as pl

    eps = episodes.filter(~pl.col(group_col).is_in(list(EXCLUDED_GROUPS)))
    rows = []
    for part in focal:
        cells = (
            eps.group_by([team_col, position_col])
            .agg((pl.col(group_col) == part).sum().alias("k"), pl.len().alias("n"))
            .filter(pl.col("n") >= MIN_EPISODES_PER_CELL)
        )
        if cells.height < 8:
            rows.append({"body_group": part, "cells": cells.height,
                         "club_var": float("nan"), "interaction_var": float("nan"),
                         "share_common": float("nan")})
            continue
        pdf = cells.to_pandas()
        pdf["share"] = pdf["k"] / pdf["n"]
        club_means = pdf.groupby(team_col)["share"].mean()
        grand = pdf["share"].mean()
        club_var = float(((club_means - grand) ** 2).mean())
        resid = pdf["share"] - pdf[team_col].map(club_means)
        inter_var = float((resid ** 2).mean())
        total = club_var + inter_var
        rows.append({"body_group": part, "cells": int(cells.height),
                     "club_var": club_var, "interaction_var": inter_var,
                     "share_common": float(club_var / total) if total > 0 else float("nan")})
    return rows


def coach_follows(episodes, coaches, *, focal, team_col="team", group_col="body_group",
                  min_clubs=2, min_episodes=40):
    """Does a club's signature travel with its head coach to another franchise?

    The only design element here that breaks the shared-roster confound. Offense and defense are
    not fully independent — an old or badly-conditioned roster is old on both sides — but a coach
    who carries an elevated share of one body part across *different* franchises, with different
    rosters and different buildings, is much harder to explain that way.

    Expect this to be power-bounded rather than decisive: few coaches run two clubs inside a
    five-season window.
    """
    import polars as pl

    eps = episodes.filter(~pl.col(group_col).is_in(list(EXCLUDED_GROUPS)))
    joined = eps.join(coaches, on=["season", team_col], how="inner")
    per = (
        joined.group_by(["coach", team_col])
        .agg(pl.len().alias("episodes"),
             *[(pl.col(group_col) == p).mean().alias(f"share_{p}") for p in focal])
        .filter(pl.col("episodes") >= min_episodes)
    )
    multi = (
        per.group_by("coach").agg(pl.col(team_col).n_unique().alias("clubs"))
        .filter(pl.col("clubs") >= min_clubs)
    )
    per = per.join(multi.select("coach"), on="coach", how="inner")
    return per, multi.height


def head_coaches(schedules):
    """Per (season, team) head coach — the club's most-frequent coach that season."""
    import polars as pl

    reg = schedules.filter(pl.col("game_type") == "REG")
    sides = [
        reg.select(["season", pl.col(f"{s}_team").alias("team"),
                    pl.col(f"{s}_coach").alias("coach")])
        for s in ("home", "away")
    ]
    stacked = pl.concat(sides).drop_nulls("coach")
    return (
        stacked.group_by(["season", "team", "coach"]).len()
        .sort("len", descending=True)
        .unique(subset=["season", "team"], keep="first")
        .drop("len")
    )


__all__ = [
    "ALR_REFERENCE", "EXCLUDED_GROUPS", "MIN_EPISODES_PER_CELL",
    "club_side_counts", "coach_follows", "composition_alr", "composition_concordance",
    "concordance", "head_coaches", "leave_one_group_out", "rate_concordance",
    "variance_decomposition",
]
