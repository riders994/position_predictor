"""Drafting situation: which team took him, and who was running it.

This layer answers a question the project has had open since the first prompt — "player archetypes,
**drafting situations**, and other factors" — but it has to answer it under a hard arithmetic
limit, and the limit shapes everything here.

Why raw team rates are not reportable
-------------------------------------
177 quarterbacks with a settled outcome, 140 of them drafted, spread over 32 franchises: a median
of **4 quarterbacks and 1 breakout per team**. Seven teams have zero breakouts and one has three.
A table of team breakout rates would show a 0%–60% spread and every bit of it would be noise. So
this module never reports a raw rate.

What it reports instead
-----------------------
**Observed minus expected, where expected comes from draft capital.** The draft is far and away
the dominant predictor of this outcome (§4.4: AUC 0.890, and largely because it allocates the
snaps a breakout requires). Teams differ enormously in the picks they spend on quarterbacks, so
comparing raw rates mostly compares draft position. Giving each quarterback an expected breakout
probability from his pick alone, and then asking whether a team beat the sum of those
expectations, removes the confound and asks the question actually worth asking: **given the
capital they spent, which regimes got more out of quarterbacks than the pick predicted?**

The null is simulated rather than assumed. Under "no team effect", a team's breakout count is
Poisson-binomial over its own quarterbacks' expected probabilities, which is drawn directly rather
than approximated.

**A detectable-effect calculation accompanies every result.** With four quarterbacks per team, an
effect has to be enormous to clear the noise. Stating how enormous turns "we found nothing" into
"nothing this size or smaller was findable", which is a different and more useful claim.

General managers
----------------
nflverse publishes head coaches (via schedules, 1999+) but **not general managers**, and no free
structured GM-by-team-season table exists. Head coach is used as the available regime proxy;
:func:`attach_gm` accepts a hand-supplied table for when a GM list is provided.
"""

from __future__ import annotations

RANDOM_STATE = 17
N_SIMULATIONS = 20000

# Below this many quarterbacks a regime is not evaluated at all — not shown with a wide interval,
# not shown at all. A one-quarterback "regime" invites reading a coincidence as a finding.
MIN_QBS_PER_REGIME = 3


def head_coach_by_team_season(schedules=None):
    """One head coach per (season, team), from nflverse schedules.

    A team appears as both home and away, so the two are stacked. Mid-season changes resolve to
    whoever coached the most games — the draft happens in the spring, so the season's dominant
    coach is the right attribution for a spring decision.
    """
    import polars as pl

    if schedules is None:
        import nflreadpy as nfl

        schedules = nfl.load_schedules()

    parts = []
    for side in ("home", "away"):
        parts.append(
            schedules.select(
                pl.col("season"),
                pl.col(f"{side}_team").alias("team"),
                pl.col(f"{side}_coach").alias("coach"),
            )
        )
    stacked = pl.concat(parts).drop_nulls()
    return (
        stacked.group_by(["season", "team", "coach"])
        .agg(games=pl.len())
        .sort(["season", "team", "games"], descending=[False, False, True])
        .group_by(["season", "team"])
        .agg(coach=pl.col("coach").first())
    )


def attach_situation(frame, coaches=None):
    """Add the drafting franchise's head coach in the quarterback's draft year."""
    from ..labels.teams import canonical_team

    df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame.copy()
    if coaches is None:
        coaches = head_coach_by_team_season()
    coach_tbl = coaches.to_pandas() if hasattr(coaches, "to_pandas") else coaches
    coach_tbl = coach_tbl.copy()
    coach_tbl["team"] = coach_tbl["team"].map(canonical_team)

    df["_team"] = df["draft_franchise"].map(
        lambda t: canonical_team(t) if isinstance(t, str) else None)
    merged = df.merge(
        coach_tbl.rename(columns={"season": "draft_season", "coach": "draft_coach"}),
        left_on=["draft_season", "_team"], right_on=["draft_season", "team"], how="left",
    ).drop(columns=["team", "_team"])
    return merged


def attach_gm(frame, gm_table):
    """Attach general manager from a supplied ``(season, team, gm)`` table.

    nflverse does not publish general managers and no free structured source exists, so this takes
    one rather than inventing one. Team codes are canonicalised on both sides before joining, so a
    hand-typed table using PFR abbreviations still lands.
    """
    from ..labels.teams import canonical_team

    df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame.copy()
    table = gm_table.to_pandas() if hasattr(gm_table, "to_pandas") else gm_table.copy()
    table["team"] = table["team"].map(canonical_team)

    df["_team"] = df["draft_franchise"].map(
        lambda t: canonical_team(t) if isinstance(t, str) else None)
    merged = df.merge(
        table.rename(columns={"season": "draft_season"}),
        left_on=["draft_season", "_team"], right_on=["draft_season", "team"], how="left",
    ).drop(columns=["team", "_team"])
    return merged


def expected_from_draft(frame, outcome="ever_sustained", *, n_splits=5,
                        random_state=RANDOM_STATE):
    """Out-of-fold P(breakout) from draft capital alone.

    Out-of-fold rather than in-sample, because an in-sample expectation absorbs part of whatever
    team effect might exist and would understate it. Log-scaled pick, since the difference between
    picks 1 and 20 matters far more than between 200 and 220.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    pick = frame["draft_pick"].to_numpy(dtype=float)
    undrafted = np.isnan(pick)
    pick = np.where(undrafted, 300.0, pick)
    X = np.column_stack([np.log(pick), undrafted.astype(float)])
    y = frame[outcome].to_numpy()

    oof = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in cv.split(X, y):
        model = LogisticRegression(max_iter=1000).fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return oof


def regime_table(frame, group_col, expected, outcome="ever_sustained", *,
                 min_qbs=MIN_QBS_PER_REGIME, n_simulations=N_SIMULATIONS,
                 random_state=RANDOM_STATE):
    """Observed vs draft-capital-expected breakouts per regime, with a simulated null.

    ``p_two_sided`` is the share of simulated seasons in which that regime's |observed − expected|
    reached what was actually seen, drawing each quarterback independently from his own expected
    probability. It is *not* corrected for the number of regimes tested; the caller reports how
    many comparisons were made so the reader can apply the discount themselves.
    """
    import numpy as np
    import pandas as pd

    df = frame.copy()
    df["_expected"] = expected
    df = df[df[group_col].notna()]

    rng = np.random.default_rng(random_state)
    rows = []
    for name, grp in df.groupby(group_col):
        n = len(grp)
        if n < min_qbs:
            continue
        p = grp["_expected"].to_numpy()
        observed = float(grp[outcome].sum())
        exp = float(p.sum())
        draws = rng.binomial(1, np.tile(p, (n_simulations, 1))).sum(axis=1)
        rows.append({
            group_col: name,
            "qbs": n,
            "observed": int(observed),
            "expected": round(exp, 2),
            "diff": round(observed - exp, 2),
            "p_two_sided": round(float(np.mean(np.abs(draws - exp) >= abs(observed - exp))), 3),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("diff", ascending=False).reset_index(drop=True) if len(out) else out


def detectable_effect(frame, group_col, expected, *, min_qbs=MIN_QBS_PER_REGIME,
                      n_simulations=N_SIMULATIONS, random_state=RANDOM_STATE, alpha=0.05):
    """How large an over-performance a typical regime would need before it was visible.

    Answers the question a null result should always be paired with: not "did we find an effect"
    but "what is the smallest effect we could have found". Reported in extra breakouts above
    expectation for a regime of the median size.
    """
    import numpy as np

    df = frame.copy()
    df["_expected"] = expected
    df = df[df[group_col].notna()]
    sizes = df.groupby(group_col).size()
    sizes = sizes[sizes >= min_qbs]
    if not len(sizes):
        return None

    median_n = int(np.median(sizes))
    base = float(df["_expected"].mean())
    rng = np.random.default_rng(random_state)
    draws = rng.binomial(median_n, base, n_simulations)
    threshold = float(np.quantile(draws, 1 - alpha))
    return {
        "regimes_evaluated": int(len(sizes)),
        "median_qbs_per_regime": median_n,
        "expected_breakouts": round(median_n * base, 2),
        "breakouts_needed": int(threshold) + 1,
        "extra_breakouts_needed": round(int(threshold) + 1 - median_n * base, 2),
    }


def group_permutation_p(frame, group_col, expected, outcome="ever_sustained", *,
                        min_qbs=MIN_QBS_PER_REGIME, n_permutations=5000,
                        random_state=RANDOM_STATE):
    """Does the grouping explain *any* variance beyond draft capital?

    Tests the whole factor at once rather than a regime at a time, which avoids the multiple-
    comparison problem entirely: the statistic is the dispersion of observed-minus-expected across
    regimes, and the null reassigns quarterbacks to regimes at random.
    """
    import numpy as np

    df = frame.copy()
    df["_expected"] = expected
    df = df[df[group_col].notna()]
    groups = df[group_col].to_numpy()
    y = df[outcome].to_numpy(dtype=float)
    p = df["_expected"].to_numpy()

    def dispersion(labels):
        diffs = []
        for name in np.unique(labels):
            mask = labels == name
            if mask.sum() < min_qbs:
                continue
            diffs.append(y[mask].sum() - p[mask].sum())
        return float(np.var(diffs)) if diffs else 0.0

    observed = dispersion(groups)
    rng = np.random.default_rng(random_state)
    hits = sum(dispersion(rng.permutation(groups)) >= observed for _ in range(n_permutations))
    return {"dispersion": round(observed, 3),
            "p_value": round((hits + 1) / (n_permutations + 1), 4)}
