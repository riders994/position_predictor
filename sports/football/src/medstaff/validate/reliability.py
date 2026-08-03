"""Reliability and power: is any of the club spread a stable property of the club?

Stage 4 showed that club residuals are larger than sampling noise. That is a weaker claim than it
sounds — a residual can be large and still be a one-off, and a leaderboard built on one-offs is a
leaderboard of luck. This stage asks the question that decides whether the grades mean anything:
**does a club's residual in one part of the data predict its residual in another?**

Two independent tests, because they fail in different ways:

**Split-half** splits each club's *players* — never its rows — into halves and correlates the two
residuals. Splitting rows would leak: a fragile player's weeks would land on both sides and
manufacture agreement out of one man's hamstring. Corrected by Spearman-Brown, since each half
carries only half the evidence.

**Temporal** grades the early seasons and asks whether that predicts the late ones. This is the
harder and more honest test — it is the one a reader actually cares about, because a grade is
only useful if it says something about the club *going forward*. Split-half can pass while
temporal fails, and that gap is itself the finding: it means the residual is a property of a
period, not of a club.

Where both fail, the honest headline is the **minimum detectable effect** — "nothing this size or
smaller was findable" is a result; "no effect" is not.
"""

from __future__ import annotations

import numpy as np

RANDOM_STATE = 17
# Preregistered thresholds. Recorded here so the gate is a decision made before seeing the
# answer rather than a line drawn around whatever came out.
GATE_SPLIT_HALF = 0.30
GATE_TEMPORAL = 0.30
GATE_PERMUTATION_P = 0.05


def _standardised_residual(observed, expected):
    """(observed - expected) / sqrt(Poisson-binomial variance), so halves are comparable."""
    var = float(np.sum(expected * (1 - expected)))
    return (float(np.sum(observed)) - float(np.sum(expected))) / np.sqrt(max(var, 1e-9))


def _club_residuals(df, *, team_col, outcome, expected_col, mask=None):
    sub = df if mask is None else df[mask]
    out = {}
    for club, grp in sub.groupby(team_col, dropna=False):
        out[club] = _standardised_residual(
            grp[outcome].to_numpy().astype(float), grp[expected_col].to_numpy()
        )
    return out


def spearman_brown(r: float) -> float:
    """Correct a half-length reliability to full length. Undefined at r = -1."""
    denom = 1.0 + r
    return float(2 * r / denom) if abs(denom) > 1e-12 else float("nan")


def split_half_reliability(frame, *, team_col="team", unit_col="gsis_id", outcome,
                           expected_col="expected", n_sim=500, seed=RANDOM_STATE):
    """Correlate two halves of each club's evidence, splitting by player.

    Splitting by *player* rather than by row is what keeps this honest: a fragile player's weeks
    are correlated, so a row-wise split would put the same man on both sides and manufacture
    agreement.
    """
    from scipy import stats

    df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    df = df[[team_col, unit_col, outcome, expected_col]].dropna(subset=[unit_col])
    rng = np.random.default_rng(seed)

    units = df[unit_col].to_numpy()
    uniq = np.unique(units)
    raw, corrected = [], []
    for _ in range(n_sim):
        side = dict(zip(uniq, rng.integers(0, 2, size=len(uniq))))
        is_a = np.array([side[u] == 0 for u in units])
        a = _club_residuals(df, team_col=team_col, outcome=outcome,
                            expected_col=expected_col, mask=is_a)
        b = _club_residuals(df, team_col=team_col, outcome=outcome,
                            expected_col=expected_col, mask=~is_a)
        clubs = sorted(set(a) & set(b))
        if len(clubs) < 4:
            continue
        r = stats.spearmanr([a[c] for c in clubs], [b[c] for c in clubs]).statistic
        if np.isnan(r):
            continue
        raw.append(float(r))
        corrected.append(spearman_brown(float(r)))
    if not raw:
        return {"n_reps": 0, "r_half": float("nan"), "r_full": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "n_reps": len(raw),
        "r_half": float(np.mean(raw)),
        "r_full": float(np.mean(corrected)),
        "ci_low": float(np.percentile(corrected, 2.5)),
        "ci_high": float(np.percentile(corrected, 97.5)),
    }


def temporal_reliability(frame, *, team_col="team", season_col="season", outcome,
                         expected_col="expected", split_season):
    """Does the early-window residual predict the late-window one?

    The harder test, and the one a reader actually cares about: a grade is only useful if it says
    something about the club going forward.
    """
    from scipy import stats

    df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    early_mask = (df[season_col] < split_season).to_numpy()
    early = _club_residuals(df, team_col=team_col, outcome=outcome,
                            expected_col=expected_col, mask=early_mask)
    late = _club_residuals(df, team_col=team_col, outcome=outcome,
                           expected_col=expected_col, mask=~early_mask)
    clubs = sorted(set(early) & set(late))
    if len(clubs) < 4:
        return {"n_clubs": len(clubs), "spearman": float("nan"), "pearson": float("nan"),
                "p_value": float("nan"), "slope": float("nan")}
    x = np.array([early[c] for c in clubs])
    y = np.array([late[c] for c in clubs])
    sp = stats.spearmanr(x, y)
    pe = stats.pearsonr(x, y)
    slope = float(np.polyfit(x, y, 1)[0]) if np.std(x) > 0 else float("nan")
    return {"n_clubs": len(clubs), "spearman": float(sp.statistic),
            "pearson": float(pe.statistic), "p_value": float(sp.pvalue), "slope": slope,
            "early": early, "late": late}


def detectable_by_group(frame, *, team_col="team", group_col="position_group",
                        expected_col="expected", alpha=0.05):
    """Minimum detectable observed-minus-expected per club x group, in the outcome's own units.

    Where a cell is too thin to grade, this is what gets reported instead of a rank.
    """
    import pandas as pd
    from scipy import stats

    df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for (club, group), grp in df.groupby([team_col, group_col], dropna=False):
        p = grp[expected_col].to_numpy()
        var = float(np.sum(p * (1 - p)))
        rows.append({team_col: club, group_col: group, "n": len(grp),
                     "expected": float(p.sum()), "sd": float(np.sqrt(var)),
                     "min_detectable": float(z * np.sqrt(var))})
    return pd.DataFrame(rows)


def reliability_gate(*, split_half_r, temporal_r, permutation_p):
    """The preregistered gate, evaluated per component.

    **The gate does not suppress anything.** Grades publish either way — that was a deliberate
    choice — so this exists to *label* each component at every appearance, not to hide it. A
    number that fails here is still printed; it is printed marked.
    """
    checks = {
        "split_half": bool(split_half_r >= GATE_SPLIT_HALF),
        "temporal": bool(temporal_r >= GATE_TEMPORAL),
        "permutation": bool(permutation_p < GATE_PERMUTATION_P),
    }
    return {"passed": all(checks.values()), **checks,
            "failed": [k for k, v in checks.items() if not v]}


__all__ = [
    "GATE_PERMUTATION_P", "GATE_SPLIT_HALF", "GATE_TEMPORAL", "RANDOM_STATE",
    "detectable_by_group", "reliability_gate", "spearman_brown", "split_half_reliability",
    "temporal_reliability",
]
