"""Composing component residuals into a club grade.

Three things happen here, in order: standardise each component, shrink it toward the league mean
by how much of its spread is real, then combine and letter.

**The letter curve is a fixed quota, by decision.** Three A's, five B's, eight C's, eight D's,
eight F's — exactly 32. This is a forced rank, so it assigns those letters whether or not the
clubs are distinguishable from one another; it does not test anything, it orders. That is a
deliberate provisional choice, to be replaced once a few seasons of reports exist and an absolute
score-to-grade mapping can be calibrated. :data:`ABSOLUTE_BANDS` is where that mapping goes, and
:func:`assign_letters` already accepts it — the switch is a parameter, not a rewrite.

Because the curve cannot express "these two clubs are the same", the report carries the
separation explicitly: every letter ships with its interval, and :func:`separation_flags` marks
each club that is not distinguishable from its neighbour.

**Shrinkage before ranking, always.** A club with thin evidence is pulled toward the league mean
in proportion to how much of the observed spread survives as signal. Without it the extremes of
any ranking are mostly the clubs with the least data.
"""

from __future__ import annotations

import numpy as np

# User-specified forced-rank curve. Sums to 32.
LETTER_QUOTAS = (("A", 3), ("B", 5), ("C", 8), ("D", 8), ("F", 8))

# The replacement, once enough seasons exist to calibrate it. Absolute composite-score cutoffs,
# highest first; unused while LETTER_QUOTAS is in force.
ABSOLUTE_BANDS: tuple[tuple[str, float], ...] = ()

# Sign convention: positive is always better. Incidence and recurrence are counts of bad things,
# so their residuals are negated; duration and returns-at-all are counts of players getting back
# on the field, so they are not.
COMPONENT_SIGN = {
    "incidence_no_history": -1.0,
    "incidence_with_history": -1.0,
    "duration": +1.0,
    "recurrence": -1.0,
    "returns_at_all": +1.0,
}

# Ordered by how plausibly a medical staff owns the outcome. Reported alongside the
# reliability-derived weights, which order them almost exactly the other way round — see the
# report. Neither is obviously right, so both ship.
ATTRIBUTABILITY_PRIOR = {
    "recurrence": 0.45,
    "duration": 0.35,
    "incidence_no_history": 0.20,
}


def component_z(residuals, *, component_col="component", team_col="team"):
    """Signed standardised residual per club x component, positive = better."""
    import pandas as pd

    df = residuals.to_pandas() if hasattr(residuals, "to_pandas") else residuals.copy()
    df = df[df[component_col].isin(COMPONENT_SIGN)].copy()
    df["sign"] = df[component_col].map(COMPONENT_SIGN)
    df["z"] = df["sign"] * df["diff"] / np.sqrt(df["var_indep"].clip(lower=1e-9))
    return pd.DataFrame(df[[team_col, component_col, "z", "n", "observed", "expected"]])


def shrinkage_factor(z) -> float:
    """Fraction of observed spread that survives as signal, for standardised residuals.

    ``z`` has sampling variance 1 by construction, so tau-squared is whatever variance is left
    over once that is subtracted, and the factor is tau2 / (tau2 + 1).
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 2:
        return 0.0
    tau2 = max(0.0, float(np.var(z, ddof=1)) - 1.0)
    return float(tau2 / (tau2 + 1.0))


def empirical_bayes(z, *, prior_mean=None):
    """Shrink standardised residuals toward the league mean; returns values and posterior sd."""
    z = np.asarray(z, dtype=float)
    k = shrinkage_factor(z)
    centre = float(np.nanmean(z)) if prior_mean is None else float(prior_mean)
    shrunk = centre + k * (z - centre)
    sd = np.full_like(shrunk, np.sqrt(k))
    return shrunk, sd, k


def two_level_shrink(cell_z, club_shrunk):
    """Shrink a club x group cell toward that club's own shrunken overall value.

    The second level matters more than the first here: club x position-group cells are thin, and
    without it the group tables would rank noise.
    """
    cell_z = np.asarray(cell_z, dtype=float)
    k = shrinkage_factor(cell_z - np.asarray(club_shrunk, dtype=float))
    shrunk = np.asarray(club_shrunk, dtype=float) + k * (
        cell_z - np.asarray(club_shrunk, dtype=float))
    return shrunk, np.full_like(shrunk, np.sqrt(k)), k


def reliability_weights(split_half: dict) -> dict:
    """Weights proportional to each component's measured split-half reliability.

    Self-limiting: a component carrying no signal gets a weight near zero without anyone having
    to decide that it should.
    """
    usable = {k: max(float(v), 0.0) for k, v in split_half.items() if np.isfinite(v)}
    total = sum(usable.values())
    if total <= 0:
        n = len(usable) or 1
        return {k: 1.0 / n for k in usable}
    return {k: v / total for k, v in usable.items()}


def composite(z_wide, weights):
    """Weighted mean of component z-scores over the components present."""
    import pandas as pd

    cols = [c for c in weights if c in z_wide.columns]
    if not cols:
        raise KeyError("no weighted component present in the score table")
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()
    values = z_wide[cols].to_numpy(dtype=float)
    mask = np.isfinite(values)
    weighted = np.where(mask, values, 0.0) @ w
    norm = mask @ w
    return pd.Series(np.where(norm > 0, weighted / np.clip(norm, 1e-9, None), np.nan),
                     index=z_wide.index)


def assign_letters(scores, *, quotas=LETTER_QUOTAS, bands=ABSOLUTE_BANDS):
    """Letters from an absolute band table if one is configured, else the forced-rank quota.

    The quota path scales proportionally if the club count is not 32, so the function does not
    silently drop or duplicate grades on a partial board.
    """
    import pandas as pd

    s = pd.Series(scores).astype(float)
    if bands:
        out = pd.Series(index=s.index, dtype=object)
        for letter, cutoff in bands:
            out[out.isna() & (s >= cutoff)] = letter
        out[out.isna()] = bands[-1][0]
        return out

    n = len(s)
    total = sum(c for _, c in quotas)
    sizes = [max(1, round(c * n / total)) for _, c in quotas] if n != total else \
        [c for _, c in quotas]
    # correct any rounding drift onto the largest band
    drift = n - sum(sizes)
    if drift:
        sizes[int(np.argmax(sizes))] += drift

    order = s.sort_values(ascending=False, kind="stable").index
    out = pd.Series(index=s.index, dtype=object)
    cursor = 0
    for (letter, _), size in zip(quotas, sizes):
        out.loc[order[cursor:cursor + size]] = letter
        cursor += size
    out.loc[order[cursor:]] = quotas[-1][0]
    return out


def separation_flags(scores, sds, *, z=1.2816):
    """For each club, whether its interval clears the club ranked immediately below it.

    A forced-rank curve cannot say "these two are the same", so this is how the report says it.
    """
    import pandas as pd

    s = pd.Series(scores).astype(float).sort_values(ascending=False, kind="stable")
    sd = pd.Series(sds).astype(float).reindex(s.index)
    below = s.shift(-1)
    below_sd = sd.shift(-1)
    gap = s - below
    pooled = np.sqrt(sd**2 + below_sd**2)
    return pd.DataFrame({
        "separated_from_next": (gap > z * pooled).fillna(False),
        "gap_to_next": gap,
    }).reindex(pd.Series(scores).index)


__all__ = [
    "ABSOLUTE_BANDS", "ATTRIBUTABILITY_PRIOR", "COMPONENT_SIGN", "LETTER_QUOTAS",
    "assign_letters", "component_z", "composite", "empirical_bayes", "reliability_weights",
    "separation_flags", "shrinkage_factor", "two_level_shrink",
]
