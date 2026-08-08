"""Fantasy scoring formats — the reception value that defines a league's points.

nflverse publishes two fantasy totals per player-week: ``fantasy_points`` (standard) and
``fantasy_points_ppr`` (full PPR). They differ by **exactly one point per reception** — verified
row-wise over every weekly row we cache (476,156 rows, 1999–2025: max absolute deviation
3.6e-15, no nulls in any of the three columns). So every format below is an exact recomputation

    points = fantasy_points + reception_value * receptions

from columns already aggregated in :data:`position_predictor.data.build.SEASON_SUM_COLS`. No new
data source, no approximation, and full PPR reproduces ``fantasy_points_ppr`` identically.

Scoring is **not** a display-layer setting here: season PPG is the model's training target
(``target_ppg_next``), so each format is a separate dataset → features → fit → board. That is why
:func:`position_predictor.utils.naming.artifact_stem` namespaces artifacts by format.
"""

from __future__ import annotations

# Points added per reception on top of nflverse standard scoring.
RECEPTION_POINTS = {"standard": 0.0, "half_ppr": 0.5, "ppr": 1.0}

# The historical (and still default) format: every artifact path predates alternate scoring.
DEFAULT_SCORING = "ppr"

# Spellings people actually write in a league config, mapped to our canonical keys.
_ALIASES = {
    "std": "standard", "standard": "standard", "non_ppr": "standard", "0ppr": "standard",
    "half": "half_ppr", "half_ppr": "half_ppr", "half_point_ppr": "half_ppr",
    "0.5_ppr": "half_ppr", "0.5ppr": "half_ppr",
    "ppr": "ppr", "full_ppr": "ppr", "1ppr": "ppr",
}


def normalize_scoring(scoring: str | None) -> str:
    """Canonicalise a scoring name (case/punctuation-insensitive); ``None`` → the default.

    Raises ``ValueError`` on an unknown format rather than silently scoring a league wrong.
    """
    if scoring is None:
        return DEFAULT_SCORING
    key = str(scoring).strip().lower().replace("-", "_").replace(" ", "_")
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"unknown scoring format {scoring!r}; expected one of {sorted(RECEPTION_POINTS)}")


def reception_points(scoring: str | None) -> float:
    """Points per reception for ``scoring``."""
    return RECEPTION_POINTS[normalize_scoring(scoring)]


def scoring_of(config) -> str:
    """The canonical scoring format of an experiment config (``target.scoring``; default PPR)."""
    return normalize_scoring(config.get("target.scoring", DEFAULT_SCORING))
