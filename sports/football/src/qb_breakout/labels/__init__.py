"""NFL-side ground truth: QB season ranks, breakout events, and cohort flags."""

from .cohort import (
    BREAKOUT_RANK,
    LATE_YEAR_THRESHOLD,
    MIN_GAMES,
    QB1_RANK,
    build_qb_careers,
    build_qb_seasons,
    rank_qb_seasons,
)

__all__ = [
    "BREAKOUT_RANK",
    "QB1_RANK",
    "LATE_YEAR_THRESHOLD",
    "MIN_GAMES",
    "build_qb_seasons",
    "rank_qb_seasons",
    "build_qb_careers",
]
