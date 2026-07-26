"""Drafting situation — franchise and regime, analysed against draft capital."""

from .regime import (
    MIN_QBS_PER_REGIME,
    attach_gm,
    attach_situation,
    detectable_effect,
    expected_from_draft,
    group_permutation_p,
    head_coach_by_team_season,
    regime_table,
)

__all__ = [
    "MIN_QBS_PER_REGIME",
    "attach_gm",
    "attach_situation",
    "detectable_effect",
    "expected_from_draft",
    "group_permutation_p",
    "head_coach_by_team_season",
    "regime_table",
]
