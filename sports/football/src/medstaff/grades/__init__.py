"""Composing residuals into club grades, and attaching staff identity where possible."""

from .compose import (
    ABSOLUTE_BANDS,
    ATTRIBUTABILITY_PRIOR,
    COMPONENT_SIGN,
    LETTER_QUOTAS,
    assign_letters,
    component_z,
    composite,
    empirical_bayes,
    reliability_weights,
    separation_flags,
    shrinkage_factor,
    two_level_shrink,
)
from .staff import attach_staff, coach_changes, head_coach_by_season, load_staff_table

__all__ = [
    "ABSOLUTE_BANDS",
    "ATTRIBUTABILITY_PRIOR",
    "COMPONENT_SIGN",
    "LETTER_QUOTAS",
    "assign_letters",
    "attach_staff",
    "coach_changes",
    "component_z",
    "composite",
    "empirical_bayes",
    "head_coach_by_season",
    "load_staff_table",
    "reliability_weights",
    "separation_flags",
    "shrinkage_factor",
    "two_level_shrink",
]
