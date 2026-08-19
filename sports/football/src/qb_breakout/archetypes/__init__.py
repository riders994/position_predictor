"""College QB archetypes — style clusters discovered from pre-NFL production."""

from .cluster import (
    CLUSTER_FEATURES,
    FIT_MIN_DROPBACKS,
    PROFILE_FEATURES,
    add_style_features,
    career_archetypes,
    choose_k,
    fit_archetypes,
    name_archetypes,
    profile_archetypes,
    quality_leakage,
    standardize_within_season,
)

__all__ = [
    "CLUSTER_FEATURES",
    "FIT_MIN_DROPBACKS",
    "PROFILE_FEATURES",
    "add_style_features",
    "career_archetypes",
    "choose_k",
    "fit_archetypes",
    "name_archetypes",
    "profile_archetypes",
    "quality_leakage",
    "standardize_within_season",
]
