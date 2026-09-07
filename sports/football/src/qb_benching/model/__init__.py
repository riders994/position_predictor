"""Fitting the benching model, and reporting it within strata."""

from .fit import (
    BASELINES, TOP_K_PER_SEASON, VOLUME_BANDS, band_profile, baseline_scores, build_pipeline,
    coefficients, cross_validate, permutation_null, precision_at_k_by_season, temporal_split,
    volume_band, within_band_auc,
)

__all__ = [
    "BASELINES",
    "TOP_K_PER_SEASON",
    "VOLUME_BANDS",
    "band_profile",
    "baseline_scores",
    "build_pipeline",
    "coefficients",
    "cross_validate",
    "permutation_null",
    "precision_at_k_by_season",
    "temporal_split",
    "volume_band",
    "within_band_auc",
]
