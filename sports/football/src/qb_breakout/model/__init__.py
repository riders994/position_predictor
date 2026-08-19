"""Stage 6: the pre-NFL-only breakout model, and its limits."""

from .features import (
    BENCHMARK,
    CATEGORICAL,
    CFBFASTR_ONLY,
    MIN_SETTLED_SEASONS,
    PORTABLE,
    feature_columns,
    modelling_frame,
)
from .fit import (
    benchmark_scores,
    build_pipeline,
    coefficients,
    cross_validate,
    permutation_null,
    precision_at_k,
    temporal_split,
)

__all__ = [
    "BENCHMARK",
    "CATEGORICAL",
    "CFBFASTR_ONLY",
    "MIN_SETTLED_SEASONS",
    "PORTABLE",
    "benchmark_scores",
    "build_pipeline",
    "coefficients",
    "cross_validate",
    "feature_columns",
    "modelling_frame",
    "permutation_null",
    "precision_at_k",
    "temporal_split",
]
