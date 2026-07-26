"""Pre-NFL evidence loaders.

``college`` is the project's primary evidence; ``recruiting`` is the high-school layer, built and
measured but demoted to an optional covariate (see ``docs/QB_BREAKOUT_PLAN.md`` §2.3).
"""

from .cfbd import (
    FIRST_PPA_SEASON,
    apply_calibration,
    build_cfbd_qb_seasons,
    compare_to_cfbfastr,
    fit_calibration,
)
from .college import (
    FIRST_PBP_SEASON,
    LAST_PBP_SEASON,
    add_career_features,
    aggregate_qb_seasons,
    build_college_qb_seasons,
    load_pbp_season,
)
from .college_link import college_coverage_report, link_college_to_cohort, normalize_school
from .link import link_recruits_to_cohort, normalize_name, recruiting_coverage_report
from .recruiting import (
    QB_POSITIONS,
    clean_measurables,
    fetch_qb_recruits,
    fetch_recruiting_class,
    parse_recruit,
)

__all__ = [
    # college — current-season extension (CFBD, needs a free API key)
    "FIRST_PPA_SEASON",
    "build_cfbd_qb_seasons",
    "compare_to_cfbfastr",
    "fit_calibration",
    "apply_calibration",
    # college (primary)
    "FIRST_PBP_SEASON",
    "LAST_PBP_SEASON",
    "load_pbp_season",
    "aggregate_qb_seasons",
    "build_college_qb_seasons",
    "add_career_features",
    "normalize_school",
    "link_college_to_cohort",
    "college_coverage_report",
    # high school (demoted)
    "QB_POSITIONS",
    "parse_recruit",
    "clean_measurables",
    "fetch_recruiting_class",
    "fetch_qb_recruits",
    "normalize_name",
    "link_recruits_to_cohort",
    "recruiting_coverage_report",
]
