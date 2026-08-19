"""Artifact naming — one stem per (position, scoring format).

Every stage names its artifacts from a stem that has always been ``f"{sport}_{position}"``
(``football_rb`` → ``data/interim/football_rb_player_seasons.parquet``,
``reports/REPORT_football_rb.md``, …). Alternate scoring formats make that ambiguous: a half-PPR
build trains on a different target, so its dataset, features, model and board are all different
artifacts — and would otherwise silently overwrite the PPR ones in place.

So the stem gains a scoring suffix for every format **except** PPR, which keeps the bare stem.
PPR is the only format that has ever existed here, so every committed path, report and
``reports/versions/`` snapshot stays exactly where it is.
"""

from __future__ import annotations

from ..scoring import DEFAULT_SCORING, scoring_of


def artifact_stem(config, *, position: str | None = None) -> str:
    """``football_rb`` under PPR (unchanged), ``football_rb_half_ppr`` otherwise."""
    sport = config.get("experiment.sport", "sport")
    position = position or config.require("experiment.position")
    stem = f"{sport}_{position}".lower()
    scoring = scoring_of(config)
    return stem if scoring == DEFAULT_SCORING else f"{stem}_{scoring}"
