"""Era-separate modeling framework (PROJECT_PLAN §6.3).

The nflverse feature coverage changes over time (box scores 1999+, snaps 2012+, NGS 2016+),
so a single model trained across a 30-year window would face **structurally** missing columns
for older seasons. Instead we define **eras** — contiguous season spans that share the same
available feature blocks — and train **one model per era** on that era's schema.

Because the schemas are **nested** (each later era only *adds* blocks), an older-era model can
score a modern player using the subset of features it was trained on. So an *X-year window
model* is the **composition of the era models whose seasons fall in the window**:

- Training rows are routed to their era model by **feature season N** (``route_by``).
- At prediction time the target is a recent, full-feature season; *each* constituent era model
  scores it using its own feature subset, and the per-era predictions are combined
  (``era_modeling.combine``). A 10-year window is ~just the newest era's model; a 30-year
  window adds the older-era models — which is exactly what makes the recency-bias study
  (§6.2) measure whether older-era relationships still carry signal.

This module is pure (no pandas/model dependency): it defines the era boundaries, season→era
assignment, the train-window → per-era season-segment mapping, and feature-column selection.
The actual per-era estimators and the combiner live in the modeling stage (§7), which consumes
:func:`era_segments_for_window`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Sentinel for an open-ended (current) era with no fixed end season.
OPEN = None


@dataclass(frozen=True)
class Era:
    """A contiguous span of seasons sharing one feature schema.

    ``end_season is None`` means open-ended (current). ``feature_blocks`` is the ordered set of
    feature-block names available throughout the era (nested across eras by construction).
    """

    name: str
    start_season: int
    end_season: int | None
    feature_blocks: tuple[str, ...]

    def contains(self, season: int) -> bool:
        if season < self.start_season:
            return False
        return self.end_season is None or season <= self.end_season


def load_eras(config) -> list[Era]:
    """Build the era list from ``config['eras']``, sorted by ``start_season`` and validated.

    Validates that eras are contiguous and non-overlapping and that feature blocks are nested
    (each era's block set ⊇ the previous era's) — the property that lets older-era models score
    modern players.
    """
    raw = config.get("eras") or []
    if not raw:
        raise ValueError("config has no 'eras' block")
    eras = sorted(
        (Era(e["name"], int(e["start_season"]),
             None if e.get("end_season") in (None, "null") else int(e["end_season"]),
             tuple(e.get("feature_blocks", [])))
         for e in raw),
        key=lambda e: e.start_season,
    )
    _validate(eras)
    return eras


def _validate(eras: list[Era]) -> None:
    prev: Era | None = None
    prev_blocks: set[str] = set()
    for era in eras:
        if prev is not None:
            if prev.end_season is None:
                raise ValueError(f"only the last era may be open-ended; {prev.name} is not")
            if era.start_season != prev.end_season + 1:
                raise ValueError(
                    f"eras must be contiguous: {prev.name} ends {prev.end_season}, "
                    f"{era.name} starts {era.start_season}")
        if not prev_blocks.issubset(set(era.feature_blocks)):
            missing = prev_blocks - set(era.feature_blocks)
            raise ValueError(
                f"era schemas must be nested; {era.name} drops blocks {sorted(missing)}")
        prev, prev_blocks = era, set(era.feature_blocks)


def assign_era(season: int, eras: list[Era]) -> str | None:
    """Return the name of the era containing ``season`` (by feature season N), or None."""
    for era in eras:
        if era.contains(season):
            return era.name
    return None


def assign_era_column(seasons, eras: list[Era]):
    """Vectorised :func:`assign_era` over an iterable of seasons → list of era names."""
    return [assign_era(int(s), eras) for s in seasons]


def train_window_bounds(window_years: int, test_seasons, *, horizon: int = 1,
                        earliest_season: int | None = None) -> tuple[int, int]:
    """Feature-season bounds ``(N_min, N_max)`` for a training window (no leakage).

    Test seasons are *label* seasons (the seasons being predicted/ranked); they are held fixed
    across windows (§6.2). The latest training **label** season is ``min(test) - 1``; its
    feature season is that minus ``horizon``. The window then spans ``window_years`` feature
    seasons back from there, clipped to ``earliest_season``.
    """
    first_test = min(int(s) for s in test_seasons)
    n_max = first_test - 1 - horizon
    n_min = n_max - window_years + 1
    if earliest_season is not None:
        n_min = max(n_min, int(earliest_season))
    return n_min, n_max


def era_segments_for_window(window_years: int, test_seasons, eras: list[Era], *,
                            horizon: int = 1, earliest_season: int | None = None):
    """Map a training window to the per-era feature-season segments it covers.

    Returns a list of ``(Era, (seg_start, seg_end))`` for every era that overlaps the window's
    feature-season range — i.e. exactly the era models that compose this X-year window, each
    with the season range it trains on. Eras outside the window are omitted (a short window
    simply has fewer era models).
    """
    n_min, n_max = train_window_bounds(
        window_years, test_seasons, horizon=horizon, earliest_season=earliest_season)
    segments = []
    for era in eras:
        seg_start = max(era.start_season, n_min)
        seg_end = n_max if era.end_season is None else min(era.end_season, n_max)
        if seg_start <= seg_end:
            segments.append((era, (seg_start, seg_end)))
    return segments


def feature_columns_for_era(era: Era, block_columns: dict[str, list[str]]) -> list[str]:
    """Resolve an era's feature blocks to concrete columns via a block→columns map.

    Unknown blocks (not yet built) are skipped, so the feature stage can grow the map
    incrementally. Order follows ``era.feature_blocks`` then column order within each block.
    """
    cols: list[str] = []
    for block in era.feature_blocks:
        cols.extend(block_columns.get(block, []))
    return cols
