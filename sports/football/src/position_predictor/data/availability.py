"""Data-availability gate for the redraft tool (use case #2).

Before we can project a *new* season we need the just-completed season's data to exist on
nflverse **and** in the local cache. Two checks live here:

- :func:`check_season_available` — a cheap, network probe that asks nflverse whether the core
  per-player datasets for a given season are published yet (so a draft in, say, June isn't run
  off a half-released season). Required datasets gate the run; optional ones (snaps/NGS) are
  reported but never block — the era ensemble degrades to an older-era schema subset if a modern
  block is missing (PROJECT_PLAN §6.3).
- :func:`datasets_needing_refresh` — compares each dataset's cache **manifest** against the
  season we want so we only re-fetch what is stale (the core caches were materialised before the
  new season published; ``rosters``/``snaps``/``NGS`` may already reach it).

Both reuse the dataset registry and loaders from :mod:`position_predictor.data.fetch`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .fetch import REGISTRY

# Core per-player datasets that must exist for the feature season before we project. ``seasonal``
# and ``weekly`` carry the fantasy points; ``rosters`` carries age/team/position attributes.
REQUIRED_DATASETS = ("seasonal", "weekly", "rosters")
# Helpful-but-not-required (coverage starts later; missing → era subset still scores).
OPTIONAL_DATASETS = ("snap_counts", "ngs_passing", "ngs_rushing", "ngs_receiving")
# Minimum plausible row counts for a *fully* published season — guards against a season that is
# only partway released (a few early weeks) reading as "available".
REQUIRED_MIN_ROWS = {"seasonal": 200, "weekly": 1000, "rosters": 500}


@dataclass
class DatasetProbe:
    name: str
    present: bool
    n_rows: int | None = None
    message: str = ""


@dataclass
class AvailabilityReport:
    season: int
    required: list[DatasetProbe] = field(default_factory=list)
    optional: list[DatasetProbe] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        """Required datasets that are absent/too small (these block the run)."""
        return [p.name for p in self.required if not p.present]

    @property
    def ok(self) -> bool:
        return not self.missing

    def __str__(self) -> str:
        lines = [f"[availability] season {self.season}: "
                 f"{'OK' if self.ok else 'NOT READY'}"]
        for p in self.required + self.optional:
            tag = "✓" if p.present else "✗"
            rows = f"{p.n_rows} rows" if p.n_rows is not None else "—"
            kind = "required" if p in self.required else "optional"
            lines.append(f"  {tag} {p.name:<13} {rows:<12} ({kind}) {p.message}")
        if not self.ok:
            lines.append(f"  → missing required: {', '.join(self.missing)}")
        return "\n".join(lines)


def _probe_one(name: str, season: int) -> DatasetProbe:
    """Load a single season of one dataset and report whether it is plausibly complete."""
    ds = REGISTRY.get(name)
    if ds is None:
        return DatasetProbe(name, False, message="unknown dataset")
    if ds.min_season is not None and season < ds.min_season:
        return DatasetProbe(name, False, message=f"coverage starts {ds.min_season}")
    try:
        df = ds.loader([season])
    except Exception as exc:  # noqa: BLE001 — an unpublished season raises; treat as absent
        return DatasetProbe(name, False, message=f"{type(exc).__name__}: {exc}")
    n = int(getattr(df, "shape", [0])[0])
    floor = REQUIRED_MIN_ROWS.get(name, 1)
    if n < floor:
        return DatasetProbe(name, False, n, message=f"only {n} rows (<{floor}); partial release?")
    return DatasetProbe(name, True, n)


def check_season_available(season: int, *, required=REQUIRED_DATASETS,
                           optional=OPTIONAL_DATASETS) -> AvailabilityReport:
    """Probe nflverse for ``season`` and report whether the required datasets are published.

    Network: one single-season pull per dataset (cheap; results discarded). ``ok`` is true only
    when every ``required`` dataset is present and plausibly complete.
    """
    return AvailabilityReport(
        season=season,
        required=[_probe_one(n, season) for n in required],
        optional=[_probe_one(n, season) for n in optional],
    )


def _manifest_max_season(name: str, manifest_dir) -> int | None:
    """Latest season recorded in a dataset's cache manifest, or ``None`` if uncached."""
    path = manifest_dir / f"{name}.json"
    if not path.exists():
        return None
    try:
        seasons = json.load(open(path)).get("seasons") or []
    except (OSError, json.JSONDecodeError):
        return None
    return max(seasons) if seasons else None


def datasets_needing_refresh(feature_season: int, *, datasets=None,
                             always_refresh=("ids", "sleeper_players", "draft_picks", "rosters"),
                             manifest_dir=None) -> list[str]:
    """Which cached datasets must be re-fetched to model ``feature_season``.

    A *seasonal* dataset is stale when it covers ``feature_season`` (``feature_season`` ≥ its
    ``min_season``) but its manifest's newest season is older than that — i.e. the cache predates
    the season's publication and must be overwritten. ``always_refresh`` datasets are snapshots /
    crosswalks (no per-season manifest) plus the draft class used for rookie counting; we always
    refresh them. Order follows the registry for stable, readable output.

    ``rosters`` is in ``always_refresh`` for a reason the staleness rule cannot express: the
    offseason block reads the season **N+1** preseason roster (see
    :func:`position_predictor.features.build.add_offseason`), but the rule only ever asks whether
    a dataset covers ``feature_season`` — which is N. So once season N's rosters were cached
    nothing ever pulled N+1's, the ``team_next`` join silently produced all-NaN, and every
    offseason feature collapsed to a constant. That is a degenerate feature block, not a stale
    one, so no amount of tightening the N-based rule would have caught it.
    """
    from ..utils.io import MANIFEST_DIR

    manifest_dir = manifest_dir or MANIFEST_DIR
    names = list(datasets) if datasets is not None else list(REGISTRY)
    stale: list[str] = []
    for name in names:
        ds = REGISTRY.get(name)
        if ds is None:
            continue
        if name in always_refresh:
            stale.append(name)
            continue
        if not ds.needs_years:
            continue
        if ds.min_season is not None and feature_season < ds.min_season:
            continue  # dataset doesn't cover this season at all — nothing to refresh
        have = _manifest_max_season(name, manifest_dir)
        if have is None or have < feature_season:
            stale.append(name)
    return stale
