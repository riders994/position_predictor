"""Fetch and cache nflverse datasets via ``nfl_data_py``.

Design goals (see docs/PROJECT_PLAN.md §2.4):
- **Reproducible:** every pull is cached to ``data/raw/<name>.parquet`` and described by a
  JSON manifest in ``data/raw/_manifests/`` recording source, seasons, pull time, row/col
  counts, and the content hash. Raw files are never edited in place.
- **Availability-aware:** each dataset declares its earliest available season; requested
  seasons are clipped accordingly (e.g. snap counts only exist 2012+).
- **Lazy + resilient:** ``nfl_data_py`` is imported only when a fetch runs, and each
  dataset is fetched independently so one failure does not abort the rest.

The dataset registry is intentionally small and explicit. Loaders are thin lambdas so
they are easy to adjust if an ``nfl_data_py`` signature changes between versions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ..utils.io import DATA_RAW, MANIFEST_DIR, ensure_dir, sha256_file, write_parquet


@dataclass(frozen=True)
class Dataset:
    """One nflverse dataset we cache.

    Attributes
    ----------
    name: cache stem (-> data/raw/<name>.parquet)
    loader: callable(years: list[int]) -> DataFrame. ``nfl_data_py`` is imported inside.
    min_season: earliest season the dataset exists for (used to clip requests).
    needs_years: whether the loader takes a season list (some loaders, e.g. ids, do not).
    large: heavy pulls (e.g. play-by-play) skipped unless explicitly included.
    note: human description.
    """

    name: str
    loader: Callable
    min_season: int | None = None
    needs_years: bool = True
    large: bool = False
    note: str = ""
    source: str = "nflverse/nfl_data_py"


def _registry() -> dict[str, Dataset]:
    """Build the dataset registry. ``nfl_data_py`` is imported lazily per loader call."""

    def L(method: str, *, stat_type: str | None = None):
        """Make a loader that calls ``nfl_data_py.<method>``."""

        def _load(years):
            import nfl_data_py as nfl

            fn = getattr(nfl, method)
            if stat_type is not None:
                return fn(stat_type, years)
            if years is None:
                return fn()
            return fn(years)

        return _load

    def _load_sleeper(years):
        """Fetch the Sleeper player universe -> fantasy-eligibility table (current snapshot).

        Sleeper exposes one unauthenticated JSON of every player it tracks, each with a
        ``fantasy_positions`` array (true fantasy eligibility, multi-position) and a
        ``gsis_id`` that joins directly to nflverse. It is a *current* snapshot, not a
        per-season history (see docs/PROJECT_PLAN.md §2 caveat); the build falls back to
        nflverse ``position_group`` for players Sleeper doesn't cover.
        """
        import pandas as pd
        import requests

        resp = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=120)
        resp.raise_for_status()
        rows = []
        for sleeper_id, p in resp.json().items():
            if p.get("sport") not in (None, "nfl"):
                continue
            fp = p.get("fantasy_positions") or []
            rows.append({
                "sleeper_id": sleeper_id,
                "gsis_id": p.get("gsis_id"),
                "full_name": p.get("full_name"),
                "position": p.get("position"),
                "fantasy_positions": ",".join(fp) if fp else None,
                "active": p.get("active"),
            })
        return pd.DataFrame(rows)

    datasets = [
        Dataset("seasonal", L("import_seasonal_data"), min_season=1999,
                note="Season totals per player (rush/rec/TD/fantasy)."),
        Dataset("weekly", L("import_weekly_data"), min_season=1999,
                note="Per-game box scores -> PPG and games played."),
        Dataset("rosters", L("import_seasonal_rosters"), min_season=1999,
                note="Age, position, team, height/weight, experience."),
        Dataset("snap_counts", L("import_snap_counts"), min_season=2012,
                note="Offensive snaps & snap share (2012+)."),
        Dataset("ngs_rushing", L("import_ngs_data", stat_type="rushing"), min_season=2016,
                note="Next Gen Stats rushing, efficiency-over-expected (2016+)."),
        Dataset("ngs_receiving", L("import_ngs_data", stat_type="receiving"), min_season=2016,
                note="Next Gen Stats receiving (2016+)."),
        Dataset("ngs_passing", L("import_ngs_data", stat_type="passing"), min_season=2016,
                note="Next Gen Stats passing — CPOE, time-to-throw, aggressiveness (2016+)."),
        Dataset("draft_picks", L("import_draft_picks"), min_season=1980,
                note="Draft capital (pick number)."),
        Dataset("combine", L("import_combine_data"), min_season=2000,
                note="Combine athletic testing (numeric)."),
        Dataset("ids", L("import_ids"), needs_years=False,
                note="Cross-source player ID crosswalk."),
        Dataset("sleeper_players", _load_sleeper, needs_years=False,
                source="sleeper/v1/players/nfl",
                note="Sleeper fantasy_positions eligibility (current snapshot; gsis_id join)."),
        Dataset("pbp", L("import_pbp_data"), min_season=1999, large=True,
                note="Play-by-play -> EPA, success rate, red-zone/usage proxies (heavy)."),
    ]
    return {d.name: d for d in datasets}


REGISTRY = _registry()

# Default set fetched by `make fetch` — excludes the heavy pbp pull.
DEFAULT_DATASETS = [n for n, d in REGISTRY.items() if not d.large]


@dataclass
class FetchResult:
    name: str
    status: str                       # "fetched" | "skipped" | "error" | "planned"
    seasons: list[int] = field(default_factory=list)
    n_rows: int | None = None
    n_cols: int | None = None
    path: str | None = None
    message: str = ""


def _clip_seasons(ds: Dataset, seasons: list[int]) -> list[int]:
    if not ds.needs_years:
        return []
    if ds.min_season is None:
        return list(seasons)
    return [s for s in seasons if s >= ds.min_season]


def fetch_dataset(ds: Dataset, seasons: list[int], *, overwrite: bool = False) -> FetchResult:
    """Fetch one dataset, cache it, and write its manifest."""
    cache_path = DATA_RAW / f"{ds.name}.parquet"
    years = _clip_seasons(ds, seasons)

    if ds.needs_years and not years:
        return FetchResult(ds.name, "skipped", message="no requested seasons in coverage")

    if cache_path.exists() and not overwrite:
        return FetchResult(ds.name, "skipped", years, path=str(cache_path),
                           message="cache exists (use overwrite=True to refresh)")

    t0 = time.time()
    try:
        df, got_years, note = _load_resilient(ds, years)
    except Exception as exc:  # one dataset failing must not abort the batch
        return FetchResult(ds.name, "error", years, message=f"{type(exc).__name__}: {exc}")

    write_parquet(df, cache_path)
    result = FetchResult(ds.name, "fetched", got_years, int(df.shape[0]), int(df.shape[1]),
                         str(cache_path), f"in {time.time() - t0:.1f}s{note}")
    _write_manifest(ds, result, cache_path)
    return result


def _load_resilient(ds: Dataset, years: list[int]):
    """Load a dataset, tolerating a not-yet-published trailing season.

    nflverse serves some player-stats datasets (``weekly``/``seasonal``) as one release
    asset *per season*, and ``nfl_data_py`` aborts the whole request if any single year's
    asset is missing (e.g. the current season before nflverse publishes it). The fast path
    is the single batch call; only if that fails do we retry **year-by-year** and keep the
    seasons that exist, so one missing trailing year cannot wipe out decades of box scores.

    Returns ``(df, fetched_years, note)``. Raises only if *no* requested year loads.
    """
    if not ds.needs_years:
        return ds.loader(None), [], ""
    try:
        return ds.loader(years), list(years), ""
    except Exception:
        if len(years) <= 1:
            raise  # nothing to salvage — surface the real error
    import pandas as pd

    frames, got, missing, last_exc = [], [], [], None
    for y in years:
        try:
            frames.append(ds.loader([y]))
            got.append(y)
        except Exception as exc:  # noqa: BLE001 — record and continue; salvage other years
            missing.append(y)
            last_exc = exc
    if not frames:
        raise last_exc  # dataset genuinely unavailable for every requested year
    note = f" (skipped unavailable {missing})" if missing else ""
    return pd.concat(frames, ignore_index=True), got, note


def _write_manifest(ds: Dataset, result: FetchResult, cache_path) -> None:
    ensure_dir(MANIFEST_DIR)
    manifest = {
        "name": ds.name,
        "source": ds.source,
        "note": ds.note,
        "seasons": result.seasons,
        "min_season": ds.min_season,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": result.n_rows,
        "n_cols": result.n_cols,
        "cache_path": str(cache_path),
        "sha256": sha256_file(cache_path),
    }
    with open(MANIFEST_DIR / f"{ds.name}.json", "w") as fh:
        json.dump(manifest, fh, indent=2)


def fetch_all(
    seasons: list[int],
    datasets: list[str] | None = None,
    *,
    include_large: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FetchResult]:
    """Fetch a set of datasets.

    Parameters
    ----------
    seasons: inclusive season list (typically ``Config.seasons()``).
    datasets: subset of registry names; default = all non-large datasets.
    include_large: also fetch datasets flagged ``large`` (e.g. play-by-play).
    overwrite: refetch even if a cache file exists.
    dry_run: report what would be fetched without calling nflverse.
    """
    names = datasets or DEFAULT_DATASETS
    results: list[FetchResult] = []
    for name in names:
        ds = REGISTRY.get(name)
        if ds is None:
            results.append(FetchResult(name, "error", message="unknown dataset"))
            continue
        if ds.large and not include_large and not (datasets and name in datasets):
            results.append(FetchResult(name, "skipped", message="large; pass include_large"))
            continue
        if dry_run:
            results.append(FetchResult(name, "planned", _clip_seasons(ds, seasons),
                                       message=ds.note))
            continue
        results.append(fetch_dataset(ds, seasons, overwrite=overwrite))
    return results
