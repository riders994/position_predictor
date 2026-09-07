"""Fetch and cache nflverse datasets via ``nflreadpy``.

Design goals (see docs/PROJECT_PLAN.md §2.4):
- **Reproducible:** every pull is cached to ``data/raw/<name>.parquet`` and described by a
  JSON manifest in ``data/raw/_manifests/`` recording source, seasons, pull time, row/col
  counts, and the content hash. Raw files are never edited in place.
- **Availability-aware:** each dataset declares its earliest available season; requested
  seasons are clipped accordingly (e.g. snap counts only exist 2012+).
- **Lazy + resilient:** ``nflreadpy`` is imported only when a fetch runs, and each
  dataset is fetched independently so one failure does not abort the rest.

The dataset registry is intentionally small and explicit. Loaders are thin wrappers so
they are easy to adjust if an ``nflreadpy`` signature changes between versions.

Provenance note (2026 migration): the previously-used ``nfl_data_py`` is deprecated upstream
and still points at the frozen ``player_stats`` release, which stops at 2024. nflverse moved
current player stats to the ``stats_player`` release (served by ``nflreadpy``). ``nflreadpy``
returns **polars** frames, so each loader converts to pandas; two datasets are then normalised
back to the column names the rest of the pipeline already expects (see ``*_RENAME`` below) so
this migration is contained to this module and the downstream cache schema is unchanged.
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
    loader: callable(years: list[int]) -> DataFrame. ``nflreadpy`` is imported inside.
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
    source: str = "nflverse/nflreadpy"


# nflverse's ``stats_player`` schema renames a few box-score columns relative to the old
# ``player_stats`` release; map them back so downstream feature code is untouched.
PLAYER_STATS_RENAME = {
    "passing_interceptions": "interceptions",
    "sacks_suffered": "sacks",
    "team": "recent_team",
}
# ``load_rosters`` keys players on ``gsis_id``; the pipeline keys on ``player_id``.
ROSTERS_RENAME = {"gsis_id": "player_id"}


def _registry() -> dict[str, Dataset]:
    """Build the dataset registry. ``nflreadpy`` is imported lazily per loader call."""

    def L(func: str, *, stat_type: str | None = None, summary_level: str | None = None,
          rename: dict[str, str] | None = None):
        """Make a loader that calls ``nflreadpy.<func>`` and returns a pandas frame.

        ``nflreadpy`` returns polars frames keyed on ``seasons`` (first positional); we convert
        to pandas and optionally rename columns to the pipeline's canonical schema. Loaders that
        take no season list (``needs_years=False``) are called with no positional argument.
        """

        def _load(years):
            import nflreadpy as nr

            fn = getattr(nr, func)
            kwargs = {}
            if stat_type is not None:
                kwargs["stat_type"] = stat_type
            if summary_level is not None:
                kwargs["summary_level"] = summary_level
            frame = fn(**kwargs) if years is None else fn(years, **kwargs)
            # Stay in polars: nflreadpy returns a polars frame, the cache is parquet, and
            # write_parquet writes polars natively — so we never materialise a pandas copy
            # (which would double peak memory on the wide multi-season pulls).
            if rename:
                frame = frame.rename({k: v for k, v in rename.items() if k in frame.columns})
            return frame

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
        Dataset("seasonal", L("load_player_stats", summary_level="reg",
                              rename=PLAYER_STATS_RENAME), min_season=1999,
                note="Season totals per player (rush/rec/TD/fantasy)."),
        Dataset("weekly", L("load_player_stats", summary_level="week",
                            rename=PLAYER_STATS_RENAME), min_season=1999,
                note="Per-game box scores -> PPG and games played."),
        Dataset("rosters", L("load_rosters", rename=ROSTERS_RENAME), min_season=1999,
                note="Age, position, team, height/weight, experience."),
        Dataset("snap_counts", L("load_snap_counts"), min_season=2012,
                note="Offensive snaps & snap share (2012+)."),
        Dataset("ngs_rushing", L("load_nextgen_stats", stat_type="rushing"), min_season=2016,
                note="Next Gen Stats rushing, efficiency-over-expected (2016+)."),
        Dataset("ngs_receiving", L("load_nextgen_stats", stat_type="receiving"), min_season=2016,
                note="Next Gen Stats receiving (2016+)."),
        Dataset("ngs_passing", L("load_nextgen_stats", stat_type="passing"), min_season=2016,
                note="Next Gen Stats passing — CPOE, time-to-throw, aggressiveness (2016+)."),
        Dataset("injuries", L("load_injuries"), min_season=2009,
                note="Weekly injury report: body part, game designation, practice status "
                     "(2009+). NB report_status is ~half-null from 2016 (the league dropped "
                     "'Probable'); body part and practice_status are the regime-invariant "
                     "columns."),
        Dataset("rosters_weekly", L("load_rosters_weekly"), min_season=2002,
                note="Weekly roster status (ACT/INA/RES-IR/PUP/practice squad) — the "
                     "report-independent absence signal; joins injuries on (gsis_id, week)."),
        Dataset("depth_charts", L("load_depth_charts"), min_season=2001,
                note="Weekly depth charts — `depth_team` is the club's own declared ordering at "
                     "each position. Registered for qb_benching, which needs a weekly role "
                     "signal that survives the 2021 `rosters_weekly` regime change: a demotion "
                     "shows as depth_team 1 -> 2 and a reserve-list move as dropping off the "
                     "chart entirely."),
        Dataset("schedules", L("load_schedules"), needs_years=False,
                note="Game context: stadium surface & roof (injury-risk confounders), "
                     "rest days, and home/away head coach."),
        Dataset("draft_picks", L("load_draft_picks"), min_season=1980,
                note="Draft capital (pick number)."),
        Dataset("combine", L("load_combine"), min_season=2000,
                note="Combine athletic testing (numeric)."),
        Dataset("ids", L("load_ff_playerids"), needs_years=False,
                note="Cross-source player ID crosswalk (fantasy positions only — carries "
                     "essentially no offensive linemen; see `players` for a complete one)."),
        Dataset("players", L("load_players"), needs_years=False,
                note="Full player universe with gsis_id<->pfr_id. Unlike `ids` and the weekly "
                     "rosters, its pfr_id coverage is not position-biased (~12% null for both "
                     "linemen and skill players), which is what makes snap joins usable."),
        Dataset("sleeper_players", _load_sleeper, needs_years=False,
                source="sleeper/v1/players/nfl",
                note="Sleeper fantasy_positions eligibility (current snapshot; gsis_id join)."),
        Dataset("pbp", L("load_pbp"), min_season=1999, large=True,
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
    asset *per season*, and the loader aborts the whole request if any single year's
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
    import polars as pl

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
    # diagonal_relaxed: tolerate columns/dtypes that drift across nflverse eras.
    return pl.concat(frames, how="diagonal_relaxed"), got, note


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
