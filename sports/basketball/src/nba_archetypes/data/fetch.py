"""Stage 1 — fetch + cache NBA source data (base layer: sportsdataverse / hoopR).

Pulls per-season tables from sportsdataverse (hoopR data releases) into ``data/raw`` as parquet,
each with a manifest (provenance + content hash) under ``data/raw/_manifests``. The granular
nba_api layer (play-type / tracking) and the Fantrax fantasy layer are separate modules.

Season = season-ENDING year (2014 = the 2013-14 season; floor 2014 per the SportVU era,
PROJECT_PLAN §2.2). sportsdataverse loaders are imported lazily so importing this module is cheap.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..utils.io import DATA_RAW, MANIFEST_DIR, ensure_dir, sha256_file, write_parquet


def _loader(fn_name: str):
    """A season-taking loader that imports sportsdataverse lazily and returns a pandas frame."""
    def load(seasons):
        import sportsdataverse.nba as nba
        return getattr(nba, fn_name)(seasons=list(seasons), return_as_pandas=True)
    return load


@dataclass
class Dataset:
    name: str
    fn_name: str
    note: str = ""
    source: str = "sportsdataverse"
    large: bool = False             # excluded from the default fetch (heavy pulls)

    @property
    def loader(self):
        return _loader(self.fn_name)


REGISTRY: dict[str, Dataset] = {
    d.name: d for d in [
        Dataset("player_season_stats", "load_nba_player_season_stats",
                note="player-season aggregate stats (long format; pivoted in build)"),
        Dataset("rosters", "load_nba_rosters", note="player-team-season rosters"),
        Dataset("team_season_stats", "load_nba_team_season_stats",
                note="team-season aggregates (pace / usage context)"),
        Dataset("shots", "load_nba_shots",
                note="shot locations / zones — powers shot-profile style features", large=True),
        Dataset("player_boxscore", "load_nba_player_boxscore",
                note="per-game player box scores", large=True),
    ]
}

# Default `make fetch` set — excludes the heavy per-event pulls (shots / boxscore).
DEFAULT_DATASETS = [n for n, d in REGISTRY.items() if not d.large]


@dataclass
class FetchResult:
    name: str
    status: str                     # "fetched" | "skipped" | "error"
    seasons: list[int] = field(default_factory=list)
    n_rows: int | None = None
    n_cols: int | None = None
    path: str | None = None
    message: str = ""


def fetch_dataset(ds: Dataset, seasons: list[int], *, overwrite: bool = False) -> FetchResult:
    """Fetch one dataset, cache it to parquet, and write its manifest."""
    cache_path = DATA_RAW / f"{ds.name}.parquet"
    if cache_path.exists() and not overwrite:
        return FetchResult(ds.name, "skipped", seasons, path=str(cache_path),
                           message="cache exists (use --overwrite to refresh)")
    t0 = time.time()
    try:
        df = _load_resilient(ds, seasons)
    except Exception as exc:        # one dataset failing must not abort the batch
        return FetchResult(ds.name, "error", seasons, message=f"{type(exc).__name__}: {exc}")
    write_parquet(df, cache_path)
    result = FetchResult(ds.name, "fetched", seasons, int(df.shape[0]), int(df.shape[1]),
                         str(cache_path), f"in {time.time() - t0:.1f}s")
    _write_manifest(ds, result, cache_path)
    return result


def _load_resilient(ds: Dataset, seasons: list[int]):
    """Batch-load; on failure retry season-by-season so one missing year can't wipe the pull."""
    try:
        return ds.loader(seasons)
    except Exception:
        if len(seasons) <= 1:
            raise
    import pandas as pd
    frames = []
    for s in seasons:
        try:
            frames.append(ds.loader([s]))
        except Exception:
            continue
    if not frames:
        raise RuntimeError(f"no seasons loaded for {ds.name}")
    return pd.concat(frames, ignore_index=True)


def _write_manifest(ds: Dataset, result: FetchResult, cache_path) -> None:
    ensure_dir(MANIFEST_DIR)
    manifest = {
        "name": ds.name,
        "source": ds.source,
        "note": ds.note,
        "seasons": result.seasons,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": result.n_rows,
        "n_cols": result.n_cols,
        "cache_path": str(cache_path),
        "sha256": sha256_file(cache_path),
    }
    with open(MANIFEST_DIR / f"{ds.name}.json", "w") as fh:
        json.dump(manifest, fh, indent=2)


def fetch_all(seasons: list[int], datasets: list[str] | None = None, *,
              include_large: bool = False, overwrite: bool = False) -> list[FetchResult]:
    """Fetch a set of datasets for ``seasons``. Defaults to the non-large set."""
    names = datasets or (list(REGISTRY) if include_large else DEFAULT_DATASETS)
    return [fetch_dataset(REGISTRY[n], seasons, overwrite=overwrite) for n in names]
