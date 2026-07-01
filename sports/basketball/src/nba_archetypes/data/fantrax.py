"""Phase 2 (success labels) — optimal-lineup season estimates via the ``max_pf`` package.

Actual win/loss is contaminated: some managers never set their lineups, so realized results
under-report roster quality. ``max_pf`` recomputes each team's **9-category "points-for"** under
*optimal* lineups from realized box scores (``methodology="hindsight"``), and with ``nash=True`` the
**mutual ceiling** — "how the season would go if *everyone* set their lineup." That optimal PF is the
clean success label for Phase 2; ``actual_pf`` is kept for reference and ``lineup_gap`` measures the
points a manager left on the table.

The 9 ``max_pf`` categories match our format exactly (fg_pct, tpm, ft_pct, pts, reb, ast, stl, blk,
to). The three optimal-PF estimates encode different *lineup-management* assumptions:

- ``m1_pf`` — **actual competition**: your optimal lineup vs opponents' *actual* (often unset) lineups.
- ``m2_pf`` — **both teams set their lineup** (the mutual/Nash ceiling): the realistic "if everyone
  manages" outcome.
- ``m3_pf`` — **maximal management**: the most aggressive optimization ceiling.

**Grade off M2 > M3 > M1** (user directive): ``m2_pf`` is the primary success label (both managers
optimize), ``m3_pf`` then ``m1_pf`` as fallbacks. ``primary_pf = m2_pf``; ``lineup_gap = m2_pf -
actual_pf`` (category wins a manager left on the table vs the both-optimize ceiling).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from ..utils.io import (DATA_PROCESSED, MANIFEST_DIR, ensure_dir, sha256_file, write_parquet)


def _run_league(league_id, *, weeks, methodology, nash, cache_dir, retries=3, backoff=30):
    """``max_pf.run`` for one league with retry/backoff (Fantrax rate-limits heavy full pulls).

    The box-score cache (``cache_dir``) is warm across retries, so a backoff usually clears a
    transient "Invalid Request" throttle without re-downloading everything.
    """
    import time

    import max_pf
    login = {"platform": "fantrax", "league_id": league_id}
    last = None
    for attempt in range(retries):
        try:
            return max_pf.run(login, weeks=weeks, methodology=methodology, nash=nash,
                              cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 - retry transient Fantrax throttles
            last = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last


def fetch_success_labels(config, *, weeks=None, methodology="hindsight", nash=True,
                         write: bool = True):
    """Run ``max_pf`` for each configured league; return a tidy team-season success table.

    Columns: ``league_id, team_id, name, periods, actual_pf, m1_pf, m2_pf, m3_pf, lineup_gap``
    (``lineup_gap = m1_pf - actual_pf`` = optimal-vs-realized category wins left on the table).
    """
    import pandas as pd

    league_ids = config.get("fantasy.league_ids", []) or []
    cache_dir = config.get("fantasy.cache_dir", ".cache")
    rows, failed = [], []
    for lid in league_ids:
        try:
            # Per-league isolation + retry/backoff: a full-season run makes many team×period
            # roster calls and Fantrax rate-limits them ("Invalid Request"); a warm cache + backoff
            # gets it through, and one league failing must not lose the others.
            res = _run_league(lid, weeks=weeks, methodology=methodology, nash=nash,
                              cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 - surface, don't abort the batch
            failed.append((lid, f"{type(exc).__name__}: {exc}"[:160]))
            continue
        for t in res:
            rows.append({
                "league_id": lid, "team_id": t.team_id, "name": t.name,
                "periods": t.periods, "actual_pf": t.actual_pf,
                "m1_pf": t.m1_pf, "m2_pf": t.m2_pf, "m3_pf": t.m3_pf,
            })
    if failed:
        for lid, msg in failed:
            print(f"[fantasy] WARNING: league {lid} failed — {msg}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["primary_pf"] = df["m2_pf"]                    # grade off M2 (both teams optimize)
        df["lineup_gap"] = df["m2_pf"] - df["actual_pf"]  # left on the table vs both-optimize ceiling

    if write and not df.empty:
        path = DATA_PROCESSED / "fantrax_success.parquet"
        ensure_dir(DATA_PROCESSED)
        write_parquet(df, path)
        _write_manifest(df, league_ids, methodology, nash, weeks, path)
    return df


def _write_manifest(df, league_ids, methodology, nash, weeks, path) -> None:
    ensure_dir(MANIFEST_DIR)
    manifest = {
        "name": "fantrax_success",
        "source": "max_pf",
        "note": "optimal-lineup 9-cat points-for (success labels) per fantasy team",
        "league_ids": list(league_ids),
        "leagues_succeeded": sorted(df["league_id"].unique().tolist()),
        "methodology": methodology,
        "nash": bool(nash),
        "weeks": "full_season" if weeks is None else str(weeks),
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "cache_path": str(path),
        "sha256": sha256_file(path),
    }
    with open(MANIFEST_DIR / "fantrax_success.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
