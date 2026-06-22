"""Snap-count aggregation with the pfr→gsis crosswalk (Stage 2 helper).

nflverse ``import_snap_counts`` keys players on ``pfr_player_id`` (Pro-Football-Reference),
**not** the gsis ``player_id`` used everywhere else in the pipeline. This module crosswalks
snaps onto the gsis id via ``import_ids`` (``pfr_id`` ↔ ``gsis_id``) and aggregates per-game
offensive snaps into a season **snap share** — the second eligibility-cutoff dimension
(PROJECT_PLAN §4.2, snaps available 2012+).

Verified on real data (see PROMPT_LOG): the crosswalk maps 100% of RB snap rows for a recent
season, and regular season is ``game_type == 'REG'`` (playoff rounds are WC/DIV/CON/SB).

``snap_share`` is **snaps-weighted across games the player was on the field**: it equals
``Σ offense_snaps / Σ team_offense_snaps`` where a game's team total is recovered as
``offense_snaps / offense_pct`` (only games with ``offense_pct`` > 0 contribute, so the share
reflects usage when active — consistent with PPG being per game-played).
"""

from __future__ import annotations

from ..utils.io import DATA_RAW, read_parquet


def crosswalk_pfr_to_gsis(ids):
    """Return a deduped ``pfr_id -> gsis_id`` map (drops rows missing either id)."""
    cols = [c for c in ("pfr_id", "gsis_id") if c in ids.columns]
    if len(cols) < 2:
        return ids.iloc[0:0][cols] if cols else ids.iloc[0:0]
    cw = ids[["pfr_id", "gsis_id"]].dropna()
    return cw.drop_duplicates("pfr_id")


def build_player_season_snaps(snap_counts, ids, *, regular_season_only: bool = True):
    """Aggregate per-game snaps into ``(player_id, season, snap_share, snaps, snaps_per_game)``.

    ``player_id`` is the gsis id (after crosswalk), matching the rest of the pipeline. Rows
    whose ``pfr_player_id`` has no gsis mapping are dropped (negligible for RBs; verified 100%
    mapped on a recent season).
    """
    import numpy as np
    import pandas as pd

    sc = snap_counts
    if regular_season_only and "game_type" in sc.columns:
        sc = sc[sc["game_type"] == "REG"]
    if sc.empty:
        return pd.DataFrame(columns=["player_id", "season", "snap_share", "snaps",
                                     "snaps_per_game"])

    cw = crosswalk_pfr_to_gsis(ids)
    sc = sc.merge(cw, left_on="pfr_player_id", right_on="pfr_id", how="left")
    sc = sc.dropna(subset=["gsis_id"]).rename(columns={"gsis_id": "player_id"})

    # Per-game team offensive snaps, recoverable only when the player took >0 snaps.
    pct = sc["offense_pct"]
    sc = sc.assign(
        _team_snaps=np.where(pct > 0, sc["offense_snaps"] / pct, np.nan),
        _active=(pct > 0).astype(int),
    )
    grp = sc.groupby(["player_id", "season"], as_index=False)
    agg = grp.agg(
        snaps=("offense_snaps", "sum"),
        _team_snaps=("_team_snaps", "sum"),
        _active_games=("_active", "sum"),
    )
    agg["snap_share"] = np.where(agg["_team_snaps"] > 0, agg["snaps"] / agg["_team_snaps"],
                                 np.nan)
    agg["snaps_per_game"] = np.where(agg["_active_games"] > 0,
                                     agg["snaps"] / agg["_active_games"], np.nan)
    return agg[["player_id", "season", "snap_share", "snaps", "snaps_per_game"]]


def load_player_season_snaps(*, regular_season_only: bool = True):
    """Read the raw snap_counts + ids caches and return the aggregated snap table (or None)."""
    snap_path = DATA_RAW / "snap_counts.parquet"
    ids_path = DATA_RAW / "ids.parquet"
    if not (snap_path.exists() and ids_path.exists()):
        return None
    return build_player_season_snaps(
        read_parquet(snap_path), read_parquet(ids_path),
        regular_season_only=regular_season_only)
