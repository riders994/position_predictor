"""Stage 2 — assemble a wide player-season table from the long ESPN stats.

``load_nba_player_season_stats`` returns *long* rows (one per player × stat); ESPN gives a single
season row per player (no multi-team splits — verified). This pivots to **one row per
(athlete_id, season)** with clean column names, parses the ``"made-attempted"`` combo fields out of
``display_value``, and flags eligibility (rotation players only). Style features come next
(features/build.py).
"""
from __future__ import annotations

from ..utils.io import DATA_INTERIM, DATA_RAW, ensure_dir, read_parquet, write_parquet

# ESPN per-game / ratio scalars (value column) -> clean names.
SCALAR_RENAME = {
    "avgPoints": "pts_pg", "avgRebounds": "reb_pg", "avgOffensiveRebounds": "oreb_pg",
    "avgDefensiveRebounds": "dreb_pg", "avgAssists": "ast_pg", "avgSteals": "stl_pg",
    "avgBlocks": "blk_pg", "avgTurnovers": "tov_pg", "avgFouls": "pf_pg",
    "avgMinutes": "min_pg", "gamesPlayed": "gp", "gamesStarted": "gs",
    "fieldGoalPct": "fg_pct", "freeThrowPct": "ft_pct", "threePointFieldGoalPct": "fg3_pct",
    "assistTurnoverRatio": "ast_tov", "stealTurnoverRatio": "stl_tov",
    "scoringEfficiency": "scoring_eff", "shootingEfficiency": "shooting_eff",
}
# "made-attempted" combo fields (value is null; parse display_value "M-A") -> (made, attempted).
COMBO_RENAME = {
    "avgFieldGoalsMade-avgFieldGoalsAttempted": ("fgm_pg", "fga_pg"),
    "avgThreePointFieldGoalsMade-avgThreePointFieldGoalsAttempted": ("fg3m_pg", "fg3a_pg"),
    "avgFreeThrowsMade-avgFreeThrowsAttempted": ("ftm_pg", "fta_pg"),
}
IDENTITY = {
    "athlete_display_name": "player_name",
    "athlete_position_abbreviation": "position",
    "team_display_name": "team",
}


def _parse_combo(series):
    """Parse a 'M-A' display_value series into (made, attempted) float frames."""
    import numpy as np
    import pandas as pd
    parts = series.astype(str).str.split("-", n=1, expand=True)
    made = pd.to_numeric(parts[0], errors="coerce")
    att = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan
    return made, att


def build_dataset(config, *, write: bool = True):
    """Pivot the long player-season stats to a wide per-(athlete, season) table.

    Eligibility (rotation players) uses ``eligibility.min_minutes_pg`` / ``min_games`` from config.
    Returns the wide DataFrame; writes it to ``data/interim`` when ``write``.
    """
    df = read_parquet(DATA_RAW / "player_season_stats.parquet")
    keys = ["athlete_id", "season"]

    # scalar stats (numeric value present)
    scal = df[df["value"].notna() & df["stat_name"].isin(SCALAR_RENAME)]
    wide = scal.pivot_table(index=keys, columns="stat_name", values="value", aggfunc="first")
    wide = wide.rename(columns=SCALAR_RENAME)

    # combo "made-attempted" fields parsed from display_value
    for raw, (made_col, att_col) in COMBO_RENAME.items():
        rows = df[df["stat_name"] == raw].drop_duplicates(keys).set_index(keys)
        if rows.empty:
            continue
        made, att = _parse_combo(rows["display_value"])
        wide = wide.join(made.rename(made_col)).join(att.rename(att_col))

    # identity (one value per athlete-season)
    ident = (df.sort_values(keys).groupby(keys)[list(IDENTITY)].first().rename(columns=IDENTITY))
    out = ident.join(wide).reset_index()

    # eligibility: rotation players only (archetypes are about role, not deep-bench noise)
    min_mpg = float(config.get("eligibility.min_minutes_pg", 15.0))
    min_gp = int(config.get("eligibility.min_games", 20))
    out["eligible"] = (out["min_pg"].fillna(0) >= min_mpg) & (out["gp"].fillna(0) >= min_gp)

    out = out.sort_values(keys).reset_index(drop=True)
    if write:
        path = DATA_INTERIM / "nba_player_seasons.parquet"
        ensure_dir(DATA_INTERIM)
        write_parquet(out, path)
    return out
