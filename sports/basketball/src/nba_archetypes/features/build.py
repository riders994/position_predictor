"""Stage 3 — engineer play-STYLE features for archetype clustering.

Style, not volume: rates (per-36), shooting splits, and tendency ratios so a star and a role player
who *play the same way* land near each other. Features are **z-scored within season** over eligible
players (era-relative comparability across the 2013+ window — PROJECT_PLAN §2.2), and each season is
tagged with its style era (E1/E2/E3). Phase 1 clusters the ``*_z`` features for eligible E2+E3 rows.
"""
from __future__ import annotations

import json

import numpy as np

from ..utils.io import DATA_INTERIM, DATA_PROCESSED, ensure_dir, read_parquet, write_parquet

# feature -> block (drives the block map; mirrors football's block→columns contract)
BLOCKS = {
    "scoring": ["pts36", "fga36", "fta36"],
    "shot_profile": ["fg3a_rate", "ft_rate", "fg3a36", "fg_pct", "fg3_pct", "ft_pct",
                     "scoring_eff", "shooting_eff"],
    "playmaking": ["ast36", "tov36", "ast_tov"],
    "rebounding": ["oreb36", "dreb36", "oreb_rate"],
    "defense": ["stl36", "blk36"],
}
IDENTITY = ["athlete_id", "season", "player_name", "position", "team", "gp", "min_pg", "eligible"]


def _div(num, den):
    import numpy as np
    return num / den.where(den != 0) if hasattr(den, "where") else np.nan


def _assign_eras(df, eras):
    """Map each season to its era name + the use-for-archetypes flag (from config)."""
    season_era, use = {}, {}
    for e in eras:
        start, end = int(e["start_season"]), e.get("end_season")
        end = 9999 if end is None else int(end)
        for s in range(start, end + 1):
            season_era[s] = e["name"]
            use[s] = bool(e.get("use_for_archetypes", False))
    df = df.copy()
    df["era"] = df["season"].map(season_era)
    df["era_use_for_archetypes"] = df["season"].map(use).fillna(False)
    return df


def build_features(config, *, write: bool = True):
    """Compute style features + within-season z-scores; tag eras. Returns the processed frame."""
    import pandas as pd

    df = read_parquet(DATA_INTERIM / "nba_player_seasons.parquet")
    m36 = 36.0 / df["min_pg"].where(df["min_pg"] != 0)  # per-36 multiplier

    out = df.copy()
    # scoring + usage rates
    for src, dst in [("pts_pg", "pts36"), ("fga_pg", "fga36"), ("fta_pg", "fta36"),
                     ("fg3a_pg", "fg3a36"), ("ast_pg", "ast36"), ("tov_pg", "tov36"),
                     ("oreb_pg", "oreb36"), ("dreb_pg", "dreb36"), ("stl_pg", "stl36"),
                     ("blk_pg", "blk36")]:
        out[dst] = df[src] * m36
    # shot profile (style tendencies)
    out["fg3a_rate"] = _div(df["fg3a_pg"], df["fga_pg"])      # 3-pt reliance
    out["ft_rate"] = _div(df["fta_pg"], df["fga_pg"])         # rim/contact proxy
    out["oreb_rate"] = _div(df["oreb_pg"], (df["oreb_pg"] + df["dreb_pg"]))  # off-reb tendency
    # carried-through shooting splits / ratios already on df: fg_pct, fg3_pct, ft_pct,
    # scoring_eff, shooting_eff, ast_tov

    # fg3_pct is unreliable for tiny 3-pt samples — a big who hits 1 of 2 gets an extreme z that used
    # to spawn a degenerate "non-shooting center" cluster (fg3_pct_z ~ +4). Neutralize it below a
    # season 3PA-volume floor (-> NaN -> 0 z at fit); shooting *volume* is still captured by
    # fg3a_rate / fg3a36, so genuine non-shooters remain distinguishable.
    min_3pa = float(config.get("features.min_3pa_for_fg3_pct", 30))
    out.loc[(df["fg3a_pg"] * df["gp"]).fillna(0) < min_3pa, "fg3_pct"] = np.nan

    feat_cols = [c for blk in BLOCKS.values() for c in blk]

    # z-score within season over ELIGIBLE players (the clustering population)
    elig = out["eligible"].fillna(False)
    z_cols = []
    for c in feat_cols:
        stats = out.loc[elig].groupby("season")[c]
        mean = out["season"].map(stats.mean())
        std = out["season"].map(stats.std()).replace(0, pd.NA)
        out[f"{c}_z"] = ((out[c] - mean) / std).astype(float)
        z_cols.append(f"{c}_z")

    out = _assign_eras(out, config.get("eras", []))
    keep = IDENTITY + ["era", "era_use_for_archetypes"] + feat_cols + z_cols
    out = out[[c for c in keep if c in out.columns]].sort_values(["season", "athlete_id"])
    out = out.reset_index(drop=True)

    if write:
        ensure_dir(DATA_PROCESSED)
        write_parquet(out, DATA_PROCESSED / "nba_archetype_features.parquet")
        block_map = {blk: [f"{c}_z" for c in cols] for blk, cols in BLOCKS.items()}
        with open(DATA_PROCESSED / "nba_archetype_feature_blocks.json", "w") as fh:
            json.dump(block_map, fh, indent=2)
    return out
