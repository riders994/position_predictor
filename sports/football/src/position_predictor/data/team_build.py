"""Stage 2 (team variant) — build the DST team-season dataset.

Fantasy DST scores a team's defense/special-teams **unit**, not an individual player, so this
mirrors :mod:`data.build`'s shape (aggregate weekly box scores -> season, attach next-season
target) but keyed on **team abbreviation** instead of ``player_id`` — deliberately not reusing
:func:`data.build.aggregate_player_seasons` / :func:`attach_next_season_target`, since a team has
no injury/retirement/eligibility concept (every team plays every season) and the raw inputs
(``team_stats``, ``schedules``) are a different shape than the player ``weekly`` frame.

Two things ``team_stats`` alone can't give us (see the DST field-mapping table in the
implementation plan / ``data/scoring.py``'s ``compute_dst_points`` docstring):

- **Points allowed** — not a ``team_stats`` column at all; it's the opponent's score in that
  game, from ``schedules``.
- **Blocked FG/XP by this team's defense** — ``team_stats`` only tracks blocks *suffered* by a
  team's own kicker (an offensive stat), not blocks *made* by its defense. This team's defensive
  blocks = its opponent's ``fg_blocked``/``pat_blocked`` in that same game, via a self-join.

Writes to the same ``data/interim/{sport}_{position}_player_seasons.parquet`` naming convention
:mod:`data.build` uses, so :func:`features.build.build_features` (with a ``DST`` branch) and
:func:`eval.projection.project_position` work unmodified — the latter only cares about the
``player_id, player_name, position, target_ppg_next`` + feature-column schema, not what kind of
entity ``player_id`` actually is. Here ``player_id`` is the canonical team abbreviation.
"""
from __future__ import annotations

from ..utils.io import DATA_INTERIM, DATA_RAW, read_parquet, write_parquet
from .build import BuildResult, label_eligibility_grid
from .scoring import compute_dst_points

# nflverse team abbreviations change on relocation; canonicalize so a franchise's history is one
# continuous entity for trajectory/regression-to-mean features (confirmed 2026-07-06: OAK->LV
# 2020, SD->LAC 2017, STL->LA 2016).
CANONICAL_TEAM = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

TEAM_STATS_SUM_COLS = [
    "def_sacks", "def_interceptions", "fumble_recovery_opp", "fumble_recovery_tds",
    "def_tds", "def_safeties", "points_allowed", "blocked_fg_defense", "blocked_xp_defense",
    "fantasy_points_dst",
]


def canonicalize_team(team):
    return CANONICAL_TEAM.get(team, team)


def _team_game_frame(team_stats, schedules):
    """One row per (team, season, week): team_stats columns + points_allowed + blocked_*_defense."""
    import pandas as pd

    ts = team_stats.copy()
    ts["team"] = ts["team"].map(canonicalize_team)
    ts["opponent_team"] = ts["opponent_team"].map(canonicalize_team)
    if "season_type" in ts.columns:
        ts = ts[ts["season_type"] == "REG"]

    # Points allowed: reshape schedules (one row per game) into one row per team-perspective.
    sched = schedules[["season", "week", "home_team", "away_team",
                       "home_score", "away_score"]].copy()
    home = sched.rename(columns={"home_team": "team", "away_team": "opponent_team",
                                 "away_score": "points_allowed"})[
        ["season", "week", "team", "opponent_team", "points_allowed"]]
    away = sched.rename(columns={"away_team": "team", "home_team": "opponent_team",
                                 "home_score": "points_allowed"})[
        ["season", "week", "team", "opponent_team", "points_allowed"]]
    pa = pd.concat([home, away], ignore_index=True)
    pa["team"] = pa["team"].map(canonicalize_team)
    pa["opponent_team"] = pa["opponent_team"].map(canonicalize_team)

    out = ts.merge(pa[["season", "week", "team", "points_allowed"]],
                   on=["season", "week", "team"], how="left")

    # This team's defensive blocks = the opponent's own fg_blocked/pat_blocked that same game.
    blocks = ts[["season", "week", "team", "fg_blocked", "pat_blocked"]].rename(
        columns={"team": "opponent_team", "fg_blocked": "blocked_fg_defense",
                 "pat_blocked": "blocked_xp_defense"})
    out = out.merge(blocks, on=["season", "week", "opponent_team"], how="left")
    out[["blocked_fg_defense", "blocked_xp_defense"]] = (
        out[["blocked_fg_defense", "blocked_xp_defense"]].fillna(0))
    return out


def aggregate_team_seasons(team_stats, schedules):
    """Aggregate per-team-game rows into one row per ``(team, season)``."""
    import numpy as np

    game = _team_game_frame(team_stats, schedules)
    game["fantasy_points_dst"] = compute_dst_points(game)

    sum_cols = [c for c in TEAM_STATS_SUM_COLS if c in game.columns]
    grouped = game.groupby(["team", "season"], as_index=False)
    out = grouped[sum_cols].sum(numeric_only=True)
    games = grouped.size().rename(columns={"size": "games"})
    out = out.merge(games, on=["team", "season"], how="left")

    out["ppr_points"] = out["fantasy_points_dst"]
    out["ppg"] = np.where(out["games"] > 0, out["ppr_points"] / out["games"], np.nan)
    out["player_id"] = out["team"]
    out["position"] = "DST"
    out["position_group"] = "DST"
    return out


def attach_team_names(season_df, teams):
    """Join the full team name (``player_name``) from the ``teams`` reference table."""
    if teams is None or len(teams) == 0 or "team_abbr" not in teams.columns:
        season_df["player_name"] = season_df["team"]
        return season_df
    names = teams[["team_abbr", "team_name"]].copy()
    names["team_abbr"] = names["team_abbr"].map(canonicalize_team)
    names = names.drop_duplicates("team_abbr").rename(
        columns={"team_abbr": "team", "team_name": "player_name"})
    out = season_df.merge(names, on="team", how="left")
    out["player_name"] = out["player_name"].fillna(out["team"])
    return out


def attach_next_season_target(season_df, *, horizon: int = 1, latest_season: int | None = None):
    """Attach season *N+horizon* PPG as the target. No injury/retirement concept for a team —
    every team plays every season (relocations are already canonicalized), so this is a plain
    shift, not the player pipeline's active/injured/retired/censored classification."""
    import numpy as np

    df = season_df
    if latest_season is None:
        latest_season = int(df["season"].max())
    nxt = df[["team", "season", "ppg", "ppr_points", "games"]].copy()
    nxt["season"] = nxt["season"] - horizon
    nxt = nxt.rename(columns={"ppg": "target_ppg_next", "ppr_points": "ppr_points_next",
                              "games": "games_next"})
    out = df.merge(nxt, on=["team", "season"], how="left")
    censored = (out["season"] + horizon) > latest_season
    out.loc[censored, ["target_ppg_next", "ppr_points_next", "games_next"]] = np.nan
    return out


def build_dataset(config, *, position: str = "DST", horizon: int | None = None,
                  write: bool = True):
    """Orchestrate the DST build: raw ``team_stats``/``schedules`` -> team-season table."""
    horizon = horizon if horizon is not None else int(config.get("target.predict_horizon", 1))
    latest_season = config.get("data.latest_completed_season")

    team_stats = read_parquet(DATA_RAW / "team_stats.parquet")
    schedules = read_parquet(DATA_RAW / "schedules.parquet")
    teams = _maybe_read("teams.parquet")

    df = aggregate_team_seasons(team_stats, schedules)
    df = attach_team_names(df, teams)
    data_max_season = int(df["season"].max())
    latest_season = (data_max_season if latest_season is None
                     else min(int(latest_season), data_max_season))
    df = attach_next_season_target(df, horizon=horizon, latest_season=latest_season)
    # eval/experiment.py reads eligible_next__g{N} columns unconditionally (produced by the same
    # generic helper the player pipeline uses) — DST has no games-played *eligibility* concept
    # (a team plays every scheduled game; there's no partial-season "backup DST"), but the column
    # still needs to exist. games_grid is just [1] in football_dst.yaml, i.e. no real filtering.
    games_grid = config.get("eligibility.candidate_games_played", [1])
    df = label_eligibility_grid(df, games_grid)
    df = df.sort_values(["season", "team"]).reset_index(drop=True)

    if not write:
        return df

    out_path = DATA_INTERIM / f"{config.stem()}_player_seasons.parquet"
    write_parquet(df, out_path)
    seasons = sorted(int(s) for s in df["season"].unique())
    result = BuildResult(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        seasons=seasons,
        n_with_target=int(df["target_ppg_next"].notna().sum()),
        status_counts={},  # no injury/retirement/censoring concept for a team-level entity
        path=str(out_path),
    )
    return df, result


def _maybe_read(name: str):
    path = DATA_RAW / name
    return read_parquet(path) if path.exists() else None
