"""Stage 2 (rookie variant) — build the rookie-year dataset.

Every other model in this project predicts season *N+1* from season *N* NFL stats. A true
incoming rookie has **zero prior NFL games**, so they have no row in that pipeline at all —
this module is a structurally different prediction: **draft-day-known information -> that same
rookie season's production**, not a next-season roll-forward. Inputs are draft capital
(round/overall pick — historically one of the single best predictors of rookie output), combine
athletic testing (optional/sparse), and landing-spot context (incumbent same-position workload on
the drafting team) — not prior NFL production, because there isn't any yet.

One row per **drafted player at the position** (from ``draft_picks``, not "players who happened
to debut") — the label is that player's production in their *own* draft-year season, computed
from ``weekly`` and **zero-filled if they had no games that year** (a bust/inactive rookie is a
real, informative outcome for a draft-day decision, not a row to exclude). This deliberately
differs from filtering to "players who debuted as rookies" — that would bias the training set
toward already-successful debuts and hide the very bust risk a live draft needs to weigh.

Writes to the same ``data/interim/{sport}_{position}_player_seasons.parquet`` naming convention
:mod:`data.build` uses (schema: ``player_id, player_name, position, season,`` target columns),
so :func:`features.build.build_features` (with an ``is_rookie`` branch) and
:func:`eval.projection.project_position` work unmodified.
"""
from __future__ import annotations

from ..utils.io import DATA_INTERIM, DATA_RAW, read_parquet, write_parquet
from .build import BuildResult, aggregate_player_seasons, label_eligibility_grid
from .team_build import canonicalize_team

# PFR draft-class position -> our fantasy position (mirrors eval/redraft.py's
# _DRAFT_POS_TO_FANTASY — duplicated rather than imported to keep this data-layer module
# independent of the eval layer; keep the two in sync if either changes).
_DRAFT_POS_TO_FANTASY = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}

# draft_picks.team uses PFR-style abbreviations, which differ from nflverse's convention (used
# everywhere else in this project — weekly.recent_team, team_stats.team, etc.) for 9 teams.
# Confirmed by diffing the two tables' team-code sets directly, not guessed. Composed with
# team_build.canonicalize_team (handles OAK->LV/SD->LAC/STL->LA relocations) so a draft pick's
# team and that team's *historical* weekly rows always resolve to the same modern label.
_PFR_TO_NFLVERSE_TEAM = {
    "GNB": "GB", "KAN": "KC", "LAR": "LA", "LVR": "LV", "NOR": "NO",
    "NWE": "NE", "SDG": "SD", "SFO": "SF", "TAM": "TB",
}


def _canonical_team(team):
    return canonicalize_team(_PFR_TO_NFLVERSE_TEAM.get(team, team))

# Season-N opportunity stat used to value a rookie's landing-spot competition, per position —
# same currency football_{rb,wr,qb,te}.yaml's features.offseason_workload_col already uses.
WORKLOAD_COL = {"QB": "attempts", "RB": "touches", "WR": "targets", "TE": "targets"}

COMBINE_COLS = ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]


def _combine_by_gsis(combine, ids):
    """Combine testing results keyed by ``gsis_id`` (join via ``ids.parquet``'s ``pfr_id``
    crosswalk). Returns ``None`` if either input is missing — combine is optional/best-effort."""
    if combine is None or ids is None:
        return None
    if "pfr_id" not in ids.columns or "gsis_id" not in ids.columns:
        return None
    xwalk = ids.dropna(subset=["gsis_id", "pfr_id"])[["gsis_id", "pfr_id"]].drop_duplicates("pfr_id")
    cols = [c for c in COMBINE_COLS if c in combine.columns]
    c = combine.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")[["pfr_id", *cols]]
    out = xwalk.merge(c, on="pfr_id", how="inner").drop(columns="pfr_id")
    return out.rename(columns={"gsis_id": "player_id"}).drop_duplicates("player_id")


def _room_context(player_seasons, *, position: str, workload_col: str):
    """Per (team, season) same-position season-total workload — the incumbent competition a
    rookie lands into, keyed by the *prior* season (joined by the caller as ``season - 1``)."""
    pos = player_seasons[player_seasons["position"] == position]
    room = pos.groupby(["recent_team", "season"], as_index=False).agg(
        room_prior_workload=(workload_col, "sum"), room_size=("player_id", "nunique"))
    return room


def build_dataset(config, *, position: str | None = None, write: bool = True):
    """Orchestrate the rookie build: raw ``weekly``/``draft_picks``/``combine`` -> rookie-season
    table keyed by ``(player_id, draft_season)``."""
    import numpy as np

    position = (position or config.require("experiment.position")).upper()
    workload_col = WORKLOAD_COL[position]

    weekly = read_parquet(DATA_RAW / "weekly.parquet")
    draft_picks = read_parquet(DATA_RAW / "draft_picks.parquet")
    combine = _maybe_read("combine.parquet")
    ids = _maybe_read("ids.parquet")

    # Season-level player production — reused for both the rookie-year label (this player, their
    # own draft season) and the room-context computation (same-position peers, prior season).
    # Canonicalize recent_team so a historical STL/OAK/SD-era row matches the same modern label
    # a draft pick's (also canonicalized) team resolves to.
    seasons = aggregate_player_seasons(weekly, scoring="RTSPORTS")
    seasons["recent_team"] = seasons["recent_team"].map(_canonical_team)

    picks = draft_picks.dropna(subset=["gsis_id"]).copy()
    picks["fantasy_position"] = picks["position"].map(_DRAFT_POS_TO_FANTASY)
    picks = picks[picks["fantasy_position"] == position][
        ["gsis_id", "season", "round", "pick", "team", "age", "pfr_player_name"]
    ].rename(columns={"gsis_id": "player_id", "season": "draft_season",
                      "team": "draft_team", "pfr_player_name": "player_name"})
    picks["draft_team"] = picks["draft_team"].map(_canonical_team)
    picks = picks.drop_duplicates("player_id")  # a player is drafted exactly once

    # Rookie-year production: this player's stats in their OWN draft season, zero-filled if none
    # (an inactive/bust rookie is a real, informative label — not excluded). EXCEPT a draft class
    # whose season hasn't been *completed* yet (season > data.latest_completed_season) — that's
    # not a real zero, it's unknown, and must stay NaN or a live board's own incoming class would
    # corrupt walk-forward validation (evaluated against a fake "everyone busted" label). This
    # doesn't affect live projection: project_position only reads FEATURES for the board season,
    # never its target.
    prod = seasons[["player_id", "season", "games", "ppr_points"]].rename(
        columns={"season": "draft_season"})
    out = picks.merge(prod, on=["player_id", "draft_season"], how="left")
    latest_completed = config.get("data.latest_completed_season")
    incomplete = (out["draft_season"] > int(latest_completed)) if latest_completed is not None \
        else False
    out.loc[~incomplete, ["games", "ppr_points"]] = out.loc[~incomplete,
                                                            ["games", "ppr_points"]].fillna(0.0)
    out["ppg"] = np.where(out["games"] > 0, out["ppr_points"] / out["games"], 0.0)
    out.loc[incomplete, "ppg"] = np.nan

    # Landing-spot context: incumbent same-position competition on the drafting team, the season
    # BEFORE the rookie arrives (proven proof of a crowded room vs. a clear path to touches).
    room = _room_context(seasons, position=position, workload_col=workload_col)
    room = room.rename(columns={"recent_team": "draft_team", "season": "_prior"})
    out["_prior"] = out["draft_season"] - 1
    out = out.merge(room, on=["draft_team", "_prior"], how="left")
    out[["room_prior_workload", "room_size"]] = out[["room_prior_workload", "room_size"]].fillna(0.0)
    out = out.drop(columns="_prior")

    # Combine testing (optional, sparse) — has_combine coverage flag added in features/build.py.
    combine_by_gsis = _combine_by_gsis(combine, ids)
    if combine_by_gsis is not None:
        out = out.merge(combine_by_gsis, on="player_id", how="left")

    out["position"] = position
    out["position_group"] = position
    out = out.rename(columns={"draft_season": "season"})
    # Same-season prediction (not a next-season shift): the label IS this season's own outcome.
    out["target_ppg_next"] = out["ppg"]
    out["ppr_points_next"] = out["ppr_points"]
    out["games_next"] = out["games"]

    # COVID 2020 is excluded as an anomalous season everywhere else in this project (empty
    # stadiums, opt-outs, depleted rosters) — same here: drop that draft class's rookie-year
    # outcomes from training rather than teach the model 2020's distorted per-game rates.
    excl = {int(s) for s in config.get("data.exclude_seasons", [])}
    if excl:
        out = out[~out["season"].isin(excl)].reset_index(drop=True)

    games_grid = config.get("eligibility.candidate_games_played", [1])
    out = label_eligibility_grid(out, games_grid)
    out = out.sort_values(["season", "pick"]).reset_index(drop=True)

    if not write:
        return out

    out_path = DATA_INTERIM / f"{config.stem()}_player_seasons.parquet"
    write_parquet(out, out_path)
    seasons_list = sorted(int(s) for s in out["season"].unique())
    result = BuildResult(
        n_rows=int(out.shape[0]),
        n_cols=int(out.shape[1]),
        seasons=seasons_list,
        n_with_target=int(out["target_ppg_next"].notna().sum()),
        status_counts={},  # no injury/retirement/censoring concept for a draft-day cohort
        path=str(out_path),
    )
    return out, result


def _maybe_read(name: str):
    path = DATA_RAW / name
    return read_parquet(path) if path.exists() else None
