"""Stage 2 — build the player-season dataset (join sources, attach target & labels).

Turns the raw nflverse caches (Stage 1) into one tidy **player-season table** keyed by
``(player_id, season)`` for a single *fantasy* position, with:

- season-*N* production aggregated from weekly box scores (games, PPR points, PPG, volume),
- player attributes from the seasonal rosters, including **age-at-season-start** computed
  from birth date (not the roster's fuzzy ``age`` field),
- the supervised **target** (``target_ppg_next``) = season *N+1* PPR PPG of the same player,
- next-season **availability** (``games_next``, 0–17) for the eligibility step (§4.2) and the
  availability model (§4.3), with retirement folded in as a zero-availability outcome.

Key methodology decisions (verified against real nflverse data — see PROMPT_LOG):

- **Games played** = count of a player's regular-season weekly rows. Verified to match the
  nflverse seasonal ``games`` column exactly (100% over a 2-season sample).
- **Fantasy position eligibility, not NFL designation.** A player is kept if they are
  fantasy-eligible at the target position. Primary signal = Sleeper ``fantasy_positions``
  (joined on ``gsis_id``); fallback = nflverse ``position_group`` (which already folds FB→RB)
  for any player Sleeper doesn't cover (notably pre-2010). An ``eligibility_source`` column
  records which was used. Sleeper is a *current* snapshot, so eligibility is treated as a
  career-level attribute applied to all of a player's seasons.
- **Missing a full season ≠ gone.** A player who misses an entire season *N+1* has **no**
  weekly rows that year (verified: J.K. Dobbins, 2021). We therefore distinguish, using
  *future* seasons:
    * ``active``      — played in *N+1* (``games_next`` > 0; PPG target defined).
    * ``injured_out`` — no *N+1* rows but the player appears in a later season → temporarily
      out; ``games_next`` = 0 (a real availability-zero example for §4.3).
    * ``retired``     — no *N+1* rows and the player never appears again → treated as a type
      of injury: ``games_next`` = 0, but ``retired_next`` flags it. We do **not** model
      "un-retirement" (no rows are created after a player's last season).
    * ``censored``    — *N+1* is beyond the last completed season → unknown future; all
      ``_next`` labels are NaN and the row is excluded from supervised training.

See docs/PROJECT_PLAN.md §3–§4 and docs/data_dictionary.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.io import DATA_INTERIM, DATA_RAW, read_parquet, write_parquet
from .snaps import load_player_season_snaps

# Per-game box-score columns summed to a season total when present. Kept defensive:
# only columns actually in the weekly frame are aggregated, so this survives schema drift.
SEASON_SUM_COLS = [
    "fantasy_points",
    "fantasy_points_ppr",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_first_downs",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
    "rushing_epa",            # additive; powers EPA-per-play efficiency features (1999+)
    "receiving_epa",
]

# Identity columns carried from the weekly frame (last non-null per player-season).
IDENTITY_COLS = ["player_name", "player_display_name", "position", "position_group", "recent_team"]

# How a player's next season is classified (see module docstring).
STATUS_ACTIVE, STATUS_INJURED, STATUS_RETIRED, STATUS_CENSORED = (
    "active", "injured_out", "retired", "censored")


@dataclass(frozen=True)
class BuildResult:
    n_rows: int
    n_cols: int
    seasons: list[int]
    n_with_target: int
    status_counts: dict
    path: str


def aggregate_player_seasons(weekly, *, regular_season_only: bool = True):
    """Aggregate per-game weekly rows into one row per ``(player_id, season)`` (all positions).

    Position filtering happens later via :func:`resolve_fantasy_eligibility`, so this keeps
    every player and carries ``position`` / ``position_group`` forward for that step.
    """
    import numpy as np
    import pandas as pd

    df = weekly
    if regular_season_only and "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]

    if df.empty:
        return pd.DataFrame(columns=["player_id", "season", "games", "ppr_points", "ppg"])

    sum_cols = [c for c in SEASON_SUM_COLS if c in df.columns]
    grouped = df.groupby(["player_id", "season"], as_index=False)
    out = grouped[sum_cols].sum(numeric_only=True)

    # Games played = number of (regular-season) weeks the player appears for that season.
    games = grouped.size().rename(columns={"size": "games"})
    out = out.merge(games, on=["player_id", "season"], how="left")

    id_cols = [c for c in IDENTITY_COLS if c in df.columns]
    if id_cols:
        sort_cols = ["player_id", "season"] + (["week"] if "week" in df.columns else [])
        ident = (
            df.sort_values(sort_cols)
            .groupby(["player_id", "season"], as_index=False)[id_cols]
            .last()
        )
        out = out.merge(ident, on=["player_id", "season"], how="left")

    out["ppr_points"] = out["fantasy_points_ppr"] if "fantasy_points_ppr" in out.columns else np.nan
    out["touches"] = out.get("carries", 0) + out.get("receptions", 0)
    out["ppg"] = np.where(out["games"] > 0, out["ppr_points"] / out["games"], np.nan)
    return out


def resolve_fantasy_eligibility(season_df, sleeper, target_position: str):
    """Flag rows fantasy-eligible at ``target_position`` and record the source.

    Primary signal: Sleeper ``fantasy_positions`` (career-level, joined on the gsis
    ``player_id``). Fallback: nflverse ``position_group`` (FB already folded into RB) for any
    player Sleeper doesn't cover. Adds ``is_eligible`` (bool) and ``eligibility_source``
    (``"sleeper"`` | ``"nflverse_fallback"``). Does not filter — callers decide.
    """
    import pandas as pd

    df = season_df.copy()
    sleeper_has = {}
    if sleeper is not None and len(sleeper) and "gsis_id" in sleeper.columns:
        s = sleeper.dropna(subset=["gsis_id"])
        for gsis, fp in zip(s["gsis_id"], s["fantasy_positions"]):
            if fp is None or (isinstance(fp, float) and pd.isna(fp)):
                continue
            sleeper_has[gsis] = target_position in str(fp).split(",")

    grp = df["position_group"] if "position_group" in df.columns else df.get("position")
    fallback_elig = (grp == target_position) if grp is not None else False

    sleeper_elig = df["player_id"].map(sleeper_has)
    df["eligibility_source"] = sleeper_elig.notna().map(
        {True: "sleeper", False: "nflverse_fallback"})
    df["is_eligible"] = sleeper_elig.where(sleeper_elig.notna(), fallback_elig).astype(bool)
    return df


def attach_roster_attributes(season_df, rosters, *, season_start_month: int = 9,
                             season_start_day: int = 1):
    """Join season-*N* attributes and compute **age-at-season-start** from birth date.

    ``age_at_season_start`` = (Sept 1 of season *N* − birth date) in years. Birth date is a
    career constant (taken as the first non-null roster value per player), so age is exact for
    every season rather than relying on the roster's coarser ``age`` field. Also joins
    experience, size, and draft capital when present.
    """
    import pandas as pd

    if rosters is None or len(rosters) == 0:
        return season_df
    if "player_id" not in rosters.columns or "season" not in rosters.columns:
        return season_df

    wanted = ["years_exp", "height", "weight", "draft_number", "entry_year", "rookie_year"]
    cols = ["player_id", "season"] + [c for c in wanted if c in rosters.columns]
    rec = rosters[cols].drop_duplicates(["player_id", "season"])
    out = season_df.merge(rec, on=["player_id", "season"], how="left")

    if "birth_date" in rosters.columns:
        bd = (rosters.dropna(subset=["birth_date"])
              .drop_duplicates("player_id")[["player_id", "birth_date"]])
        out = out.merge(bd, on="player_id", how="left")
        birth = pd.to_datetime(out["birth_date"], errors="coerce")
        ref = pd.to_datetime(
            out["season"].astype("Int64").astype(str)
            + f"-{season_start_month:02d}-{season_start_day:02d}",
            errors="coerce")
        out["age_at_season_start"] = (ref - birth).dt.days / 365.25
    return out


def attach_snap_share(season_df, snaps):
    """Left-join season-*N* ``snap_share`` (and ``snaps``, ``snaps_per_game``) onto each row.

    ``snaps`` is the crosswalked season snap table from :mod:`snaps`. Values are NaN for
    seasons before snap coverage (2012+) or players without a pfr→gsis mapping; downstream
    snap-based eligibility flags are left NA for those rows rather than treated as zero.
    """
    if snaps is None or len(snaps) == 0:
        return season_df
    return season_df.merge(
        snaps[["player_id", "season", "snap_share", "snaps", "snaps_per_game"]],
        on=["player_id", "season"], how="left")


def attach_next_season_target(season_df, *, horizon: int = 1, latest_season: int | None = None):
    """Attach season *N+horizon* labels with retirement/injury/censoring classification.

    See the module docstring for the four ``status_next`` outcomes. ``target_ppg_next`` is
    defined only for ``active`` rows (a PPG requires games played); ``games_next`` is 0 for
    ``injured_out``/``retired`` (real availability-zero labels) and NaN when ``censored``.
    ``latest_season`` is the last completed season; rows whose next season exceeds it are
    censored. Defaults to the max season present.
    """
    import numpy as np

    df = season_df
    if latest_season is None:
        latest_season = int(df["season"].max())

    shift_cols = ["ppg", "ppr_points", "games"]
    rename = {"ppg": "target_ppg_next", "ppr_points": "ppr_points_next", "games": "games_next"}
    if "snap_share" in df.columns:  # carry the N+1 snap-share for the snap eligibility dim
        shift_cols.append("snap_share")
        rename["snap_share"] = "snap_share_next"
    nxt = df[["player_id", "season", *shift_cols]].copy()
    nxt["season"] = nxt["season"] - horizon  # align N+1 stats onto the season-N row
    nxt = nxt.rename(columns=rename)
    out = df.merge(nxt, on=["player_id", "season"], how="left")

    last_season = df.groupby("player_id")["season"].max().rename("_last_season")
    out = out.merge(last_season, on="player_id", how="left")

    played = out["games_next"].notna()
    censored = (out["season"] + horizon) > latest_season
    absent_observed = (~played) & (~censored)

    out.loc[absent_observed, "games_next"] = 0
    out.loc[absent_observed, "ppr_points_next"] = 0.0
    if "snap_share_next" in out.columns:  # out all year -> observed 0 snap share
        out.loc[absent_observed, "snap_share_next"] = 0.0
    # target_ppg_next: keep NaN for absent (no PPG) and censored (unknown).

    out["status_next"] = np.select(
        [played, censored, out["_last_season"] <= out["season"]],
        [STATUS_ACTIVE, STATUS_CENSORED, STATUS_RETIRED],
        default=STATUS_INJURED,
    )
    out["retired_next"] = out["status_next"] == STATUS_RETIRED
    return out.drop(columns="_last_season")


def label_eligibility_grid(season_df, games_grid, snap_grid=None):
    """Add nullable boolean candidate-cutoff flags for the next-season ranking universe.

    ``eligible_next__g{G}`` from ``games_next`` (NA when censored); ``eligible_next__s{S}``
    from ``snap_share_next`` (NA when censored *or* snap data is missing — pre-2012 / unmapped).
    Both dimensions are materialised so §4.2 can report rank metrics across the cutoff grid.
    """
    import pandas as pd

    df = season_df.copy()
    g_obs = df["games_next"].notna()
    for g in games_grid:
        col = f"eligible_next__g{g}"
        df[col] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        df.loc[g_obs, col] = (df.loc[g_obs, "games_next"] >= g)

    if snap_grid and "snap_share_next" in df.columns:
        s_obs = df["snap_share_next"].notna()
        for s in snap_grid:
            col = f"eligible_next__s{_fmt_snap(s)}"
            df[col] = pd.Series(pd.NA, index=df.index, dtype="boolean")
            df.loc[s_obs, col] = (df.loc[s_obs, "snap_share_next"] >= s)
    return df


def _fmt_snap(s) -> str:
    """Column-safe snap-cutoff suffix, e.g. 0.30 -> '30' (percent, no dot)."""
    return f"{round(float(s) * 100):02d}"


def build_dataset(config, *, position: str | None = None, horizon: int | None = None,
                  write: bool = True):
    """Orchestrate Stage 2: raw caches -> eligible player-season table -> interim parquet.

    Reads ``weekly`` (required), ``rosters`` and ``sleeper_players`` (optional) from
    ``data/raw``. Returns the DataFrame; when ``write`` also persists it and returns
    ``(df, BuildResult)``.
    """
    position = position or config.require("experiment.position")
    horizon = horizon if horizon is not None else int(config.get("target.predict_horizon", 1))
    games_grid = config.get("eligibility.candidate_games_played", [4, 6, 8, 10, 12])
    snap_grid = config.get("eligibility.candidate_snap_share", [0.30, 0.40, 0.50])
    latest_season = config.get("data.latest_completed_season")

    weekly = read_parquet(DATA_RAW / "weekly.parquet")
    rosters = _maybe_read("rosters.parquet")
    sleeper = _maybe_read("sleeper_players.parquet")
    snaps = load_player_season_snaps()

    df = aggregate_player_seasons(weekly)
    df = resolve_fantasy_eligibility(df, sleeper, position)
    df = df[df["is_eligible"]].drop(columns="is_eligible").reset_index(drop=True)
    df = attach_roster_attributes(df, rosters)
    df = attach_snap_share(df, snaps)
    # The censoring boundary cannot exceed the seasons actually present: an N+1 outcome is only
    # knowable if season N+1 is in the data. Clamp the (possibly aspirational) configured
    # ``latest_completed_season`` to the max season on hand — otherwise feature rows whose N+1 is
    # unpublished (e.g. the current season nflverse hasn't released) get mislabeled ``retired``
    # (left the league) instead of ``censored`` (unknown future).
    data_max_season = int(df["season"].max())
    latest_season = data_max_season if latest_season is None \
        else min(int(latest_season), data_max_season)
    df = attach_next_season_target(df, horizon=horizon, latest_season=latest_season)
    df = label_eligibility_grid(df, games_grid, snap_grid)
    df = df.sort_values(["season", "player_id"]).reset_index(drop=True)

    if not write:
        return df

    sport = config.get("experiment.sport", "sport")
    out_path = DATA_INTERIM / f"{sport}_{position}_player_seasons.parquet".lower()
    write_parquet(df, out_path)
    seasons = sorted(int(s) for s in df["season"].unique())
    result = BuildResult(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        seasons=seasons,
        n_with_target=int(df["target_ppg_next"].notna().sum()),
        status_counts=df["status_next"].value_counts().to_dict(),
        path=str(out_path),
    )
    return df, result


def _maybe_read(name: str):
    path = DATA_RAW / name
    return read_parquet(path) if path.exists() else None
