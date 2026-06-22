"""Stage 4 — engineered features (block-organised, era-aware, no-leakage).

Turns the Stage-2 player-season table (``data/interim``) into the modeling matrix
(``data/processed``). Features are grouped into **blocks** matching the era schemas
(PROJECT_PLAN §6.3); each ``add_*`` function returns the columns it produces, and
:func:`build_features` assembles the **block→columns map** that
``eras.feature_columns_for_era`` consumes to pick each era model's inputs.

Leakage rules (PROJECT_PLAN §6.4):
- Every feature is computed from data **through season N** only; the label is season *N+1*.
- Trajectory / regression-to-mean baselines use **prior** seasons (``shift(1)`` within player),
  so season *N* is compared against its own past, never its future.
- Within-season ranks (positional finish) use only that season's rows.

All maths guards divide-by-zero (→ NaN). ``pandas``/``numpy`` are imported lazily.
"""

from __future__ import annotations

# Quantitative-only (PROJECT_PLAN §5): no qualitative inputs anywhere in this module.

PROCESSED_NAME = "{sport}_{position}_features.parquet"
BLOCKMAP_NAME = "{sport}_{position}_feature_blocks.json"


def _div(df, num, den):
    """Column-safe division returning NaN where the denominator is 0/NaN/missing."""
    import numpy as np
    if num not in df.columns or den not in df.columns:
        return None
    d = df[den].where(df[den] != 0)
    return df[num] / d if d is not None else np.nan


def _sorted(df):
    return df.sort_values(["player_id", "season"]).reset_index(drop=True)


# --------------------------------------------------------------------------- blocks

def add_production(df):
    """Prior-season production levels + within-season positional finish."""
    out = df.copy()
    out["total_yards"] = out.get("rushing_yards", 0) + out.get("receiving_yards", 0)
    out["total_tds"] = out.get("rushing_tds", 0) + out.get("receiving_tds", 0)
    # Positional finish in season N (rank 1 = best); uses only season-N rows -> no leakage.
    out["finish_ppr_rank"] = out.groupby("season")["ppr_points"].rank(
        ascending=False, method="min")
    out["finish_ppg_rank"] = out.groupby("season")["ppg"].rank(ascending=False, method="min")
    cols = ["ppg", "ppr_points", "rushing_yards", "receiving_yards", "total_yards",
            "rushing_tds", "receiving_tds", "total_tds", "receptions",
            "finish_ppr_rank", "finish_ppg_rank"]
    return out, [c for c in cols if c in out.columns]


def add_team_context(df, team):
    """Team-relative usage shares + team pace / run-rate proxies (numeric §5 proxies).

    ``team`` is the team-season aggregate from :func:`team_season_context`. Shares use the
    player's season-N team (``recent_team``); traded players are measured vs their last team
    (documented approximation).
    """
    out = df.merge(team, on=["recent_team", "season"], how="left")
    out["rush_att_share"] = _div(out, "carries", "team_rush_att")
    out["target_share"] = _div(out, "targets", "team_targets")
    out["touch_share"] = _div(out, "touches", "team_plays")
    out["team_run_rate"] = _div(out, "team_rush_att", "team_plays")
    out["team_plays_pg"] = _div(out, "team_plays", "team_games")
    cols = ["rush_att_share", "target_share", "touch_share", "team_run_rate", "team_plays_pg"]
    return out, cols


def add_air_yards(df):
    """Receiving air-yards opportunity (RB/WR; 1999+). Depth + share of the team's passing game.

    Air yards measure *intended* receiving opportunity independent of catches/yards, so they are
    a leakage-free demand signal that survives a down year. Runs after :func:`add_team_context`
    (which merges the team-season totals), so ``team_air_yards`` / ``team_targets`` are present.
    ``wopr`` is the standard Weighted Opportunity Rating = 1.5·target_share + 0.7·air_yards_share.

    **Dormant — not in the default pipeline.** Measured in the air-yards back-apply (RB v4 / WR v2
    probe): flat for WR, slightly negative for RB tree models. WOPR is by construction a linear
    combination of ``target_share`` + ``air_yards_share``, and ``target_share``/``targets`` are
    already features, so air yards add ~no incremental signal over existing target volume. Kept
    (tested) for re-use; re-enable by adding the call in :func:`build_features` + the ``air_yards``
    block to a config's era schemas. See PROMPT_LOG.
    """
    out = df.copy()
    out["adot"] = _div(out, "receiving_air_yards", "targets")         # average depth of target
    out["air_yards_pg"] = _div(out, "receiving_air_yards", "games")
    out["air_yards_share"] = _div(out, "receiving_air_yards", "team_air_yards")
    out["racr"] = _div(out, "receiving_yards", "receiving_air_yards")  # air-yards conversion
    tgt_share = _div(out, "targets", "team_targets")
    if tgt_share is not None and "air_yards_share" in out.columns:
        out["wopr"] = 1.5 * tgt_share + 0.7 * out["air_yards_share"]
    cols = ["receiving_air_yards", "adot", "air_yards_pg", "air_yards_share", "racr", "wopr"]
    return out, [c for c in cols if c in out.columns]


def add_volume(df):
    """Per-game usage and weighted opportunity (season-N volume)."""
    out = df.copy()
    for base in ["carries", "targets", "receptions", "touches"]:
        out[f"{base}_pg"] = _div(out, base, "games")
    out["weighted_opportunities"] = out.get("carries", 0) + 2.0 * out.get("targets", 0)
    out["wo_pg"] = _div(out, "weighted_opportunities", "games")
    cols = ["carries", "targets", "receptions", "touches",
            "carries_pg", "targets_pg", "receptions_pg", "touches_pg",
            "weighted_opportunities", "wo_pg"]
    return out, [c for c in cols if c in out.columns]


def add_efficiency(df):
    """Per-play efficiency incl. weekly EPA (all available 1999+)."""
    out = df.copy()
    out["yards_per_carry"] = _div(out, "rushing_yards", "carries")
    out["yards_per_rec"] = _div(out, "receiving_yards", "receptions")
    out["yards_per_target"] = _div(out, "receiving_yards", "targets")
    out["yards_per_touch"] = _div(out, "total_yards", "touches")
    out["catch_rate"] = _div(out, "receptions", "targets")
    out["rush_td_rate"] = _div(out, "rushing_tds", "carries")
    out["rec_td_rate"] = _div(out, "receiving_tds", "targets")
    out["rush_fd_rate"] = _div(out, "rushing_first_downs", "carries")
    out["rec_fd_rate"] = _div(out, "receiving_first_downs", "receptions")
    out["ppr_per_touch"] = _div(out, "ppr_points", "touches")
    out["rush_epa_per_att"] = _div(out, "rushing_epa", "carries")
    out["rec_epa_per_target"] = _div(out, "receiving_epa", "targets")
    cols = ["yards_per_carry", "yards_per_rec", "yards_per_target", "yards_per_touch",
            "catch_rate", "rush_td_rate", "rec_td_rate", "rush_fd_rate", "rec_fd_rate",
            "ppr_per_touch", "rush_epa_per_att", "rec_epa_per_target"]
    return out, [c for c in cols if c in out.columns]


# --------------------------------------------------------- passing blocks (QB)

def add_passing_production(df):
    """Prior-season passing production levels + within-season positional finish (QB).

    QB fantasy production is already captured by ``ppr_points`` / ``ppg`` (nflverse
    ``fantasy_points_ppr`` uses the standard 4-pt passing-TD scoring); this block adds the
    passing volume/scoring levels and the same leakage-free positional finish ranks the
    skill-position :func:`add_production` produces.
    """
    out = df.copy()
    out["total_tds"] = out.get("passing_tds", 0) + out.get("rushing_tds", 0)
    out["finish_ppr_rank"] = out.groupby("season")["ppr_points"].rank(
        ascending=False, method="min")
    out["finish_ppg_rank"] = out.groupby("season")["ppg"].rank(ascending=False, method="min")
    cols = ["ppg", "ppr_points", "passing_yards", "passing_tds", "interceptions",
            "completions", "rushing_yards", "rushing_tds", "total_tds",
            "finish_ppr_rank", "finish_ppg_rank"]
    return out, [c for c in cols if c in out.columns]


def add_passing_volume(df):
    """Per-game passing volume + dropbacks (season-N volume for QBs)."""
    out = df.copy()
    out["dropbacks"] = out.get("attempts", 0) + out.get("sacks", 0)
    for base in ["attempts", "completions", "passing_air_yards", "dropbacks"]:
        out[f"{base}_pg"] = _div(out, base, "games")
    cols = ["attempts", "completions", "dropbacks", "passing_air_yards",
            "attempts_pg", "completions_pg", "passing_air_yards_pg", "dropbacks_pg"]
    return out, [c for c in cols if c in out.columns]


def add_passing_efficiency(df):
    """Per-attempt passing efficiency incl. EPA (all available 1999+)."""
    out = df.copy()
    out["completion_pct"] = _div(out, "completions", "attempts")
    out["yards_per_attempt"] = _div(out, "passing_yards", "attempts")
    out["yards_per_completion"] = _div(out, "passing_yards", "completions")
    out["pass_td_rate"] = _div(out, "passing_tds", "attempts")
    out["int_rate"] = _div(out, "interceptions", "attempts")
    out["sack_rate"] = _div(out, "sacks", "dropbacks")
    out["air_yards_per_att"] = _div(out, "passing_air_yards", "attempts")
    out["pass_fd_rate"] = _div(out, "passing_first_downs", "attempts")
    out["pass_epa_per_db"] = _div(out, "passing_epa", "dropbacks")
    cols = ["completion_pct", "yards_per_attempt", "yards_per_completion", "pass_td_rate",
            "int_rate", "sack_rate", "air_yards_per_att", "pass_fd_rate", "pass_epa_per_db"]
    # pacr / dakota are season-summable efficiency proxies if the build carried them through.
    cols += [c for c in ["pacr", "dakota"] if c in out.columns]
    return out, [c for c in cols if c in out.columns]


def add_qb_rushing(df):
    """Rushing volume/efficiency for mobile QBs — a real, persistent fantasy edge."""
    out = df.copy()
    out["rush_yards_pg"] = _div(out, "rushing_yards", "games")
    out["carries_pg"] = _div(out, "carries", "games")
    out["yards_per_carry"] = _div(out, "rushing_yards", "carries")
    out["rush_epa_per_att"] = _div(out, "rushing_epa", "carries")
    cols = ["carries", "rushing_yards", "rushing_tds", "carries_pg", "rush_yards_pg",
            "yards_per_carry", "rush_epa_per_att"]
    return out, [c for c in cols if c in out.columns]


def add_ngs_passing(df, ngs_pass):
    """Next Gen Stats passing — CPOE, time-to-throw, aggressiveness (2016+; NaN before).

    Same coverage logic as :func:`add_ngs_efficiency`: NGS is sparse and its missingness is
    informative (sub-qualifying QBs), so ``has_ngs_pass`` exposes the coverage flag explicitly.
    """
    out = df
    if ngs_pass is not None and len(ngs_pass):
        out = out.merge(ngs_pass, on=["player_id", "season"], how="left")
    out["has_ngs_pass"] = (out["cpoe"].notna().astype(int)
                           if "cpoe" in out.columns else 0)
    cols = [c for c in ["cpoe", "expected_completion_pct", "avg_time_to_throw",
                        "aggressiveness", "avg_air_yards_to_sticks", "avg_intended_air_yards",
                        "ngs_passer_rating", "has_ngs_pass"] if c in out.columns]
    return out, cols


def add_player_attrs(df):
    """Age-curve terms, body composition, draft capital (all numeric)."""
    out = df.copy()
    if "age_at_season_start" in out.columns:
        out["age"] = out["age_at_season_start"]
        out["age_sq"] = out["age"] ** 2  # RB age cliff is non-linear
    if {"height", "weight"}.issubset(out.columns):
        out["bmi"] = 703.0 * out["weight"] / (out["height"] ** 2)
    if "draft_number" in out.columns:
        import pandas as pd
        # nflverse stores draft_number as a (numeric-valued) string; coerce so the resulting
        # capital column is genuinely numeric for the models, not an object/string dtype.
        draft_number = pd.to_numeric(out["draft_number"], errors="coerce")
        out["is_undrafted"] = draft_number.isna().astype(int)
        # Undrafted -> sentinel just past the last pick so "capital" stays monotone & numeric.
        out["draft_capital"] = draft_number.fillna(261)
    cols = [c for c in ["age", "age_sq", "bmi", "years_exp", "height", "weight",
                        "draft_capital", "is_undrafted"] if c in out.columns]
    return out, cols


def add_availability(df):
    """Durability + workload/wear proxies (inputs to the §4.3 availability model too)."""
    import numpy as np
    out = _sorted(df)
    out["season_length"] = np.where(out["season"] >= 2021, 17, 16)
    out["games_missed"] = (out["season_length"] - out["games"]).clip(lower=0)
    out["availability_rate"] = _div(out, "games", "season_length")
    g = out.groupby("player_id", group_keys=False)
    # Multi-year availability through N (inclusive ≤ N, so no leakage).
    out["avail_rate_3yr"] = g["availability_rate"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    out["career_touches"] = g["touches"].transform(lambda s: s.cumsum())
    out["touches_3yr"] = g["touches"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    out["touches_pg_3yr"] = _div(out, "touches_3yr", "games")  # rough recent wear rate
    cols = ["games", "games_missed", "availability_rate", "avail_rate_3yr",
            "career_touches", "touches_3yr", "touches_pg_3yr"]
    return out, [c for c in cols if c in out.columns]


def add_snap_usage(df):
    """Snap-share level + YoY delta (2012+; NaN before snap coverage)."""
    out = _sorted(df)
    g = out.groupby("player_id", group_keys=False)
    out["snap_share_delta"] = out["snap_share"] - g["snap_share"].shift(1)
    cols = [c for c in ["snap_share", "snaps_per_game", "snap_share_delta"] if c in out.columns]
    return out, cols


def add_trajectory(df):
    """YoY deltas and a multi-year PPG slope — direction of travel into season N."""
    import numpy as np
    out = _sorted(df)
    g = out.groupby("player_id", group_keys=False)
    for col in ["ppg", "touches_pg", "target_share", "yards_per_carry"]:
        if col in out.columns:
            out[f"{col}_delta1"] = out[col] - g[col].shift(1)
    out["seasons_played"] = g.cumcount() + 1  # career season index (≤ N)

    def _slope(s):
        # OLS slope of the last ≤3 PPG values vs season index; NaN with <2 points.
        def f(x):
            x = x.dropna()
            if len(x) < 2:
                return np.nan
            t = np.arange(len(x))
            return np.polyfit(t, x.values, 1)[0]
        return s.rolling(3, min_periods=2).apply(f, raw=False)

    out["ppg_slope3"] = g["ppg"].transform(_slope)
    cols = [c for c in ["ppg_delta1", "touches_pg_delta1", "target_share_delta1",
                        "yards_per_carry_delta1", "ppg_slope3", "seasons_played"]
            if c in out.columns]
    return out, cols


def add_regression_mean(df):
    """Sustainability signals: season-N rates vs the player's own prior baseline."""
    out = _sorted(df)
    out["td_per_touch"] = _div(out, "total_tds", "touches")
    g = out.groupby("player_id", group_keys=False)
    # Prior-career means (expanding, shifted by 1 -> strictly past seasons).
    for col in ["td_per_touch", "ppg", "yards_per_carry"]:
        if col in out.columns:
            prior = g[col].transform(lambda s: s.shift(1).expanding().mean())
            out[f"{col}_vs_prior"] = out[col] - prior
    cols = [c for c in ["td_per_touch", "td_per_touch_vs_prior", "ppg_vs_prior",
                        "yards_per_carry_vs_prior"] if c in out.columns]
    return out, cols


def add_ngs_efficiency(df, ngs_rush, ngs_rec):
    """Next Gen Stats efficiency-over-expected + coverage flags (2016+; NaN before, by design).

    NGS only covers *qualified* players (a rush-attempt threshold for rushing), so even
    post-2016 these columns are sparse — and the missingness is **informative**: lacking NGS
    rushing ≈ a low-rush-volume / receiving-profile back. ``has_ngs_rush`` / ``has_ngs_rec``
    expose that as explicit 0/1 features so every model type (not just NaN-native trees) can use
    the coverage signal; the NGS values' contribution itself is settled by the §7.3 ablation.
    """
    out = df
    if ngs_rush is not None and len(ngs_rush):
        out = out.merge(ngs_rush, on=["player_id", "season"], how="left")
    if ngs_rec is not None and len(ngs_rec):
        out = out.merge(ngs_rec, on=["player_id", "season"], how="left")
    out["has_ngs_rush"] = (out["ryoe_per_att"].notna().astype(int)
                           if "ryoe_per_att" in out.columns else 0)
    out["has_ngs_rec"] = (out["avg_separation"].notna().astype(int)
                          if "avg_separation" in out.columns else 0)
    cols = [c for c in ["ryoe_per_att", "rush_pct_over_expected", "ngs_efficiency",
                        "avg_time_to_los", "pct_attempts_8plus_box",
                        "yac_above_expected", "avg_separation",
                        "has_ngs_rush", "has_ngs_rec"] if c in out.columns]
    return out, cols


# --------------------------------------------------------------------- raw aggregates

def team_season_context(weekly, *, regular_season_only: bool = True):
    """Team-season totals from weekly box scores (all positions) for usage shares & pace."""
    df = weekly
    if regular_season_only and "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    g = df.groupby(["recent_team", "season"], as_index=False)
    agg = g.agg(team_rush_att=("carries", "sum"),
                team_targets=("targets", "sum"),
                team_air_yards=("receiving_air_yards", "sum") if "receiving_air_yards" in df.columns
                else ("targets", "sum"),
                team_pass_att=("attempts", "sum") if "attempts" in df.columns
                else ("targets", "sum"))
    games = (df.groupby(["recent_team", "season"])["week"].nunique()
             .rename("team_games").reset_index())
    agg = agg.merge(games, on=["recent_team", "season"], how="left")
    agg["team_plays"] = agg["team_rush_att"] + agg["team_pass_att"]
    return agg


def ngs_season(ngs, kind: str, *, regular_season_only: bool = True):
    """Reduce raw NGS to one season row per player (the ``week == 0`` summary) and rename."""
    if ngs is None or len(ngs) == 0:
        return None
    df = ngs
    if "season_type" in df.columns and regular_season_only:
        df = df[df["season_type"] == "REG"]
    if "week" in df.columns:
        df = df[df["week"] == 0]  # nflverse season-summary row
    df = df.rename(columns={"player_gsis_id": "player_id"})
    if kind == "rushing":
        ren = {"rush_yards_over_expected_per_att": "ryoe_per_att",
               "rush_pct_over_expected": "rush_pct_over_expected",
               "efficiency": "ngs_efficiency",
               "avg_time_to_los": "avg_time_to_los",
               "percent_attempts_gte_eight_defenders": "pct_attempts_8plus_box"}
    elif kind == "passing":
        ren = {"completion_percentage_above_expectation": "cpoe",
               "expected_completion_percentage": "expected_completion_pct",
               "avg_time_to_throw": "avg_time_to_throw",
               "aggressiveness": "aggressiveness",
               "avg_air_yards_to_sticks": "avg_air_yards_to_sticks",
               "avg_intended_air_yards": "avg_intended_air_yards",
               "passer_rating": "ngs_passer_rating"}
    else:
        ren = {"avg_yac_above_expectation": "yac_above_expected",
               "avg_separation": "avg_separation"}
    keep = ["player_id", "season"] + [k for k in ren if k in df.columns]
    return df[keep].rename(columns=ren).drop_duplicates(["player_id", "season"])


# ------------------------------------------------------------------------ orchestrator

def add_offseason(df, rosters, draft_picks, *, position="RB", workload_col="touches",
                  horizon=1, udfa_pick=300):
    """Offseason context for the season-*N+horizon* prediction (PROJECT_PLAN §5, Sept-1 cutoff).

    Quantifies the roster/draft dynamics the market reacts to but prior-season box scores miss:
    a **changed team**, a **drafted replacement** at the player's position (and how high a pick),
    and how **crowded / proven** the position room is. Every input — the N+1 draft (April) and the
    N+1 preseason roster — predates the season, so these are legitimately known at draft time (by
    **Sept 1**): *not* leakage, and exactly the information ECR already has. This is the one
    feature block that reads season *N+1* (preseason) data; all others use only data through *N*.

    Position-agnostic: ``position`` selects which drafted rookies / roster-mates count as
    competition, and ``workload_col`` is the season-*N* opportunity stat used to value that
    competition (RB ``touches``, WR/TE ``targets``, QB ``pass_attempts``). ``rosters`` and
    ``draft_picks`` are the raw nflverse tables. Returns ``(df, columns)``; the block is skipped
    (empty list) if an input is missing.
    """
    out = df.copy()
    if rosters is None or draft_picks is None or "recent_team" not in out.columns:
        return out, []
    wcol = workload_col if workload_col in out.columns else "touches"
    out["_label"] = out["season"] + horizon  # the season being predicted (N+1)

    # season-N workload per player, to value proven competition in the new position room.
    season_load = out[["player_id", "season", wcol]].dropna(subset=["player_id"])

    # --- the player's team in the prediction season (from the N+1 preseason roster) ---
    ros = rosters[["player_id", "season", "team"]].dropna(subset=["player_id", "team"])
    team_next = ros.drop_duplicates(["player_id", "season"]).rename(
        columns={"season": "_label", "team": "team_next"})
    out = out.merge(team_next, on=["player_id", "_label"], how="left")
    out["changed_team_next"] = (
        out["team_next"].notna() & (out["team_next"] != out["recent_team"])).astype(int)

    # --- rookie at the player's position the team drafted in the N+1 draft (capital = best pick) ---
    pos_draft = draft_picks[draft_picks.get("position") == position].dropna(subset=["pick"])
    cap = pos_draft.groupby(["season", "team"]).agg(
        rookie_draft_capital_next=("pick", "min"),
        rookie_count_next=("pick", "size")).reset_index().rename(
        columns={"season": "_label", "team": "team_next"})
    out = out.merge(cap, on=["_label", "team_next"], how="left")
    out["rookie_drafted_next"] = out["rookie_draft_capital_next"].notna().astype(int)
    out["rookie_draft_capital_next"] = out["rookie_draft_capital_next"].fillna(udfa_pick)
    out["rookie_count_next"] = out["rookie_count_next"].fillna(0)

    # --- proven workload in the new position room (N+1 roster mates' prior-season workload) ---
    pos_ros = rosters[rosters.get("position") == position][
        ["player_id", "season", "team"]].dropna().drop_duplicates(["player_id", "season"])
    pos_ros["_prior"] = pos_ros["season"] - 1
    pos_ros = pos_ros.merge(
        season_load.rename(columns={"season": "_prior", wcol: "_load"}),
        on=["player_id", "_prior"], how="left")
    pos_ros["_load"] = pos_ros["_load"].fillna(0.0)
    room = pos_ros.groupby(["team", "season"]).agg(
        room_prior_workload_next=("_load", "sum"),
        room_size_next=("player_id", "nunique")).reset_index().rename(
        columns={"season": "_label", "team": "team_next"})
    out = out.merge(room, on=["_label", "team_next"], how="left")
    # exclude the player's own prior workload -> only the *competition's* proven workload.
    out["room_prior_workload_next"] = (
        out["room_prior_workload_next"].fillna(0.0) - out[wcol].fillna(0.0)
    ).clip(lower=0.0)
    out["room_size_next"] = out["room_size_next"].fillna(1).astype(float)

    # --- vacated opportunity: season-N workload on the player's N+1 team that DEPARTED ---
    # Workload from same-position players who were on a team in season N but are NOT back on it
    # in N+1 (left, retired, or cut) is *opportunity that opened up*. A player joining or staying
    # on that team sees it; their own returning workload is excluded (it didn't vacate). This is
    # the RB-v3 complement to room competition — the lever WR analysis surfaced (PROJECT_PLAN §12).
    trans = out[["recent_team", "season", "team_next", wcol]].copy()
    trans["_w"] = trans[wcol].fillna(0.0)
    trans["_stay"] = trans["_w"] * (trans["team_next"] == trans["recent_team"]).astype(float)
    vac = trans.groupby(["recent_team", "season"]).agg(
        _team=("_w", "sum"), _ret=("_stay", "sum")).reset_index()
    vac["vacated_workload_next"] = (vac["_team"] - vac["_ret"]).clip(lower=0.0)
    vac = vac.rename(columns={"recent_team": "team_next"})  # index by the team being joined
    out = out.merge(vac[["team_next", "season", "vacated_workload_next"]],
                    on=["team_next", "season"], how="left")
    out["vacated_workload_next"] = out["vacated_workload_next"].fillna(0.0)

    out = out.drop(columns=["_label", "team_next"])
    cols = ["changed_team_next", "rookie_drafted_next", "rookie_draft_capital_next",
            "rookie_count_next", "room_prior_workload_next", "room_size_next",
            "vacated_workload_next"]
    out[cols] = out[cols].astype(float)
    return out, cols


def build_features(config, *, write: bool = True):
    """Stage 4 orchestrator: interim table + raw aggregates -> processed matrix + block map.

    Returns ``df`` (or ``(df, block_columns, path)`` when ``write``). ``block_columns`` maps
    each feature block to the columns it produced, for era-aware feature selection (§6.3).
    """
    import json

    from ..utils.io import (DATA_INTERIM, DATA_PROCESSED, DATA_RAW, ensure_dir,
                            read_parquet, write_parquet)

    sport = config.get("experiment.sport", "sport")
    position = config.require("experiment.position")

    interim = DATA_INTERIM / f"{sport}_{position}_player_seasons.parquet".lower()
    df = read_parquet(interim)

    def _raw(name):
        p = DATA_RAW / name
        return read_parquet(p) if p.exists() else None

    weekly = _raw("weekly.parquet")
    team = team_season_context(weekly) if weekly is not None else None
    ngs_rush = ngs_season(_raw("ngs_rushing.parquet"), "rushing")
    ngs_rec = ngs_season(_raw("ngs_receiving.parquet"), "receiving")
    ngs_pass = ngs_season(_raw("ngs_passing.parquet"), "passing")
    rosters = _raw("rosters.parquet")
    draft_picks = _raw("draft_picks.parquet")
    horizon = int(config.get("target.predict_horizon", 1))
    workload_col = config.get("features.offseason_workload_col", "touches")
    is_qb = str(position).upper() == "QB"

    block_columns: dict[str, list[str]] = {}

    # Position-specific production / usage / efficiency blocks. QBs score off passing (+ rushing
    # for mobile QBs); skill positions score off rushing/receiving. Shared blocks below are
    # column-defensive and apply to both.
    if is_qb:
        df, block_columns["production"] = add_passing_production(df)
        df, block_columns["volume"] = add_passing_volume(df)
        df, block_columns["rushing"] = add_qb_rushing(df)
        df, block_columns["efficiency"] = add_passing_efficiency(df)
    else:
        df, block_columns["production"] = add_production(df)
        df, block_columns["volume"] = add_volume(df)
        if team is not None:
            df, block_columns["team_context"] = add_team_context(df, team)
        df, block_columns["efficiency"] = add_efficiency(df)

    df, block_columns["player_attrs"] = add_player_attrs(df)
    df, block_columns["availability"] = add_availability(df)
    df, block_columns["snap_usage"] = add_snap_usage(df)
    df, block_columns["trajectory"] = add_trajectory(df)
    df, block_columns["regression_mean"] = add_regression_mean(df)
    if is_qb:
        df, block_columns["ngs_passing"] = add_ngs_passing(df, ngs_pass)
    else:
        df, block_columns["ngs_efficiency"] = add_ngs_efficiency(df, ngs_rush, ngs_rec)
    df, offseason_cols = add_offseason(df, rosters, draft_picks, position=position,
                                       workload_col=workload_col, horizon=horizon)
    if offseason_cols:
        block_columns["offseason"] = offseason_cols

    df = df.sort_values(["season", "player_id"]).reset_index(drop=True)

    if not write:
        return df, block_columns

    out_path = DATA_PROCESSED / PROCESSED_NAME.format(sport=sport, position=position).lower()
    map_path = DATA_PROCESSED / BLOCKMAP_NAME.format(sport=sport, position=position).lower()
    write_parquet(df, out_path)
    ensure_dir(DATA_PROCESSED)
    with open(map_path, "w") as fh:
        json.dump(block_columns, fh, indent=2)
    return df, block_columns, str(out_path)
