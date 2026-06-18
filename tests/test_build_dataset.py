"""Tests for the Stage 2 build logic (pure DataFrame functions, no nflverse / network).

Synthetic frames exercise season aggregation, fantasy-eligibility resolution (Sleeper +
nflverse fallback), age-at-season-start, the retirement/injury/censoring next-season
classification, and the eligibility candidate grid.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data.build import (  # noqa: E402
    STATUS_EXCLUDED,
    aggregate_player_seasons,
    apply_season_exclusion,
    attach_next_season_target,
    attach_roster_attributes,
    attach_snap_share,
    label_eligibility_grid,
    resolve_fantasy_eligibility,
)


def _weekly():
    """RBs A (2020,2022,2023 — misses 2021), B (2020 retires), a FB, and a WR.

    A's gap year 2021 = injured_out (returns 2022); B never returns after 2020 = retired.
    """
    def row(pid, season, week, pos, grp, ppr, name, team="TM"):
        return dict(player_id=pid, season=season, week=week, season_type="REG",
                    position=pos, position_group=grp, player_name=name, recent_team=team,
                    fantasy_points_ppr=ppr, carries=10, receptions=2,
                    rushing_yards=40, receiving_yards=10)

    rows = [
        # A: 2020 two games + a playoff row (must be excluded)
        row("A", 2020, 1, "RB", "RB", 10.0, "Back A"),
        row("A", 2020, 2, "RB", "RB", 20.0, "Back A"),
        dict(player_id="A", season=2020, week=19, season_type="POST", position="RB",
             position_group="RB", player_name="Back A", recent_team="TM",
             fantasy_points_ppr=99.0, carries=99, receptions=9,
             rushing_yards=999, receiving_yards=99),
        # A skips 2021 entirely (injury) then returns
        row("A", 2022, 1, "RB", "RB", 15.0, "Back A"),
        row("A", 2023, 1, "RB", "RB", 12.0, "Back A"),
        # B: 2020 only, never returns -> retired
        row("B", 2020, 1, "RB", "RB", 8.0, "Back B"),
        # FB designation but RB position_group (fallback should make eligible)
        row("F", 2020, 1, "FB", "RB", 5.0, "Full F"),
        # WR -> not RB-eligible
        row("W", 2020, 1, "WR", "WR", 30.0, "Wide W"),
    ]
    return pd.DataFrame(rows)


def test_aggregate_excludes_playoffs_keeps_all_positions():
    agg = aggregate_player_seasons(_weekly())
    # all positions retained at this stage (filtering happens later)
    assert set(agg["player_id"]) == {"A", "B", "F", "W"}
    a20 = agg[(agg.player_id == "A") & (agg.season == 2020)].iloc[0]
    assert a20["games"] == 2          # playoff week 19 excluded
    assert a20["ppr_points"] == 30.0
    assert a20["ppg"] == 15.0
    assert a20["carries"] == 20       # playoff carries=99 excluded


def test_fantasy_eligibility_sleeper_then_fallback():
    agg = aggregate_player_seasons(_weekly())
    # Sleeper knows A (RB-eligible) and W (WR-only); F/B absent -> fallback to position_group
    sleeper = pd.DataFrame([
        dict(gsis_id="A", fantasy_positions="RB"),
        dict(gsis_id="W", fantasy_positions="WR"),
    ])
    res = resolve_fantasy_eligibility(agg, sleeper, "RB")
    by_pid = res.drop_duplicates("player_id").set_index("player_id")
    assert by_pid.loc["A", "is_eligible"] and by_pid.loc["A", "eligibility_source"] == "sleeper"
    assert not by_pid.loc["W", "is_eligible"]  # WR-only via sleeper
    assert by_pid.loc["F", "is_eligible"]      # FB folds to RB via position_group fallback
    assert by_pid.loc["F", "eligibility_source"] == "nflverse_fallback"
    assert by_pid.loc["B", "is_eligible"]      # B is RB via position_group fallback


def test_eligibility_no_sleeper_uses_fallback():
    agg = aggregate_player_seasons(_weekly())
    res = resolve_fantasy_eligibility(agg, None, "RB")
    elig = set(res[res.is_eligible]["player_id"])
    assert elig == {"A", "B", "F"}  # all RB position_group; WR excluded
    assert (res["eligibility_source"] == "nflverse_fallback").all()


def test_age_at_season_start_from_birth_date():
    agg = aggregate_player_seasons(_weekly())
    rosters = pd.DataFrame([
        dict(player_id="A", season=2020, birth_date="1996-09-01", years_exp=2,
             height=70, weight=215),
        dict(player_id="A", season=2022, birth_date="1996-09-01", years_exp=4,
             height=70, weight=215),
    ])
    out = attach_roster_attributes(agg, rosters)
    a20 = out[(out.player_id == "A") & (out.season == 2020)].iloc[0]
    # born 1996-09-01, season start 2020-09-01 -> exactly 24 years
    assert abs(a20["age_at_season_start"] - 24.0) < 0.02
    assert a20["years_exp"] == 2


def test_next_season_status_active_injured_retired_censored():
    agg = aggregate_player_seasons(_weekly())
    rb = agg[agg.player_id.isin(["A", "B"])]
    out = attach_next_season_target(rb, horizon=1, latest_season=2023)
    g = lambda pid, yr: out[(out.player_id == pid) & (out.season == yr)].iloc[0]  # noqa: E731

    # A 2020 -> 2021 missed but returns 2022 == injured_out, games_next 0, target NaN
    a20 = g("A", 2020)
    assert a20["status_next"] == "injured_out"
    assert a20["games_next"] == 0
    assert pd.isna(a20["target_ppg_next"])
    assert not a20["retired_next"]

    # A 2022 -> 2023 active: target = 2023 PPG (12.0), games_next 1
    a22 = g("A", 2022)
    assert a22["status_next"] == "active"
    assert a22["target_ppg_next"] == 12.0
    assert a22["games_next"] == 1

    # A 2023 -> 2024 is beyond latest_season(2023) == censored, labels NaN
    a23 = g("A", 2023)
    assert a23["status_next"] == "censored"
    assert pd.isna(a23["games_next"])

    # B 2020 -> never returns == retired (treated as availability-zero)
    b20 = g("B", 2020)
    assert b20["status_next"] == "retired"
    assert b20["games_next"] == 0
    assert b20["retired_next"]


def test_eligibility_grid_nullable_for_censored():
    agg = aggregate_player_seasons(_weekly())
    rb = agg[agg.player_id == "A"]
    out = attach_next_season_target(rb, horizon=1, latest_season=2023)
    out = label_eligibility_grid(out, games_grid=[1, 2])
    a22 = out[(out.player_id == "A") & (out.season == 2022)].iloc[0]
    assert bool(a22["eligible_next__g1"]) is True   # plays 1 game in 2023
    assert bool(a22["eligible_next__g2"]) is False
    a23 = out[(out.player_id == "A") & (out.season == 2023)].iloc[0]
    assert pd.isna(a23["eligible_next__g1"])         # censored -> NA, not False


def test_snap_share_carries_to_next_and_grid():
    agg = aggregate_player_seasons(_weekly())
    rb = agg[agg.player_id == "A"]
    snaps = pd.DataFrame([
        dict(player_id="A", season=2022, snap_share=0.8, snaps=400, snaps_per_game=40),
        dict(player_id="A", season=2023, snap_share=0.35, snaps=20, snaps_per_game=20),
    ])
    df = attach_snap_share(rb, snaps)
    df = attach_next_season_target(df, horizon=1, latest_season=2023)
    df = label_eligibility_grid(df, games_grid=[1], snap_grid=[0.30, 0.50])
    a22 = df[(df.player_id == "A") & (df.season == 2022)].iloc[0]
    # 2022 row's next-season (2023) snap share = 0.35 -> clears 0.30, not 0.50
    assert abs(a22["snap_share_next"] - 0.35) < 1e-9
    assert bool(a22["eligible_next__s30"]) is True
    assert bool(a22["eligible_next__s50"]) is False


def test_snap_share_zero_when_out_all_year():
    # B retires after 2020 -> out all of 2021 -> observed-zero snap share, flags False not NA
    agg = aggregate_player_seasons(_weekly())
    rb = agg[agg.player_id == "B"]
    snaps = pd.DataFrame([dict(player_id="B", season=2020, snap_share=0.5, snaps=300,
                               snaps_per_game=30)])
    df = attach_snap_share(rb, snaps)
    df = attach_next_season_target(df, horizon=1, latest_season=2023)
    df = label_eligibility_grid(df, games_grid=[1], snap_grid=[0.30])
    b20 = df[(df.player_id == "B") & (df.season == 2020)].iloc[0]
    assert b20["snap_share_next"] == 0.0
    assert bool(b20["eligible_next__s30"]) is False


def test_empty_weekly_returns_empty_frame():
    empty = pd.DataFrame(columns=["player_id", "season", "position", "season_type", "week"])
    assert len(aggregate_player_seasons(empty)) == 0


def test_apply_season_exclusion_drops_feature_season_keeps_players_and_nulls_label():
    # Player A spans 2018-2022 (played COVID 2020); B is 2019-only. horizon=1, exclude 2020.
    df = pd.DataFrame([
        dict(player_id="A", season=2018, target_ppg_next=10.0, games_next=16,
             status_next="active"),
        dict(player_id="A", season=2019, target_ppg_next=11.0, games_next=15,
             status_next="active"),  # label season = 2020 -> excluded
        dict(player_id="A", season=2020, target_ppg_next=12.0, games_next=10,
             status_next="active"),  # feature season 2020 -> dropped
        dict(player_id="A", season=2021, target_ppg_next=13.0, games_next=17,
             status_next="active"),
        dict(player_id="A", season=2022, target_ppg_next=14.0, games_next=17,
             status_next="active"),
        dict(player_id="B", season=2019, target_ppg_next=9.0, games_next=16,
             status_next="active"),  # label 2020 -> excluded
    ])
    out = apply_season_exclusion(df, [2020], horizon=1)

    # 2020 feature-season rows are gone entirely...
    assert (out["season"] != 2020).all()
    # ...but the player is NOT removed — A still has its other seasons (no survivorship bias).
    assert set(out.loc[out.player_id == "A", "season"]) == {2018, 2019, 2021, 2022}

    # label season 2020 (the 2019 rows) is kept as history but nulled + flagged, not supervised.
    a19 = out[(out.player_id == "A") & (out.season == 2019)].iloc[0]
    assert pd.isna(a19["target_ppg_next"]) and pd.isna(a19["games_next"])
    assert a19["status_next"] == STATUS_EXCLUDED
    # a clean pair (2021 -> label 2022) is untouched.
    a21 = out[(out.player_id == "A") & (out.season == 2021)].iloc[0]
    assert a21["target_ppg_next"] == 13.0 and a21["status_next"] == "active"


def test_apply_season_exclusion_noop_when_empty():
    df = pd.DataFrame([dict(player_id="A", season=2020, target_ppg_next=12.0,
                            games_next=10, status_next="active")])
    assert apply_season_exclusion(df, [], horizon=1).equals(df)
