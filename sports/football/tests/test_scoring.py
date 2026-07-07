"""Tests for the RT Sports custom fantasy-point formula (data/scoring.py)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data.build import aggregate_player_seasons  # noqa: E402
from position_predictor.data.scoring import (  # noqa: E402
    compute_dst_points,
    compute_points,
    compute_rtsports_points,
)


def _row(**kw):
    base = dict(
        player_id="P", season=2024, week=1, season_type="REG", position="QB",
        position_group="QB", player_name="Player", recent_team="TM",
        passing_yards=0, passing_tds=0, passing_2pt_conversions=0, interceptions=0,
        rushing_yards=0, rushing_tds=0, rushing_2pt_conversions=0,
        receptions=0, receiving_yards=0, receiving_tds=0, receiving_2pt_conversions=0,
        special_teams_tds=0, fantasy_points_ppr=0.0,
        fg_made_0_19=0, fg_made_20_29=0, fg_made_30_39=0, fg_made_40_49=0,
        fg_made_50_59=0, fg_made_60_=0, pat_made=0,
    )
    base.update(kw)
    return base


def test_passing_td_is_6_not_4():
    df = pd.DataFrame([_row(passing_tds=1)])
    assert compute_rtsports_points(df).iloc[0] == 6


def test_passing_yardage_and_300_bonus():
    under = pd.DataFrame([_row(passing_yards=299)])
    over = pd.DataFrame([_row(passing_yards=300)])
    assert compute_rtsports_points(under).iloc[0] == 299 * 0.04
    assert compute_rtsports_points(over).iloc[0] == 300 * 0.04 + 5


def test_rushing_100_bonus_and_no_fumble_penalty():
    df = pd.DataFrame([_row(rushing_yards=100, rushing_tds=1)])
    # 100*0.10 + 6 (TD) + 5 (100-yd bonus) = 21; no fumble columns even considered.
    assert compute_rtsports_points(df).iloc[0] == 21


def test_receiving_full_ppr_plus_100_bonus():
    df = pd.DataFrame([_row(receptions=5, receiving_yards=100, receiving_tds=1)])
    # 5*1 + 100*0.10 + 6 (TD) + 5 (100-yd bonus) = 26
    assert compute_rtsports_points(df).iloc[0] == 26


def test_interception_thrown_penalty():
    df = pd.DataFrame([_row(interceptions=2)])
    assert compute_rtsports_points(df).iloc[0] == -4


def test_return_td():
    df = pd.DataFrame([_row(special_teams_tds=1)])
    assert compute_rtsports_points(df).iloc[0] == 6


def test_fg_scoring_tiers():
    cases = [
        (dict(fg_made_0_19=1), 3), (dict(fg_made_20_29=1), 3), (dict(fg_made_30_39=1), 3),
        (dict(fg_made_40_49=1), 4), (dict(fg_made_50_59=1), 5), (dict(fg_made_60_=1), 6),
    ]
    for kw, expected in cases:
        df = pd.DataFrame([_row(**kw)])
        assert compute_rtsports_points(df).iloc[0] == expected, kw


def test_pat_made_and_no_miss_penalty():
    df = pd.DataFrame([_row(pat_made=3)])
    assert compute_rtsports_points(df).iloc[0] == 3


def test_kicker_multi_fg_game():
    # A 3-FG game: one 35-yarder (3), one 45-yarder (4), one 55-yarder (5), + 2 PATs.
    df = pd.DataFrame([_row(fg_made_30_39=1, fg_made_40_49=1, fg_made_50_59=1, pat_made=2)])
    assert compute_rtsports_points(df).iloc[0] == 3 + 4 + 5 + 2


def test_compute_points_returns_none_for_ppr():
    df = pd.DataFrame([_row()])
    assert compute_points(df, "PPR") is None
    assert compute_points(df, "ppr") is None


def test_compute_points_case_insensitive():
    df = pd.DataFrame([_row(passing_tds=1)])
    col, series = compute_points(df, "rtsports")
    assert col == "fantasy_points_rtsports"
    assert series.iloc[0] == 6


def test_aggregate_player_seasons_uses_rtsports_when_selected():
    weekly = pd.DataFrame([
        _row(week=1, rushing_yards=100, rushing_tds=1, fantasy_points_ppr=999.0),
        _row(week=2, rushing_yards=50, fantasy_points_ppr=999.0),
    ])
    out = aggregate_player_seasons(weekly, scoring="RTSPORTS")
    # Week 1: 100*0.10 + 6 + 5 = 21; week 2: 50*0.10 = 5. Season total = 26, not fantasy_points_ppr.
    assert out.loc[0, "ppr_points"] == 26
    assert out.loc[0, "ppg"] == 13.0


def test_aggregate_player_seasons_defaults_to_ppr():
    weekly = pd.DataFrame([_row(week=1, fantasy_points_ppr=12.5)])
    out = aggregate_player_seasons(weekly)
    assert out.loc[0, "ppr_points"] == 12.5


# --------------------------------------------------------------- DST (team-level)

def _team_row(**kw):
    # points_allowed=0 defaults every case into the top (16.5-pt) tier, so each test below adds
    # that baseline explicitly rather than pretending points-allowed doesn't contribute.
    base = dict(
        def_sacks=0, def_interceptions=0, fumble_recovery_opp=0, fumble_recovery_tds=0,
        def_tds=0, def_safeties=0, points_allowed=0, blocked_fg_defense=0, blocked_xp_defense=0,
    )
    base.update(kw)
    return base


PA_0 = 16.5  # points_allowed=0 -> the 0-2-pts-allowed tier, present in every case below


def test_dst_sacks_and_interceptions():
    df = pd.DataFrame([_team_row(def_sacks=3, def_interceptions=2)])
    assert compute_dst_points(df).iloc[0] == 3 * 1 + 2 * 2 + PA_0


def test_dst_fumble_recovered_and_fumble_return_td():
    df = pd.DataFrame([_team_row(fumble_recovery_opp=1, fumble_recovery_tds=1, def_tds=1)])
    # 1 fumble recovered (2) + 1 fumble-return TD (6); def_tds=1 fully explained by the fumble TD
    # -> int_return_td approximation contributes 0.
    assert compute_dst_points(df).iloc[0] == 2 + 6 + PA_0


def test_dst_int_return_td_approximation():
    # def_tds=1 with no fumble_recovery_tds -> the whole TD is attributed to an INT return.
    df = pd.DataFrame([_team_row(def_tds=1)])
    assert compute_dst_points(df).iloc[0] == 6 + PA_0


def test_dst_safety():
    df = pd.DataFrame([_team_row(def_safeties=1)])
    assert compute_dst_points(df).iloc[0] == 2 + PA_0


def test_dst_blocked_kicks():
    df = pd.DataFrame([_team_row(blocked_fg_defense=1, blocked_xp_defense=1)])
    assert compute_dst_points(df).iloc[0] == 2 + 2 + PA_0


def test_dst_points_allowed_tiers():
    cases = [(0, 16.5), (2, 16.5), (3, 13.0), (6, 13.0), (7, 8.5), (13, 8.5),
             (14, 5.0), (20, 5.0), (21, 1.0), (27, 1.0), (28, -2.0), (45, -2.0)]
    for pa, expected in cases:
        df = pd.DataFrame([_team_row(points_allowed=pa)])
        assert compute_dst_points(df).iloc[0] == expected, pa
