"""Tests for Stage 4 feature blocks (pure DataFrame logic; no nflverse / network).

Emphasis on the no-leakage contract: trajectory deltas and regression-to-mean baselines must
use only *prior* seasons, so a player's first season has NaN for those.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.features.build import (  # noqa: E402
    add_air_yards,
    add_availability,
    add_efficiency,
    add_ngs_efficiency,
    add_ngs_passing,
    add_offseason,
    add_passing_efficiency,
    add_passing_production,
    add_passing_volume,
    season_health,
    add_player_attrs,
    add_production,
    add_qb_rushing,
    add_regression_mean,
    add_snap_usage,
    add_team_context,
    add_trajectory,
    add_volume,
    ngs_season,
    offseason_degenerate,
    team_season_context,
)


def _panel():
    """Player A over 2012–2014 (+ B 2012). Counts chosen for easy hand-checks."""
    rows = [
        dict(player_id="A", season=2012, recent_team="AAA", games=16,
             carries=200, targets=40, receptions=30, rushing_yards=800, receiving_yards=300,
             rushing_tds=8, receiving_tds=2, rushing_first_downs=50, receiving_first_downs=15,
             rushing_epa=20.0, receiving_epa=10.0, ppr_points=240.0, ppg=15.0, touches=230,
             snap_share=0.70, snaps_per_game=40.0, age_at_season_start=24.0, years_exp=2,
             height=70, weight=215, draft_number=40.0),
        dict(player_id="A", season=2013, recent_team="AAA", games=8,
             carries=100, targets=20, receptions=15, rushing_yards=380, receiving_yards=150,
             rushing_tds=3, receiving_tds=1, rushing_first_downs=22, receiving_first_downs=7,
             rushing_epa=5.0, receiving_epa=4.0, ppr_points=112.0, ppg=14.0, touches=115,
             snap_share=0.55, snaps_per_game=38.0, age_at_season_start=25.0, years_exp=3,
             height=70, weight=215, draft_number=40.0),
        dict(player_id="A", season=2014, recent_team="AAA", games=16,
             carries=240, targets=50, receptions=40, rushing_yards=1000, receiving_yards=350,
             rushing_tds=10, receiving_tds=3, rushing_first_downs=60, receiving_first_downs=20,
             rushing_epa=30.0, receiving_epa=12.0, ppr_points=288.0, ppg=18.0, touches=280,
             snap_share=0.75, snaps_per_game=42.0, age_at_season_start=26.0, years_exp=4,
             height=70, weight=215, draft_number=40.0),
        dict(player_id="B", season=2012, recent_team="BBB", games=16,
             carries=120, targets=60, receptions=48, rushing_yards=480, receiving_yards=400,
             rushing_tds=2, receiving_tds=4, rushing_first_downs=25, receiving_first_downs=22,
             rushing_epa=2.0, receiving_epa=15.0, ppr_points=200.0, ppg=12.5, touches=168,
             snap_share=0.60, snaps_per_game=36.0, age_at_season_start=28.0, years_exp=6,
             height=72, weight=225, draft_number=np.nan),  # undrafted
    ]
    return pd.DataFrame(rows)


def _row(df, pid, season):
    return df[(df.player_id == pid) & (df.season == season)].iloc[0]


def test_production_totals_and_finish_rank():
    out, cols = add_production(_panel())
    a12 = _row(out, "A", 2012)
    assert a12["total_yards"] == 1100
    assert a12["total_tds"] == 10
    # 2012: A (240 ppr) ranks above B (200) -> rank 1
    assert _row(out, "A", 2012)["finish_ppr_rank"] == 1
    assert _row(out, "B", 2012)["finish_ppr_rank"] == 2
    assert "total_yards" in cols


def test_volume_per_game_and_weighted_opp():
    out, cols = add_volume(_panel())
    a12 = _row(out, "A", 2012)
    assert a12["carries_pg"] == 200 / 16
    assert a12["weighted_opportunities"] == 200 + 2 * 40
    assert "wo_pg" in cols


def test_team_context_shares():
    team = pd.DataFrame([
        dict(recent_team="AAA", season=2012, team_rush_att=400, team_targets=160,
             team_pass_att=540, team_plays=940, team_games=16),
    ])
    df = _panel()[_panel().recent_team == "AAA"]
    out, cols = add_team_context(df, team)
    a12 = _row(out, "A", 2012)
    assert a12["rush_att_share"] == 200 / 400
    assert a12["target_share"] == 40 / 160
    assert abs(a12["team_run_rate"] - 400 / 940) < 1e-9
    assert "team_plays_pg" in cols


def test_efficiency_ratios_and_zero_guard():
    out, _ = add_efficiency(add_production(_panel())[0])
    a12 = _row(out, "A", 2012)
    assert a12["yards_per_carry"] == 800 / 200
    assert a12["catch_rate"] == 30 / 40
    assert abs(a12["rush_epa_per_att"] - 20.0 / 200) < 1e-9
    # zero-carry row -> ypc NaN, not error/inf
    z = pd.DataFrame([dict(player_id="Z", season=2012, carries=0, rushing_yards=0,
                           receptions=1, targets=2, receiving_yards=5, touches=1,
                           total_yards=5, rushing_tds=0, receiving_tds=0,
                           rushing_first_downs=0, receiving_first_downs=0,
                           ppr_points=1.0, rushing_epa=0.0, receiving_epa=0.0)])
    zout, _ = add_efficiency(z)
    assert pd.isna(zout.iloc[0]["yards_per_carry"])


def test_player_attrs_age_curve_bmi_draft():
    out, cols = add_player_attrs(_panel())
    a12 = _row(out, "A", 2012)
    assert a12["age"] == 24.0
    assert a12["age_sq"] == 576.0
    assert abs(a12["bmi"] - 703.0 * 215 / 70**2) < 1e-9
    assert a12["draft_capital"] == 40.0 and a12["is_undrafted"] == 0
    b12 = _row(out, "B", 2012)
    assert b12["is_undrafted"] == 1 and b12["draft_capital"] == 261  # sentinel


def test_availability_season_length_and_career():
    out, cols = add_availability(_panel())
    a12, a13 = _row(out, "A", 2012), _row(out, "A", 2013)
    assert a12["season_length"] == 16              # pre-2021
    assert a13["games_missed"] == 8                # 16 - 8
    assert a13["availability_rate"] == 0.5
    # career touches accumulate through N
    a14 = _row(out, "A", 2014)
    assert a14["career_touches"] == 230 + 115 + 280
    # 17-game season after 2021
    later = pd.DataFrame([dict(player_id="C", season=2022, games=17, touches=300)])
    lout, _ = add_availability(later)
    assert lout.iloc[0]["season_length"] == 17


def test_snap_usage_delta_uses_prior():
    out, _ = add_snap_usage(_panel())
    assert pd.isna(_row(out, "A", 2012)["snap_share_delta"])  # first season -> NaN
    assert abs(_row(out, "A", 2013)["snap_share_delta"] - (0.55 - 0.70)) < 1e-9


def test_trajectory_no_leakage():
    # need ppg + touches_pg present; run volume first for touches_pg
    df, _ = add_volume(_panel())
    out, cols = add_trajectory(df)
    a12, a13 = _row(out, "A", 2012), _row(out, "A", 2013)
    assert pd.isna(a12["ppg_delta1"])              # first season has no prior
    assert abs(a13["ppg_delta1"] - (14.0 - 15.0)) < 1e-9
    assert a12["seasons_played"] == 1 and a13["seasons_played"] == 2
    assert pd.isna(a12["ppg_slope3"])              # <2 points -> NaN
    assert not pd.isna(_row(out, "A", 2014)["ppg_slope3"])


def test_regression_mean_uses_strictly_prior():
    out, cols = add_regression_mean(_panel())
    a12 = _row(out, "A", 2012)
    # first season: no prior baseline -> vs_prior is NaN (no leakage from future)
    assert pd.isna(a12["ppg_vs_prior"])
    a13 = _row(out, "A", 2013)
    # 2013 ppg (14) vs prior mean (just 2012 = 15) -> -1
    assert abs(a13["ppg_vs_prior"] - (14.0 - 15.0)) < 1e-9
    assert "td_per_touch" in cols


def test_ngs_season_reduces_and_renames():
    raw = pd.DataFrame([
        dict(player_gsis_id="A", season=2016, season_type="REG", week=0,
             rush_yards_over_expected_per_att=0.5, efficiency=4.0,
             percent_attempts_gte_eight_defenders=20.0, avg_time_to_los=2.8,
             rush_pct_over_expected=3.0),
        dict(player_gsis_id="A", season=2016, season_type="REG", week=5,
             rush_yards_over_expected_per_att=9.9, efficiency=9.9,
             percent_attempts_gte_eight_defenders=9.9, avg_time_to_los=9.9,
             rush_pct_over_expected=9.9),  # weekly row -> dropped
    ])
    season = ngs_season(raw, "rushing")
    assert len(season) == 1
    assert season.iloc[0]["ryoe_per_att"] == 0.5
    assert season.iloc[0]["pct_attempts_8plus_box"] == 20.0
    # merge into a panel
    out, cols = add_ngs_efficiency(_panel().assign(season=2016), season, None)
    assert "ryoe_per_att" in cols
    assert _row(out, "A", 2016)["ryoe_per_att"] == 0.5


def test_ngs_coverage_flags():
    season = pd.DataFrame([dict(player_id="A", season=2016, ryoe_per_att=0.5)])
    # A has NGS rushing, B does not; no NGS receiving for either
    panel = _panel().assign(season=2016)
    out, cols = add_ngs_efficiency(panel, season, None)
    assert "has_ngs_rush" in cols and "has_ngs_rec" in cols
    assert _row(out, "A", 2016)["has_ngs_rush"] == 1   # covered (receiving-back signal absent)
    assert _row(out, "B", 2016)["has_ngs_rush"] == 0   # uncovered -> informative flag
    assert (out["has_ngs_rec"] == 0).all()             # no NGS receiving merged


def test_ngs_flags_present_even_without_ngs_data():
    # when NGS files are absent entirely, flags still exist (all 0) so the schema is stable
    out, cols = add_ngs_efficiency(_panel(), None, None)
    assert out["has_ngs_rush"].eq(0).all()
    assert "has_ngs_rush" in cols and "has_ngs_rec" in cols


def test_team_season_context_aggregates():
    weekly = pd.DataFrame([
        dict(recent_team="AAA", season=2012, week=1, season_type="REG",
             carries=20, targets=15, attempts=30),
        dict(recent_team="AAA", season=2012, week=2, season_type="REG",
             carries=25, targets=18, attempts=35),
        dict(recent_team="AAA", season=2012, week=1, season_type="POST",
             carries=99, targets=99, attempts=99),  # playoff -> excluded
    ])
    out = team_season_context(weekly)
    row = out.iloc[0]
    assert row["team_rush_att"] == 45
    assert row["team_targets"] == 33
    assert row["team_pass_att"] == 65
    assert row["team_games"] == 2
    assert row["team_plays"] == 45 + 65


def test_add_offseason_quantifies_n_plus_1_context():
    # Player X: 2022 on TEN (200 touches), moves to BAL for 2023; Y is the BAL incumbent.
    df = pd.DataFrame([
        dict(player_id="X", season=2022, recent_team="TEN", touches=200.0),
        dict(player_id="Y", season=2022, recent_team="BAL", touches=120.0),
    ])
    rosters = pd.DataFrame([  # 2023 BAL room = X, Y, rookie Z
        dict(player_id="X", season=2023, team="BAL", position="RB"),
        dict(player_id="Y", season=2023, team="BAL", position="RB"),
        dict(player_id="Z", season=2023, team="BAL", position="RB"),
    ])
    draft = pd.DataFrame([dict(season=2023, team="BAL", position="RB", pick=100)])

    out, cols = add_offseason(df, rosters, draft, position="RB", workload_col="touches",
                              horizon=1)
    x = out[out.player_id == "X"].iloc[0]
    assert x["changed_team_next"] == 1.0                  # TEN -> BAL
    assert x["rookie_drafted_next"] == 1.0 and x["rookie_draft_capital_next"] == 100.0
    # competition = OTHER RBs' prior touches in BAL's 2023 room (Y=120; rookie Z=0)
    assert x["room_prior_workload_next"] == 120.0
    assert x["room_size_next"] == 3.0
    y = out[out.player_id == "Y"].iloc[0]
    assert y["changed_team_next"] == 0.0                  # stayed on BAL
    assert y["room_prior_workload_next"] == 200.0         # now competes with X's 200
    assert set(cols) == {"changed_team_next", "rookie_drafted_next", "rookie_draft_capital_next",
                         "rookie_count_next", "room_prior_workload_next", "room_size_next",
                         "vacated_workload_next"}


def test_add_offseason_position_aware_workload():
    # WR model: workload currency is targets, and only WRs count as competition.
    df = pd.DataFrame([
        dict(player_id="W", season=2022, recent_team="LAR", touches=10.0, targets=150.0),
        dict(player_id="V", season=2022, recent_team="LAR", touches=5.0, targets=90.0),
    ])
    rosters = pd.DataFrame([
        dict(player_id="W", season=2023, team="LAR", position="WR"),
        dict(player_id="V", season=2023, team="LAR", position="WR"),
        dict(player_id="R", season=2023, team="LAR", position="RB"),  # RB ignored for WR room
    ])
    draft = pd.DataFrame([
        dict(season=2023, team="LAR", position="WR", pick=20),
        dict(season=2023, team="LAR", position="RB", pick=3),   # RB pick must NOT count
    ])
    out, _ = add_offseason(df, rosters, draft, position="WR", workload_col="targets", horizon=1)
    w = out[out.player_id == "W"].iloc[0]
    assert w["rookie_draft_capital_next"] == 20.0          # the WR pick, not the RB pick=3
    assert w["room_prior_workload_next"] == 90.0           # V's targets; RB R excluded
    assert w["room_size_next"] == 2.0                      # W + V (RB excluded)


def test_add_offseason_vacated_opportunity():
    # OLD team 2022: A (100 touches) leaves to NEW, B (50) stays. C joins OLD from X.
    df = pd.DataFrame([
        dict(player_id="A", season=2022, recent_team="OLD", touches=100.0),
        dict(player_id="B", season=2022, recent_team="OLD", touches=50.0),
        dict(player_id="C", season=2022, recent_team="X", touches=10.0),
    ])
    rosters = pd.DataFrame([
        dict(player_id="A", season=2023, team="NEW", position="RB"),  # A left OLD
        dict(player_id="B", season=2023, team="OLD", position="RB"),  # B stayed
        dict(player_id="C", season=2023, team="OLD", position="RB"),  # C joined OLD
    ])
    draft = pd.DataFrame([dict(season=2023, team="OLD", position="RB", pick=50)])
    out, cols = add_offseason(df, rosters, draft, position="RB", workload_col="touches")
    assert "vacated_workload_next" in cols
    g = out.set_index("player_id")["vacated_workload_next"]
    # OLD vacated A's 100 touches (A left); B's 50 stays -> not vacated.
    assert g["B"] == 100.0     # B (stays OLD) sees A's vacated workload
    assert g["C"] == 100.0     # C (joins OLD) sees the same opportunity
    assert g["A"] == 0.0       # A's new team NEW vacated nothing


def test_add_offseason_sentinel_and_none_safe():
    df = pd.DataFrame([dict(player_id="X", season=2022, recent_team="TEN", touches=100.0)])
    rosters = pd.DataFrame([dict(player_id="X", season=2023, team="TEN", position="RB")])
    draft = pd.DataFrame([dict(season=2023, team="DAL", position="RB", pick=5)])  # not TEN
    out, _ = add_offseason(df, rosters, draft, position="RB", horizon=1, udfa_pick=300)
    r = out.iloc[0]
    assert r["changed_team_next"] == 0.0                  # stayed on TEN
    assert r["rookie_drafted_next"] == 0.0                # TEN drafted no RB
    assert r["rookie_draft_capital_next"] == 300.0        # sentinel = no threat
    out2, cols2 = add_offseason(df, None, draft)          # missing input -> skipped, no crash
    assert cols2 == [] and "changed_team_next" not in out2.columns


# --------------------------------------------------------------------- QB passing blocks

def _qb_panel():
    """QB A over 2015–2016 (NGS era). Counts chosen for easy hand-checks."""
    rows = [
        dict(player_id="A", season=2015, recent_team="AAA", games=16,
             attempts=500, completions=325, passing_yards=4000, passing_tds=30,
             interceptions=10, sacks=20, passing_air_yards=4200, passing_first_downs=200,
             passing_epa=120.0, carries=40, rushing_yards=200, rushing_tds=2,
             rushing_epa=5.0, ppr_points=320.0, ppg=20.0, touches=40),
        dict(player_id="A", season=2016, recent_team="AAA", games=16,
             attempts=550, completions=360, passing_yards=4500, passing_tds=35,
             interceptions=8, sacks=25, passing_air_yards=4600, passing_first_downs=220,
             passing_epa=140.0, carries=50, rushing_yards=300, rushing_tds=3,
             rushing_epa=8.0, ppr_points=360.0, ppg=22.5, touches=50),
        dict(player_id="B", season=2015, recent_team="BBB", games=16,
             attempts=400, completions=240, passing_yards=2800, passing_tds=18,
             interceptions=14, sacks=35, passing_air_yards=3200, passing_first_downs=150,
             passing_epa=30.0, carries=20, rushing_yards=80, rushing_tds=1,
             rushing_epa=1.0, ppr_points=220.0, ppg=13.75, touches=20),
    ]
    return pd.DataFrame(rows)


def test_passing_production_finish_rank_and_total_tds():
    out, cols = add_passing_production(_qb_panel())
    a15 = _row(out, "A", 2015)
    assert a15["total_tds"] == 32                 # 30 passing + 2 rushing
    assert a15["finish_ppr_rank"] == 1            # A (320) > B (220) in 2015
    assert _row(out, "B", 2015)["finish_ppr_rank"] == 2
    assert "passing_yards" in cols and "interceptions" in cols


def test_passing_volume_dropbacks_and_per_game():
    out, cols = add_passing_volume(_qb_panel())
    a15 = _row(out, "A", 2015)
    assert a15["dropbacks"] == 520                # 500 attempts + 20 sacks
    assert a15["attempts_pg"] == 500 / 16
    assert "dropbacks_pg" in cols


def test_passing_efficiency_rates_and_epa():
    out, cols = add_passing_efficiency(add_passing_volume(_qb_panel())[0])
    a15 = _row(out, "A", 2015)
    assert a15["completion_pct"] == 325 / 500
    assert a15["yards_per_attempt"] == 4000 / 500
    assert a15["int_rate"] == 10 / 500
    assert a15["sack_rate"] == 20 / 520            # sacks / dropbacks
    assert a15["pass_epa_per_db"] == 120.0 / 520
    assert "pass_td_rate" in cols


def test_qb_rushing_block():
    out, cols = add_qb_rushing(_qb_panel())
    a16 = _row(out, "A", 2016)
    assert a16["rush_yards_pg"] == 300 / 16
    assert a16["yards_per_carry"] == 300 / 50
    assert "rushing_tds" in cols


def test_add_ngs_passing_merges_and_flags():
    df = _qb_panel()
    ngs = pd.DataFrame([
        dict(player_id="A", season=2015, cpoe=3.5, avg_time_to_throw=2.7,
             aggressiveness=18.0, ngs_passer_rating=98.0),
    ])
    out, cols = add_ngs_passing(df, ngs)
    assert _row(out, "A", 2015)["cpoe"] == 3.5
    assert _row(out, "A", 2015)["has_ngs_pass"] == 1   # covered
    assert _row(out, "B", 2015)["has_ngs_pass"] == 0   # not in NGS frame
    assert "cpoe" in cols and "has_ngs_pass" in cols


def test_add_ngs_passing_flag_present_without_ngs_data():
    out, cols = add_ngs_passing(_qb_panel(), None)
    assert (out["has_ngs_pass"] == 0).all()
    assert "has_ngs_pass" in cols


def test_ngs_season_passing_renames():
    raw = pd.DataFrame([
        dict(player_gsis_id="A", season=2016, season_type="REG", week=0,
             completion_percentage_above_expectation=2.1, avg_time_to_throw=2.6,
             aggressiveness=15.0, passer_rating=95.0),
        dict(player_gsis_id="A", season=2016, season_type="REG", week=5,
             completion_percentage_above_expectation=9.9, avg_time_to_throw=9.9,
             aggressiveness=99.0, passer_rating=99.0),  # non-summary week, dropped
    ])
    out = ngs_season(raw, "passing")
    assert list(out["player_id"]) == ["A"]
    assert out.iloc[0]["cpoe"] == 2.1                  # week==0 summary row only
    assert out.iloc[0]["ngs_passer_rating"] == 95.0


def test_add_air_yards_depth_share_and_wopr():
    # Dormant block (measured no-gain), but kept correct & tested. Simulates a post-team_context
    # frame: team_air_yards / team_targets already merged.
    df = pd.DataFrame([
        dict(player_id="A", season=2015, games=16, targets=120, receiving_yards=1200,
             receiving_air_yards=1500, team_air_yards=3000, team_targets=400),
    ])
    out, cols = add_air_yards(df)
    r = out.iloc[0]
    assert r["adot"] == 1500 / 120                    # depth of target
    assert r["air_yards_share"] == 1500 / 3000        # 0.5
    assert r["racr"] == 1200 / 1500                   # air-yards conversion
    # WOPR = 1.5*target_share + 0.7*air_yards_share = 1.5*(120/400) + 0.7*0.5
    assert abs(r["wopr"] - (1.5 * (120 / 400) + 0.7 * 0.5)) < 1e-9
    assert "wopr" in cols and "adot" in cols


def test_offseason_degenerate_detects_missing_n1_join():
    bc = {"offseason": ["changed_team_next", "room_prior_workload_next",
                        "rookie_draft_capital_next", "room_size_next"]}
    df = pd.DataFrame({
        "season": [2024, 2024, 2025, 2025],
        # 2024 board: real per-player signal (varies). 2025 live board: N+1 join missing →
        # every offseason column is constant (all-zero or a sentinel default like 300 / 1).
        "changed_team_next": [0, 1, 0, 0],
        "room_prior_workload_next": [120.0, 300.0, 0.0, 0.0],
        "rookie_draft_capital_next": [40.0, 300.0, 300.0, 300.0],
        "room_size_next": [2, 3, 1, 1],
    })
    assert offseason_degenerate(df, bc, 2025) is True    # constant defaults → degenerate
    assert offseason_degenerate(df, bc, 2024) is False   # varies across players → healthy
    assert offseason_degenerate(df, {"offseason": []}, 2025) is False  # no block → not flagged
    assert offseason_degenerate(df, bc, 1999) is False   # season absent → not flagged


# -- season_health (report context; deliberately NOT a feature block) -----------------------

def _weekly(rows):
    """rows: (player_id, week) pairs for team 'AAA' in 2024, plus the team's own game weeks."""
    return pd.DataFrame([{"player_id": p, "season": 2024, "week": w, "season_type": "REG",
                          "recent_team": "AAA"} for p, w in rows])


def _team_playing(weeks):
    """A filler player who plays every one of the team's weeks, so the team's schedule is known."""
    return [("team_filler", w) for w in weeks]


TEAM_WEEKS = [1, 2, 3, 4, 5, 6, 7, 9, 10]        # week 8 is the bye


def test_season_health_does_not_count_a_bye_as_a_missed_game():
    """Week 8 is the team's bye — a player present all year must show no absence at all."""
    w = _weekly(_team_playing(TEAM_WEEKS) + [("ironman", x) for x in TEAM_WEEKS])
    h = season_health(w).set_index("player_id")
    r = h.loc["ironman"]
    assert r.team_games == len(TEAM_WEEKS) == 9
    assert r.games_played == 9
    assert r.longest_absence == 0
    assert r.weeks_to_season_end == 0 and r.finished_season == 1


def test_season_health_separates_came_back_from_still_out():
    """The whole point of the block: two players missing four games, opposite signals."""
    came_back = [("came_back", x) for x in [1, 2, 3, 4, 5, 9, 10]]      # out 6,7 + bye, returned
    still_out = [("still_out", x) for x in [1, 2, 3, 4, 5]]             # last played wk 5
    h = season_health(_weekly(_team_playing(TEAM_WEEKS) + came_back + still_out)
                      ).set_index("player_id")
    a, b = h.loc["came_back"], h.loc["still_out"]
    assert a.games_played == b.games_played + 2
    assert a.weeks_to_season_end == 0 and a.finished_season == 1 and a.returned_after_absence == 1
    assert b.weeks_to_season_end == 4 and b.finished_season == 0 and b.returned_after_absence == 0


def test_season_health_injury_report_is_optional_and_absence_means_zero():
    """A player never listed on the report gets zeros — that is the truth, not a missing value."""
    w = _weekly(_team_playing(TEAM_WEEKS) + [("hurt", x) for x in [1, 2, 3]]
                + [("healthy", x) for x in TEAM_WEEKS])
    assert "inj_weeks_out" not in season_health(w).columns        # no injuries table → omitted
    inj = pd.DataFrame([
        {"gsis_id": "hurt", "season": 2024, "week": 4, "game_type": "REG",
         "report_status": "Out", "practice_status": "Did Not Participate In Practice",
         "report_primary_injury": "Knee"},
        {"gsis_id": "hurt", "season": 2024, "week": 10, "game_type": "REG",
         "report_status": "Out", "practice_status": "Did Not Participate In Practice",
         "report_primary_injury": "Knee"},
    ])
    h = season_health(w, inj).set_index("player_id")
    assert h.loc["hurt"].inj_weeks_out == 2
    assert h.loc["hurt"].inj_out_at_end == 1          # ruled out in the closing stretch
    assert h.loc["healthy"].inj_weeks_out == 0 and h.loc["healthy"].inj_report_weeks == 0
    assert h.loc["healthy"].inj_out_at_end == 0


def test_season_health_handles_a_player_who_drops_off_the_report_on_ir():
    """IR players vanish from the injury report, so weekly appearances must carry the absence."""
    w = _weekly(_team_playing(TEAM_WEEKS) + [("ir", x) for x in [1, 2]])
    inj = pd.DataFrame([{"gsis_id": "ir", "season": 2024, "week": 3, "game_type": "REG",
                         "report_status": "Out", "practice_status": "Did Not Participate In "
                         "Practice", "report_primary_injury": "Achilles"}])
    r = season_health(w, inj).set_index("player_id").loc["ir"]
    assert r.games_played == 2
    assert r.weeks_to_season_end == 7          # never came back, though the report goes quiet
    assert r.inj_weeks_out == 1                # only the one week he was listed
