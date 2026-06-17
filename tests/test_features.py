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
    add_availability,
    add_efficiency,
    add_ngs_efficiency,
    add_offseason,
    add_player_attrs,
    add_production,
    add_regression_mean,
    add_snap_usage,
    add_team_context,
    add_trajectory,
    add_volume,
    ngs_season,
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

    out, cols = add_offseason(df, rosters, draft, horizon=1)
    x = out[out.player_id == "X"].iloc[0]
    assert x["changed_team_next"] == 1.0                  # TEN -> BAL
    assert x["rookie_rb_drafted_next"] == 1.0 and x["rookie_rb_capital_next"] == 100.0
    # competition = OTHER RBs' prior touches in BAL's 2023 room (Y=120; rookie Z=0)
    assert x["backfield_prior_touches_next"] == 120.0
    assert x["backfield_rb_count_next"] == 3.0
    y = out[out.player_id == "Y"].iloc[0]
    assert y["changed_team_next"] == 0.0                  # stayed on BAL
    assert y["backfield_prior_touches_next"] == 200.0     # now competes with X's 200
    assert set(cols) == {"changed_team_next", "rookie_rb_drafted_next", "rookie_rb_capital_next",
                         "rookie_rb_count_next", "backfield_prior_touches_next",
                         "backfield_rb_count_next"}


def test_add_offseason_sentinel_and_none_safe():
    df = pd.DataFrame([dict(player_id="X", season=2022, recent_team="TEN", touches=100.0)])
    rosters = pd.DataFrame([dict(player_id="X", season=2023, team="TEN", position="RB")])
    draft = pd.DataFrame([dict(season=2023, team="DAL", position="RB", pick=5)])  # not TEN
    out, _ = add_offseason(df, rosters, draft, horizon=1, udfa_pick=300)
    r = out.iloc[0]
    assert r["changed_team_next"] == 0.0                  # stayed on TEN
    assert r["rookie_rb_drafted_next"] == 0.0             # TEN drafted no RB
    assert r["rookie_rb_capital_next"] == 300.0           # sentinel = no threat
    out2, cols2 = add_offseason(df, None, draft)          # missing input -> skipped, no crash
    assert cols2 == [] and "changed_team_next" not in out2.columns
