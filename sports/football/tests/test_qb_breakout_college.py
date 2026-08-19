"""Tests for the college production layer: aggregation, career shape, and the cohort join.

Covers the things that are silently wrong rather than loudly broken — sacks leaking into rushing,
truncated careers reading as short ones, and a name join that guesses.
"""

import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qb_breakout.data.college import (  # noqa: E402
    FIRST_PBP_SEASON,
    _normalise_dtypes,
    add_career_features,
    aggregate_qb_seasons,
    mask_unreliable_flags,
    season_flag_quality,
)
from qb_breakout.data.college_link import link_college_to_cohort, normalize_school  # noqa: E402


def _play(season=2015, team="Oklahoma", game=1, passer=None, rusher=None, *, attempt=False,
          completion=False, pass_yds=0.0, pass_td=False, interception=False, sack=False,
          sack_yds=0.0, rush=False, rush_yds=0.0, rush_td=False, epa=0.0, success=False,
          opponent=1):
    return {
        "season": season, "week": 1, "game_id": game, "start.pos_team.name": team,
        "def_pos_team": opponent, "passer_player_name": passer, "rusher_player_name": rusher,
        "pass_attempt": attempt, "completion": completion, "yds_receiving": pass_yds,
        "pass_td": pass_td, "int": interception, "sack": sack, "yds_sacked": sack_yds,
        "rush": rush, "yds_rushed": rush_yds, "rush_td": rush_td, "EPA": epa,
        "EPA_success": success, "scrimmage_play": True,
    }


def _pbp(rows):
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------------- aggregation


def test_dropbacks_include_sacks_but_rushing_does_not():
    """NCAA charges sack yardage to rushing; splitting them is the point of using play-by-play.

    A QB who is sacked has not run the ball. Folding sacks into rushing is what makes official
    college rushing totals understate mobile QBs.
    """
    rows = (
        [_play(passer="QB", attempt=True, completion=True, pass_yds=10.0) for _ in range(60)]
        + [_play(passer="QB", sack=True, sack_yds=-7.0) for _ in range(5)]
        + [_play(rusher="QB", rush=True, rush_yds=8.0) for _ in range(10)]
    )
    out = aggregate_qb_seasons(_pbp(rows)).to_dicts()[0]
    assert out["dropbacks"] == 65          # 60 attempts + 5 sacks
    assert out["attempts"] == 60
    assert out["sacks"] == 5
    assert out["rush_att"] == 10           # sacks excluded
    assert out["rush_yds"] == 80.0         # sack yardage not netted out
    assert out["sack_rate"] == pytest.approx(5 / 65)


def test_rush_share_separates_mobile_from_pocket_quarterbacks():
    pocket = [_play(passer="P", attempt=True) for _ in range(90)] + [
        _play(rusher="P", rush=True, rush_yds=1.0) for _ in range(10)]
    mobile = [_play(passer="M", team="Auburn", attempt=True) for _ in range(60)] + [
        _play(rusher="M", team="Auburn", rush=True, rush_yds=6.0) for _ in range(40)]
    out = aggregate_qb_seasons(_pbp(pocket + mobile))
    shares = dict(zip(out["player"], out["rush_share"]))
    assert shares["P"] == pytest.approx(0.10)
    assert shares["M"] == pytest.approx(0.40)


def test_low_volume_passers_are_dropped_as_non_quarterbacks():
    """A wildcat back or a holder throwing twice is not a QB-season."""
    rows = ([_play(passer="Real", attempt=True) for _ in range(60)]
            + [_play(passer="TrickPlay", attempt=True) for _ in range(3)])
    out = aggregate_qb_seasons(_pbp(rows))
    assert set(out["player"]) == {"Real"}


def test_efficiency_uses_all_dropbacks_not_just_attempts():
    rows = ([_play(passer="QB", attempt=True, epa=1.0, success=True) for _ in range(60)]
            + [_play(passer="QB", sack=True, epa=-3.0) for _ in range(20)])
    out = aggregate_qb_seasons(_pbp(rows)).to_dicts()[0]
    assert out["pass_epa_per_db"] == pytest.approx((60 * 1.0 + 20 * -3.0) / 80)
    assert out["pass_success_rate"] == pytest.approx(60 / 80)


def test_flag_columns_typed_as_floats_are_cast():
    """2009 ships `rush` as Float64 where every other season has Boolean."""
    df = pl.DataFrame({"rush": [1.0, 0.0], "pass_attempt": [True, False]})
    out = _normalise_dtypes(df)
    assert out.schema["rush"] == pl.Boolean
    assert out["rush"].to_list() == [True, False]


# --------------------------------------------------------------------- unpopulated play flags


def _seasons(rows):
    return pl.DataFrame(rows)


def test_a_season_that_never_records_a_flag_is_nulled_not_zeroed():
    """cfbfastR records no interceptions in 2006-2013; that must not read as ball security.

    This is the defect's whole danger: an unpopulated flag sums to a clean zero, so nothing is
    missing and nothing looks wrong — the QB simply appears never to have thrown a pick.
    """
    rows = ([_season_row(player=f"A{i}", season=2010, interceptions=0, int_rate=0.0)
             for i in range(5)]
            + [_season_row(player=f"B{i}", season=2015) for i in range(5)])
    out = mask_unreliable_flags(_seasons(rows))

    bad = out.filter(pl.col("season") == 2010)
    assert bad["int_rate"].null_count() == 5      # nulled, not left at 0.0
    assert bad["interceptions"].null_count() == 5
    assert set(bad["int_recorded"].to_list()) == {0}

    good = out.filter(pl.col("season") == 2015)
    assert good["int_rate"].null_count() == 0
    assert set(good["int_recorded"].to_list()) == {1}


def test_a_missing_sack_flag_invalidates_everything_derived_from_it():
    """2013 has no sack plays at all, so sack rate *and* adjusted yards per dropback are void."""
    rows = [_season_row(player=f"A{i}", season=2013, sacks=0, sack_yds=0.0, sack_rate=0.0)
            for i in range(5)]
    out = mask_unreliable_flags(_seasons(rows))
    for col in ("sacks", "sack_yds", "sack_rate", "adj_yards_per_dropback"):
        assert out[col].null_count() == 5, col


def test_masking_is_idempotent():
    """Re-running the repair on already-repaired data must not change it."""
    rows = ([_season_row(player=f"A{i}", season=2010, interceptions=0, int_rate=0.0)
             for i in range(5)]
            + [_season_row(player=f"B{i}", season=2015) for i in range(5)])
    once = mask_unreliable_flags(_seasons(rows))
    assert mask_unreliable_flags(once).equals(once)


def test_flag_quality_reports_the_evidence_per_season():
    rows = ([_season_row(player=f"A{i}", season=2010, interceptions=0, int_rate=0.0)
             for i in range(3)]
            + [_season_row(player=f"B{i}", season=2015) for i in range(3)])
    q = season_flag_quality(_seasons(rows))
    ints = q.filter(pl.col("flag") == "interceptions").sort("season")
    assert ints["recorded"].to_list() == [0, 1]
    assert ints["rate"].to_list()[0] == 0.0


def test_a_career_total_is_not_summed_across_an_unrecorded_season():
    """8 + (unrecorded) is not 8. A partial total is worse than no total."""
    rows = mask_unreliable_flags(_seasons([
        _season_row(player="QB", season=2012, interceptions=0, int_rate=0.0),
        _season_row(player="QB", season=2015, interceptions=8),
        _season_row(player="Clean", season=2015, interceptions=8),
        _season_row(player="Clean", season=2016, interceptions=6),
    ]))
    career = add_career_features(rows).to_pandas().set_index("player")
    assert pd.isna(career.loc["QB", "career_int"])
    assert career.loc["QB", "int_seasons_recorded"] == 1
    assert career.loc["Clean", "career_int"] == 14


# ------------------------------------------------------------------------------- career shape


def _season_row(player="QB", season=2015, team="Oklahoma", epa=0.2, **kw):
    base = dict(player=player, season=season, team=team, games=12, dropbacks=300, attempts=280,
                completions=180, pass_yds=2500.0, pass_td=20, interceptions=8, sacks=20,
                sack_yds=-120.0, pass_epa=60.0, pass_epa_per_db=epa, pass_success_rate=0.5,
                n_opponents=12, rush_att=60, rush_yds=300.0, rush_td=4,
                rush_epa_per_att=0.1, rush_success_rate=0.45, completion_pct=0.64,
                yards_per_attempt=8.9, td_rate=0.07, int_rate=0.03, sack_rate=0.07,
                adj_yards_per_dropback=8.7, rush_yds_per_att=5.0, rush_share=0.17,
                pass_yds_per_game=208.0)
    base.update(kw)
    return base


def test_career_spanning_the_coverage_floor_is_flagged_as_truncated():
    """Rodgers reads as a one-year starter only because 2004 is the first covered season."""
    out = add_career_features(_seasons([_season_row(season=FIRST_PBP_SEASON)])).to_dicts()[0]
    assert out["college_career_truncated"] == 1
    assert out["n_college_seasons"] == 1


def test_career_starting_after_the_floor_is_not_flagged():
    rows = [_season_row(season=2013), _season_row(season=2014), _season_row(season=2015)]
    out = add_career_features(_seasons(rows)).to_dicts()[0]
    assert out["college_career_truncated"] == 0
    assert out["n_college_seasons"] == 3


def test_epa_trend_and_peak_season_capture_career_shape():
    """Improving across college is the pre-NFL analogue of the late-breakout hypothesis."""
    rows = [_season_row(season=2013, epa=0.0), _season_row(season=2014, epa=0.1),
            _season_row(season=2015, epa=0.4)]
    out = add_career_features(_seasons(rows)).to_dicts()[0]
    assert out["epa_trend"] == pytest.approx(0.4)
    assert out["breakout_season_idx"] == 3
    assert out["peaked_in_final_season"] == 1


def test_peaking_early_is_distinguished_from_peaking_late():
    rows = [_season_row(season=2013, epa=0.5), _season_row(season=2014, epa=0.1),
            _season_row(season=2015, epa=0.0)]
    out = add_career_features(_seasons(rows)).to_dicts()[0]
    assert out["breakout_season_idx"] == 1
    assert out["peaked_in_final_season"] == 0
    assert out["epa_trend"] == pytest.approx(-0.5)


def test_multiple_teams_flag_a_transfer():
    rows = [_season_row(season=2013, team="Texas Tech"), _season_row(season=2015, team="Oklahoma")]
    out = add_career_features(_seasons(rows)).to_dicts()[0]
    assert out["transferred"] == 1
    assert out["n_teams"] == 2


# ------------------------------------------------------------------------------ school names


@pytest.mark.parametrize("a,b", [
    ("Michigan St.", "Michigan State"),
    ("Utah St.", "Utah State"),
    ("Ole Miss", "Mississippi"),
    ("Pitt", "Pittsburgh"),
    ("USC", "Southern California"),
    ("Miami (FL)", "Miami"),
])
def test_school_spellings_across_sources_compare_equal(a, b):
    assert normalize_school(a) == normalize_school(b)


def test_distinct_schools_stay_distinct():
    assert normalize_school("Michigan") != normalize_school("Michigan State")


# ------------------------------------------------------------------------------------ joining


def _careers(rows):
    return pd.DataFrame(rows)


def _college(rows):
    return pd.DataFrame(rows)


def test_unique_in_window_match_attaches_the_college_profile():
    careers = _careers([dict(player_id="A", player_name="Baker Mayfield", entry_season=2018,
                             draft_college="Oklahoma")])
    college = _college([dict(player="Baker Mayfield", last_season=2017, final_team="Oklahoma",
                             final_epa_per_db=0.47)])
    out = link_college_to_cohort(careers, college)
    assert out.loc[0, "college_matched"]
    assert out.loc[0, "final_epa_per_db"] == 0.47


def test_college_columns_that_collide_are_prefixed_not_duplicated():
    """Both frames have `last_season`, meaning different things."""
    careers = _careers([dict(player_id="A", player_name="X", entry_season=2018,
                             draft_college="Oklahoma", last_season=2024)])
    college = _college([dict(player="X", last_season=2017, final_team="Oklahoma")])
    out = link_college_to_cohort(careers, college)
    assert out["last_season"].iloc[0] == 2024          # NFL side untouched
    assert out["college_last_season"].iloc[0] == 2017  # college side prefixed
    assert list(out.columns).count("last_season") == 1


def test_school_breaks_a_tie_between_same_named_quarterbacks():
    careers = _careers([dict(player_id="A", player_name="John Smith", entry_season=2018,
                             draft_college="Michigan St.")])
    college = _college([
        dict(player="John Smith", last_season=2017, final_team="Michigan State", tag="right"),
        dict(player="John Smith", last_season=2016, final_team="Alabama", tag="wrong"),
    ])
    out = link_college_to_cohort(careers, college)
    assert out.loc[0, "college_match_reason"] == "school_match"
    assert out.loc[0, "tag"] == "right"


def test_college_career_ending_after_nfl_entry_is_rejected():
    careers = _careers([dict(player_id="A", player_name="John Smith", entry_season=2018,
                             draft_college="Oklahoma")])
    college = _college([dict(player="John Smith", last_season=2020, final_team="Oklahoma")])
    out = link_college_to_cohort(careers, college)
    assert not out.loc[0, "college_matched"]
    assert out.loc[0, "college_match_reason"] == "name_match_out_of_window"


def test_indistinguishable_candidates_are_refused():
    careers = _careers([dict(player_id="A", player_name="John Smith", entry_season=2018,
                             draft_college=None)])
    college = _college([
        dict(player="John Smith", last_season=2016, final_team="Alabama"),
        dict(player="John Smith", last_season=2016, final_team="Auburn"),
    ])
    out = link_college_to_cohort(careers, college)
    assert not out.loc[0, "college_matched"]
    assert out.loc[0, "college_match_reason"] == "ambiguous"


def test_undrafted_quarterbacks_can_still_match_without_a_school():
    """Romo/Keenum/Hill have no draft_college, so school cannot be required."""
    careers = _careers([dict(player_id="A", player_name="Tony Romo", entry_season=2003,
                             draft_college=None)])
    college = _college([dict(player="Tony Romo", last_season=2002, final_team="Eastern Illinois")])
    out = link_college_to_cohort(careers, college)
    assert out.loc[0, "college_matched"]
