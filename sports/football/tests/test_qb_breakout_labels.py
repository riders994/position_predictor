"""Tests for the late-breakout QB label engine (pure; synthetic fixtures).

The label is the whole project — a QB in the wrong cell is a training example pointing the model
the wrong way — so these cover the decisions that are easy to get quietly wrong: how a breakout
is defined, how careers are indexed in time, and how team identity is compared.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qb_breakout.labels.cohort import (  # noqa: E402
    build_qb_careers,
    build_qb_seasons,
    first_sustained_breakout,
    rank_qb_seasons,
    resolve_entry,
)
from qb_breakout.labels.teams import canonical_team  # noqa: E402


def _weekly(rows):
    """Build a weekly frame from (player, season, week, team, points) tuples."""
    return pd.DataFrame(
        [
            dict(player_id=p, player_display_name=p, position="QB", season_type="REG",
                 season=s, week=w, recent_team=t, fantasy_points_ppr=pts, attempts=30)
            for p, s, w, t, pts in rows
        ]
    )


def _season(player, season, team, ppg, weeks=16):
    return [(player, season, w, team, ppg) for w in range(1, weeks + 1)]


def _field(seasons, n=25, ppg=18.0):
    """Filler starters so ranks are meaningful.

    Ranking is within-season, so a fixture holding one QB makes that QB rank 1 and therefore a
    breakout no matter how badly he played. Tests about anything other than ranking need a field
    around the subject.
    """
    rows = []
    for s in seasons:
        for i in range(n):
            rows += _season(f"Filler{i}", s, "DAL", ppg)
    return rows


# --------------------------------------------------------------------------- season aggregation


def test_games_and_ppg_come_from_regular_season_weeks_only():
    weekly = _weekly(_season("QB1", 2015, "GB", 20.0, weeks=10))
    weekly = pd.concat([weekly, _weekly([("QB1", 2015, 20, "GB", 99.0)]).assign(
        season_type="POST")])
    out = build_qb_seasons(weekly)
    assert out.loc[0, "games"] == 10
    assert out.loc[0, "ppg"] == pytest.approx(20.0)


def test_non_qb_rows_are_dropped():
    weekly = _weekly(_season("QB1", 2015, "GB", 20.0))
    weekly = pd.concat([weekly, _weekly(_season("RB1", 2015, "GB", 25.0)).assign(position="RB")])
    out = build_qb_seasons(weekly)
    assert set(out["player_id"]) == {"QB1"}


# ------------------------------------------------------------------------------------- ranking


def test_seasons_below_the_games_cutoff_are_unranked_and_cannot_break_out():
    rows = _season("Starter", 2015, "GB", 10.0, weeks=16) + _season("Backup", 2015, "GB", 40.0,
                                                                    weeks=3)
    ranked = rank_qb_seasons(build_qb_seasons(_weekly(rows)), min_games=7)
    backup = ranked[ranked.player_id == "Backup"].iloc[0]
    # A three-game hot streak outscores everyone but is not a breakout.
    assert pd.isna(backup["ppg_rank"])
    assert not backup["is_breakout"]
    assert ranked[ranked.player_id == "Starter"].iloc[0]["is_breakout"]


# --------------------------------------------------------------------------- sustained breakout


def _ranked_from_ranks(player, ranks_by_season):
    """Hand-build a ranked frame so sustain logic is tested independently of scoring."""
    return pd.DataFrame([
        dict(player_id=player, season=s, ppg_rank=r, team="GB",
             is_breakout=(r is not None and r <= 20))
        for s, r in ranks_by_season.items()
    ])


def test_one_off_top20_season_does_not_count_as_sustained():
    """The Mayfield-2018 case: rank 20 once, then nothing, is not a breakout."""
    ranked = _ranked_from_ranks("Baker", {2018: 20, 2019: 27, 2020: 24, 2021: 28})
    out = first_sustained_breakout(ranked, latest_season=2021).iloc[0]
    assert pd.isna(out["sustained_season"])
    assert not out["sustained_censored"]


def test_tier_that_holds_counts_from_its_first_season():
    ranked = _ranked_from_ranks("Baker", {2018: 20, 2019: 27, 2023: 17, 2024: 4, 2025: 19})
    out = first_sustained_breakout(ranked, latest_season=2025).iloc[0]
    assert out["sustained_season"] == 2023


def test_seasons_missed_entirely_count_against_the_window():
    """A QB benched for two years has not sustained a tier, even with no bad ranks on record."""
    ranked = _ranked_from_ranks("Gap", {2015: 10, 2018: 12})
    out = first_sustained_breakout(ranked, latest_season=2020).iloc[0]
    assert pd.isna(out["sustained_season"])


def test_open_window_is_censored_not_rejected():
    """The Darnold case: one top-20 season with the window still running is pending, not a no."""
    ranked = _ranked_from_ranks("Darnold", {2024: 9})
    out = first_sustained_breakout(ranked, latest_season=2024).iloc[0]
    assert pd.isna(out["sustained_season"])
    assert out["sustained_censored"]


# --------------------------------------------------------------------------- franchise identity


@pytest.mark.parametrize("a,b", [
    ("GNB", "GB"),      # PFR vs nflverse abbreviation styles
    ("NWE", "NE"),
    ("SFO", "SF"),
    ("SDG", "LAC"),     # Chargers relocation — same building
    ("STL", "LA"),      # Rams relocation
    ("OAK", "LV"),      # Raiders relocation
    ("JAC", "JAX"),
])
def test_same_franchise_written_two_ways_compares_equal(a, b):
    assert canonical_team(a) == canonical_team(b)


def test_genuinely_different_franchises_stay_distinct():
    assert canonical_team("NYJ") != canonical_team("NYG")
    assert canonical_team("CLE") != canonical_team("BAL")


def test_relocation_flag_is_false_when_only_the_abbreviation_style_differs():
    """Aaron Rodgers was drafted GNB and broke out for GB — he never left."""
    rows = (_season("Rodgers", 2005, "GB", 3.0, weeks=3)
            + _season("Rodgers", 2008, "GB", 22.0)
            + _season("Rodgers", 2009, "GB", 23.0))
    ranked = rank_qb_seasons(build_qb_seasons(_weekly(rows)))
    draft = pd.DataFrame([dict(gsis_id="Rodgers", season=2005, round=1, pick=24, team="GNB",
                               position="QB", college="California")])
    careers = build_qb_careers(ranked, draft, None, latest_season=2010)
    row = careers.iloc[0]
    assert row["sustained_relocated"] == 0
    assert row["late_sustained"] == 1  # 2008 is NFL year 4


# ------------------------------------------------------------------------------ entry resolution


def test_draft_season_beats_first_observed_season():
    draft = pd.DataFrame([dict(player_id="A", draft_season=2005)])
    rosters = pd.DataFrame([dict(player_id="A", entry_year=2007, rookie_year=2007)])
    out = resolve_entry(["A"], draft, rosters).iloc[0]
    assert out["entry_season"] == 2005
    assert out["entry_source"] == "draft"


def test_roster_entry_year_covers_undrafted_players():
    draft = pd.DataFrame(columns=["player_id", "draft_season"])
    rosters = pd.DataFrame([dict(player_id="A", entry_year=2003, rookie_year=2003)])
    out = resolve_entry(["A"], draft, rosters).iloc[0]
    assert out["entry_season"] == 2003
    assert out["entry_source"] == "roster"


def test_veteran_predating_the_data_window_is_dropped_not_guessed():
    """Testaverde-style: first appears in 1999 with no entry signal, so he leaves the cohort.

    Imputing entry from the first observed season would call a 12-year veteran a rookie and put
    every later season at the wrong NFL year index.
    """
    rows = _season("Vinny", 1999, "NYJ", 18.0) + _season("Vinny", 2000, "NYJ", 18.0)
    ranked = rank_qb_seasons(build_qb_seasons(_weekly(rows)))
    careers = build_qb_careers(ranked, pd.DataFrame(columns=["gsis_id", "season", "position"]),
                               None, latest_season=2005)
    assert "Vinny" not in set(careers["player_id"])


def test_debut_inside_the_window_falls_back_to_first_season():
    rows = _season("Later", 2010, "NYJ", 18.0) + _season("Later", 2011, "NYJ", 18.0)
    ranked = rank_qb_seasons(build_qb_seasons(_weekly(rows)))
    careers = build_qb_careers(ranked, pd.DataFrame(columns=["gsis_id", "season", "position"]),
                               None, latest_season=2015)
    row = careers[careers.player_id == "Later"].iloc[0]
    assert row["entry_season"] == 2010
    assert row["entry_source"] == "first_season"


# ------------------------------------------------------------------------------------ censoring


def test_recent_entrant_is_censored_rather_than_labelled_a_failure():
    rows = _season("Rookie", 2025, "NYJ", 5.0) + _field([2025])
    ranked = rank_qb_seasons(build_qb_seasons(_weekly(rows)))
    draft = pd.DataFrame([dict(gsis_id="Rookie", season=2025, round=1, pick=5, team="NYJ",
                               position="QB", college="Somewhere")])
    careers = build_qb_careers(ranked, draft, None, latest_season=2025)
    row = careers[careers.player_id == "Rookie"].iloc[0]
    assert row["censored"]
    assert row["cell"] == "censored"
