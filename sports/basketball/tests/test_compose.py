"""No-network tests for Phase-2 composition logic (Fantrax fetch covered by the live run)."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.eval.compose import _norm, season_from_date, team_composition  # noqa: E402


def test_norm_strips_punct_suffix_and_accents():
    assert _norm("Jaren Jackson Jr.") == "jaren jackson"
    assert _norm("D'Angelo Russell") == "dangelo russell"
    # accent / "font" folding: Fantrax keeps diacritics, ESPN often doesn't — both fold equal
    assert _norm("Nikola Jokić") == _norm("Nikola Jokic") == "nikola jokic"
    assert _norm("Luka Dončić") == "luka doncic"
    assert _norm("Alperen Şengün") == "alperen sengun"
    assert _norm("Nikola Jović") == "nikola jovic"


def test_season_from_date_ending_year():
    assert season_from_date(date(2024, 12, 15)) == 2025   # Oct-Dec -> next year
    assert season_from_date(date(2025, 3, 1)) == 2025     # Jan-Sep -> same year


def _membership():
    return pd.DataFrame({
        "season": [2025, 2025, 2025],
        "player_name": ["Rudy Gobert", "Stephen Curry", "Nikola Jokic"],
        "arch": [0, 1, 0], "arch_name": ["Rim", "Shooter", "Rim"],
        "p0": [1.0, 0.0, 0.6], "p1": [0.0, 1.0, 0.4],
    })


def test_team_composition_shares_counts_and_match_rate():
    rosters = pd.DataFrame({
        "league_id": ["L", "L", "L"], "season": [2025, 2025, 2025],
        "team_id": ["T1", "T1", "T1"], "team_name": ["Team One"] * 3,
        "player_name": ["Rudy Gobert", "Stephen Curry", "Some Rookie"],  # 2 match, 1 doesn't
    })
    comp, match_rate, unmatched = team_composition(rosters, _membership())
    assert abs(match_rate - 2 / 3) < 1e-9
    assert list(unmatched["player_name"]) == ["Some Rookie"]
    row = comp.iloc[0]
    assert row["n_roster"] == 3 and row["n_matched"] == 2
    # matched soft vectors Gobert [1,0] + Curry [0,1] -> mean shares 0.5 / 0.5
    assert row["comp_Rim"] == 0.5 and row["comp_Shooter"] == 0.5
    # hard labels: Gobert -> Rim, Curry -> Shooter
    assert row["n_Rim"] == 1 and row["n_Shooter"] == 1


def test_team_composition_accent_folding_matches():
    # roster carries the diacritic, membership doesn't (or vice versa) -> still matches
    mem = _membership()
    mem.loc[mem.player_name == "Nikola Jokic", "player_name"] = "Nikola Jokic"  # ESPN: no accent
    rosters = pd.DataFrame({
        "league_id": ["L"], "season": [2025], "team_id": ["T"], "team_name": ["T"],
        "player_name": ["Nikola Jokić"],   # Fantrax: accented
    })
    comp, match_rate, _ = team_composition(rosters, mem)
    assert match_rate == 1.0 and comp.iloc[0]["n_matched"] == 1


def test_team_composition_alias_escape_hatch():
    rosters = pd.DataFrame({
        "league_id": ["L"], "season": [2025], "team_id": ["T"], "team_name": ["T"],
        "player_name": ["Steph Curry"],    # nickname; won't normalize to "Stephen Curry"
    })
    comp, mr, un = team_composition(rosters, _membership())
    assert mr == 0.0                                          # no match without an alias
    comp2, mr2, _ = team_composition(rosters, _membership(),
                                     aliases={"Steph Curry": "Stephen Curry"})
    assert mr2 == 1.0 and comp2.iloc[0]["comp_Shooter"] == 1.0
