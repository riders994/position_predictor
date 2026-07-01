"""No-network tests for the Yahoo Phase-2 augmentation logic (the live pull is exercised by a run)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.data.yahoo import (  # noqa: E402
    _end_season, _matchup_team_keys, _regular_weeks, season_long_composition, tally_category_wins)


def test_end_season_adds_one_to_yahoo_start_year():
    # Yahoo labels NBA seasons by START year; archetype membership is keyed by END year.
    assert _end_season({"season": "2023"}) == 2024
    assert _end_season({"season": 2015}) == 2016


def test_regular_weeks_excludes_playoffs():
    assert _regular_weeks({"start_week": 1, "playoff_start_week": 22, "end_week": 24}) == list(range(1, 22))
    # falls back to end_week when no playoff_start_week
    assert _regular_weeks({"start_week": 2, "end_week": 5}) == [2, 3, 4]


def _matchup(team_a, team_b, winners):
    """Build a minimal Yahoo matchup dict. ``winners`` maps stat_id -> team_key or '' (tie)."""
    def team(tk):
        return {"team": [[{"team_key": tk}, {"name": tk}]]}
    stat_winners = []
    for sid, tk in winners.items():
        sw = {"stat_id": sid}
        if tk:
            sw["winner_team_key"] = tk
        else:
            sw["is_tied"] = 1
        stat_winners.append({"stat_winner": sw})
    return {"is_playoffs": 0, "is_consolation": 0, "stat_winners": stat_winners,
            "0": {"teams": {"count": 2, "0": team(team_a), "1": team(team_b)}}}


def test_matchup_team_keys():
    mu = _matchup("L.t.1", "L.t.2", {"5": "L.t.1"})
    assert _matchup_team_keys(mu) == ["L.t.1", "L.t.2"]


def test_tally_category_wins_winner_and_ties():
    # A wins 6 cats, B wins 2, 1 tie -> A 6.5, B 2.5; both "played" all 9
    winners = {"5": "A", "8": "A", "10": "A", "12": "A", "15": "A", "16": "A",
               "17": "B", "18": "B", "19": ""}
    won, played = tally_category_wins([_matchup("A", "B", winners)], ["A", "B"])
    assert won["A"] == 6.5 and won["B"] == 2.5
    assert played["A"] == 9 and played["B"] == 9
    # symmetric: total wins == total plays / 2
    assert sum(won.values()) == sum(played.values()) / 2


def test_tally_skips_playoff_matchups():
    mu = _matchup("A", "B", {"5": "A"})
    mu["is_playoffs"] = 1
    won, played = tally_category_wins([mu], ["A", "B"])
    assert played["A"] == 0 and won["A"] == 0.0


def _membership():
    return pd.DataFrame({
        "season": [2024, 2024],
        "player_name": ["Rudy Gobert", "Stephen Curry"],
        "arch": [0, 1], "arch_name": ["Rim", "Shooter"],
        "p0": [1.0, 0.0], "p1": [0.0, 1.0],
    })


def test_season_long_composition_weights_by_weeks_on_roster():
    # Gobert rostered 3 weeks, Curry 1 week -> weeks-weighted shares 0.75 Rim / 0.25 Shooter.
    team_weeks = pd.DataFrame({
        "league_key": ["L"] * 4, "team_key": ["L.t.1"] * 4, "team_name": ["One"] * 4,
        "season": [2024] * 4, "week": [1, 2, 3, 1],
        "player_name": ["Rudy Gobert", "Rudy Gobert", "Rudy Gobert", "Stephen Curry"],
    })
    comp, match_rate, unmatched = season_long_composition(team_weeks, _membership())
    assert match_rate == 1.0 and len(unmatched) == 0
    row = comp.iloc[0]
    assert row["comp_Rim"] == 0.75 and row["comp_Shooter"] == 0.25
    assert row["n_player_weeks"] == 4 and row["matched_player_weeks"] == 4
    # hard counts are distinct players (not weeks-weighted)
    assert row["n_Rim"] == 1 and row["n_Shooter"] == 1


def test_season_long_composition_match_rate_is_weeks_weighted():
    # 1 matched player held all season (3 wk) + 1 unmatched 1-wk streamer -> match_rate 3/4, not 1/2.
    team_weeks = pd.DataFrame({
        "league_key": ["L"] * 4, "team_key": ["L.t.1"] * 4, "team_name": ["One"] * 4,
        "season": [2024] * 4, "week": [1, 2, 3, 1],
        "player_name": ["Rudy Gobert"] * 3 + ["Some Streamer"],
    })
    comp, match_rate, unmatched = season_long_composition(team_weeks, _membership())
    assert match_rate == 0.75
    assert list(unmatched["player_name"]) == ["Some Streamer"]
    assert unmatched.iloc[0]["weeks"] == 1
