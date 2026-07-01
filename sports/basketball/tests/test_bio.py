"""Tests for the Phase-3 age source (name mapping + age math; network calls monkeypatched)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.data import bio  # noqa: E402


def test_norm_folds_accents_and_suffixes():
    assert bio._norm("Nikola Jokić") == "nikola jokic"
    assert bio._norm("Gary Payton II") == "gary payton"
    assert bio._norm("Karl-Anthony Towns") == "karlanthony towns"


def test_age_on_computes_midseason_age():
    # born 1994-02-19; at Feb 1 of 2020 that's just shy of 26
    assert bio._age_on("1994-02-19", 2020) == 25.9
    assert bio._age_on(None, 2020) is None
    assert bio._age_on("not-a-date", 2020) is None


def test_map_players_splits_unambiguous_unmatched_ambiguous(monkeypatch):
    monkeypatch.setattr(bio, "_nba_name_index", lambda: {
        "nikola jokic": [1], "marcus williams": [2, 3]})   # one clean, one ambiguous
    need = pd.DataFrame({"athlete_id": [10, 11, 12],
                         "player_name": ["Nikola Jokić", "Marcus Williams", "Nobody Here"]})
    mapped, unmatched, ambiguous = bio.map_players(need)
    assert mapped.set_index("athlete_id")["nba_id"].to_dict() == {10: 1}
    assert unmatched == ["Nobody Here"]
    assert ambiguous == ["Marcus Williams"]


def test_build_player_ages_derives_age(monkeypatch, tmp_path):
    monkeypatch.setattr(bio, "_nba_name_index", lambda: {"ada lovelace": [99]})
    monkeypatch.setattr(bio, "fetch_birthdates",
                        lambda ids, **kw: pd.DataFrame({"nba_id": [99], "birthdate": ["1990-06-15"]}))
    monkeypatch.setattr(bio, "_espn_dob", lambda: {})
    mem = pd.DataFrame({"athlete_id": [1, 1], "player_name": ["Ada Lovelace"] * 2,
                        "season": [2019, 2020]})
    ages, info = bio.build_player_ages(mem, write=False)
    got = ages.set_index("season")["age"].to_dict()
    assert got[2019] == bio._age_on("1990-06-15", 2019)
    assert got[2020] == bio._age_on("1990-06-15", 2020)
    assert info["coverage"] == 1.0
