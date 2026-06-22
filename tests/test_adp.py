"""Tests for the FantasyFootballCalculator ADP benchmark (no network — fetch is stubbed)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data import adp  # noqa: E402
from position_predictor.eval.keeper import _norm  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def _cfg(position):
    return Config({"experiment": {"sport": "football", "position": position}})


def _raw():
    return pd.DataFrame([
        {"name": "Bijan Robinson", "position": "RB", "team": "ATL", "adp": 5.0, "times_drafted": 9},
        {"name": "Breece Hall", "position": "RB", "team": "NYJ", "adp": 2.0, "times_drafted": 9},
        {"name": "Ja'Marr Chase", "position": "WR", "team": "CIN", "adp": 1.0, "times_drafted": 9},
        {"name": "Nobody Special", "position": "RB", "team": "FA", "adp": 99.0, "times_drafted": 1},
    ])


def test_build_adp_benchmark_filters_maps_and_ranks(monkeypatch):
    monkeypatch.setattr(adp, "fetch_ffc_adp", lambda season, teams=12: _raw())
    name_id_map = {_norm("Bijan Robinson"): "id_bijan", _norm("Breece Hall"): "id_breece"}

    out, match = adp.build_adp_benchmark(_cfg("RB"), 2024, name_id_map, write=False)

    assert set(out["player_id"]) == {"id_bijan", "id_breece"}        # WR filtered, unmatched dropped
    assert match == {"ranked": 3, "matched": 2}                       # 3 RB rows ranked, 2 matched
    # lower ADP = earlier pick = rank 1
    assert out.loc[out["player_id"] == "id_breece", "adp_pos_rank"].iloc[0] == 1
    assert out.loc[out["player_id"] == "id_bijan", "adp_pos_rank"].iloc[0] == 2
    assert (out["season"] == 2024).all()


def test_build_adp_benchmark_empty_when_no_match(monkeypatch):
    monkeypatch.setattr(adp, "fetch_ffc_adp", lambda season, teams=12: _raw())
    out, match = adp.build_adp_benchmark(_cfg("RB"), 2024, {}, write=False)
    assert out.empty and match == {"ranked": 3, "matched": 0}
