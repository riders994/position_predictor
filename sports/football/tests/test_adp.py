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
    assert match == {"ranked": 3, "matched": 2, "source": "FFC"}      # 3 RB rows ranked, 2 matched
    # lower ADP = earlier pick = rank 1
    assert out.loc[out["player_id"] == "id_breece", "adp_pos_rank"].iloc[0] == 1
    assert out.loc[out["player_id"] == "id_bijan", "adp_pos_rank"].iloc[0] == 2
    assert (out["season"] == 2024).all()


def test_build_adp_benchmark_empty_when_no_match(monkeypatch):
    monkeypatch.setattr(adp, "fetch_ffc_adp", lambda season, teams=12: _raw())
    out, match = adp.build_adp_benchmark(_cfg("RB"), 2024, {}, write=False)
    assert out.empty and match == {"ranked": 3, "matched": 0, "source": "FFC"}


def test_falls_back_to_fantasypros_when_ffc_empty(monkeypatch):
    # FFC has a per-season hole (e.g. 2025) -> fall back to the FantasyPros board.
    def _ffc_down(season, teams=12):
        raise RuntimeError("FFC returned no ADP players")
    monkeypatch.setattr(adp, "fetch_ffc_adp", _ffc_down)
    monkeypatch.setattr(adp, "fetch_fantasypros_adp", lambda season: _raw())
    name_id_map = {_norm("Bijan Robinson"): "id_bijan", _norm("Breece Hall"): "id_breece"}

    out, match = adp.build_adp_benchmark(_cfg("RB"), 2025, name_id_map, write=False)

    assert set(out["player_id"]) == {"id_bijan", "id_breece"}
    assert match == {"ranked": 3, "matched": 2, "source": "FantasyPros"}


def test_fetch_fantasypros_adp_parses_table(monkeypatch):
    html = (
        "<table id='data'><tbody>"
        "<tr><td>1</td>"
        "<td class='player-label'><a class='fp-id-19788' fp-player-name=\"Ja'Marr Chase\" "
        "href='#'>Ja'Marr Chase</a> <small>CIN</small> <small>(10)</small></td>"
        "<td>WR1</td><td>1</td><td>1</td><td>1.0</td><td>1.0</td></tr>"
        "<tr><td>2</td>"
        "<td class='player-label'><a class='fp-id-23133' fp-player-name=\"Bijan Robinson\" "
        "href='#'>Bijan Robinson</a> <small>ATL</small> <small>(5)</small></td>"
        "<td>RB1</td><td>2</td><td>3</td><td>2.5</td><td>2.4</td></tr>"
        "</tbody></table>"
    )

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return html.encode()
    import urllib.request as ur
    monkeypatch.setattr(ur, "urlopen", lambda *a, **k: _Resp())

    df = adp.fetch_fantasypros_adp(2025)
    assert list(df.columns) == ["name", "position", "team", "adp", "times_drafted"]
    assert len(df) == 2
    assert df.iloc[0]["name"] == "Ja'Marr Chase"
    assert df.iloc[0]["position"] == "WR" and df.iloc[0]["team"] == "CIN"
    assert df.iloc[0]["adp"] == 1.0
    assert df.iloc[1]["position"] == "RB" and df.iloc[1]["adp"] == 2.5
