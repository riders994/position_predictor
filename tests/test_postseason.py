"""Tests for the postseason report (orchestration + summary; no network/model)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor import eras as eras_mod  # noqa: E402
from position_predictor.data.availability import AvailabilityReport, DatasetProbe  # noqa: E402
from position_predictor.eval import postseason as ps  # noqa: E402
from position_predictor.eval import projection as proj  # noqa: E402
from position_predictor.models import era_ensemble as ee_mod  # noqa: E402
from position_predictor.utils import io as io_mod  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402
from position_predictor.utils.io import DATA_PROCESSED, ensure_dir  # noqa: E402


def _ok(season):
    return AvailabilityReport(season=season,
                              required=[DatasetProbe("seasonal", True, 300),
                                        DatasetProbe("weekly", True, 5000),
                                        DatasetProbe("rosters", True, 900)], optional=[])


def _bad(season):
    return AvailabilityReport(season=season,
                              required=[DatasetProbe("weekly", False, 0, "unpublished")],
                              optional=[])


# -- leak-safe projection param -----------------------------------------------------------

def test_project_position_excludes_board_season_from_training(monkeypatch):
    """feature_season=Y boards Y and trains only on season<Y (the no-leakage backtest)."""
    captured = {}

    class FakeEns:
        def __init__(self, *a, **k):
            pass

        def fit(self, train):
            captured["seasons"] = sorted(train["season"].unique().tolist())
            return self

        def predict(self, board):
            return list(range(len(board)))

    df = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(6)],
        "season": [2020, 2021, 2022, 2023, 2023, 2022],
        "target_ppg_next": [10.0, 11.0, 12.0, 13.0, 9.0, 8.0],  # all known (post-hoc)
        "player_display_name": [f"P {i}" for i in range(6)],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    monkeypatch.setattr(eras_mod, "load_eras", lambda cfg: [])
    monkeypatch.setattr(io_mod, "read_parquet", lambda path: df)
    monkeypatch.setattr(ee_mod, "EraEnsemble", FakeEns)
    blocks = ensure_dir(DATA_PROCESSED) / "football_testpos_feature_blocks.json"
    blocks.write_text('{"blk": ["x"]}')
    try:
        cfg = Config({"experiment": {"sport": "football", "position": "TESTPOS"},
                      "target": {"predict_horizon": 1}})
        out = proj.project_position(cfg, feature_season=2023, write=False)
    finally:
        blocks.unlink()

    assert captured["seasons"] == [2020, 2021, 2022]   # 2023 (the board/answer) excluded
    assert (out["feature_season"] == 2023).all()
    assert (out["proj_season"] == 2024).all()
    assert len(out) == 2                                # the two 2023 board rows


# -- pure summary helpers -----------------------------------------------------------------

def test_rank_accuracy_and_rank_error():
    actual_ppg = pd.Series([20.0, 18.0, 16.0, 14.0])
    actual_rank = pd.Series([1, 2, 3, 4])
    score = pd.Series([19.0, 17.0, 21.0, 10.0])   # ranks the 3rd-best first
    pred_rank = pd.Series([2, 3, 1, 4])
    m = ps._rank_accuracy(actual_ppg, actual_rank, score, pred_rank, (12, 24))
    assert m["n"] == 4
    assert -1.0 <= m["spearman"] <= 1.0
    assert m["mean_abs_rank_err"] == pytest.approx((1 + 1 + 2 + 0) / 4)


def test_highlights_partition():
    board = pd.DataFrame({
        "player_name": ["Hit", "Bust", "Value", "Reach"],
        "model_rank": [1, 2, 30, 5],
        "ecr_rank": [1, 2, 25, 6],
        "adp_rank": [1, 3, 40, 6],
        "actual_rank": [1, 50, 4, 60],
        "actual_ppg": [22.0, 3.0, 18.0, 2.0],
    })
    h = ps._highlights(board, tier1=12)
    assert any(r["player_name"] == "Hit" for r in h["hits"])
    assert any(r["player_name"] == "Bust" for r in h["busts"])
    assert h["values"][0]["player_name"] == "Value"      # actual 4 vs adp 40 = +36
    assert h["reaches"][0]["player_name"] == "Reach"      # actual 60 vs adp 6 = -54


# -- orchestration ------------------------------------------------------------------------

def test_resolve_season_autodetects_latest_published(monkeypatch):
    ready = {2024}
    monkeypatch.setattr(ps, "check_season_available",
                        lambda s: _ok(s) if s in ready else _bad(s))
    season, rep = ps._resolve_season(None, Config({}))
    assert season == 2024 and rep.ok


def test_build_postseason_gate_failure(monkeypatch):
    monkeypatch.setattr(ps, "check_season_available", lambda s: _bad(s))
    res = ps.build_postseason_report([Config({"experiment": {"position": "RB"}})],
                                     season=2025, refresh=False)
    assert not res.ready and res.board is None and "weekly" in res.availability.missing


def test_build_postseason_aggregates_positions(monkeypatch):
    monkeypatch.setattr(ps, "check_season_available", lambda s: _ok(s))

    def fake_pos_report(cfg, season):
        pos = cfg.require("experiment.position").upper()
        board = pd.DataFrame({"position": [pos, pos], "player_name": [f"{pos}1", f"{pos}2"],
                              "actual_rank": [1, 2], "model_rank": [1, 3], "ecr_rank": [1, 2],
                              "adp_rank": [1, 2], "actual_ppg": [20.0, 15.0]})
        summary = [{"position": pos, "source": "model", "spearman": 0.9, "n": 2}]
        return board, summary, {"hits": [], "busts": []}, {"ranked": 5, "matched": 4}

    monkeypatch.setattr(ps, "_position_report", fake_pos_report)
    res = ps.build_postseason_report(
        [Config({"experiment": {"position": "RB"}, "data": {"earliest_season": 1999}}),
         Config({"experiment": {"position": "WR"}, "data": {"earliest_season": 1999}})],
        season=2024, refresh=False)
    assert res.ready
    assert set(res.board["position"]) == {"RB", "WR"} and len(res.board) == 4
    assert {s["position"] for s in res.summary} == {"RB", "WR"}
    assert res.adp_match["RB"] == {"ranked": 5, "matched": 4}
