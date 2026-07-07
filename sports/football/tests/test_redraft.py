"""Tests for the redraft orchestration + rookie trim (no network/model — stages stubbed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data.availability import AvailabilityReport, DatasetProbe  # noqa: E402
from position_predictor.eval import redraft  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402

DRAFT_SEASON = 2026
FEATURE_SEASON = 2025


def _config(position):
    return Config({
        "experiment": {"sport": "football", "position": position},
        "data": {"earliest_season": 1999, "latest_completed_season": FEATURE_SEASON},
        "target": {"predict_horizon": 1},
    })


def _fake_proj(position, n=100, season=DRAFT_SEASON):
    return pd.DataFrame({
        "player_id": [f"{position}{i}" for i in range(n)],
        "player_name": [f"{position} {i}" for i in range(n)],
        "position": position,
        "proj_ppg": [float(n - i) for i in range(n)],
        "proj_pos_rank": list(range(1, n + 1)),
        "feature_season": FEATURE_SEASON,
        "proj_season": season,
    })


def _ok_report():
    return AvailabilityReport(
        season=FEATURE_SEASON,
        required=[DatasetProbe("seasonal", True, 300), DatasetProbe("weekly", True, 5000),
                  DatasetProbe("rosters", True, 900)],
        optional=[])


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub the gate + the three stages so run_redraft is pure orchestration."""
    monkeypatch.setattr(redraft, "check_season_available", lambda *a, **k: _ok_report())
    monkeypatch.setattr(redraft, "build_dataset", lambda cfg, write=True: None)
    monkeypatch.setattr(redraft, "build_features", lambda cfg, write=True: None)
    monkeypatch.setattr(
        redraft, "project_position",
        lambda cfg, write=False: _fake_proj(cfg.require("experiment.position").upper()))


# -- estimate_rookie_count ----------------------------------------------------------------

def test_estimate_rookie_count_counts_only_top_n_rookies():
    ecr = pd.DataFrame({
        "pos": ["QB"] * 5,
        "player": [f"Q{i}" for i in range(5)],
        "ecr": [1, 2, 3, 4, 5],
    })
    rookies = {"QB": {redraft._norm("Q1"), redraft._norm("Q3")}}
    # top 3 = Q0,Q1,Q2 -> only Q1 is a rookie inside the top 3.
    assert redraft.estimate_rookie_count(ecr, rookies, "QB", 3) == 1
    assert redraft.estimate_rookie_count(ecr, rookies, "QB", 5) == 2


def test_estimate_rookie_count_no_market_returns_zero():
    assert redraft.estimate_rookie_count(None, {"QB": {"x"}}, "QB", 20) == 0
    assert redraft.estimate_rookie_count(pd.DataFrame(), {}, "QB", 20) == 0


# -- run_redraft --------------------------------------------------------------------------

def test_run_redraft_full_n_without_rookie_adjustment(stub_pipeline):
    # Explicit top_n so this orchestration test doesn't depend on DEFAULT_TOP_N's actual values.
    res = redraft.run_redraft(
        [_config("QB"), _config("RB"), _config("WR")],
        draft_season=DRAFT_SEASON, refresh=False, top_n={"QB": 20, "RB": 50, "WR": 75},
        rookie_context=(None, {}, "no market"))
    assert res.ready
    counts = res.board.groupby("position").size().to_dict()
    assert counts == {"QB": 20, "RB": 50, "WR": 75}
    assert (res.board["proj_season"] == DRAFT_SEASON).all()


def test_run_redraft_trims_by_rookie_count(stub_pipeline, monkeypatch):
    monkeypatch.setattr(redraft, "estimate_rookie_count", lambda *a, **k: 3)
    res = redraft.run_redraft(
        [_config("QB")], draft_season=DRAFT_SEASON, refresh=False, top_n={"QB": 20},
        rookie_context=(pd.DataFrame(), {"QB": {"x"}}, ""))
    counts = res.board.groupby("position").size().to_dict()
    assert counts == {"QB": 17}                       # 20 − 3 rookies
    assert res.summaries[0].rookies_subtracted == 3
    assert res.summaries[0].returned == 17


def test_run_redraft_gate_failure_stops(monkeypatch):
    bad = AvailabilityReport(season=FEATURE_SEASON,
                             required=[DatasetProbe("weekly", False, 0, "unpublished")],
                             optional=[])
    monkeypatch.setattr(redraft, "check_season_available", lambda *a, **k: bad)
    res = redraft.run_redraft([_config("QB")], draft_season=DRAFT_SEASON, refresh=False)
    assert not res.ready
    assert res.board is None
    assert "weekly" in res.availability.missing


def test_run_redraft_rejects_wrong_projection_season(monkeypatch):
    monkeypatch.setattr(redraft, "check_season_available", lambda *a, **k: _ok_report())
    monkeypatch.setattr(redraft, "build_dataset", lambda cfg, write=True: None)
    monkeypatch.setattr(redraft, "build_features", lambda cfg, write=True: None)
    # projects 2025, not the requested 2026 -> stale data guard fires.
    monkeypatch.setattr(redraft, "project_position",
                        lambda cfg, write=False: _fake_proj("QB", season=2025))
    with pytest.raises(RuntimeError, match="!= draft season"):
        redraft.run_redraft([_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
                            rookie_context=(None, {}, ""))
