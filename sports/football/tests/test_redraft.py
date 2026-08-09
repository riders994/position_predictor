"""Tests for the redraft orchestration + rookie trim (no network/model — stages stubbed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data.availability import AvailabilityReport, DatasetProbe  # noqa: E402
from position_predictor.eval import redraft  # noqa: E402
from position_predictor.eval.league import league_from_dict  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402

DRAFT_SEASON = 2026
FEATURE_SEASON = 2025


def _league(**over):
    """The default 12-team 1QB PPR shape, with overrides."""
    base = {"name": "test", "label": "test", "scoring": "ppr", "teams": 12,
            "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
            "flex_positions": ["RB", "WR"], "roster_size": 16}
    return league_from_dict({**base, **over})


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
    res = redraft.run_redraft(
        [_config("QB"), _config("RB"), _config("WR")],
        draft_season=DRAFT_SEASON, refresh=False, rookie_context=(None, {}, "no market"))
    assert res.ready
    counts = res.board.groupby("position").size().to_dict()
    # League-derived depth for 12-team 1QB with a TE-eligible flex (the flex share splits three
    # ways), scaled so the four positions together cover all 12*16 = 192 picks:
    # QB 23, RB 55, WR 83 (TE 31 isn't requested here).
    assert counts == {"QB": 23, "RB": 55, "WR": 83}
    assert (res.board["proj_season"] == DRAFT_SEASON).all()


def test_run_redraft_trims_by_rookie_count(stub_pipeline, monkeypatch):
    monkeypatch.setattr(redraft, "estimate_rookie_count", lambda *a, **k: 3)
    res = redraft.run_redraft(
        [_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
        rookie_context=(pd.DataFrame(), {"QB": {"x"}}, ""))
    counts = res.board.groupby("position").size().to_dict()
    assert counts == {"QB": 20}                       # 23 − 3 rookies
    assert res.summaries[0].rookies_subtracted == 3
    assert res.summaries[0].returned == 20


def test_explicit_top_n_overrides_league_depth(stub_pipeline):
    res = redraft.run_redraft([_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
                              top_n={"QB": 8}, rookie_context=(None, {}, ""))
    assert res.board.groupby("position").size().to_dict() == {"QB": 8}


# -- board_depth --------------------------------------------------------------------------

def test_board_depth_follows_started_slots():
    # A 10-team 2QB league starts 20 QBs, so a top-20 board would be all starters and no bench.
    assert redraft.board_depth(_league(teams=10, starters={"QB": 2, "RB": 2, "WR": 2, "TE": 1,
                                                           "FLEX": 1}))["QB"] == 35
    # 1QB stays far shallower.
    assert redraft.board_depth(_league())["QB"] == 23


def test_board_depth_counts_flex_eligibility():
    """A TE-eligible flex (Underdog) deepens the TE board; an RB/WR flex does not."""
    shape = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1}
    rb_wr = redraft.board_depth(_league(starters=shape, flex_positions=["RB", "WR"]))
    rb_wr_te = redraft.board_depth(_league(starters=shape, flex_positions=["RB", "WR", "TE"]))
    assert rb_wr_te["TE"] > rb_wr["TE"]


def test_board_depth_never_shallower_than_defaults():
    tiny = redraft.board_depth(_league(teams=4))
    for pos, floor in redraft.DEFAULT_TOP_N.items():
        assert tiny[pos] >= floor


def test_board_depth_covers_every_pick_in_the_draft():
    """An 18-round Underdog draft spends 216 picks; the board must not stop short of that."""
    deep = _league(teams=12, roster_size=18,
                   starters={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1},
                   flex_positions=["RB", "WR", "TE"])
    assert sum(redraft.board_depth(deep).values()) >= deep.total_picks


def test_explicit_depth_overrides_are_not_rescaled():
    deep = _league(teams=12, roster_size=18, top_n={"QB": 12})
    assert redraft.board_depth(deep)["QB"] == 12
    assert redraft.board_depth(deep, overrides={"QB": 5})["QB"] == 5


# -- leagues ------------------------------------------------------------------------------

def test_one_pipeline_pass_per_scoring_not_per_league(stub_pipeline, monkeypatch):
    """Three leagues over two formats must build datasets twice, not three times."""
    calls = []
    monkeypatch.setattr(redraft, "build_dataset",
                        lambda cfg, write=True: calls.append(cfg.get("target.scoring")))
    leagues = [_league(name="a", scoring="ppr"), _league(name="b", scoring="half_ppr"),
               _league(name="c", scoring="half_ppr", teams=10)]
    res = redraft.run_redraft([_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
                              leagues=leagues, rookie_context=(None, {}, ""))
    assert sorted(calls) == ["half_ppr", "ppr"]        # one pass per format, not per league
    assert [lb.league.name for lb in res.leagues] == ["a", "b", "c"]
    assert all(lb.board is not None and not lb.board.empty for lb in res.leagues)


def test_board_carries_vorp_and_overall_rank(stub_pipeline):
    res = redraft.run_redraft([_config("QB"), _config("RB")], draft_season=DRAFT_SEASON,
                              refresh=False, rookie_context=(None, {}, ""))
    board = res.leagues[0].board
    assert {"vorp", "proj_overall_rank", "league"} <= set(board.columns)
    assert board["vorp"].is_monotonic_decreasing
    assert list(board["proj_overall_rank"]) == list(range(1, len(board) + 1))
    assert set(res.leagues[0].replacement) == {"QB", "RB", "WR", "TE"}


def test_two_qb_league_lifts_qbs_up_the_board(stub_pipeline):
    """The whole point of a roster config: doubling started QBs must raise QB board position."""
    cfgs = [_config("QB"), _config("RB"), _config("WR")]
    res = redraft.run_redraft(
        cfgs, draft_season=DRAFT_SEASON, refresh=False, rookie_context=(None, {}, ""),
        leagues=[_league(name="one", starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}),
                 _league(name="two", starters={"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1})])
    one, two = (lb.board for lb in res.leagues)
    qb_top_one = int(one[one["position"] == "QB"]["proj_overall_rank"].min())
    qb_top_two = int(two[two["position"] == "QB"]["proj_overall_rank"].min())
    assert qb_top_two < qb_top_one
    assert res.leagues[1].starters["QB"] == 24        # 12 teams x 2 slots


def test_board_cut_at_total_picks(stub_pipeline):
    res = redraft.run_redraft(
        [_config("QB"), _config("RB"), _config("WR")], draft_season=DRAFT_SEASON, refresh=False,
        rookie_context=(None, {}, ""), leagues=[_league(teams=10, roster_size=12)])
    assert len(res.leagues[0].board) <= 120           # 10 teams x 12 rounds


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
