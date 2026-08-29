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


# -- market board per league ----------------------------------------------------------------

def test_rookie_context_is_fetched_once_per_board_not_per_league(stub_pipeline, monkeypatch):
    """Two half-PPR leagues differing only in QB shape need two boards; a third shares one."""
    seen = []

    def fake_ctx(season, *, ecr_type=redraft.REDRAFT_OVERALL):
        seen.append(ecr_type)
        return (None, {}, "")

    monkeypatch.setattr(redraft, "_rookie_market_context", fake_ctx)
    leagues = [_league(name="one", scoring="half_ppr"),
               _league(name="two", scoring="half_ppr",
                       starters={"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}),
               _league(name="three", scoring="half_ppr", teams=10)]
    res = redraft.run_redraft([_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
                              leagues=leagues)
    assert sorted(seen) == ["ro", "rsf"]              # one fetch per board, not per league
    assert [lb.ecr_type for lb in res.leagues] == ["ro", "rsf", "ro"]


def test_two_qb_league_pulls_the_superflex_board(stub_pipeline, monkeypatch):
    """The rookie count for a 2QB league must come off the superflex chart."""
    asked = {}

    def fake_ctx(season, *, ecr_type=redraft.REDRAFT_OVERALL):
        # a board whose only rookie sits inside the QB top 5 on rsf but not on ro
        top = ["rookie qb"] if ecr_type == "rsf" else ["vet qb"]
        asked[ecr_type] = True
        return (pd.DataFrame({"player": top, "pos": ["QB"], "ecr": [1.0]}),
                {"QB": {"rookie qb"}}, "")

    monkeypatch.setattr(redraft, "_rookie_market_context", fake_ctx)
    res = redraft.run_redraft(
        [_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
        leagues=[_league(name="sf", starters={"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1})])
    assert asked == {"rsf": True}
    assert res.leagues[0].summaries[0].rookies_subtracted == 1


def test_pipeline_passes_split_on_board_as_well_as_scoring(stub_pipeline, monkeypatch):
    """Same format, different QB shape -> different rookie trim, so two passes are required."""
    calls = []
    monkeypatch.setattr(redraft, "build_dataset",
                        lambda cfg, write=True: calls.append(cfg.get("target.scoring")))
    leagues = [_league(name="one", scoring="ppr"),
               _league(name="two", scoring="ppr",
                       starters={"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1})]
    redraft.run_redraft([_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
                        leagues=leagues, rookie_context=(None, {}, ""))
    assert calls == ["ppr", "ppr"]


def test_rookie_note_names_the_board_that_failed(stub_pipeline, monkeypatch):
    monkeypatch.setattr(redraft, "_rookie_market_context",
                        lambda s, *, ecr_type=redraft.REDRAFT_OVERALL:
                        (None, {}, f"no {ecr_type} board"))
    res = redraft.run_redraft(
        [_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
        leagues=[_league(name="one"),
                 _league(name="two", starters={"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1})])
    assert res.rookie_note == "ro: no ro board; rsf: no rsf board"


def test_report_names_the_market_board(stub_pipeline):
    res = redraft.run_redraft(
        [_config("QB"), _config("RB")], draft_season=DRAFT_SEASON, refresh=False,
        rookie_context=(None, {}, ""),
        leagues=[_league(name="sf", starters={"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1})])
    md = redraft.render_markdown(res, res.leagues[0])
    assert "**superflex** ECR" in md


# -- rookie trim is per league, not per pooled pass ------------------------------------------

def _ecr_board(rookie_at):
    """A QB market board of 20 names with the single rookie at ECR rank ``rookie_at``."""
    players = [f"vet {i}" for i in range(20)]
    players[rookie_at - 1] = "rookie qb"
    return pd.DataFrame({"player": players, "pos": "QB",
                         "ecr": [float(i + 1) for i in range(20)]})


def test_rookie_count_uses_each_leagues_own_depth(stub_pipeline, monkeypatch):
    """Two leagues share a pass; the rookie sits inside the deep board only.

    Regression: the count used to be taken once at the *pooled* depth, so the shallow league was
    charged for a rookie the market ranks below its own board.
    """
    monkeypatch.setattr(redraft, "_rookie_market_context",
                        lambda s, *, ecr_type=redraft.REDRAFT_OVERALL:
                        (_ecr_board(rookie_at=8), {"QB": {"rookie qb"}}, ""))
    res = redraft.run_redraft(
        [_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
        leagues=[_league(name="shallow", top_n={"QB": 5}),
                 _league(name="deep", top_n={"QB": 12})])
    shallow, deep = res.leagues
    assert shallow.summaries[0].rookies_subtracted == 0   # rookie is ECR QB8, below a top-5 board
    assert deep.summaries[0].rookies_subtracted == 1
    # counts are of *model* rows; the rookie also rides the board as a market-placed pick
    model = lambda b: b[b["source"] == "model"]           # noqa: E731
    assert len(model(shallow.board)) == 5                 # 5 − 0
    assert len(model(deep.board)) == 11                   # 12 − 1
    assert shallow.summaries[0].returned == 5 and deep.summaries[0].returned == 11


def test_board_columns_record_the_leagues_own_trim(stub_pipeline, monkeypatch):
    """`requested_top_n` / `rookies_subtracted` ride on the CSV, so they must be per league."""
    monkeypatch.setattr(redraft, "_rookie_market_context",
                        lambda s, *, ecr_type=redraft.REDRAFT_OVERALL:
                        (_ecr_board(rookie_at=8), {"QB": {"rookie qb"}}, ""))
    res = redraft.run_redraft(
        [_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
        leagues=[_league(name="shallow", top_n={"QB": 5}),
                 _league(name="deep", top_n={"QB": 12})])
    shallow, deep = (lb.board[lb.board["source"] == "model"] for lb in res.leagues)
    assert set(shallow["requested_top_n"]) == {5} and set(shallow["rookies_subtracted"]) == {0}
    assert set(deep["requested_top_n"]) == {12} and set(deep["rookies_subtracted"]) == {1}


# -- the board is a pick order --------------------------------------------------------------

def test_market_rookies_reads_slots_over_modeled_positions_only():
    """The `ro` scrape carries IDP rows; they are not draftable and must not consume a slot."""
    ecr = pd.DataFrame({
        "player": ["vet wr", "some lb", "rookie rb", "vet qb"],
        "pos": ["WR", "LB", "RB", "QB"],
        "ecr": [1.0, 2.0, 3.0, 4.0]})
    got = redraft.market_rookies(ecr, {"RB": {"rookie rb"}}, limit=50)
    assert got == [{"player_name": "rookie rb", "position": "RB",
                    "market_slot": 2, "market_ecr": 3.0}]      # slot 2, not 3 — the LB is skipped


def test_market_rookies_respects_the_limit():
    ecr = pd.DataFrame({"player": [f"r{i}" for i in range(5)], "pos": "RB",
                        "ecr": [float(i + 1) for i in range(5)]})
    names = {"RB": {f"r{i}" for i in range(5)}}
    assert [r["market_slot"] for r in redraft.market_rookies(ecr, names, limit=3)] == [1, 2, 3]


def test_insert_market_rookies_makes_a_contiguous_pick_order():
    board = pd.DataFrame({"player_name": [f"p{i}" for i in range(5)], "position": "RB",
                          "proj_ppg": [10.0, 9.0, 8.0, 7.0, 6.0], "vorp": [4.0, 3, 2, 1, 0],
                          "proj_overall_rank": range(1, 6)})
    out = redraft.insert_market_rookies(
        board, [{"player_name": "rook", "position": "WR", "market_slot": 3, "market_ecr": 3.0}],
        limit=6)
    assert list(out["player_name"]) == ["p0", "p1", "rook", "p2", "p3", "p4"]
    assert list(out["proj_overall_rank"]) == [1, 2, 3, 4, 5, 6]
    assert list(out["source"]) == ["model"] * 2 + ["market_rookie"] + ["model"] * 3
    assert pd.isna(out.loc[2, "proj_ppg"]) and pd.isna(out.loc[2, "vorp"])


def test_insert_market_rookies_cuts_to_the_limit():
    board = pd.DataFrame({"player_name": [f"p{i}" for i in range(5)], "position": "RB",
                          "proj_ppg": [10.0, 9, 8, 7, 6], "vorp": [4.0, 3, 2, 1, 0],
                          "proj_overall_rank": range(1, 6)})
    out = redraft.insert_market_rookies(
        board, [{"player_name": "rook", "position": "WR", "market_slot": 1, "market_ecr": 1.0}],
        limit=3)
    assert list(out["player_name"]) == ["rook", "p0", "p1"]


def test_insert_market_rookies_appends_a_slot_past_the_board():
    board = pd.DataFrame({"player_name": ["p0"], "position": "RB", "proj_ppg": [10.0],
                          "vorp": [1.0], "proj_overall_rank": [1]})
    out = redraft.insert_market_rookies(
        board, [{"player_name": "rook", "position": "WR", "market_slot": 40, "market_ecr": 40.0}],
        limit=10)
    assert list(out["player_name"]) == ["p0", "rook"]


def test_board_is_a_pick_order_with_rookies_in_it(stub_pipeline, monkeypatch):
    """End to end: the rookie occupies its market slot and the ranks stay contiguous."""
    monkeypatch.setattr(redraft, "_rookie_market_context",
                        lambda s, *, ecr_type=redraft.REDRAFT_OVERALL:
                        (_ecr_board(rookie_at=3), {"QB": {"rookie qb"}}, ""))
    res = redraft.run_redraft([_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
                              leagues=[_league(name="one", top_n={"QB": 10})])
    board = res.leagues[0].board
    assert list(board["proj_overall_rank"]) == list(range(1, len(board) + 1))
    rookie = board[board["source"] == "market_rookie"]
    assert len(rookie) == 1 and int(rookie["proj_overall_rank"].iloc[0]) == 3
    assert rookie["market_ecr"].iloc[0] == 3.0


def test_rookie_beyond_the_last_pick_does_not_shorten_the_board(stub_pipeline, monkeypatch):
    """A rookie the market ranks past the final pick takes no slot, so he is not subtracted.

    Regression: subtracting at position depth while inserting within the pick limit used two
    different windows, and boards came up short of their own draft (ppr_1qb was 190 of 192).
    """
    # 120 QBs; the only rookie is ECR QB100 — inside a top-120 board, past a 70-pick draft.
    players = [f"vet {i}" for i in range(120)]
    players[99] = "late rookie"
    ecr = pd.DataFrame({"player": players, "pos": "QB",
                        "ecr": [float(i + 1) for i in range(120)]})
    monkeypatch.setattr(redraft, "_rookie_market_context",
                        lambda s, *, ecr_type=redraft.REDRAFT_OVERALL:
                        (ecr, {"QB": {"late rookie"}}, ""))
    lg = _league(name="tiny", teams=10, roster_size=7, top_n={"QB": 120})   # 70 total picks
    res = redraft.run_redraft([_config("QB")], draft_season=DRAFT_SEASON, refresh=False,
                              leagues=[lg])
    lb = res.leagues[0]
    assert lb.summaries[0].rookies_subtracted == 0        # ECR 100 is past pick 70
    assert len(lb.board) == lg.total_picks == 70          # board still fills the draft
    assert not (lb.board["source"] == "market_rookie").any()


def test_league_boards_are_exactly_their_draft_length(stub_pipeline, monkeypatch):
    """The board is a pick order, so it must have one row per pick — no holes, no overrun."""
    cycle = ["QB", "RB", "WR", "TE"]
    ecr = pd.DataFrame({"player": [f"p{i}" for i in range(200)],
                        "pos": [cycle[i % 4] for i in range(200)],
                        "ecr": [float(i + 1) for i in range(200)]})
    monkeypatch.setattr(redraft, "_rookie_market_context",
                        lambda s, *, ecr_type=redraft.REDRAFT_OVERALL:
                        (ecr, {"QB": {"p8"}, "RB": {"p13"}, "WR": {"p30"}, "TE": {"p71"}}, ""))
    leagues = [_league(name="a", teams=10, roster_size=8),
               _league(name="b", teams=12, roster_size=8, scoring="half_ppr")]
    res = redraft.run_redraft([_config(p) for p in cycle], draft_season=DRAFT_SEASON,
                              refresh=False, leagues=leagues)
    for lb in res.leagues:
        assert len(lb.board) == lb.league.total_picks, lb.league.name
        assert list(lb.board["proj_overall_rank"]) == list(range(1, lb.league.total_picks + 1))
