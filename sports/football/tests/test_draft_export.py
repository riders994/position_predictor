"""Tests for the live-draft JSON export (synthetic frames; no model, no network)."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eval.draft_export import (  # noqa: E402
    build_draft_export,
    bye_weeks,
    choose_adp_board,
    export_league,
)
from position_predictor.eval.league import league_from_dict  # noqa: E402
from position_predictor.eval.redraft import LeagueBoard  # noqa: E402


def _league(**over):
    base = {"name": "t", "label": "T", "scoring": "ppr", "teams": 4,
            "starters": {"QB": 1, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0}, "roster_size": 3}
    return league_from_dict({**base, **over})


def _board():
    return pd.DataFrame([
        {"player_id": "g-rb", "position": "RB", "player_name": "Run Back", "proj_ppg": 15.0,
         "proj_pos_rank": 1, "vorp": 5.0, "proj_overall_rank": 1, "bench_insurance": 3.0,
         "source": "model", "market_ecr": None},
        {"player_id": None, "position": "WR", "player_name": "Rook Wide", "proj_ppg": None,
         "proj_pos_rank": None, "vorp": None, "proj_overall_rank": 2, "bench_insurance": None,
         "source": "market_rookie", "market_ecr": 2.0},
        {"player_id": "g-qb", "position": "QB", "player_name": "Quarter Back", "proj_ppg": 20.0,
         "proj_pos_rank": 1, "vorp": 4.0, "proj_overall_rank": 3, "bench_insurance": 1.0,
         "source": "model", "market_ecr": None},
    ])


def _lb(league=None):
    return LeagueBoard(league=league or _league(), board=_board(),
                       replacement={"QB": 16.0, "RB": 10.0, "WR": 9.0, "TE": 5.0},
                       ecr_type="ro", market_board=_ecr())


def _ecr():
    return pd.DataFrame({"player": ["Run Back", "Rook Wide", "Quarter Back", "Vet Wide"],
                         "id": [11, 12, 13, 14], "pos": ["RB", "WR", "QB", "WR"],
                         "ecr": [1.0, 2.0, 3.0, 4.0],
                         "scrape_date": ["2026-08-30"] * 4})


def _rosters():
    return pd.DataFrame({"season": 2026, "team": ["AAA", "BBB", "AAA", "CCC"],
                         "position": ["RB", "WR", "QB", "WR"],
                         "full_name": ["Run Back", "Rook Wide", "Quarter Back", "Vet Wide"],
                         "football_name": ["Run", "Rook", "Quarter", "Vet"],
                         "last_name": ["Back", "Wide", "Back", "Wide"],
                         "player_id": ["g-rb", "g-rook", "g-qb", "g-vet"]})


def _schedules():
    # weeks 1-3; AAA sits out week 2, BBB week 1, CCC week 3
    games = [(1, "AAA", "CCC"), (2, "BBB", "CCC"), (3, "AAA", "BBB")]
    return pd.DataFrame([{"season": 2026, "game_type": "REG", "week": w, "home_team": h,
                          "away_team": a} for w, h, a in games])


def _ffc(n_extra=0):
    rows = [("Run Back", "RB", "AAA", 1.2), ("Rook Wide", "WR", "BBB", 2.5),
            ("Quarter Back", "QB", "AAA", 4.0), ("Vet Wide", "WR", "CCC", 3.1)]
    rows += [(f"Deep {i}", "RB", "ZZZ", 50.0 + i) for i in range(n_extra)]
    return pd.DataFrame([{"ffc_id": 100 + i, "name": n, "position": p, "team": t, "adp": a,
                          "stdev": 0.5, "bye": None} for i, (n, p, t, a) in enumerate(rows)])


def _ids():
    return pd.DataFrame({"fantasypros_id": ["11", "12", "13", "14"],
                         "gsis_id": ["g-rb", "g-rook", "g-qb", "g-vet"]})


def _export(lb=None, adp=None, source="ffc_ppr"):
    lb = lb or _lb()
    return build_draft_export(lb, draft_season=2026, feature_season=2025,
                              adp_board=_ffc() if adp is None else adp, adp_source=source,
                              ecr_df=lb.market_board, rosters=_rosters(),
                              schedules=_schedules(), ids=_ids(), generated_at="fixed")


def test_bye_weeks_is_the_one_week_a_team_has_no_game():
    assert bye_weeks(_schedules(), 2026) == {"AAA": 2, "BBB": 1, "CCC": 3}


def test_export_carries_board_rows_then_market_only_players_with_adp():
    out = _export()
    ids = [p["id"] for p in out["players"]]
    # board order first (the rookie keeps its slot, gains a gsis id via the ADP match), then the
    # market-only veteran the model never put on its board
    assert ids == ["g-rb", "g-rook", "g-qb", "g-vet"]
    by = {p["id"]: p for p in out["players"]}
    assert by["g-rook"]["value"] is None and by["g-rook"]["market_adp"] == 2.5
    assert by["g-vet"]["source"] == "market_only" and by["g-vet"]["board_rank"] is None
    assert by["g-rb"]["value"] == 15.0 and by["g-rb"]["market_ecr"] == 1.0
    assert (by["g-rb"]["team"], by["g-rb"]["bye"]) == ("AAA", 2)
    assert out["policy"]["name"] == "market_window" and out["policy"]["window"] == 4
    assert out["policy"]["position_caps"]["QB"] == 2
    json.dumps(out, allow_nan=False)          # strict JSON: no NaN anywhere


def test_thin_ffc_board_falls_back_to_ecr_rank():
    lg = _league()
    board, source, warnings = choose_adp_board(lg, _ffc().head(2), _ecr(), ffc_label="ffc_ppr")
    assert source == "ecr_rank" and "falls back" in warnings[0]
    assert list(board["adp"]) == [1, 2, 3, 4]
    # 12 picks: 4 + 8 modeled rows clear the 85% coverage bar
    board, source, warnings = choose_adp_board(lg, _ffc(n_extra=8), _ecr(), ffc_label="ffc_ppr")
    assert source == "ffc_ppr" and not warnings


def test_ecr_fallback_matches_players_by_fantasypros_id():
    lb = _lb()
    board, source, _ = choose_adp_board(lb.league, None, lb.market_board, ffc_label="ffc_ppr")
    out = _export(lb, adp=board, source=source)
    assert [p["market_adp"] for p in out["players"]] == [1, 2, 3, 4]
    assert out["players"][-1]["id"] == "g-vet"


def test_backtest_coverage_is_flagged():
    assert _export()["policy"]["backtested"] is False                     # 4 teams
    lb = _lb(_league(teams=10, roster_size=3))
    assert _export(lb)["policy"]["backtested"] is True
    sf = _lb(_league(teams=10, starters={"QB": 1, "RB": 1, "WR": 1, "FLEX": 1},
                     flex_positions=["QB", "RB", "WR"], roster_size=4))
    out = _export(sf)
    assert out["policy"]["backtested"] is False and out["league"]["is_superflex"]


def test_export_league_uses_the_2qb_board_for_superflex_and_survives_a_failed_fetch():
    calls = []

    def fetch(season, *, scoring, fmt):
        calls.append(fmt)
        return _ffc(n_extra=10)          # 14 rows for 16 picks: covered

    sf = _lb(_league(starters={"QB": 2, "RB": 1, "WR": 1}, scoring="half_ppr", roster_size=4))
    out = export_league(sf, draft_season=2026, feature_season=2025, rosters=_rosters(),
                        schedules=_schedules(), ids=_ids(), fetch_ffc=fetch)
    assert calls == ["2qb"] and out["market"]["adp_source"] == "ffc_2qb"
    assert any("PPR-scored" in w for w in out["warnings"])

    def broken(*_, **__):
        raise OSError("down")

    out = export_league(_lb(), draft_season=2026, feature_season=2025, rosters=_rosters(),
                        schedules=_schedules(), ids=_ids(), fetch_ffc=broken)
    assert out["market"]["adp_source"] == "ecr_rank"
    assert any("fetch failed" in w for w in out["warnings"])
