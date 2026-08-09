"""Tests for keeper valuation (pure logic; synthetic projections, no model/network)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eval.keeper import (  # noqa: E402
    build_board,
    evaluate_keepers,
    replacement_levels,
    resolve_players,
)


def _proj():
    rows = []
    for i, v in enumerate([30, 28, 26, 24]):
        rows.append(dict(player_id=f"QB{i}", player_name=f"Q {i}", position="QB",
                         proj_ppg=v, proj_pos_rank=i + 1))
    for i, v in enumerate([20, 18, 16, 14, 12]):
        rows.append(dict(player_id=f"RB{i}", player_name=f"R {i}", position="RB",
                         proj_ppg=v, proj_pos_rank=i + 1))
    for i, v in enumerate([22, 19, 17, 15, 13]):
        rows.append(dict(player_id=f"WR{i}", player_name=f"W {i}", position="WR",
                         proj_ppg=v, proj_pos_rank=i + 1))
    return pd.DataFrame(rows)


ROSTER = {"RB": 1, "WR": 1, "FLEX": 1}


def test_replacement_levels_flex_allocation_and_first_non_starter():
    repl, starters = replacement_levels(_proj(), teams=2, fmt="1qb", roster=ROSTER)
    # QB: round(2*1)=2 starters -> replacement = 3rd QB (26).
    # Dedicated RB=2, WR=2; 2 flex slots go to best-remaining: WR17 then RB16 -> RB3, WR3.
    # ROSTER has no TE slot -> 0 TE starters (starters always reports every modeled position).
    assert starters == {"QB": 2, "RB": 3, "WR": 3, "TE": 0}
    assert repl["QB"] == 26      # pool[2]
    assert repl["RB"] == 14      # pool[3]
    assert repl["WR"] == 15      # pool[3]


def test_te_dedicated_slot_replacement():
    # TE gets its own slot + replacement level but is NOT flex-eligible (flex stays RB/WR).
    proj = pd.concat([_proj(), pd.DataFrame(
        [dict(player_id=f"TE{i}", player_name=f"T {i}", position="TE",
              proj_ppg=v, proj_pos_rank=i + 1)
         for i, v in enumerate([16, 13, 11, 9, 7])])], ignore_index=True)
    roster = {"RB": 1, "WR": 1, "TE": 1, "FLEX": 1}
    repl, starters = replacement_levels(proj, teams=2, fmt="1qb", roster=roster)
    # TE: 2 teams * 1 slot = 2 started -> replacement = 3rd TE (11). Flex untouched by TE.
    assert starters["TE"] == 2
    assert repl["TE"] == 11       # pool[2]
    assert starters["RB"] == 3 and starters["WR"] == 3


def test_format_changes_qb_depth():
    _, s1 = replacement_levels(_proj(), teams=2, fmt="1qb", roster=ROSTER)
    _, s2 = replacement_levels(_proj(), teams=2, fmt="2qb", roster=ROSTER)
    assert s1["QB"] == 2 and s2["QB"] == 4   # 2qb doubles started QBs


def test_invalid_format_raises():
    with pytest.raises(ValueError):
        replacement_levels(_proj(), teams=12, fmt="3qb")


def test_build_board_vorp_and_overall_rank():
    board, repl, _ = build_board(_proj(), teams=2, fmt="1qb", roster=ROSTER)
    top = board.iloc[0]
    # best VORP here is WR0: 22 - 15 = 7 (beats RB0 20-14=6 and QB0 30-26=4).
    assert top["player_id"] == "WR0" and top["vorp"] == 7
    assert list(board["proj_overall_rank"]) == list(range(1, len(board) + 1))
    assert board["vorp"].is_monotonic_decreasing


def test_build_board_breaks_vorp_ties_deterministically():
    """VORP is rounded to 2dp so ties are common; pandas' default sort is not stable, which
    reordered tied players between otherwise identical runs. Ties break on projection, then id."""
    proj = pd.DataFrame([
        # Three RBs landing on the same VORP; input order is deliberately scrambled.
        dict(player_id="RBb", player_name="b", position="RB", proj_ppg=20.0, proj_pos_rank=2),
        dict(player_id="RBa", player_name="a", position="RB", proj_ppg=20.0, proj_pos_rank=1),
        dict(player_id="RBc", player_name="c", position="RB", proj_ppg=25.0, proj_pos_rank=3),
        dict(player_id="RBd", player_name="d", position="RB", proj_ppg=10.0, proj_pos_rank=4),
    ])
    board, _, _ = build_board(proj, teams=2, roster={"RB": 1, "QB": 0})
    # Equal VORP -> equal proj_ppg -> player_id ascending.
    assert list(board["player_id"]) == ["RBc", "RBa", "RBb", "RBd"]

    # Shuffling the input must not change the board.
    shuffled = build_board(proj.iloc[::-1].reset_index(drop=True),
                           teams=2, roster={"RB": 1, "QB": 0})[0]
    assert list(shuffled["player_id"]) == list(board["player_id"])


def test_resolve_players_normalises_and_flags_unmatched():
    board = pd.DataFrame([
        dict(player_id="x", player_name="A.J. Brown", position="WR",
             proj_ppg=18.0, proj_pos_rank=4, vorp=7.0, proj_overall_rank=5),
    ])
    picks = pd.DataFrame([
        dict(player="AJ Brown", pick=30),       # punctuation-insensitive match
        dict(player="Sam LaPorta", pick=60),    # TE, not on board -> unmatched
    ])
    matched, unmatched = resolve_players(board, picks)
    assert len(matched) == 1 and matched.iloc[0]["player_id"] == "x"
    assert matched.iloc[0]["pick"] == 30
    assert list(unmatched["player"]) == ["Sam LaPorta"]


def test_evaluate_keepers_surplus_and_sort():
    board, _, _ = build_board(_proj(), teams=2, fmt="1qb", roster=ROSTER)
    # keep WR0 (a top board slot) cheaply, and QB3 (low slot) expensively.
    picks = pd.DataFrame([
        dict(player="W 0", pick=40),
        dict(player="Q 3", pick=5),
    ])
    ranked, _ = evaluate_keepers(board, picks)
    # WR0: surplus = 40 - its (top) overall rank; QB3: 5 - a low rank -> negative.
    assert ranked.iloc[0]["player_id"] == "WR0"
    assert ranked.iloc[0]["surplus"] > ranked.iloc[1]["surplus"]
    assert ranked.iloc[0]["keep"] and not ranked.iloc[-1]["keep"]
    assert ranked.iloc[0]["pos_rank"] == "WR1"
