"""Golden cases for ports of the draft policies (the live draft tool reimplements them in JS).

Every expected pick below is produced by calling the real policy in :mod:`eval.draftsim` on a
small, hand-checkable draft. ``tests/fixtures/draft_golden.json`` is the committed output, and
``tests/test_draft_golden.py`` fails if it drifts from what the policies now return. Regenerate
with ``uv run python -m position_predictor.eval.draft_golden > tests/fixtures/draft_golden.json``
only after deciding the behaviour change is intended, because a port somewhere is pinned to it.

Pool semantics a port must reproduce:

* ``board_rank`` is the model's VORP board order; only rows with a ``value`` are walked.
* ``adp`` null = the market doesn't rank the player. For the market's order such players queue
  behind every ranked player, in pool order (``Pool._adp_filled``).
* ``value`` null = the model can't score him (a rookie, or a veteran the model has no projection
  for). Model-driven choices skip him; market-driven choices can take him.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .draftsim import (MODELED_POS, POS_INDEX, AdpPolicy, BoardPolicy, DraftSim, MarketWindowPolicy,
                       Pool, lineup_value, open_positions)
from .league import league_from_dict

SCHEMA_VERSION = 1

LEAGUE = {"name": "golden", "label": "4-team 1QB PPR", "scoring": "ppr", "teams": 4,
          "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
          "flex_positions": ["RB", "WR", "TE"], "roster_size": 9}

# (id, position, model value, market adp, source). Board rows first, in VORP board order; then pool
# rows that are not on the board. value None = unscored, adp None = unranked.
PLAYERS = [
    ("rb1", "RB", 20.0, 1.5, "model"), ("wr1", "WR", 19.0, 2.4, "model"),
    ("rk_rb", "RB", None, 6.0, "market_rookie"), ("qb1", "QB", 24.0, 9.0, "model"),
    ("wr2", "WR", 17.5, 5.0, "model"), ("rb2", "RB", 16.0, 4.2, "model"),
    ("te1", "TE", 14.0, 7.5, "model"), ("wr3", "WR", 15.5, 3.3, "model"),
    ("qb2", "QB", 22.0, 12.3, "model"), ("rb3", "RB", 14.5, 8.2, "model"),
    ("wr4", "WR", 14.0, 10.1, "model"), ("rk_wr", "WR", None, 11.0, "market_rookie"),
    ("rb4", "RB", 16.5, 13.4, "model"), ("te2", "TE", 11.0, 15.0, "model"),
    ("wr5", "WR", 12.5, 14.2, "model"), ("qb3", "QB", 20.5, 18.0, "model"),
    ("rb5", "RB", 11.5, 16.6, "model"), ("wr6", "WR", 11.0, 17.1, "model"),
    ("rb6", "RB", 10.0, 19.5, "model"), ("wr7", "WR", 9.5, 21.0, "model"),
    ("te3", "TE", 9.0, 20.2, "model"), ("qb4", "QB", 19.0, 25.0, "model"),
    ("rb7", "RB", 8.5, 22.8, "model"), ("wr8", "WR", 8.0, 24.0, "model"),
    ("rb8", "RB", 7.0, 27.5, "model"), ("wr9", "WR", 6.5, 26.1, "model"),
    ("te4", "TE", 7.5, 29.0, "model"), ("qb5", "QB", 17.0, 31.0, "model"),
    # Off-board pool rows: a veteran the market ranks but the model doesn't score, and an
    # unranked, unscored body (the market's order reaches him only after every ranked player).
    ("vet_rb", "RB", None, 23.5, "market_only"), ("unr_wr", "WR", None, None, "market_only"),
]
IDS = [p[0] for p in PLAYERS]
ON_BOARD = [p[4] != "market_only" for p in PLAYERS]


def build_pool() -> Pool:
    value = np.array([np.nan if p[2] is None else p[2] for p in PLAYERS])
    adp = np.array([np.inf if p[3] is None else p[3] for p in PLAYERS])
    board = np.array([i for i, on in enumerate(ON_BOARD) if on and np.isfinite(value[i])])
    return Pool(position=[POS_INDEX[p[1]] for p in PLAYERS], adp=adp, value=value,
                board_order=board, player_id=np.array(IDS))


def _state(sim: DraftSim, taken_by_others, mine):
    state = sim.new_state()
    for pid in taken_by_others:
        sim.take(state, 1, IDS.index(pid))
    for pid in mine:
        sim.take(state, 0, IDS.index(pid))
    return state


def board_ranked(sim: DraftSim, state, team: int, scan: int) -> list[dict]:
    """A display ordering consistent with :class:`BoardPolicy` (its first entry is the pick):
    every allowed player who improves the lineup, in board order, never re-sorted; else the
    insurance scan's candidates by their summed insurance gain, ties to the earlier board row."""
    ok, mine = sim.allowed(state, team), state.values[team]
    cur = lineup_value(mine, sim.league)
    out = []
    for i in sim.pool.board_order:
        if not ok[i]:
            continue
        p = MODELED_POS[sim.pool.position[i]]
        mine[p].append(float(sim.pool.value[i]))
        gain = lineup_value(mine, sim.league) - cur
        mine[p].pop()
        if gain > 1e-9:
            out.append({"id": IDS[i], "phase": "starter", "gain": round(gain, 4)})
    if out:
        return out
    absences = []
    for p_out, v_out in [(p, v) for p, vals in mine.items() for v in vals]:
        lineup = {p: list(v) for p, v in mine.items()}
        lineup[p_out].remove(v_out)
        absences.append((lineup, lineup_value(lineup, sim.league)))
    scored = []
    for rank, i in enumerate(i for i in sim.pool.board_order if ok[i]):
        if rank >= scan:
            break
        p, v = MODELED_POS[sim.pool.position[i]], float(sim.pool.value[i])
        gain = 0.0
        for lineup, base in absences:
            lineup[p].append(v)
            gain += lineup_value(lineup, sim.league) - base
            lineup[p].pop()
        scored.append((-round(gain, 9), rank, i, gain))
    return [{"id": IDS[i], "phase": "bench", "insurance_sum": round(g, 4)}
            for _, _, i, g in sorted(scored)]


def _case(name, note, policy, taken_by_others, mine, **params):
    league = league_from_dict(LEAGUE)
    sim = DraftSim(league, build_pool())
    state = _state(sim, taken_by_others, mine)
    out = {"name": name, "note": note, "policy": policy, "params": params,
           "taken_by_others": list(taken_by_others), "my_roster": list(mine),
           "open_positions": [p for p, need in zip(MODELED_POS, open_positions(sim, state, 0))
                              if need]}
    if policy == "adp":
        out["expected_pick"] = IDS[AdpPolicy().pick(sim, state, 0, 0)]
    elif policy == "board":
        pol = BoardPolicy(bench="insurance", **params)
        out["expected_pick"] = IDS[pol.pick(sim, state, 0, 0)]
        out["expected_ranked"] = board_ranked(sim, state, 0, pol.scan)
        assert out["expected_ranked"][0]["id"] == out["expected_pick"], name
    elif policy in ("market_window", "market_window_depth"):
        bench = "insurance" if policy == "market_window_depth" else "market"
        pol = MarketWindowPolicy(bench=bench, **params)
        out["expected_market_pick"] = IDS[AdpPolicy().pick(sim, state, 0, 0)]
        out["expected_window_pick"] = IDS[pol._window_pick(sim, state, 0, 0)]
        out["expected_pick"] = IDS[pol.pick(sim, state, 0, 0)]
    else:
        raise ValueError(policy)
    return out


# One shared bench state: every starting slot filled, a QB and TE already backed up by nobody.
_BENCH_OTHERS = ["rb1", "wr1", "qb2", "rb4", "wr5", "te2"]
_BENCH_MINE = ["qb1", "rb2", "rb3", "wr2", "wr3", "te1", "wr4"]
# Rookie run: every scored RB near the market's pick is gone, so the RB window holds only rk_rb.
_RUN_OTHERS = ["rb1", "wr1", "wr3", "rb2", "wr2", "rb3"]


def cases() -> list[dict]:
    return [
        # -- BoardPolicy(bench="insurance"): lineup_value, caps, insurance ----------------------
        _case("board_empty", "nothing drafted: the first scored board row", "board", [], []),
        _case("board_rookie_skipped", "rk_rb is next on the board but unscored, so skipped",
              "board", ["rb1", "wr1"], []),
        _case("board_qb_cap", "I hold 2 QBs (cap = qb_starters + 1): no QB allowed", "board",
              ["rb1", "wr1"], ["qb1", "qb2"]),
        _case("board_upgrade_counts", "RB slots and FLEX hold weak RBs; a better RB still gains",
              "board", ["rb1", "wr1", "wr2", "rb2"],
              ["rb5", "rb6", "rb7", "qb1", "wr3", "wr4", "te1"]),
        _case("board_bench_insurance", "lineup full: the insurance scan decides", "board",
              _BENCH_OTHERS, _BENCH_MINE),
        _case("board_bench_small_scan", "same state, scan=3: only 3 allowed board rows scored",
              "board", _BENCH_OTHERS, _BENCH_MINE, scan=3),
        # -- AdpPolicy: need rule, caps, rookies --------------------------------------------------
        _case("adp_empty", "best ADP overall", "adp", [], []),
        _case("adp_takes_rookie", "the market takes unscored players", "adp", _RUN_OTHERS, []),
        _case("adp_need_forces_te", "only TE is open: TE beats better-ADP players", "adp",
              ["rb1"], ["qb1", "rb2", "rb3", "wr1", "wr2", "rb4"]),
        _case("adp_qb_cap", "best ADP left is a QB but I hold 2 QBs (cap)", "adp",
              ["rb1", "wr1", "wr2", "rb2", "wr3", "rk_rb", "te1", "rb3", "wr4", "rk_wr", "rb4",
               "wr5", "te2", "rb5", "wr6"], ["qb1", "qb2"]),
        _case("adp_lineup_full", "no open slot: best ADP within caps", "adp",
              _BENCH_OTHERS, _BENCH_MINE),
        # -- MarketWindowPolicy (window defaults to teams = 4 picks) ----------------------------
        _case("window_model_overrides", "market says wr3 (3.3); wr2 (5.0) is inside the window "
              "and the model projects him higher", "market_window", ["rb1", "wr1"], []),
        _case("window_rookie_run", "market pick is rk_rb and no scored RB is in the window: "
              "fall back to the market's pick", "market_window", _RUN_OTHERS, []),
        _case("window_need_forces_te", "need rule picks the position (TE)", "market_window",
              ["rb1"], ["qb1", "rb2", "rb3", "wr1", "wr2", "rb4"]),
        _case("window_qb_cap", "QB capped; the RB window holds unscored vet_rb, which the model "
              "can't take", "market_window",
              ["rb1", "wr1", "wr2", "rb2", "wr3", "rk_rb", "te1", "rb3", "wr4", "rk_wr", "rb4",
               "wr5", "te2", "rb5", "wr6"], ["qb1", "qb2"]),
        _case("window_default_width", "market says rb2 (4.2); rb4 (13.4) is outside a 4-pick "
              "window", "market_window", ["rb1", "wr1", "wr3", "wr2"], []),
        _case("window_wide", "same state, window=12: rb4 is inside and the model projects him "
              "highest", "market_window", ["rb1", "wr1", "wr3", "wr2"], [], window=12),
        _case("window_bench_market", "lineup full, bench='market': keeps drafting by the window",
              "market_window", _BENCH_OTHERS, _BENCH_MINE),
        _case("window_bench_insurance", "lineup full, bench='insurance': roster-aware insurance",
              "market_window_depth", _BENCH_OTHERS, _BENCH_MINE),
        _case("window_depth_starters", "lineup not full: market_window_depth = market_window",
              "market_window_depth", ["rb1", "wr1"], []),
    ]


def golden() -> dict:
    return {"schema_version": SCHEMA_VERSION, "league": LEAGUE,
            "players": [{"id": i, "pos": p, "value": v, "adp": a, "source": s,
                         "board_rank": k + 1 if ON_BOARD[k] else None}
                        for k, (i, p, v, a, s) in enumerate(PLAYERS)],
            "cases": cases()}


if __name__ == "__main__":
    json.dump(golden(), sys.stdout, indent=1)
    sys.stdout.write("\n")
