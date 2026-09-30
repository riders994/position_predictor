"""The committed golden cases must match what the draft policies return now.

A JS port of the policies (the live draft tool) is pinned to ``tests/fixtures/draft_golden.json``.
If this fails, a policy's behaviour changed: decide whether that is intended, then regenerate
(see :mod:`position_predictor.eval.draft_golden`) and tell the port.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eval.draft_golden import golden  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "draft_golden.json"


def test_fixture_matches_the_policies():
    assert json.loads(FIXTURE.read_text()) == json.loads(json.dumps(golden()))


def test_fixture_covers_every_policy_and_the_edge_cases():
    cases = {c["name"]: c for c in golden()["cases"]}
    assert {c["policy"] for c in cases.values()} == {"adp", "board", "market_window",
                                                     "market_window_depth"}
    # rookie run: the window has no scored player, so the market's (rookie) pick stands
    assert cases["window_rookie_run"]["expected_pick"] == "rk_rb"
    assert cases["adp_need_forces_te"]["expected_pick"] == "te1"
    assert cases["adp_qb_cap"]["expected_pick"] != "qb3"
    # board order, not the biggest gain: rb3 (+6.0) is taken ahead of rb4 (+8.0)
    ranked = cases["board_upgrade_counts"]["expected_ranked"]
    assert [r["id"] for r in ranked[:2]] == ["rb3", "rb4"] and ranked[1]["gain"] > ranked[0]["gain"]
    # the two bench rules disagree on the same state
    assert (cases["window_bench_market"]["expected_pick"]
            != cases["window_bench_insurance"]["expected_pick"])
