"""Tests for the Phase-2 archetype value guide (value-weighted exposure + coefficients + labels)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.eval.archetype_value import (  # noqa: E402
    classify, exposure_coefficients, value_weighted_exposure)


def _membership(n_arch=3):
    """Two players with known one-hot-ish membership vectors and archetype names."""
    rows = []
    for aid, arch in [(1, 0), (2, 1)]:
        probs = np.full(n_arch, 0.0)
        probs[arch] = 1.0
        rows.append({"athlete_id": aid, "season": 2020, "arch": arch, "arch_name": f"A{arch}",
                     **{f"p{j}": probs[j] for j in range(n_arch)}})
    return pd.DataFrame(rows)


def test_value_weighted_exposure_is_magnitude_not_share():
    mem = _membership()
    values = pd.DataFrame({"athlete_id": [1, 2], "season": [2020, 2020],
                           "value": [5.0, 1.0], "prior_value": [4.0, 2.0]})
    rosters = pd.DataFrame({"sim_league_id": ["L"] * 2, "team_id": [0, 0], "season": [2020, 2020],
                            "athlete_id": [1, 2]})
    exp = value_weighted_exposure(rosters, values, mem)
    row = exp.iloc[0]
    # exposure keeps magnitude: A0 gets player-1's value (5 actual / 4 prior), A1 gets player-2's
    assert row["expa_A0"] == 5.0 and row["expa_A1"] == 1.0
    assert row["expp_A0"] == 4.0 and row["expp_A1"] == 2.0


def test_value_weighted_exposure_clips_negative_value():
    mem = _membership()
    values = pd.DataFrame({"athlete_id": [1, 2], "season": [2020, 2020],
                           "value": [-3.0, 2.0], "prior_value": [1.0, 1.0]})
    rosters = pd.DataFrame({"sim_league_id": ["L", "L"], "team_id": [0, 0], "season": [2020, 2020],
                            "athlete_id": [1, 2]})
    exp = value_weighted_exposure(rosters, values, mem)
    assert exp.iloc[0]["expa_A0"] == 0.0            # negative value clipped to 0


def _planted_table(seed=0):
    """8 seasons x 40 teams; cat_win_rate driven by exposure to archetype 0 (prior) -> recoverable."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(8):
        for _ in range(40):
            e0, e1, e2 = rng.uniform(0, 5, 3)
            y = 0.5 + 0.03 * e0 - 0.01 * e2 + rng.normal(0, 0.01)
            rows.append({"season": s, "sim_cat_win_rate": y,
                         "expp_A0": e0, "expp_A1": e1, "expp_A2": e2,
                         "expa_A0": e0, "expa_A1": e1, "expa_A2": e2})
    return pd.DataFrame(rows)


def test_exposure_coefficients_recover_planted_signal():
    rows = exposure_coefficients(_planted_table(), kind="prior", n_boot=100)
    top = rows[0]
    assert top["archetype"] == "A0"                 # the driver archetype ranks first
    assert top["coef_std"] > 0 and top["sign_stability"] >= 0.95
    assert top["top_minus_bottom"] > 0              # winners hold more A0 exposure
    a2 = next(r for r in rows if r["archetype"] == "A2")
    assert a2["coef_std"] < 0                        # the penalized archetype is negative


def test_classify_labels_prioritize_trap_avoid():
    prior = [{"archetype": "Engine", "coef_std": 0.02, "sign_stability": 1.0, "top_minus_bottom": 2.0},
             {"archetype": "Trap", "coef_std": 0.0, "sign_stability": 0.5, "top_minus_bottom": -2.0},
             {"archetype": "Bad", "coef_std": -0.005, "sign_stability": 0.99, "top_minus_bottom": -0.2}]
    actual = [{"archetype": "Engine", "coef_std": 0.035}, {"archetype": "Trap", "coef_std": 0.023},
              {"archetype": "Bad", "coef_std": 0.001}]
    labels = {r["archetype"]: r["label"] for r in classify(prior, actual)}
    assert labels == {"Engine": "PRIORITIZE", "Trap": "trap", "Bad": "avoid"}
