"""No-network unit tests for archetype-discovery helpers (GMM fit covered by the live run)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.archetypes.discover import (  # noqa: E402
    _signature,
    archetype_stability,
    feature_cols,
)
from nba_archetypes.utils.config import Config  # noqa: E402


def test_feature_cols_flattens_block_map():
    block_map = {"shot_profile": ["fg3a_rate_z", "ft_rate_z"], "defense": ["blk36_z"]}
    assert feature_cols(block_map) == ["fg3a_rate_z", "ft_rate_z", "blk36_z"]


def test_signature_reports_top_hi_and_lo_features():
    zcols = ["fg3a_rate_z", "blk36_z", "oreb_rate_z", "ast36_z"]
    row = pd.Series({"size": 50, "name": "X",
                     "fg3a_rate_z": -1.6, "blk36_z": 2.3, "oreb_rate_z": 2.0, "ast36_z": 0.1})
    hi, lo = _signature(row, zcols, n=2)
    # highest two (+) first, lowest two (−)
    assert hi.startswith("blk36 +2.3") and "oreb_rate +2.0" in hi
    assert lo.startswith("fg3a_rate -1.6")


def test_config_names_cover_k():
    cfg = Config.load(ROOT / "config" / "nba_archetypes.yaml")
    k = int(cfg.get("archetypes.k"))
    names = cfg.get("archetypes.names")
    assert set(names) == set(range(k)) and len(names) == k    # one provisional name per component


def test_archetype_stability_counts_consecutive_pairs_only():
    # P1: stays A in 2018->2019, switches A->B in 2019->2020 (3 consecutive pairs total).
    # P2: 2018 then 2020 (gap) -> no consecutive pair. P3: single season -> none.
    mem = pd.DataFrame({
        "athlete_id": [1, 1, 1, 2, 2, 3],
        "season":     [2018, 2019, 2020, 2018, 2020, 2019],
        "arch":       [0, 0, 1, 0, 0, 2],
        "arch_name":  ["A", "A", "B", "A", "A", "C"],
    })
    st = archetype_stability(mem)
    assert st["n_pairs"] == 2                       # P1 18->19, P1 19->20; P2 has a season gap
    assert st["overall"] == 0.5                     # one stay (A->A), one switch (A->B)
    assert st["per_archetype"]["A"] == 0.5          # A: one stayed, one left


def test_archetype_stability_empty_when_no_consecutive_pairs():
    mem = pd.DataFrame({"athlete_id": [1, 2], "season": [2019, 2021],
                        "arch": [0, 1], "arch_name": ["A", "B"]})
    st = archetype_stability(mem)
    assert st["n_pairs"] == 0 and st["per_archetype"] == {}
