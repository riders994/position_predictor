"""No-network unit tests for the build (pivot/parse) + feature (era/rate) logic."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.data.build import COMBO_RENAME, SCALAR_RENAME, _parse_combo  # noqa: E402
from nba_archetypes.features.build import BLOCKS, _assign_eras, _div  # noqa: E402


def test_parse_combo_made_attempted():
    made, att = _parse_combo(pd.Series(["2-6", "0-0", "10-12", None, "bad"]))
    assert list(made[:3]) == [2.0, 0.0, 10.0]
    assert list(att[:3]) == [6.0, 0.0, 12.0]
    assert pd.isna(made.iloc[3]) and pd.isna(att.iloc[4])  # None / unparseable -> NaN


def test_rename_maps_cover_the_combos_and_scalars():
    # every combo maps to a (made, attempted) pair; scalars include the rate/ratio basics
    assert all(isinstance(v, tuple) and len(v) == 2 for v in COMBO_RENAME.values())
    assert {"avgPoints", "avgMinutes", "gamesPlayed"} <= set(SCALAR_RENAME)


def test_div_guards_zero_denominator():
    out = _div(pd.Series([3.0, 5.0]), pd.Series([6.0, 0.0]))
    assert out.iloc[0] == 0.5 and pd.isna(out.iloc[1])      # x/0 -> NaN, not inf


def test_assign_eras_maps_seasons_and_archetype_flag():
    eras = [
        {"name": "E1", "start_season": 2014, "end_season": 2017, "use_for_archetypes": False},
        {"name": "E2", "start_season": 2018, "end_season": 2020, "use_for_archetypes": True},
        {"name": "E3", "start_season": 2021, "end_season": None, "use_for_archetypes": True},
    ]
    df = pd.DataFrame({"season": [2014, 2019, 2022, 2025]})
    out = _assign_eras(df, eras)
    assert list(out["era"]) == ["E1", "E2", "E3", "E3"]      # E3 end_season=None is open-ended
    assert list(out["era_use_for_archetypes"]) == [False, True, True, True]


def test_block_map_is_disjoint_and_nonempty():
    cols = [c for v in BLOCKS.values() for c in v]
    assert len(cols) == len(set(cols)) and len(cols) >= 15   # no feature in two blocks
    assert {"shot_profile", "playmaking", "rebounding", "defense", "scoring"} == set(BLOCKS)
