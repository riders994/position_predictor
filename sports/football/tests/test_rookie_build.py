"""Tests for the rookie-year build (pure DataFrame functions, no network).

Covers the pieces that are genuinely novel here (team-code canonicalization across the
draft_picks/weekly convention mismatch, the combine join, and room-context computation) —
aggregate_player_seasons/label_eligibility_grid themselves are already covered by
test_build_dataset.py and test_scoring.py.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data.rookie_build import (  # noqa: E402
    _canonical_team,
    _combine_by_gsis,
    _room_context,
)


def test_canonical_team_fixes_pfr_style_codes():
    # draft_picks.team uses PFR abbreviations that differ from nflverse's convention.
    assert _canonical_team("GNB") == "GB"
    assert _canonical_team("KAN") == "KC"
    assert _canonical_team("LVR") == "LV"
    assert _canonical_team("NOR") == "NO"
    assert _canonical_team("NWE") == "NE"
    assert _canonical_team("SFO") == "SF"
    assert _canonical_team("TAM") == "TB"


def test_canonical_team_composes_with_relocation():
    # LAR (PFR) -> LA (nflverse code) -> LA (already-canonical, Rams didn't move again).
    assert _canonical_team("LAR") == "LA"
    # SDG (PFR) -> SD (nflverse code for the pre-2017 Chargers) -> LAC (relocation canonical).
    assert _canonical_team("SDG") == "LAC"


def test_canonical_team_passes_through_unmapped_codes():
    assert _canonical_team("PHI") == "PHI"
    assert _canonical_team("KC") == "KC"  # already nflverse-style, not PFR-style


def test_combine_by_gsis_joins_via_pfr_id_crosswalk():
    ids = pd.DataFrame({
        "gsis_id": ["00-001", "00-002", "00-003"],
        "pfr_id": ["PlayA00", "PlayB00", None],
    })
    combine = pd.DataFrame({
        "pfr_id": ["PlayA00", "PlayB00", "PlayC00"],  # PlayC has no gsis_id -> dropped
        "forty": [4.4, 4.6, 4.5],
        "bench": [15, 20, 18],
    })
    out = _combine_by_gsis(combine, ids)
    assert set(out["player_id"]) == {"00-001", "00-002"}
    assert out.set_index("player_id").loc["00-001", "forty"] == 4.4


def test_combine_by_gsis_returns_none_when_missing():
    assert _combine_by_gsis(None, pd.DataFrame({"gsis_id": [], "pfr_id": []})) is None
    assert _combine_by_gsis(pd.DataFrame({"pfr_id": []}), None) is None


def test_room_context_sums_same_position_workload_per_team_season():
    seasons = pd.DataFrame({
        "player_id": ["A", "B", "C"],
        "position": ["RB", "RB", "WR"],
        "recent_team": ["PHI", "PHI", "PHI"],
        "season": [2022, 2022, 2022],
        "touches": [200, 50, 999],  # C is a WR, must not count toward the RB room
    })
    room = _room_context(seasons, position="RB", workload_col="touches")
    row = room[(room["recent_team"] == "PHI") & (room["season"] == 2022)].iloc[0]
    assert row["room_prior_workload"] == 250
    assert row["room_size"] == 2


def test_room_context_empty_for_team_season_with_no_position_players():
    seasons = pd.DataFrame({
        "player_id": ["A"], "position": ["WR"], "recent_team": ["PHI"],
        "season": [2022], "targets": [100],
    })
    room = _room_context(seasons, position="RB", workload_col="targets")
    assert room.empty
