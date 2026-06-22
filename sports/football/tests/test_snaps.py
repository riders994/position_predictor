"""Tests for the snap-count crosswalk + season aggregation (pure DataFrame logic)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data.snaps import (  # noqa: E402
    build_player_season_snaps,
    crosswalk_pfr_to_gsis,
)


def _ids():
    return pd.DataFrame([
        dict(pfr_id="P1", gsis_id="G1"),
        dict(pfr_id="P2", gsis_id="G2"),
        dict(pfr_id="P1", gsis_id="G1"),  # dup -> deduped
        dict(pfr_id=None, gsis_id="G9"),  # dropped
    ])


def _snaps():
    def r(pfr, season, week, snaps, pct, gt="REG"):
        return dict(pfr_player_id=pfr, season=season, week=week, game_type=gt,
                    offense_snaps=snaps, offense_pct=pct, player="x", position="RB")
    return pd.DataFrame([
        r("P1", 2022, 1, 50, 1.0),    # team = 50
        r("P1", 2022, 2, 30, 0.6),    # team = 50
        r("P1", 2022, 20, 99, 1.0, gt="DIV"),  # playoff -> excluded
        r("P2", 2022, 1, 0, 0.0),     # 0-snap game -> contributes nothing to share
        r("PX", 2022, 1, 40, 0.8),    # unmapped pfr -> dropped
    ])


def test_crosswalk_dedupes_and_drops_nulls():
    cw = crosswalk_pfr_to_gsis(_ids())
    assert set(cw["pfr_id"]) == {"P1", "P2"}
    assert len(cw) == 2


def test_snap_share_weighted_and_excludes_playoffs():
    out = build_player_season_snaps(_snaps(), _ids())
    g1 = out[out.player_id == "G1"].iloc[0]
    # (50+30) / (50+50) = 0.8 ; playoff week 20 excluded
    assert abs(g1["snap_share"] - 0.8) < 1e-9
    assert g1["snaps"] == 80
    assert abs(g1["snaps_per_game"] - 40.0) < 1e-9
    assert g1["player_id"] == "G1"  # crosswalked to gsis id


def test_zero_snap_game_yields_nan_share():
    out = build_player_season_snaps(_snaps(), _ids())
    g2 = out[out.player_id == "G2"].iloc[0]
    # only a 0-snap game -> no recoverable team snaps -> NaN share, 0 snaps
    assert pd.isna(g2["snap_share"])
    assert g2["snaps"] == 0


def test_unmapped_pfr_dropped():
    out = build_player_season_snaps(_snaps(), _ids())
    assert "PX" not in set(out["player_id"])  # never mapped to a gsis id


def test_empty_snaps_returns_empty():
    empty = pd.DataFrame(columns=["pfr_player_id", "season", "game_type",
                                  "offense_snaps", "offense_pct"])
    assert len(build_player_season_snaps(empty, _ids())) == 0
