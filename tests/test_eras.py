"""Tests for the era-separate modeling framework (pure logic, no data deps)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eras import (  # noqa: E402
    Era,
    assign_era,
    era_segments_for_window,
    feature_columns_for_era,
    load_eras,
    train_window_bounds,
)
from position_predictor.utils.config import Config  # noqa: E402


def test_load_eras_from_real_config_is_valid_and_nested():
    cfg = Config.load(ROOT / "config" / "football_rb.yaml")
    eras = load_eras(cfg)
    assert [e.name for e in eras] == ["boxscore", "snaps", "ngs"]
    assert eras[-1].end_season is None  # ngs open-ended
    # nested schemas: each era's blocks superset the previous
    for prev, cur in zip(eras, eras[1:]):
        assert set(prev.feature_blocks).issubset(set(cur.feature_blocks))
    # snap_usage appears at snaps, ngs_efficiency only at ngs
    assert "snap_usage" not in eras[0].feature_blocks
    assert "snap_usage" in eras[1].feature_blocks
    assert "ngs_efficiency" in eras[2].feature_blocks


def test_assign_era_by_feature_season():
    cfg = Config.load(ROOT / "config" / "football_rb.yaml")
    eras = load_eras(cfg)
    assert assign_era(1999, eras) == "boxscore"
    assert assign_era(2011, eras) == "boxscore"
    assert assign_era(2012, eras) == "snaps"
    assert assign_era(2015, eras) == "snaps"
    assert assign_era(2016, eras) == "ngs"
    assert assign_era(2030, eras) == "ngs"   # open-ended
    assert assign_era(1990, eras) is None     # before coverage


def test_train_window_bounds_no_leakage_and_clipping():
    test_seasons = [2021, 2022, 2023, 2024, 2025]
    # horizon 1: latest train label = 2020 -> latest train feature season N = 2019
    assert train_window_bounds(10, test_seasons, earliest_season=1999) == (2010, 2019)
    assert train_window_bounds(20, test_seasons, earliest_season=1999) == (2000, 2019)
    # 30yr would start 1990 but clips to data earliest 1999
    assert train_window_bounds(30, test_seasons, earliest_season=1999) == (1999, 2019)


def test_era_segments_compose_window():
    cfg = Config.load(ROOT / "config" / "football_rb.yaml")
    eras = load_eras(cfg)
    test_seasons = [2021, 2022, 2023, 2024, 2025]

    # 10-year window (feature seasons 2010–2019) dips 2 seasons into boxscore, then snaps+ngs
    seg10 = {e.name: span for e, span in
             era_segments_for_window(10, test_seasons, eras, earliest_season=1999)}
    assert seg10["boxscore"] == (2010, 2011)   # only the tail of the boxscore era
    assert seg10["snaps"] == (2012, 2015)
    assert seg10["ngs"] == (2016, 2019)

    # 5-year window (2015–2019) -> snaps tail + ngs only, no boxscore at all
    seg5 = {e.name: span for e, span in
            era_segments_for_window(5, test_seasons, eras, earliest_season=1999)}
    assert "boxscore" not in seg5
    assert seg5["snaps"] == (2015, 2015)
    assert seg5["ngs"] == (2016, 2019)

    # 30-year window (1999–2019) -> all three era models, boxscore clipped to data start
    seg30 = {e.name: span for e, span in
             era_segments_for_window(30, test_seasons, eras, earliest_season=1999)}
    assert seg30["boxscore"] == (1999, 2011)
    assert seg30["snaps"] == (2012, 2015)
    assert seg30["ngs"] == (2016, 2019)


def test_load_eras_rejects_non_nested_schema():
    class FakeCfg:
        def get(self, k, default=None):
            return [
                dict(name="a", start_season=1999, end_season=2011, feature_blocks=["x", "y"]),
                dict(name="b", start_season=2012, end_season=None, feature_blocks=["x"]),  # drops y
            ] if k == "eras" else default
    with pytest.raises(ValueError, match="nested"):
        load_eras(FakeCfg())


def test_load_eras_rejects_gap():
    class FakeCfg:
        def get(self, k, default=None):
            return [
                dict(name="a", start_season=1999, end_season=2011, feature_blocks=["x"]),
                dict(name="b", start_season=2014, end_season=None, feature_blocks=["x"]),  # gap
            ] if k == "eras" else default
    with pytest.raises(ValueError, match="contiguous"):
        load_eras(FakeCfg())


def test_feature_columns_for_era_resolves_blocks():
    era = Era("snaps", 2012, 2015, ("volume", "snap_usage", "ngs_efficiency"))
    block_columns = {
        "volume": ["carries", "targets"],
        "snap_usage": ["snap_share"],
        # ngs_efficiency intentionally absent (not built yet) -> skipped, no error
    }
    cols = feature_columns_for_era(era, block_columns)
    assert cols == ["carries", "targets", "snap_share"]
