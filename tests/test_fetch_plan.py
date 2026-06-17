"""Tests for the fetch stage's pure logic (no network / nflverse import needed).

Covers config loading, season clipping by dataset coverage, and the dry-run fetch plan.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data.fetch import REGISTRY, _clip_seasons, fetch_all  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def test_config_loads_and_lists_seasons():
    cfg = Config.load(ROOT / "config" / "football_rb.yaml")
    seasons = cfg.seasons()
    assert seasons[0] == cfg.get("data.earliest_season")
    assert seasons[-1] == cfg.get("data.latest_completed_season")
    assert seasons == list(range(seasons[0], seasons[-1] + 1))
    # quantitative-only and availability model are configured per the design
    assert cfg.get("features.quantitative_only") is True
    assert cfg.get("availability_model.enabled") is True


def test_season_clipping_respects_coverage():
    seasons = list(range(1999, 2026))
    # snap counts only exist 2012+
    assert min(_clip_seasons(REGISTRY["snap_counts"], seasons)) == 2012
    # NGS only 2016+
    assert min(_clip_seasons(REGISTRY["ngs_rushing"], seasons)) == 2016
    # ids takes no years
    assert _clip_seasons(REGISTRY["ids"], seasons) == []
    # seasonal goes back to 1999
    assert min(_clip_seasons(REGISTRY["seasonal"], seasons)) == 1999


def test_dry_run_excludes_large_by_default():
    seasons = list(range(2010, 2026))
    results = {r.name: r for r in fetch_all(seasons, dry_run=True)}
    assert "pbp" not in results  # large pull excluded from the default set
    assert results["seasonal"].status == "planned"
    assert all(r.status in {"planned", "skipped"} for r in results.values())


def test_large_dataset_included_when_requested():
    seasons = list(range(2020, 2026))
    results = {r.name: r for r in fetch_all(seasons, datasets=["pbp"], dry_run=True)}
    assert results["pbp"].status == "planned"
