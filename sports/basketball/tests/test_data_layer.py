"""No-network unit tests for the basketball data layer + config."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.data.fetch import DEFAULT_DATASETS, REGISTRY, Dataset  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402


def test_config_loads_and_season_range():
    cfg = Config.load(ROOT / "config" / "nba_archetypes.yaml")
    assert cfg.get("experiment.sport") == "basketball"
    seasons = cfg.seasons()
    assert seasons[0] == 2014 and seasons[-1] == 2026          # 2013-14 floor .. 2025-26
    # Phase 1 uses E2+E3 only; Phase 2/3 use all eras.
    eras = cfg.get("eras")
    assert [e["name"] for e in eras] == ["E1", "E2", "E3"]
    assert [e["name"] for e in eras if e["use_for_archetypes"]] == ["E2", "E3"]
    assert cfg.get("fantasy.league_ids") == ["blk3bn3clw9njuhc", "wserh14rmbbpqtcg"]


def test_registry_and_default_excludes_large():
    assert {"player_season_stats", "rosters", "team_season_stats"} <= set(REGISTRY)
    # heavy per-event pulls are opt-in (not in the default fetch set)
    assert "shots" in REGISTRY and REGISTRY["shots"].large
    assert "shots" not in DEFAULT_DATASETS and "player_boxscore" not in DEFAULT_DATASETS
    assert "player_season_stats" in DEFAULT_DATASETS


def test_dataset_loader_is_callable_without_importing_source():
    # Building a loader must not import sportsdataverse (lazy) — just yields a callable.
    ds = Dataset("x", "load_nba_player_season_stats")
    assert callable(ds.loader)
