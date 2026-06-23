"""Tests for the nflreadpy loader wiring in the fetch registry.

These guard the 2026 ``nfl_data_py`` -> ``nflreadpy`` migration: that each dataset calls the
right ``nflreadpy`` function with the right keyword args and renames ``stats_player`` columns
back to the pipeline's canonical schema. The loaders stay in polars (no pandas copy) so the
fetch path can cache wide multi-season pulls without doubling peak memory. A fake ``nflreadpy``
module is injected so nothing touches the network or the real library.
"""

import sys
import types
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data.fetch import REGISTRY  # noqa: E402

_LOADERS = (
    "load_player_stats", "load_rosters", "load_snap_counts", "load_nextgen_stats",
    "load_draft_picks", "load_combine", "load_ff_playerids", "load_pbp",
)


def _install(monkeypatch, frame: pl.DataFrame) -> list[dict]:
    """Inject a fake ``nflreadpy`` whose every loader records its call and returns ``frame``."""
    calls: list[dict] = []
    mod = types.ModuleType("nflreadpy")

    def make(name):
        def fn(seasons=None, *, stat_type=None, summary_level=None):
            calls.append({"func": name, "seasons": seasons,
                          "stat_type": stat_type, "summary_level": summary_level})
            return frame.clone()
        return fn

    for name in _LOADERS:
        setattr(mod, name, make(name))
    monkeypatch.setitem(sys.modules, "nflreadpy", mod)
    return calls


def test_player_stats_loaders_summary_level_and_renames(monkeypatch):
    raw = pl.DataFrame({
        "player_id": ["00-1"],
        "passing_interceptions": [3],
        "sacks_suffered": [5],
        "team": ["KC"],
        "fantasy_points_ppr": [120.0],
    })
    calls = _install(monkeypatch, raw)
    out = REGISTRY["seasonal"].loader([2024])

    assert calls == [{"func": "load_player_stats", "seasons": [2024],
                      "stat_type": None, "summary_level": "reg"}]
    assert isinstance(out, pl.DataFrame)  # stays in polars — no pandas materialisation
    # stats_player columns normalised back to the pipeline's canonical names
    assert "interceptions" in out.columns and "passing_interceptions" not in out.columns
    assert "sacks" in out.columns and "sacks_suffered" not in out.columns
    assert "recent_team" in out.columns and "team" not in out.columns
    # untouched columns pass through unchanged
    assert out["fantasy_points_ppr"][0] == 120.0

    # weekly is the same loader with the per-week summary level (multi-season passthrough)
    REGISTRY["weekly"].loader([2023, 2024])
    assert calls[1] == {"func": "load_player_stats", "seasons": [2023, 2024],
                        "stat_type": None, "summary_level": "week"}


def test_rosters_renames_gsis_id_to_player_id(monkeypatch):
    _install(monkeypatch, pl.DataFrame({"gsis_id": ["00-1"], "age": [25]}))
    out = REGISTRY["rosters"].loader([2024])
    assert "player_id" in out.columns and "gsis_id" not in out.columns


def test_ngs_loaders_pass_stat_type(monkeypatch):
    calls = _install(monkeypatch, pl.DataFrame({"player_id": ["x"]}))
    for name, expected in (("ngs_rushing", "rushing"),
                           ("ngs_receiving", "receiving"),
                           ("ngs_passing", "passing")):
        calls.clear()
        REGISTRY[name].loader([2024])
        assert calls[0]["func"] == "load_nextgen_stats"
        assert calls[0]["stat_type"] == expected


def test_yearless_loader_called_without_seasons(monkeypatch):
    # ``ids`` takes no season list; fetch invokes the loader with ``None``.
    calls = _install(monkeypatch, pl.DataFrame({"gsis_id": ["x"]}))
    REGISTRY["ids"].loader(None)
    assert calls[0]["func"] == "load_ff_playerids"
    assert calls[0]["seasons"] is None


def test_rename_skips_columns_not_present(monkeypatch):
    # a frame missing some rename keys must not raise (only present keys are mapped)
    _install(monkeypatch, pl.DataFrame({"player_id": ["x"], "team": ["KC"]}))
    out = REGISTRY["seasonal"].loader([2024])
    assert "recent_team" in out.columns
    assert "interceptions" not in out.columns  # source column was absent
