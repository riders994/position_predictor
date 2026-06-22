"""Tests for the redraft data-availability gate (no network — loaders are stubbed)."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data import availability as av  # noqa: E402
from position_predictor.data.fetch import REGISTRY, Dataset  # noqa: E402

SEASON = 2025


def _stub_dataset(name, *, rows=None, raises=False, min_season=1999):
    """A registry Dataset whose loader returns ``rows`` rows (or raises)."""
    def loader(years):
        if raises:
            raise RuntimeError("not published")
        return pd.DataFrame({"x": range(rows)})
    return Dataset(name, loader, min_season=min_season)


def _install(monkeypatch, **specs):
    for name, ds in specs.items():
        monkeypatch.setitem(REGISTRY, name, ds)


def test_available_when_required_present(monkeypatch):
    _install(monkeypatch,
             seasonal=_stub_dataset("seasonal", rows=300),
             weekly=_stub_dataset("weekly", rows=5000),
             rosters=_stub_dataset("rosters", rows=900),
             snap_counts=_stub_dataset("snap_counts", rows=8000, min_season=2012))
    rep = av.check_season_available(SEASON, optional=("snap_counts",))
    assert rep.ok and rep.missing == []
    assert all(p.present for p in rep.required)
    assert rep.optional[0].present


def test_partial_release_fails_gate(monkeypatch):
    # weekly only a few hundred rows -> below floor -> treated as not-yet-complete.
    _install(monkeypatch,
             seasonal=_stub_dataset("seasonal", rows=300),
             weekly=_stub_dataset("weekly", rows=200),
             rosters=_stub_dataset("rosters", rows=900))
    rep = av.check_season_available(SEASON, optional=())
    assert not rep.ok
    assert "weekly" in rep.missing


def test_optional_missing_still_ok(monkeypatch):
    _install(monkeypatch,
             seasonal=_stub_dataset("seasonal", rows=300),
             weekly=_stub_dataset("weekly", rows=5000),
             rosters=_stub_dataset("rosters", rows=900),
             ngs_passing=_stub_dataset("ngs_passing", raises=True, min_season=2016))
    rep = av.check_season_available(SEASON, optional=("ngs_passing",))
    assert rep.ok
    assert not rep.optional[0].present


def _write_manifest(d: Path, name: str, seasons):
    (d / f"{name}.json").write_text(json.dumps({"seasons": seasons}))


def test_datasets_needing_refresh(tmp_path):
    _write_manifest(tmp_path, "seasonal", list(range(1999, 2025)))   # stale (max 2024)
    _write_manifest(tmp_path, "rosters", list(range(1999, 2026)))    # up to date (max 2025)
    # snap_counts: no manifest -> uncached -> stale; ids: always-refresh snapshot.
    stale = av.datasets_needing_refresh(
        2025, datasets=["seasonal", "rosters", "snap_counts", "ids"], manifest_dir=tmp_path)
    assert stale == ["seasonal", "snap_counts", "ids"]


def test_refresh_skips_uncovered_dataset(tmp_path):
    # Feature season predates NGS coverage (min_season 2016) -> nothing to refresh there.
    stale = av.datasets_needing_refresh(
        2010, datasets=["ngs_passing"], always_refresh=(), manifest_dir=tmp_path)
    assert stale == []
