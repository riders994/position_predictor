"""Tests for the CFBD extension: key handling, calibration honesty, and source tagging.

The risk this layer introduces is not a crash — it is a silently spliced feature. A CFBD value
mapped onto the cfbfastR scale looks exactly like a measured one unless the code refuses to let it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qb_breakout.data import cfbd as cfbd_mod  # noqa: E402
from qb_breakout.data.cfbd import (  # noqa: E402
    api_key,
    apply_calibration,
    build_cfbd_qb_seasons,
    fit_calibration,
)


# ------------------------------------------------------------------------------- credentials


def test_api_key_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("CFBD_API_KEY", "  from-env  ")
    assert api_key() == "from-env"


def test_api_key_falls_back_to_the_config_file(monkeypatch, tmp_path):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    key_file = tmp_path / "api_key"
    key_file.write_text("from-file\n")
    monkeypatch.setattr(cfbd_mod, "KEY_FILE", key_file)
    assert api_key() == "from-file"


def test_missing_key_fails_loudly_with_instructions(monkeypatch, tmp_path):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    monkeypatch.setattr(cfbd_mod, "KEY_FILE", tmp_path / "nope")
    with pytest.raises(SystemExit, match="CFBD_API_KEY"):
        api_key()


# -------------------------------------------------------------------------------- calibration


def _linear_overlap(n=200, slope=2.0, intercept=0.5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.25, 0.15, n)
    y = (x - intercept) / slope + rng.normal(0, noise, n)
    return pd.DataFrame({"src": x, "tgt": y})


def test_calibration_recovers_a_clean_linear_relationship():
    cal = fit_calibration(_linear_overlap(noise=0.0), source_col="src", target_col="tgt")
    assert cal["r2"] == pytest.approx(1.0, abs=1e-6)
    assert cal["noise_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_noise_ratio_reports_the_spread_the_mapping_loses():
    """The number that decides whether a feature is portable, so it has to mean what it says."""
    clean = fit_calibration(_linear_overlap(noise=0.001), source_col="src", target_col="tgt")
    dirty = fit_calibration(_linear_overlap(noise=0.20), source_col="src", target_col="tgt")
    assert clean["noise_ratio"] < 0.1
    assert dirty["noise_ratio"] > 0.5
    # resid_sd and target_sd are in target units, and their ratio is the reported noise.
    assert dirty["noise_ratio"] == pytest.approx(dirty["resid_sd"] / dirty["target_sd"], abs=1e-3)


def test_calibrated_values_are_marked_as_estimates():
    """A mapped value must never be indistinguishable from a measured one."""
    cal = {"slope": 2.0, "intercept": 0.1}
    out = apply_calibration(pd.DataFrame({"ppa_pass": [0.2, 0.4]}), cal)
    assert list(out["pass_epa_per_db_est"]) == pytest.approx([0.5, 0.9])
    assert "pass_epa_per_db" not in out.columns   # the measured name is not claimed
    assert set(out["efficiency_is_estimated"]) == {1}


def test_calibration_ignores_rows_missing_either_side():
    df = pd.DataFrame({"src": [0.1, 0.2, np.nan, 0.4], "tgt": [1.0, 2.0, 3.0, np.nan]})
    assert fit_calibration(df, source_col="src", target_col="tgt")["n"] == 2


# ------------------------------------------------------------------------------- season build


def _fake_api(monkeypatch, passing, rushing, ppa):
    def _get(path, params, session=None):
        if path == "/stats/player/season":
            return passing if params["category"] == "passing" else rushing
        if path == "/ppa/players/season":
            return ppa
        raise AssertionError(path)

    monkeypatch.setattr(cfbd_mod, "_get", _get)


def _stat(pid, name, team, stat_type, value, category):
    return {"playerId": pid, "player": name, "team": team, "statType": stat_type,
            "stat": str(value), "category": category}


def test_season_build_pivots_stats_and_joins_ppa(monkeypatch):
    passing = [_stat("1", "QB One", "Oklahoma", t, v, "passing") for t, v in
               [("ATT", 400), ("COMPLETIONS", 260), ("YDS", 3500), ("TD", 30), ("INT", 8)]]
    rushing = [_stat("1", "QB One", "Oklahoma", t, v, "rushing") for t, v in
               [("CAR", 100), ("YDS", 500), ("TD", 6)]]
    ppa = [{"id": "1", "position": "QB", "conference": "Big 12",
            "averagePPA": {"all": 0.3, "pass": 0.35, "rush": 0.1},
            "totalPPA": {"all": 100.0}}]
    _fake_api(monkeypatch, passing, rushing, ppa)

    out = build_cfbd_qb_seasons([2023])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["attempts"] == 400
    assert row["completion_pct"] == pytest.approx(0.65)
    assert row["ppa_pass"] == 0.35
    assert row["source"] == "cfbd"
    assert row["rush_share"] == pytest.approx(100 / 500)


def test_low_volume_passers_are_excluded(monkeypatch):
    passing = [_stat("9", "Punter", "Duke", "ATT", 3, "passing"),
               _stat("9", "Punter", "Duke", "COMPLETIONS", 1, "passing")]
    _fake_api(monkeypatch, passing, [], [])
    assert build_cfbd_qb_seasons([2023]).empty


def test_running_backs_do_not_become_quarterback_rows(monkeypatch):
    """Rushing is joined onto passers, so a 200-carry back with no attempts must not appear."""
    passing = [_stat("1", "QB One", "Oklahoma", "ATT", 400, "passing"),
               _stat("1", "QB One", "Oklahoma", "COMPLETIONS", 250, "passing")]
    rushing = [_stat("2", "RB Two", "Oklahoma", "CAR", 200, "rushing"),
               _stat("2", "RB Two", "Oklahoma", "YDS", 1200, "rushing")]
    _fake_api(monkeypatch, passing, rushing, [])
    out = build_cfbd_qb_seasons([2023])
    assert list(out["player"]) == ["QB One"]
    assert out.iloc[0]["rush_att"] == 0.0
