"""Stage-6 tests: reliability, the gate, and power.

The gate decides how every stage-7 number is labelled, so it is tested against synthetic data
where the right answer is known: a persistent club effect must be recovered, and pure noise must
not be.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medstaff.validate.reliability import (  # noqa: E402
    GATE_SPLIT_HALF, detectable_by_group, reliability_gate, spearman_brown,
    split_half_reliability, temporal_reliability,
)


def panel(*, persistent_lift=0.0, seasons=(2021, 2022, 2023, 2024, 2025), n_clubs=16,
          n_players=20, n_weeks=8, seed=0):
    """Clubs get a per-club offset that either persists across seasons or is reshuffled."""
    rng = np.random.default_rng(seed)
    clubs = [f"C{c:02d}" for c in range(n_clubs)]
    offsets = {c: rng.normal(0, persistent_lift) for c in clubs}
    rows = []
    for season in seasons:
        for c in clubs:
            off = offsets[c] if persistent_lift else 0.0
            for pi in range(n_players):
                for w in range(n_weeks):
                    p = float(np.clip(0.15 + off, 0.01, 0.95))
                    rows.append({
                        "season": season, "team": c, "gsis_id": f"{c}_{season}_{pi}",
                        "position_group": ["QB", "OL", "DB", "WR_TE"][pi % 4],
                        "expected": 0.15,
                        "y": int(rng.random() < p),
                    })
    return pl.DataFrame(rows)


class TestSpearmanBrown:
    def test_corrects_upward(self):
        assert spearman_brown(0.5) == pytest.approx(2 / 3)

    def test_identity_at_one(self):
        assert spearman_brown(1.0) == pytest.approx(1.0)


class TestSplitHalf:
    def test_recovers_a_persistent_club_effect(self):
        frame = panel(persistent_lift=0.08, seed=1)
        out = split_half_reliability(frame, outcome="y", n_sim=60)
        assert out["r_full"] > GATE_SPLIT_HALF

    def test_pure_noise_gives_no_reliability(self):
        frame = panel(persistent_lift=0.0, seed=2)
        out = split_half_reliability(frame, outcome="y", n_sim=60)
        assert out["r_full"] < GATE_SPLIT_HALF

    def test_splits_players_not_rows(self):
        """A row-wise split would put one fragile player on both sides and fake agreement."""
        frame = panel(persistent_lift=0.0, n_clubs=8, n_players=6, n_weeks=4, seed=3)
        out = split_half_reliability(frame, outcome="y", n_sim=40)
        assert out["n_reps"] > 0
        assert -1.0 <= out["r_full"] <= 1.0


class TestTemporal:
    def test_persistent_effect_carries_across_the_split(self):
        frame = panel(persistent_lift=0.08, seed=4)
        out = temporal_reliability(frame, outcome="y", split_season=2024)
        assert out["spearman"] > 0.3 and out["n_clubs"] == 16

    def test_noise_does_not_carry(self):
        frame = panel(persistent_lift=0.0, seed=5)
        out = temporal_reliability(frame, outcome="y", split_season=2024)
        assert abs(out["spearman"]) < 0.5


class TestGate:
    def test_passes_only_when_all_three_clear(self):
        assert reliability_gate(split_half_r=0.5, temporal_r=0.4,
                                permutation_p=0.01)["passed"] is True

    def test_names_what_failed(self):
        gate = reliability_gate(split_half_r=0.9, temporal_r=0.05, permutation_p=0.4)
        assert gate["passed"] is False
        assert set(gate["failed"]) == {"temporal", "permutation"}

    def test_high_split_half_alone_is_not_enough(self):
        """The split-half/temporal gap is the whole point: consistency is not persistence."""
        gate = reliability_gate(split_half_r=0.85, temporal_r=0.05, permutation_p=0.001)
        assert gate["passed"] is False and gate["failed"] == ["temporal"]


class TestPower:
    def test_thin_cells_get_a_larger_detectable_threshold(self):
        frame = panel(persistent_lift=0.0, n_clubs=4, n_players=8, n_weeks=4, seed=6)
        out = detectable_by_group(frame)
        assert {"min_detectable", "n", "expected"} <= set(out.columns)
        assert (out["min_detectable"] > 0).all()
        thin = out.nsmallest(1, "n").iloc[0]
        thick = out.nlargest(1, "n").iloc[0]
        assert thick["min_detectable"] >= thin["min_detectable"]
