"""Stage-7 tests: composing residuals into letters.

The forced-rank curve is the risky part — it will assign three A's and eight F's to any input,
including noise — so these tests pin the things that keep it honest: shrinkage before ranking,
the exact quota, no raw rate ever reaching the board, and the recalibration path working.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medstaff.grades.compose import (  # noqa: E402
    COMPONENT_SIGN, LETTER_QUOTAS, assign_letters, component_z, composite, empirical_bayes,
    reliability_weights, separation_flags, shrinkage_factor, two_level_shrink,
)
from medstaff.grades.staff import attach_staff, coach_changes, head_coach_by_season  # noqa: E402

CLUBS = [f"C{i:02d}" for i in range(32)]


class TestQuota:
    def test_quota_sums_to_thirty_two(self):
        assert sum(n for _, n in LETTER_QUOTAS) == 32
        assert [ltr for ltr, _ in LETTER_QUOTAS] == ["A", "B", "C", "D", "F"]

    def test_exact_counts_on_a_full_board(self):
        scores = pd.Series(np.arange(32, dtype=float), index=CLUBS)
        letters = assign_letters(scores)
        counts = letters.value_counts().to_dict()
        assert counts == {"A": 3, "B": 5, "C": 8, "D": 8, "F": 8}

    def test_best_score_gets_an_A_and_worst_an_F(self):
        scores = pd.Series(np.arange(32, dtype=float), index=CLUBS)
        letters = assign_letters(scores)
        assert letters[CLUBS[31]] == "A" and letters[CLUBS[0]] == "F"

    def test_curve_assigns_letters_even_to_pure_noise(self):
        """Documents the cost of a forced rank: it orders, it does not test."""
        rng = np.random.default_rng(0)
        scores = pd.Series(rng.normal(size=32), index=CLUBS)
        counts = assign_letters(scores).value_counts().to_dict()
        assert counts["A"] == 3 and counts["F"] == 8

    def test_partial_board_scales_without_dropping_clubs(self):
        scores = pd.Series(np.arange(16, dtype=float), index=CLUBS[:16])
        letters = assign_letters(scores)
        assert len(letters) == 16 and letters.notna().all()

    def test_absolute_bands_override_the_curve(self):
        """The recalibration path: a score-to-grade table replaces the forced rank."""
        scores = pd.Series([3.0, 1.0, -1.0], index=["A1", "B1", "F1"])
        bands = (("A", 2.0), ("B", 0.0), ("F", -99.0))
        letters = assign_letters(scores, bands=bands)
        assert letters.to_dict() == {"A1": "A", "B1": "B", "F1": "F"}
        assert (letters == "A").sum() == 1, "no quota is enforced under absolute bands"


class TestSign:
    def test_bad_outcomes_are_negated_so_positive_is_always_better(self):
        assert COMPONENT_SIGN["incidence_no_history"] == -1.0
        assert COMPONENT_SIGN["recurrence"] == -1.0
        assert COMPONENT_SIGN["duration"] == +1.0
        assert COMPONENT_SIGN["returns_at_all"] == +1.0

    def test_more_injuries_than_expected_scores_worse(self):
        residuals = pd.DataFrame({
            "team": ["A", "B"], "component": ["recurrence"] * 2,
            "diff": [5.0, -5.0], "var_indep": [4.0, 4.0],
            "n": [100, 100], "observed": [20, 10], "expected": [15.0, 15.0],
        })
        z = component_z(residuals).set_index("team")["z"]
        assert z["A"] < 0 < z["B"]


class TestShrinkage:
    def test_pure_noise_shrinks_almost_fully(self):
        rng = np.random.default_rng(1)
        z = rng.normal(0, 1, 32)
        assert shrinkage_factor(z) < 0.3

    def test_real_spread_survives(self):
        rng = np.random.default_rng(1)
        z = rng.normal(0, 3, 32)
        assert shrinkage_factor(z) > 0.7

    def test_shrinkage_pulls_toward_the_mean(self):
        z = np.array([10.0, 0.0, -10.0])
        shrunk, sd, k = empirical_bayes(z)
        assert abs(shrunk[0]) < abs(z[0]) and 0 <= k <= 1
        assert (sd >= 0).all()

    def test_two_level_pulls_cells_toward_their_own_club(self):
        cells = np.array([5.0, -5.0, 5.0, -5.0])
        club = np.array([2.0, 2.0, -2.0, -2.0])
        shrunk, _sd, _k = two_level_shrink(cells, club)
        # each cell must end up between its raw value and its club's value
        for c, b, s in zip(cells, club, shrunk):
            assert min(c, b) - 1e-9 <= s <= max(c, b) + 1e-9


class TestWeights:
    def test_weights_sum_to_one_and_track_reliability(self):
        w = reliability_weights({"a": 0.8, "b": 0.4, "c": 0.0})
        assert sum(w.values()) == pytest.approx(1.0)
        assert w["a"] > w["b"] > w["c"]

    def test_a_component_with_no_signal_gets_no_weight(self):
        w = reliability_weights({"a": 1.0, "b": 0.0})
        assert w["b"] == pytest.approx(0.0)

    def test_composite_ignores_missing_components(self):
        wide = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, 4.0]}, index=["X", "Y"])
        out = composite(wide, {"a": 0.5, "b": 0.5})
        assert out["X"] == pytest.approx(1.0), "X has only component a"
        assert out["Y"] == pytest.approx(3.0)


class TestSeparation:
    def test_wide_intervals_mean_nothing_separates(self):
        scores = pd.Series([1.0, 0.9, 0.8], index=list("XYZ"))
        sds = pd.Series([1.0, 1.0, 1.0], index=list("XYZ"))
        out = separation_flags(scores, sds)
        assert not out["separated_from_next"].any()

    def test_a_large_gap_separates(self):
        scores = pd.Series([10.0, 0.0], index=list("XY"))
        sds = pd.Series([0.1, 0.1], index=list("XY"))
        out = separation_flags(scores, sds)
        assert bool(out.loc["X", "separated_from_next"])


class TestNoRawRate:
    def test_component_z_never_exposes_an_unadjusted_rate(self):
        """The direct analogue of the sibling project's no-raw-rate test.

        A rate without its expectation is exactly the number this project exists to refuse.
        """
        residuals = pd.DataFrame({
            "team": ["A"], "component": ["duration"], "diff": [1.0], "var_indep": [1.0],
            "n": [10], "observed": [5], "expected": [4.0],
            "rate_obs": [0.5], "rate_exp": [0.4],
        })
        out = component_z(residuals)
        assert "rate" not in out.columns
        assert not [c for c in out.columns if c.startswith("rate")]


class TestStaff:
    def test_attach_staff_is_a_no_op_without_a_table(self):
        grades = pl.DataFrame({"team": ["KC"], "score": [1.0]})
        assert attach_staff(grades, None).equals(grades)

    def test_attach_staff_canonicalises_relocated_codes(self):
        grades = pl.DataFrame({"team": ["LAC"], "score": [1.0]})
        table = pl.DataFrame({"team": ["SD"], "head_athletic_trainer": ["Someone"]})
        out = attach_staff(grades, table)
        assert out["head_athletic_trainer"].to_list() == ["Someone"]

    def test_coach_change_flags_a_multi_regime_window(self):
        coaches = pl.DataFrame({
            "season": [2021, 2022, 2021, 2022], "team": ["KC", "KC", "CAR", "CAR"],
            "coach": ["Andy Reid", "Andy Reid", "Matt Rhule", "Frank Reich"],
        })
        out = coach_changes(coaches).sort("team")
        got = dict(zip(out["team"].to_list(), out["regime_changed"].to_list()))
        assert got == {"CAR": True, "KC": False}

    def test_head_coach_resolved_from_both_home_and_away(self):
        sched = pl.DataFrame({
            "season": [2023], "game_type": ["REG"], "home_team": ["KC"], "away_team": ["BUF"],
            "home_coach": ["Andy Reid"], "away_coach": ["Sean McDermott"],
        })
        out = head_coach_by_season(sched)
        assert set(out["team"].to_list()) == {"KC", "BUF"}
