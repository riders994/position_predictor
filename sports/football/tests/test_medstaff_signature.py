"""Stage-5 tests: the cross-position-group signature.

The two specifications exist because they answer different questions, and the test that matters
most is the one pinning that difference: composition must be invariant to a club's overall injury
*level* while rate must not be. Without that, reporting both is theatre.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medstaff.signature.concordance import (  # noqa: E402
    ALR_REFERENCE, EXCLUDED_GROUPS, club_side_counts, coach_follows, composition_alr,
    composition_concordance, concordance, head_coaches, leave_one_group_out,
    variance_decomposition,
)

FOCAL = ["knee", "ankle", "back", "hip", "concussion"]


def episodes(rows):
    return pl.DataFrame(rows)


def ep(team, side, group, *, season=2023, gsis="P", pos="WR_TE"):
    return {"team": team, "side": side, "body_group": group, "season": season,
            "gsis_id": gsis, "position_group": pos}


class TestCounts:
    def test_unlabelled_spells_are_excluded_not_pooled(self):
        """`unknown` is 21.6% of episodes and concentrated in long absences."""
        rows = [ep("KC", "OFF", "knee"), ep("KC", "OFF", "unknown"),
                ep("KC", "OFF", "other"), ep("KC", "OFF", "illness")]
        counts = club_side_counts(episodes(rows))
        assert counts["body_group"].to_list() == ["knee"]
        assert set(EXCLUDED_GROUPS) >= {"unknown", "other", "illness"}


class TestCompositionSpec:
    def _counts(self, knee, ref, side="OFF", team="KC"):
        return pl.DataFrame([
            {"team": team, "side": side, "body_group": "knee", "episodes": knee},
            {"team": team, "side": side, "body_group": ALR_REFERENCE, "episodes": ref},
        ])

    def test_composition_is_invariant_to_overall_level(self):
        """The test that makes reporting two specifications meaningful.

        Scaling every count up leaves the *shares* untouched, so composition must not move.
        """
        small = composition_alr(self._counts(4, 8), focal=["knee"])
        large = composition_alr(self._counts(40, 80), focal=["knee"])
        # ALR is not exactly scale-free because of the continuity correction, but must be close
        assert small["alr_knee"][0] == pytest.approx(large["alr_knee"][0], abs=0.15)

    def test_composition_moves_when_the_mix_changes(self):
        even = composition_alr(self._counts(8, 8), focal=["knee"])["alr_knee"][0]
        knee_heavy = composition_alr(self._counts(16, 8), focal=["knee"])["alr_knee"][0]
        assert knee_heavy > even

    def test_zero_count_does_not_produce_negative_infinity(self):
        """Cells hold single-digit counts, so a zero must not send the log to -inf."""
        out = composition_alr(self._counts(0, 6), focal=["knee"])
        assert np.isfinite(out["alr_knee"][0])

    def test_missing_focal_part_is_filled_not_dropped(self):
        counts = pl.DataFrame([
            {"team": "KC", "side": "OFF", "body_group": ALR_REFERENCE, "episodes": 5}])
        out = composition_alr(counts, focal=["concussion"])
        assert np.isfinite(out["alr_concussion"][0])


class TestConcordance:
    def test_perfect_agreement(self):
        out = concordance([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert out["spearman"] == pytest.approx(1.0) and out["n"] == 5

    def test_nans_are_dropped_pairwise(self):
        out = concordance([1, 2, np.nan, 4, 5], [1, 2, 3, np.nan, 5])
        assert out["n"] == 3

    def test_too_few_points_returns_nan_not_a_number(self):
        out = concordance([1, 2], [1, 2])
        assert np.isnan(out["spearman"])

    def test_side_split_uses_offense_against_defense(self):
        rows = []
        for club in "ABCDEFGH":
            weight = ord(club) - 65
            for side in ("OFF", "DEF"):
                for _ in range(3 + weight):
                    rows.append(ep(club, side, "concussion"))
                for _ in range(10):
                    rows.append(ep(club, side, ALR_REFERENCE))
        out = composition_concordance(episodes(rows), focal=["concussion"])
        assert out["concussion"]["spearman"] > 0.8, "a planted signature must be recovered"

    def test_no_signature_gives_no_concordance(self):
        rng = np.random.default_rng(0)
        rows = []
        for club in "ABCDEFGH":
            for side in ("OFF", "DEF"):
                for _ in range(int(rng.integers(2, 12))):
                    rows.append(ep(club, side, "concussion"))
                for _ in range(10):
                    rows.append(ep(club, side, ALR_REFERENCE))
        out = composition_concordance(episodes(rows), focal=["concussion"])
        assert abs(out["concussion"]["spearman"]) < 0.75


class TestLeaveOneGroupOut:
    def test_drops_exactly_one_group_each_time(self):
        # each side needs >=2 position groups, or dropping one empties the side entirely
        rows = []
        for club in "ABCDEF":
            for side, positions in (("OFF", ("OL", "WR_TE")), ("DEF", ("DL", "DB"))):
                for pos in positions:
                    rows.extend([ep(club, side, "concussion", pos=pos)] * 4)
                    rows.extend([ep(club, side, ALR_REFERENCE, pos=pos)] * 6)
        out = leave_one_group_out(episodes(rows), focal=["concussion"])
        dropped = {r["dropped_group"] for r in out}
        assert dropped == {"OL", "WR_TE", "DL", "DB"}
        assert all(r["body_group"] == "concussion" for r in out)


class TestVarianceDecomposition:
    def test_common_share_is_higher_when_the_excess_spans_groups(self):
        """A club-wide excess must read as common; a one-group excess as interaction."""
        spread, narrow_rows = [], []
        for club in "ABCDEFGH":
            lift = 8 if club in "ABCD" else 0
            for pos in ("OL", "DL", "DB", "WR_TE"):
                spread.extend([ep(club, "OFF", "concussion", pos=pos)] * (2 + lift))
                spread.extend([ep(club, "OFF", ALR_REFERENCE, pos=pos)] * 20)
                only = lift if pos == "OL" else 0
                narrow_rows.extend([ep(club, "OFF", "concussion", pos=pos)] * (2 + only))
                narrow_rows.extend([ep(club, "OFF", ALR_REFERENCE, pos=pos)] * 20)
        wide = variance_decomposition(episodes(spread), focal=["concussion"])[0]
        narrow = variance_decomposition(episodes(narrow_rows), focal=["concussion"])[0]
        assert wide["share_common"] > narrow["share_common"]


class TestCoaches:
    def test_head_coach_resolved_from_both_sides(self):
        sched = pl.DataFrame({
            "season": [2023, 2023], "game_type": ["REG", "REG"],
            "home_team": ["KC", "BUF"], "away_team": ["BUF", "KC"],
            "home_coach": ["Andy Reid", "Sean McDermott"],
            "away_coach": ["Sean McDermott", "Andy Reid"],
        })
        out = head_coaches(sched)
        got = dict(zip(out["team"].to_list(), out["coach"].to_list()))
        assert got == {"KC": "Andy Reid", "BUF": "Sean McDermott"}

    def test_only_multi_club_coaches_are_returned(self):
        rows = []
        for team in ("IND", "CAR", "KC"):
            rows.extend([ep(team, "OFF", "concussion", season=2023)] * 50)
        eps = episodes(rows)
        coaches = pl.DataFrame({
            "season": [2023, 2023, 2023], "team": ["IND", "CAR", "KC"],
            "coach": ["Frank Reich", "Frank Reich", "Andy Reid"],
        })
        table, n_multi = coach_follows(eps, coaches, focal=["concussion"], min_episodes=10)
        assert n_multi == 1
        assert set(table["coach"].to_list()) == {"Frank Reich"}
