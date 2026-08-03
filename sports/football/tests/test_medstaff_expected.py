"""Stage-4 tests: the expectation models and their nulls.

The load-bearing test here is ``test_loto_recovers_an_injected_club_effect``. Leave-one-team-out
is more expensive than random k-fold and the whole design rests on it being necessary, so that
necessity is demonstrated rather than asserted: an effect of known size is injected, and LOTO has
to recover it while random k-fold shrinks it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medstaff.expected.datasets import (  # noqa: E402
    duration_frame, incidence_frame, recurrence_frame, returns_at_all_frame,
)
from medstaff.expected.models import (  # noqa: E402
    build_hazard_pipeline, detectable_effect, feature_columns, fit_predict_kfold,
    fit_predict_loto, overdispersion, player_block_null, poisson_binomial_null,
    team_observed_expected, two_sided_p, whole_factor_permutation,
)

NUMERIC = ["age", "week"]
CATEGORICAL = ["position_group"]


def synthetic(n_clubs=12, n_players=25, n_weeks=10, bad_club="C00", bad_lift=0.25, seed=0):
    """A panel where exactly one club has an elevated injury rate."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clubs):
        club = f"C{c:02d}"
        for p in range(n_players):
            age = float(rng.normal(26, 3))
            for w in range(1, n_weeks + 1):
                base = 0.10 + 0.004 * (age - 26)
                prob = base + (bad_lift if club == bad_club else 0.0)
                rows.append({
                    "team": club, "gsis_id": f"{club}_P{p:03d}", "season": 2023, "week": w,
                    "age": age, "position_group": ["QB", "RB", "WR_TE", "OL"][p % 4],
                    "onset": int(rng.random() < min(max(prob, 0.001), 0.99)),
                })
    return pl.DataFrame(rows)


class TestFeatureGuard:
    def test_team_identity_is_rejected_from_the_design_matrix(self):
        """A club that could see its own identity could explain away its own residual."""
        frame = synthetic(n_clubs=3, n_players=3, n_weeks=2)
        with pytest.raises(ValueError, match="club identity leaked"):
            feature_columns(frame, ["age", "team"], CATEGORICAL)

    def test_clean_columns_pass(self):
        frame = synthetic(n_clubs=3, n_players=3, n_weeks=2)
        assert feature_columns(frame, NUMERIC, CATEGORICAL) == ["age", "week", "position_group"]

    def test_pipeline_builds_and_predicts_probabilities(self):
        frame = synthetic(n_clubs=3, n_players=8, n_weeks=5)
        pipe = build_hazard_pipeline(NUMERIC, CATEGORICAL)
        X = frame[[*NUMERIC, *CATEGORICAL]].to_pandas()
        pipe.fit(X, np.asarray(frame["onset"]))
        p = pipe.predict_proba(X)[:, 1]
        assert ((p >= 0) & (p <= 1)).all()


class TestFoldScheme:
    def test_loto_recovers_an_injected_club_effect_while_kfold_shrinks_it(self):
        """The test that justifies leave-one-team-out over the cheaper random k-fold.

        A random fold contains some of the held-out club's own rows, so the model learns that
        club's elevated rate and predicts it back — shrinking the residual toward zero. LOTO
        cannot, so the residual survives.
        """
        frame = synthetic(bad_club="C00", bad_lift=0.25, seed=3)

        p_loto = fit_predict_loto(frame, NUMERIC, CATEGORICAL, "onset")
        p_kfold = fit_predict_kfold(frame, NUMERIC, CATEGORICAL, "onset")

        oe_loto = team_observed_expected(frame, p_loto, outcome="onset")
        oe_kfold = team_observed_expected(frame, p_kfold, outcome="onset")

        d_loto = oe_loto.set_index("team").loc["C00", "diff"]
        d_kfold = oe_kfold.set_index("team").loc["C00", "diff"]

        assert d_loto > 0, "the injected effect must show up as a positive residual"
        assert d_loto > d_kfold, (
            f"LOTO must not shrink the effect the way k-fold does "
            f"(loto={d_loto:.1f}, kfold={d_kfold:.1f})"
        )

    def test_loto_scores_every_row(self):
        frame = synthetic(n_clubs=4, n_players=10, n_weeks=4)
        p = fit_predict_loto(frame, NUMERIC, CATEGORICAL, "onset")
        assert len(p) == frame.height and not np.isnan(p).any()


class TestObservedExpected:
    def test_diff_and_variance(self):
        frame = pl.DataFrame({
            "team": ["A", "A", "B", "B"], "onset": [1, 0, 1, 1], "gsis_id": list("wxyz"),
        })
        oe = team_observed_expected(frame, np.array([0.5, 0.5, 0.5, 0.5]),
                                    outcome="onset").set_index("team")
        assert oe.loc["A", "observed"] == 1 and oe.loc["A", "expected"] == pytest.approx(1.0)
        assert oe.loc["A", "diff"] == pytest.approx(0.0)
        assert oe.loc["B", "diff"] == pytest.approx(1.0)
        assert oe.loc["A", "var_indep"] == pytest.approx(0.5)

    def test_no_raw_rate_is_presented_without_its_expectation(self):
        """Both rates ship together — a raw rate alone is the thing this project refuses."""
        frame = pl.DataFrame({"team": ["A"], "onset": [1], "gsis_id": ["x"]})
        oe = team_observed_expected(frame, np.array([0.3]), outcome="onset")
        assert {"rate_obs", "rate_exp"} <= set(oe.columns)


class TestNulls:
    def test_poisson_binomial_centres_on_the_expected_total(self):
        p = np.full(200, 0.2)
        draws = poisson_binomial_null(p, n_sim=3000, seed=1)
        assert draws.mean() == pytest.approx(40, abs=2)

    def test_player_block_null_is_wider_when_weeks_are_correlated(self):
        """A fragile player is fragile all season, so independent Bernoulli understates spread."""
        # 20 players x 10 weeks; half the players are high-risk throughout
        blocks = np.repeat(np.arange(20), 10)
        p = np.where(blocks < 10, 0.05, 0.45)
        indep = poisson_binomial_null(p, n_sim=3000, seed=1)
        blocked = player_block_null(p, blocks, n_sim=3000, seed=1)
        assert blocked.std() > indep.std(), (
            f"block null must be more conservative (block={blocked.std():.2f}, "
            f"indep={indep.std():.2f})"
        )

    def test_two_sided_p_is_bounded_and_symmetric(self):
        draws = np.random.default_rng(0).normal(10, 2, 5000)
        assert 0 < two_sided_p(10, draws) <= 1
        assert two_sided_p(30, draws) < 0.01
        assert two_sided_p(10 - 6, draws) == pytest.approx(two_sided_p(10 + 6, draws), abs=0.05)

    def test_whole_factor_permutation_finds_a_real_club_effect(self):
        frame = synthetic(bad_club="C00", bad_lift=0.30, seed=5)
        p = fit_predict_loto(frame, NUMERIC, CATEGORICAL, "onset")
        res = whole_factor_permutation(frame, p, outcome="onset", n_sim=300)
        assert res["p_value"] < 0.05

    def test_whole_factor_permutation_is_null_when_no_club_differs(self):
        frame = synthetic(bad_club="__none__", bad_lift=0.0, seed=7)
        p = fit_predict_loto(frame, NUMERIC, CATEGORICAL, "onset")
        res = whole_factor_permutation(frame, p, outcome="onset", n_sim=300)
        assert res["p_value"] > 0.05

    def test_permutation_is_reproducible_under_a_fixed_seed(self):
        frame = synthetic(n_clubs=6, n_players=10, n_weeks=5)
        p = fit_predict_loto(frame, NUMERIC, CATEGORICAL, "onset")
        a = whole_factor_permutation(frame, p, outcome="onset", n_sim=100, seed=42)
        b = whole_factor_permutation(frame, p, outcome="onset", n_sim=100, seed=42)
        assert a["p_value"] == b["p_value"]


class TestOverdispersionAndPower:
    def test_pure_noise_leaves_no_signal(self):
        rng = np.random.default_rng(0)
        var = np.full(32, 25.0)
        diff = rng.normal(0, 5.0, 32)
        out = overdispersion(diff, var)
        assert out["tau2"] == pytest.approx(0.0, abs=12)
        assert out["intraclass"] < 0.35

    def test_real_spread_shows_up_as_signal(self):
        rng = np.random.default_rng(0)
        var = np.full(32, 25.0)
        diff = rng.normal(0, 5.0, 32) + rng.normal(0, 15.0, 32)
        out = overdispersion(diff, var)
        assert out["tau2"] > 0 and out["intraclass"] > 0.5

    def test_detectable_effect_scales_with_sampling_variance(self):
        small = detectable_effect(np.full(32, 4.0))
        large = detectable_effect(np.full(32, 100.0))
        assert large["min_detectable"] > small["min_detectable"]
        assert small["min_detectable"] == pytest.approx(1.96 * 2, abs=0.05)


class TestFrames:
    def _episodes(self):
        return pl.DataFrame([
            {"season": 2023, "gsis_id": "P1", "team": "KC", "position_group": "WR_TE",
             "body_group": "knee", "severity_worst": "DNP", "is_severe": False,
             "used_reserve": False, "onset_week": 2, "end_week": 4, "return_week": 5,
             "games_missed": 3, "weeks_elapsed": 3, "at_risk": True, "recur_games_to": 1},
            {"season": 2023, "gsis_id": "P2", "team": "KC", "position_group": "OL",
             "body_group": "ankle", "severity_worst": "LIMITED", "is_severe": False,
             "used_reserve": True, "onset_week": 10, "end_week": 17, "return_week": None,
             "games_missed": 8, "weeks_elapsed": 8, "at_risk": False, "recur_games_to": None},
        ])

    def test_incidence_marks_onset_weeks_only(self):
        exposure = pl.DataFrame([
            {"season": 2023, "gsis_id": "P1", "week": w, "team": "KC"} for w in (1, 2, 5, 6)
        ])
        frame = incidence_frame(exposure, self._episodes())
        got = dict(zip(frame["week"].to_list(), frame["onset"].to_list()))
        assert got == {1: 0, 2: 1, 5: 0, 6: 0}

    def test_duration_expands_to_one_row_per_episode_week(self):
        frame = duration_frame(self._episodes())
        p1 = frame.filter(pl.col("gsis_id") == "P1").sort("week")
        assert p1.height == 3, "weeks 2, 3, 4"
        assert p1["returned"].to_list() == [0, 0, 1]

    def test_censored_episode_never_records_a_return(self):
        """A spell that ran out of season is not a fast return and not a dropped row."""
        frame = duration_frame(self._episodes())
        p2 = frame.filter(pl.col("gsis_id") == "P2")
        assert p2.height == 8 and p2["returned"].sum() == 0

    def test_recurrence_risk_set_is_returns_only(self):
        panel = pl.DataFrame([
            {"season": 2023, "gsis_id": "P1", "week": w, "team_played": True}
            for w in range(1, 18)
        ] + [
            {"season": 2023, "gsis_id": "P2", "week": w, "team_played": True}
            for w in range(1, 18)
        ])
        frame = recurrence_frame(self._episodes(), panel, horizon=6)
        assert set(frame["gsis_id"].to_list()) == {"P1"}, "P2 never returned, so cannot recur"

    def test_recurrence_stops_at_the_recurrence(self):
        panel = pl.DataFrame([
            {"season": 2023, "gsis_id": "P1", "week": w, "team_played": True}
            for w in range(1, 18)
        ])
        frame = recurrence_frame(self._episodes(), panel, horizon=6).sort("games_since_return")
        assert frame["recurred"].to_list() == [0, 1]

    def test_returns_at_all_is_its_own_outcome(self):
        """Otherwise a club scores well by having unrecovered cases quietly censored."""
        frame = returns_at_all_frame(self._episodes())
        got = dict(zip(frame["gsis_id"].to_list(), frame["returned_at_all"].to_list()))
        assert got == {"P1": 1, "P2": 0}
