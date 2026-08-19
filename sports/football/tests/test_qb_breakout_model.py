"""Tests for stage 6, concentrated on the ways a small-N model lies.

None of these check that the model is accurate — at 35 positives it is not entitled to be. They
check that it cannot cheat: that unsettled outcomes stay out, that draft position never becomes a
feature, that the permutation null is a real null, and that the two tiers are what they claim.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qb_breakout.model.features import (  # noqa: E402
    BENCHMARK,
    CFBFASTR_ONLY,
    PORTABLE,
    feature_columns,
    modelling_frame,
)
from qb_breakout.model.fit import (  # noqa: E402
    cross_validate,
    draft_band,
    opportunity_profile,
    permutation_null,
    precision_at_k,
    within_band_auc,
)


def _cohort(n=120, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "player_name": [f"QB {i}" for i in range(n)],
        "archetype_name": rng.choice(["pocket_quick", "runner_downfield"], n),
        "seasons_elapsed": rng.integers(1, 15, n),
        "sustained_censored": False,
        "sustained_season": np.where(signal > 0.8, 2015.0, np.nan),
        "late_sustained": np.nan,
        "entry_season": rng.integers(2005, 2020, n),
        "draft_pick": rng.integers(1, 260, n).astype(float),
        "career_games": rng.integers(1, 150, n),
        **{c: signal + rng.normal(0, 0.5, n) for c in PORTABLE},
        **{c: signal + rng.normal(0, 0.5, n) for c in CFBFASTR_ONLY},
    })


# ------------------------------------------------------------------------------ who is in scope


def test_an_unsettled_career_is_not_taught_as_a_negative():
    """A 2024 entrant who has not broken out has not failed to; he has not finished."""
    cohort = _cohort()
    cohort.loc[:9, "seasons_elapsed"] = 1
    cohort.loc[:9, "sustained_season"] = np.nan

    frame = modelling_frame(cohort, min_settled_seasons=5)
    assert (frame["seasons_elapsed"] >= 5).all()
    assert not frame["player_id"].isin([f"P{i}" for i in range(10)]).any()


def test_a_censored_career_is_excluded_even_if_it_is_old_enough():
    """A trigger whose confirmation window is unfinished is unknown, not negative."""
    cohort = _cohort()
    cohort.loc[:4, "seasons_elapsed"] = 12
    cohort.loc[:4, "sustained_censored"] = True
    frame = modelling_frame(cohort)
    assert not frame["player_id"].isin([f"P{i}" for i in range(5)]).any()


def test_quarterbacks_without_a_college_profile_are_dropped():
    cohort = _cohort()
    cohort.loc[:3, "archetype_name"] = None
    frame = modelling_frame(cohort)
    assert frame["archetype_name"].notna().all()


def test_the_outcome_is_the_two_bar_label_not_the_single_bar_diagnostic():
    """`ever_breakout` is the top-15 trigger alone; the project's label is the sustained one."""
    cohort = _cohort()
    cohort["ever_breakout"] = 1          # a decoy that must not be picked up
    frame = modelling_frame(cohort)
    expected = frame["sustained_season"].notna().astype(int)
    assert (frame["ever_sustained"] == expected).all()
    assert frame["ever_sustained"].sum() < len(frame)


# ------------------------------------------------------------------------------------- tiers


def test_draft_position_is_never_a_feature():
    """The league's opinion is a benchmark. A model reading it partly just reports the scouts."""
    for tier in ("portable", "full"):
        numeric, categorical = feature_columns(tier)
        assert not set(numeric + categorical) & set(BENCHMARK)


def test_the_portable_tier_is_a_strict_subset_of_the_full_one():
    """Otherwise the tiers differ by more than efficiency and the comparison means nothing."""
    p_num, p_cat = feature_columns("portable")
    f_num, f_cat = feature_columns("full")
    assert set(p_num) < set(f_num)
    assert p_cat == f_cat
    assert set(f_num) - set(p_num) == set(CFBFASTR_ONLY)


def test_the_portable_tier_excludes_every_efficiency_feature():
    """Portability is the whole point: one cfbfastR-only column makes it un-scoreable today."""
    numeric, _ = feature_columns("portable")
    assert not set(numeric) & set(CFBFASTR_ONLY)
    assert not any("epa" in c or "success" in c for c in numeric)


def test_interception_rate_stays_out_of_both_tiers():
    """Portable across sources, but absent from cfbfastR for most pre-2014 careers (§2.6)."""
    for tier in ("portable", "full"):
        numeric, _ = feature_columns(tier)
        assert "final_int_rate" not in numeric


# ------------------------------------------------------------------------------- evaluation


def test_shuffled_labels_score_around_chance():
    """If the null is not centred on 0.5 the evaluation has a leak, and every p-value is wrong."""
    frame = modelling_frame(_cohort(140, seed=3))
    numeric, categorical = feature_columns("portable")
    null = permutation_null(frame, numeric, categorical, n_permutations=12, random_state=5)
    assert 0.35 < null.mean() < 0.65


def test_real_labels_beat_shuffled_ones_when_signal_exists():
    """The fixture has real signal in it, so the evaluation must be able to see it."""
    frame = modelling_frame(_cohort(140, seed=3))
    numeric, categorical = feature_columns("portable")
    rows, _ = cross_validate(frame, numeric, categorical, n_repeats=2)
    observed = pd.DataFrame(rows)["auc"].mean()
    null = permutation_null(frame, numeric, categorical, n_permutations=12, random_state=5)
    assert observed > null.mean() + 2 * null.std()


def test_out_of_fold_scores_cover_every_quarterback_exactly_once():
    """A pooled score that included in-fold predictions would be optimistic and invisible."""
    frame = modelling_frame(_cohort(100, seed=1))
    numeric, categorical = feature_columns("portable")
    _, oof = cross_validate(frame, numeric, categorical, n_repeats=1)
    assert len(oof) == len(frame)
    assert np.isfinite(oof).all()
    assert (oof != 0).all()


def test_precision_at_k_reads_the_top_of_the_ranking():
    y = [0, 0, 1, 1]
    assert precision_at_k(y, [0.1, 0.2, 0.9, 0.8], k=2) == pytest.approx(1.0)
    assert precision_at_k(y, [0.9, 0.8, 0.2, 0.1], k=2) == pytest.approx(0.0)


# --------------------------------------------------------------------------------- the draft


def test_undrafted_quarterbacks_land_in_the_last_band():
    frame = pd.DataFrame({"draft_pick": [5.0, 50.0, 200.0, np.nan]})
    assert list(draft_band(frame)) == ["R1", "R2-3", "R4+/UDFA", "R4+/UDFA"]


def test_opportunity_profile_exposes_the_snap_confound():
    """The draft allocates the playing time a fantasy breakout requires; the table must show it."""
    frame = pd.DataFrame({
        "draft_pick": [1.0, 2.0, 200.0, 201.0],
        "career_games": [100, 90, 5, 4],
        "ever_sustained": [1, 1, 0, 0],
    })
    out = opportunity_profile(frame).set_index("draft_band")
    assert out.loc["R1", "mean_nfl_games"] > out.loc["R4+/UDFA", "mean_nfl_games"]
    assert out.loc["R1", "rate"] > out.loc["R4+/UDFA", "rate"]


def test_within_band_auc_holds_the_draft_fixed():
    """Globally a college model re-reports the draft; the question is what it adds inside a band."""
    frame = pd.DataFrame({
        "draft_pick": [1.0, 2.0, 3.0, 4.0, 200.0, 201.0, 202.0, 203.0],
        "ever_sustained": [1, 0, 1, 0, 1, 0, 1, 0],
    })
    perfect = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    out = within_band_auc(frame, perfect).set_index("draft_band")
    assert out.loc["R1", "model_auc"] == pytest.approx(1.0)
    assert out.loc["R4+/UDFA", "model_auc"] == pytest.approx(1.0)
