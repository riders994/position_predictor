"""Tests for the drafting-situation layer.

The danger here is not a crash, it is a plausible table. Team-level cells hold four quarterbacks,
so any statistic that does not explicitly control for draft capital and state its own power will
produce a ranking that looks like insight and is noise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qb_breakout.situation.regime import (  # noqa: E402
    attach_gm,
    attach_situation,
    detectable_effect,
    expected_from_draft,
    group_permutation_p,
    head_coach_by_team_season,
    regime_table,
)


def _schedules():
    return pl.DataFrame({
        "season": [2015] * 4 + [2016] * 2,
        "home_team": ["GB", "GB", "GB", "CHI", "GB", "GB"],
        "home_coach": ["Mike McCarthy"] * 3 + ["John Fox"] + ["Mike McCarthy"] * 2,
        "away_team": ["CHI", "CHI", "MIN", "GB", "CHI", "MIN"],
        "away_coach": ["John Fox", "John Fox", "Mike Zimmer", "Mike McCarthy",
                       "John Fox", "Mike Zimmer"],
    })


# --------------------------------------------------------------------------------- regime data


def test_a_coach_is_found_from_both_home_and_away_games():
    out = head_coach_by_team_season(_schedules()).to_pandas()
    got = dict(zip(zip(out["season"], out["team"]), out["coach"]))
    assert got[(2015, "GB")] == "Mike McCarthy"
    assert got[(2015, "MIN")] == "Mike Zimmer"     # appears only as an away team


def test_a_midseason_change_resolves_to_whoever_coached_most():
    """The draft is a spring decision, so the season's dominant coach is the right attribution."""
    sched = pl.DataFrame({
        "season": [2015] * 4,
        "home_team": ["NYJ"] * 4,
        "home_coach": ["Real Coach", "Real Coach", "Real Coach", "Interim Guy"],
        "away_team": ["BUF"] * 4,
        "away_coach": ["Rex Ryan"] * 4,
    })
    out = head_coach_by_team_season(sched).to_pandas()
    assert out[out["team"] == "NYJ"]["coach"].iloc[0] == "Real Coach"


def test_franchise_abbreviations_are_canonicalised_before_joining():
    """Draft frames use PFR codes (GNB) and schedules use nflverse codes (GB)."""
    frame = pd.DataFrame({"draft_franchise": ["GNB"], "draft_season": [2015]})
    out = attach_situation(frame, coaches=_schedules().pipe(head_coach_by_team_season))
    assert out["draft_coach"].iloc[0] == "Mike McCarthy"


def test_undrafted_quarterbacks_do_not_acquire_a_regime():
    frame = pd.DataFrame({"draft_franchise": [None], "draft_season": [2015.0]})
    out = attach_situation(frame, coaches=_schedules().pipe(head_coach_by_team_season))
    assert pd.isna(out["draft_coach"].iloc[0])


def test_a_supplied_gm_table_joins_on_canonical_teams():
    frame = pd.DataFrame({"draft_franchise": ["GNB", "SDG"], "draft_season": [2015, 2015]})
    gms = pd.DataFrame({"season": [2015, 2015], "team": ["GB", "LAC"],
                        "draft_gm": ["Ted Thompson", "Tom Telesco"]})
    out = attach_gm(frame, gms)
    assert list(out["draft_gm"]) == ["Ted Thompson", "Tom Telesco"]


# ------------------------------------------------------------------- expectation and testing


def _cohort(n=120, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.integers(1, 260, n).astype(float)
    p = 1 / (1 + np.exp((pick - 60) / 40))
    return pd.DataFrame({
        "draft_pick": pick,
        "draft_franchise": rng.choice(list("ABCDEFGH"), n),
        "ever_sustained": rng.binomial(1, p),
    })


def test_expected_probability_tracks_draft_position():
    frame = _cohort(200, seed=2)
    exp = expected_from_draft(frame)
    early = exp[frame["draft_pick"] < 40].mean()
    late = exp[frame["draft_pick"] > 180].mean()
    assert early > late


def test_undrafted_players_get_an_expectation_rather_than_a_crash():
    frame = _cohort(150, seed=4)
    frame.loc[:19, "draft_pick"] = np.nan
    exp = expected_from_draft(frame)
    assert np.isfinite(exp).all()
    assert (exp > 0).all()


def test_regimes_below_the_floor_are_omitted_entirely():
    """A one-quarterback 'regime' invites reading a coincidence as a finding."""
    frame = _cohort(120, seed=5)
    frame.loc[frame.index[:1], "draft_franchise"] = "TINY"
    table = regime_table(frame, "draft_franchise", expected_from_draft(frame), n_simulations=200)
    assert "TINY" not in set(table["draft_franchise"])


def test_a_team_that_only_drafted_early_is_not_credited_for_it():
    """Raw rates reward draft capital. Expectation has to absorb it, or the table is a pick list."""
    n = 160
    rng = np.random.default_rng(7)
    pick = np.where(np.arange(n) < 40, rng.integers(1, 20, n), rng.integers(150, 250, n)).astype(float)
    p = 1 / (1 + np.exp((pick - 60) / 40))
    frame = pd.DataFrame({
        "draft_pick": pick,
        # "RICH" drafts only in the top 20 and breaks out at exactly the rate its picks imply.
        "draft_franchise": np.where(np.arange(n) < 40, "RICH", "POOR"),
        "ever_sustained": rng.binomial(1, p),
    })
    table = regime_table(frame, "draft_franchise", expected_from_draft(frame), n_simulations=2000)
    rich = table[table["draft_franchise"] == "RICH"].iloc[0]
    assert abs(rich["diff"]) < 0.35 * rich["qbs"]      # credited for the pick, not the outcome
    assert rich["p_two_sided"] > 0.05


def test_random_group_labels_do_not_look_like_a_team_effect():
    """Checked across seeds, not one.

    A calibrated test *should* return p < 0.05 about a twentieth of the time under the null, so
    asserting it on a single random grouping is a coin-flip that fails one run in twenty. What
    must hold is the distribution: most groupings unremarkable, and no systematic bias towards
    finding an effect that is not there.
    """
    pvals = []
    for seed in range(10):
        frame = _cohort(200, seed=100 + seed)
        rng = np.random.default_rng(200 + seed)
        frame["random_group"] = rng.choice(list("ABCDEFGH"), len(frame))
        pvals.append(group_permutation_p(frame, "random_group", expected_from_draft(frame),
                                         n_permutations=200)["p_value"])
    assert np.median(pvals) > 0.2
    assert sum(p < 0.05 for p in pvals) <= 2


def test_a_planted_regime_effect_is_detected():
    """The test has to be able to find something, or a null result means nothing."""
    frame = _cohort(200, seed=13)
    frame.loc[frame["draft_franchise"] == "A", "ever_sustained"] = 1
    frame.loc[frame["draft_franchise"] == "B", "ever_sustained"] = 0
    out = group_permutation_p(frame, "draft_franchise", expected_from_draft(frame),
                              n_permutations=300)
    assert out["p_value"] < 0.05


def test_detectable_effect_reports_a_threshold_above_expectation():
    """A null result is only informative next to the smallest effect that could have been seen."""
    frame = _cohort(160, seed=15)
    power = detectable_effect(frame, "draft_franchise", expected_from_draft(frame),
                              n_simulations=2000)
    assert power["breakouts_needed"] > power["expected_breakouts"]
    assert power["extra_breakouts_needed"] > 0
    assert power["median_qbs_per_regime"] >= 3


def test_no_raw_breakout_rate_is_ever_produced():
    """The module's central promise: a column of team rates would be all noise and all quotable."""
    frame = _cohort(120, seed=17)
    table = regime_table(frame, "draft_franchise", expected_from_draft(frame), n_simulations=200)
    assert "rate" not in table.columns
    assert set(table.columns) == {"draft_franchise", "qbs", "observed", "expected", "diff",
                                  "p_two_sided"}


def test_expected_column_is_not_left_on_the_input_frame():
    frame = _cohort(80, seed=19)
    before = list(frame.columns)
    regime_table(frame, "draft_franchise", expected_from_draft(frame), n_simulations=100)
    assert list(frame.columns) == before


@pytest.mark.parametrize("factor", ["draft_franchise"])
def test_table_is_sorted_by_over_performance(factor):
    frame = _cohort(160, seed=21)
    table = regime_table(frame, factor, expected_from_draft(frame), n_simulations=200)
    assert list(table["diff"]) == sorted(table["diff"], reverse=True)
