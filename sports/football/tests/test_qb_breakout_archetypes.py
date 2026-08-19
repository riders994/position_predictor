"""Tests for the college QB archetype taxonomy.

The failure modes here are all silent. A taxonomy can cluster on the calendar, or on quality while
claiming to describe style, or rename every archetype when someone changes a random seed — and in
every case it still produces a tidy table of four clusters that looks exactly right.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qb_breakout.archetypes.cluster import (  # noqa: E402
    add_style_features,
    career_archetypes,
    fit_archetypes,
    name_archetypes,
    quality_leakage,
    standardize_within_season,
)


def _season(player, season=2015, *, rush_att=40, rush_yds=200.0, rush_td=2, pass_td=20,
            completions=200, pass_yds=2400.0, dropbacks=300, epa=0.15, team="Oklahoma"):
    return {
        "player": player, "season": season, "team": team, "dropbacks": dropbacks,
        "attempts": dropbacks - 20, "completions": completions, "pass_yds": pass_yds,
        "pass_td": pass_td, "rush_att": rush_att, "rush_yds": rush_yds, "rush_td": rush_td,
        "rush_share": rush_att / (dropbacks + rush_att),
        "rush_yds_per_att": rush_yds / rush_att if rush_att else None,
        "pass_epa": epa * dropbacks, "pass_epa_per_db": epa, "pass_success_rate": 0.45,
        "td_rate": 0.06, "completion_pct": 0.64, "games": 12,
    }


def _population(n=240, seed=0):
    """A synthetic league with two real style groups, so clustering has something to find."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        mobile = i % 2 == 0
        rows.append(_season(
            f"QB{i}", season=2010 + i % 5,
            rush_att=int(rng.normal(120 if mobile else 25, 8)),
            rush_yds=float(rng.normal(700 if mobile else 40, 40)),
            rush_td=int(max(0, rng.normal(10 if mobile else 1, 1))),
            pass_td=int(rng.normal(20, 2)),
            completions=int(rng.normal(200, 10)),
            pass_yds=float(rng.normal(2900 if mobile else 2200, 80)),
            epa=float(rng.normal(0.15, 0.1)),
        ))
    return pl.DataFrame(rows)


# ------------------------------------------------------------------------------ style features


def test_depth_is_measured_per_completion_not_per_attempt():
    """Yards per attempt mixes depth with accuracy; the taxonomy wants only depth."""
    row = add_style_features(pl.DataFrame([
        _season("QB", completions=200, pass_yds=2400.0)
    ])).to_dicts()[0]
    assert row["yards_per_completion"] == pytest.approx(12.0)


def test_goal_line_role_is_a_share_of_his_own_touchdowns():
    row = add_style_features(pl.DataFrame([
        _season("QB", pass_td=15, rush_td=5)
    ])).to_dicts()[0]
    assert row["rush_td_share"] == pytest.approx(0.25)


def test_a_quarterback_with_no_touchdowns_gets_null_not_zero():
    """0/0 is unknown, and an unknown goal-line role must not read as 'never runs one in'."""
    row = add_style_features(pl.DataFrame([
        _season("QB", pass_td=0, rush_td=0)
    ])).to_dicts()[0]
    assert row["rush_td_share"] is None


# ------------------------------------------------------------------------------------- era


def test_standardizing_is_within_season_so_archetypes_are_not_dates():
    """College offense drifted hard across the span. Raw features cluster on the calendar.

    Two eras with the same internal spread but very different levels must standardise onto the
    same scale — otherwise the first split a clustering finds is simply a decade.
    """
    rows = ([_season(f"A{i}", season=2005, rush_att=10 + 2 * i) for i in range(6)]
            + [_season(f"B{i}", season=2019, rush_att=100 + 2 * i) for i in range(6)])
    out = standardize_within_season(add_style_features(pl.DataFrame(rows)))

    old = out.filter(pl.col("season") == 2005)["z_rush_share"]
    new = out.filter(pl.col("season") == 2019)["z_rush_share"]
    assert old.mean() == pytest.approx(0.0, abs=1e-9)
    assert new.mean() == pytest.approx(0.0, abs=1e-9)
    # Raw rush share differs ~10x across the eras; standardised, the two eras overlap.
    assert old.min() == pytest.approx(new.min(), abs=0.15)


def test_a_season_with_no_spread_yields_null_not_nan():
    """NaN is not null, so it survives a null check and reaches the clustering as a coordinate."""
    rows = [_season("Solo", season=2007)]
    out = standardize_within_season(add_style_features(pl.DataFrame(rows)))
    assert out["z_rush_share"].null_count() == 1
    assert not out["z_rush_share"].is_nan().any()


# --------------------------------------------------------------------------- volume gating


def test_thin_seasons_are_assigned_but_do_not_define_the_centroids():
    """Rate features on small denominators are the classic source of artefact clusters.

    A 60-dropback season has a noise-dominated profile; it should still receive an archetype, but
    it must not get to pull a centroid towards itself.
    """
    thin = [_season(f"T{i}", season=2010 + i % 5, dropbacks=60, rush_att=5 + i, rush_yds=10.0 + i)
            for i in range(20)]
    df = pl.concat([_population(), pl.DataFrame(thin)], how="diagonal_relaxed")

    assigned, _ = fit_archetypes(df, k=2)
    thin_rows = assigned.filter(pl.col("dropbacks") == 60)
    assert set(thin_rows["fitted_on"].to_list()) == {0}
    assert thin_rows["archetype"].null_count() == 0


# ------------------------------------------------------------------------------------ naming


def test_names_come_from_centroid_position_not_the_label_integer():
    """k-means label integers are a seed artefact; names built on them silently permute."""
    df = _population()
    a, _ = fit_archetypes(df, k=2, random_state=1)
    b, _ = fit_archetypes(df, k=2, random_state=1234)

    keys = ["season", "team", "player"]
    joined = a.select([*keys, "archetype_name"]).join(b.select([*keys, "archetype_name"]), on=keys)
    agreement = (joined["archetype_name"] == joined["archetype_name_right"]).mean()
    assert agreement > 0.95


def test_the_mobile_group_is_named_a_runner():
    """A name that does not track the thing it describes is worse than a number."""
    assigned, _ = fit_archetypes(_population(), k=2)
    by_name = (assigned.filter(pl.col("archetype").is_not_null())
               .group_by("archetype_name").agg(pl.col("rush_share").mean()).to_dicts())
    runners = [r for r in by_name if r["archetype_name"].startswith("runner")]
    pockets = [r for r in by_name if r["archetype_name"].startswith("pocket")]
    assert runners and pockets
    assert min(r["rush_share"] for r in runners) > max(p["rush_share"] for p in pockets)


def test_two_centroids_in_one_quadrant_do_not_share_a_name():
    """A reused name would make two different things indistinguishable downstream."""
    df = pl.DataFrame({
        "archetype": [0, 1, 2],
        "z_rush_share": [1.0, 2.0, -1.0],
        "z_yards_per_completion": [1.0, 1.5, -1.0],
    })
    names = name_archetypes(df)
    slugs = [s for s, _ in names.values()]
    assert len(set(slugs)) == 3
    # The more extreme of the two keeps the plain name.
    assert names[1][0] == "runner_downfield"
    assert names[0][0].startswith("runner_downfield_")


# ------------------------------------------------------------------------- quality leakage


def test_leakage_is_near_zero_when_quality_is_independent_of_archetype():
    rng = np.random.default_rng(3)
    df = pl.DataFrame({
        "season": [2015] * 200,
        "archetype": [i % 4 for i in range(200)],
        "fitted_on": [1] * 200,
        "pass_epa_per_db": rng.normal(0.1, 0.1, 200),
    })
    assert quality_leakage(df, columns=("pass_epa_per_db",))[0]["eta_squared"] < 0.05


def test_leakage_is_near_one_when_the_clusters_are_the_quality_ordering():
    """The measurement has to be able to catch a leaderboard wearing a taxonomy's clothes."""
    df = pl.DataFrame({
        "season": [2015] * 200,
        "archetype": [i % 4 for i in range(200)],
        "fitted_on": [1] * 200,
        "pass_epa_per_db": [0.1 * (i % 4) for i in range(200)],
    })
    assert quality_leakage(df, columns=("pass_epa_per_db",))[0]["eta_squared"] > 0.95


# --------------------------------------------------------------------------- career rollup


def _assigned(rows):
    return pl.DataFrame(rows).with_columns(
        archetype_name=pl.col("archetype").cast(pl.String),
        archetype_label=pl.col("archetype").cast(pl.String),
    )


def test_a_career_takes_its_final_season_archetype():
    """Matching the project's `final_*` preference: a clipped career still has a real last year."""
    career = career_archetypes(_assigned([
        {"player": "QB", "season": 2012, "archetype": 0},
        {"player": "QB", "season": 2013, "archetype": 0},
        {"player": "QB", "season": 2014, "archetype": 3},
    ])).to_dicts()[0]
    assert career["archetype"] == 3
    assert career["archetype_first"] == 0
    assert career["archetype_modal"] == 0        # the majority is *not* what the career takes
    assert career["archetype_changed"] == 1
    assert career["archetype_stability"] == pytest.approx(1 / 3)


def test_a_quarterback_who_never_changed_style_is_marked_stable():
    career = career_archetypes(_assigned([
        {"player": "QB", "season": 2012, "archetype": 2},
        {"player": "QB", "season": 2013, "archetype": 2},
    ])).to_dicts()[0]
    assert career["archetype_changed"] == 0
    assert career["archetype_stability"] == pytest.approx(1.0)
