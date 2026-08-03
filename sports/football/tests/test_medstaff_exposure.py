"""Stage-3 tests: exposure, confounders, and the risk set.

The failure mode this stage guards against is subtle: a covariate that is missing *unevenly*
looks like a finding. The crosswalk tests exist because the obvious ID sources drop almost every
offensive lineman, and position correlates with body part — which is exactly what stage 5
compares.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medstaff.exposure.context import (  # noqa: E402
    MAX_YEARS_EXP, build_risk_set, club_home_surface, game_context, id_crosswalk,
    injury_history, normalize_surface, player_attributes,
)


class TestNormalizeSurface:
    @pytest.mark.parametrize("raw,expected", [
        ("grass", "grass"),
        ("grass ", "grass"),          # trailing space — a distinct raw value, 93 rows
        ("Grass", "grass"),
        ("fieldturf", "turf"), ("matrixturf", "turf"), ("sportturf", "turf"),
        ("astroturf", "turf"), ("a_turf", "turf"),
        ("", None), (None, None), ("   ", None),
    ])
    def test_mapping(self, raw, expected):
        assert normalize_surface(raw) == expected

    def test_turf_brands_collapse_to_one_level(self):
        brands = {normalize_surface(b) for b in
                  ("fieldturf", "matrixturf", "sportturf", "astroturf", "a_turf")}
        assert brands == {"turf"}, "brand is not an injury-risk distinction"


class TestIdCrosswalk:
    """`players` is the only unbiased route — see the module docstring."""

    def _sources(self):
        players = pl.DataFrame({"gsis_id": ["G1", "G2", "G3"],
                                "pfr_id": ["P1", "P2", "P3"]})
        # the fantasy table carries skill players only
        ids = pl.DataFrame({"gsis_id": ["G1"], "pfr_id": ["P1"]})
        # weekly rosters have pfr_id null for essentially every lineman
        rosters = pl.DataFrame({"gsis_id": ["G1", "G3"], "pfr_id": ["P1", None]})
        return players, ids, rosters

    def test_players_covers_what_the_biased_sources_miss(self):
        players, ids, rosters = self._sources()
        assert id_crosswalk(players, ids, rosters).height == 3
        assert id_crosswalk(None, ids, rosters).height == 1, "the biased sources alone lose two"

    def test_one_row_per_pfr_id(self):
        players, ids, rosters = self._sources()
        xw = id_crosswalk(players, ids, rosters)
        assert xw["pfr_id"].n_unique() == xw.height

    def test_missing_every_source_raises(self):
        with pytest.raises(KeyError):
            id_crosswalk(pl.DataFrame({"gsis_id": ["G1"]}))


class TestGameContext:
    def _sched(self):
        return pl.DataFrame({
            "season": [2023, 2023], "week": [1, 2], "game_type": ["REG", "REG"],
            "home_team": ["KC", "KC"], "away_team": ["DET", "BUF"],
            "home_rest": [7, 4], "away_rest": [7, 10],
            "surface": ["grass ", "fieldturf"], "roof": ["outdoors", "dome"],
            "temp": [70, None], "div_game": [False, False],
        })

    def test_both_clubs_get_a_row_per_game(self):
        ctx = game_context(self._sched())
        assert ctx.height == 4
        assert set(ctx["team"].to_list()) == {"KC", "DET", "BUF"}

    def test_short_week_and_indoor_flags(self):
        ctx = game_context(self._sched()).sort(["week", "team"])
        wk2 = ctx.filter((pl.col("week") == 2) & (pl.col("team") == "KC")).row(0, named=True)
        assert wk2["short_week"] is True and wk2["rest_days"] == 4
        assert wk2["indoor"] is True
        wk1 = ctx.filter((pl.col("week") == 1) & (pl.col("team") == "KC")).row(0, named=True)
        assert wk1["short_week"] is False and wk1["indoor"] is False
        assert wk1["surface"] == "grass", "the trailing-space variant must normalise"

    def test_home_surface_is_a_club_property(self):
        home = club_home_surface(self._sched())
        assert home.height == 1 and home.row(0, named=True)["team"] == "KC"


class TestPlayerAttributes:
    def test_age_experience_and_bmi(self):
        rosters = pl.DataFrame({
            "season": [2023], "week": [1], "gsis_id": ["G1"], "position": ["WR"],
            "position_group": ["WR_TE"], "side": ["OFF"],
            "birth_date": [__import__("datetime").date(1998, 9, 1)],
            "years_exp": [3], "height": [72.0], "weight": [200],
        })
        row = player_attributes(rosters).row(0, named=True)
        assert row["age"] == pytest.approx(25.0, abs=0.05)
        assert row["bmi"] == pytest.approx(703 * 200 / 72**2, rel=1e-6)
        assert row["rookie"] is False

    def test_impossible_experience_is_clipped_not_dropped(self):
        """One 2023 roster row claims 29 years. The row is real exposure; the number is not."""
        rosters = pl.DataFrame({
            "season": [2023], "week": [1], "gsis_id": ["G1"], "position": ["T"],
            "position_group": ["OL"], "side": ["OFF"], "birth_date": [None],
            "years_exp": [29], "height": [78.0], "weight": [310],
        })
        out = player_attributes(rosters)
        assert out.height == 1
        assert out.row(0, named=True)["years_exp"] == MAX_YEARS_EXP


class TestInjuryHistory:
    def _injuries(self):
        rows = []
        for season, week in [(2019, 1), (2019, 2), (2021, 1)]:
            rows.append({"season": season, "week": week, "gsis_id": "G1",
                         "game_type": "REG", "body_part_group": "knee"})
        rows.append({"season": 2019, "week": 3, "gsis_id": "G1",
                     "game_type": "REG", "body_part_group": "illness"})
        return pl.DataFrame(rows)

    def test_history_is_strictly_prior(self):
        """A season must never contribute to its own covariate, or it leaks the outcome."""
        overall, _ = injury_history(self._injuries(), seasons=(2021,))
        row = overall.filter(pl.col("gsis_id") == "G1").row(0, named=True)
        assert row["prior_designated_weeks"] == 2, "2021's own week must not count"
        assert row["prior_injury_seasons"] == 1

    def test_reaches_back_past_the_comparable_window(self):
        """The report is comparable from 2009 even though episodes start in 2021."""
        overall, _ = injury_history(self._injuries(), seasons=(2021,))
        assert overall.height == 1, "2019 evidence is used for a 2021 row"

    def test_illness_is_not_injury_history(self):
        _, by_group = injury_history(self._injuries(), seasons=(2021,))
        assert "illness" not in by_group["body_part_group"].to_list()


class TestRiskSet:
    def _panel(self, states):
        return pl.DataFrame([
            {"season": 2023, "week": w, "gsis_id": "G1", "team": "KC",
             "week_state": s, "team_played": played, "on_practice_squad": ps}
            for w, s, played, ps in states
        ])

    def test_weeks_inside_an_episode_are_not_at_risk(self):
        """Counting them would turn one long absence into weeks of injury-free exposure."""
        panel = self._panel([
            (1, "available", True, False), (2, "impaired", True, False),
            (3, "impaired", True, False), (4, "available", True, False),
        ])
        episodes = pl.DataFrame([{
            "season": 2023, "gsis_id": "G1", "onset_week": 2, "end_week": 3, "return_week": 4,
        }])
        risk = build_risk_set(panel, episodes)
        assert sorted(risk["week"].to_list()) == [1, 4]

    def test_practice_squad_and_byes_excluded(self):
        panel = self._panel([
            (1, "available", True, False), (2, "bye", False, False),
            (3, "available", True, True),
        ])
        risk = build_risk_set(panel, pl.DataFrame())
        assert risk["week"].to_list() == [1]

    def test_weeks_since_return_counts_from_the_return(self):
        panel = self._panel([(w, "available", True, False) for w in range(1, 7)])
        episodes = pl.DataFrame([{
            "season": 2023, "gsis_id": "G1", "onset_week": 2, "end_week": 2, "return_week": 3,
        }])
        risk = build_risk_set(panel, episodes).sort("week")
        got = dict(zip(risk["week"].to_list(), risk["weeks_since_return"].to_list()))
        assert 2 not in got, "the injured week itself is not at risk"
        assert got[3] == 0 and got[4] == 1 and got[6] == 3
        assert got[1] is None, "no return has happened yet"
