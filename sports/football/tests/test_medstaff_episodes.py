"""Stage-2 tests: episode construction and recurrence.

Episodes are the unit every later stage aggregates, and none of the failure modes here are
loud — a bad rule does not crash, it quietly changes every count in the project. So each edge
case from MEDSTAFF_PLAN §3 gets a named test built from a hand-written weekly panel, where the
right answer is obvious by inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medstaff.episodes.build import (  # noqa: E402
    CENSOR_OFF_ROSTER, CENSOR_PRACTICE_SQUAD, CENSOR_SEASON_END, CENSOR_TEAM_CHANGE,
    build_episodes, practice_severity, team_week_games,
)
from medstaff.episodes.recurrence import attach_recurrence  # noqa: E402


def week(w, state, *, team="KC", group=None, reserve=False, played=True, severity=None,
         gsis="P1", season=2023, severe=False):
    return {
        "season": season, "week": w, "gsis_id": gsis, "team": team, "position": "WR",
        "position_group": "WR_TE", "side": "OFF", "week_state": state,
        "report_body_group": group, "report_body_raw": group, "is_severe": severe,
        "severity": severity, "on_reserve": reserve, "team_played": played,
    }


def panel(rows):
    return pl.DataFrame(rows)


def episodes(rows):
    return build_episodes(panel(rows))


class TestBasicSpell:
    def test_simple_episode_opens_and_closes(self):
        eps = episodes([
            week(1, "available"),
            week(2, "impaired", group="knee", severity="DNP"),
            week(3, "impaired", group="knee", severity="LIMITED"),
            week(4, "available"),
        ])
        assert eps.height == 1
        e = eps.row(0, named=True)
        assert e["onset_week"] == 2 and e["return_week"] == 4 and e["end_week"] == 3
        assert e["games_missed"] == 2 and e["weeks_elapsed"] == 2
        assert e["censored"] is False and e["never_returned"] is False
        assert e["body_group"] == "knee"

    def test_available_week_is_the_only_terminator(self):
        """Report noise: unlisted but inactive in week 3 is not a return."""
        eps = episodes([
            week(1, "impaired", group="hamstring"),
            week(2, "impaired", group="hamstring"),
            week(3, "out_other"),
            week(4, "impaired", group="hamstring"),
            week(5, "available"),
        ])
        assert eps.height == 1, "an inactive gap must not split the spell"
        assert eps.row(0, named=True)["games_missed"] == 4

    def test_inactive_week_alone_never_starts_an_episode(self):
        """A healthy scratch is not an injury."""
        eps = episodes([week(1, "available"), week(2, "out_other"), week(3, "available")])
        assert eps.height == 0


class TestByeWeeks:
    def test_bye_continues_spell_and_is_not_a_game_missed(self):
        eps = episodes([
            week(1, "impaired", group="ankle"),
            week(2, "bye", played=False),
            week(3, "impaired", group="ankle"),
            week(4, "available"),
        ])
        assert eps.height == 1
        e = eps.row(0, named=True)
        assert e["n_byes"] == 1
        assert e["games_missed"] == 2, "the bye is elapsed time but not a missed game"
        assert e["weeks_elapsed"] == 3


class TestTeamChange:
    def test_trade_censors_onset_club_and_opens_a_new_spell(self):
        """Rehab credit belongs to whoever did the rehab."""
        eps = episodes([
            week(1, "impaired", group="knee", team="KC"),
            week(2, "impaired", group="knee", team="KC"),
            week(3, "impaired", group="knee", team="BUF"),
            week(4, "available", team="BUF"),
        ])
        assert eps.height == 2
        first, second = eps.row(0, named=True), eps.row(1, named=True)
        assert first["team"] == "KC"
        assert first["censored"] is True and first["censor_reason"] == CENSOR_TEAM_CHANGE
        assert first["return_week"] is None
        assert second["team"] == "BUF" and second["return_week"] == 4


class TestRosterGap:
    """Gaps are resolved by ``build_panel`` now, which labels each missing week bye vs
    off-roster; ``build_episodes`` just reads the state. See TestFillInteriorWeeks."""

    def test_off_roster_weeks_censor_the_spell(self):
        """Released then re-signed: the club was not rehabbing him in between."""
        eps = episodes([
            week(1, "impaired", group="back"),
            week(2, "impaired", group="back"),
            week(3, "off_roster", played=True),
            week(4, "off_roster", played=True),
            week(5, "impaired", group="back"),
            week(6, "available"),
        ])
        assert eps.height == 2
        assert eps.row(0, named=True)["censor_reason"] == CENSOR_OFF_ROSTER
        assert eps.row(0, named=True)["end_week"] == 2

    def test_a_bye_inside_a_long_spell_does_not_split_it(self):
        """The bug: a bye read as off-roster halved every long absence at the bye week."""
        rows = ([week(w, "impaired", group="knee", reserve=True) for w in range(1, 14)]
                + [week(14, "bye", played=False)]
                + [week(w, "impaired", group="knee", reserve=True) for w in range(15, 19)])
        eps = episodes(rows)
        assert eps.height == 1, "one season-long spell, not two"
        e = eps.row(0, named=True)
        assert e["weeks_elapsed"] == 18 and e["games_missed"] == 17
        assert e["n_byes"] == 1
        assert e["censor_reason"] == CENSOR_SEASON_END


class TestPracticeSquad:
    def test_practice_squad_move_censors(self):
        eps = episodes([
            week(1, "impaired", group="hip"),
            week(2, "practice_squad"),
            week(3, "available"),
        ])
        assert eps.height == 1
        assert eps.row(0, named=True)["censor_reason"] == CENSOR_PRACTICE_SQUAD

    def test_practice_squad_week_never_starts_an_episode(self):
        eps = episodes([week(1, "practice_squad"), week(2, "practice_squad")])
        assert eps.height == 0


class TestSeasonBoundary:
    def test_unresolved_spell_censors_at_season_end(self):
        eps = episodes([
            week(16, "available"),
            week(17, "impaired", group="knee", reserve=True),
        ])
        assert eps.height == 1
        e = eps.row(0, named=True)
        assert e["censor_reason"] == CENSOR_SEASON_END
        assert e["never_returned"] is True and e["return_week"] is None

    def test_episode_never_spans_seasons(self):
        rows = [
            week(17, "impaired", group="knee", season=2022),
            week(1, "impaired", group="knee", season=2023),
            week(2, "available", season=2023),
        ]
        eps = build_episodes(panel(rows))
        assert eps.height == 2
        assert set(eps["season"].to_list()) == {2022, 2023}


class TestReserveAndSeverity:
    def test_reserve_weeks_counted_and_flagged(self):
        eps = episodes([
            week(1, "impaired", group="knee", reserve=True),
            week(2, "impaired", group="knee", reserve=True),
            week(3, "available"),
        ])
        e = eps.row(0, named=True)
        assert e["used_reserve"] is True and e["weeks_on_reserve"] == 2

    def test_worst_severity_wins_not_onset(self):
        """Clubs list a knock before the diagnosis firms up."""
        eps = episodes([
            week(1, "impaired", group="knee", severity="FULL"),
            week(2, "impaired", group="knee", severity="DNP"),
            week(3, "available"),
        ])
        e = eps.row(0, named=True)
        assert e["severity_onset"] == "FULL"
        assert e["severity_worst"] == "DNP"

    def test_severe_flag_is_sticky_across_the_spell(self):
        eps = episodes([
            week(1, "impaired", group="knee", severe=False),
            week(2, "impaired", group="knee", severe=True),
            week(3, "available"),
        ])
        assert eps.row(0, named=True)["is_severe"] is True


class TestGroupChange:
    def test_primary_group_flip_stays_one_spell_and_is_flagged(self):
        """The report carries one primary part and clubs flip it; splitting invents episodes."""
        eps = episodes([
            week(1, "impaired", group="knee"),
            week(2, "impaired", group="ankle"),
            week(3, "available"),
        ])
        assert eps.height == 1
        e = eps.row(0, named=True)
        assert e["body_group"] == "knee", "the spell keeps the group it opened with"
        assert e["group_changed"] is True


class TestPracticeSeverity:
    @pytest.mark.parametrize("raw,expected", [
        ("Did Not Participate In Practice", "DNP"),
        ("Limited Participation in Practice", "LIMITED"),
        ("Full Participation in Practice", "FULL"),
        ("", None), (None, None), ("   ", None),
    ])
    def test_mapping(self, raw, expected):
        assert practice_severity(raw) == expected


class TestTeamWeekGames:
    def test_both_sides_of_each_game_are_listed(self):
        sched = pl.DataFrame({
            "season": [2023, 2023], "week": [1, 1], "game_type": ["REG", "POST"],
            "home_team": ["KC", "SF"], "away_team": ["DET", "DAL"],
        })
        tw = team_week_games(sched)
        assert set(tw["team"].to_list()) == {"KC", "DET"}, "postseason excluded"


class TestRecurrence:
    def _panel_for(self, weeks_played, gsis="P1", season=2023):
        return pl.DataFrame([
            {"season": season, "gsis_id": gsis, "week": w, "team_played": True}
            for w in weeks_played
        ])

    def test_recurrence_within_horizon_counts_available_games_not_weeks(self):
        eps = episodes([
            week(1, "impaired", group="hamstring"),
            week(2, "available"),
            week(3, "available"),
            week(4, "impaired", group="hamstring"),
            week(5, "available"),
        ])
        out = attach_recurrence(eps, self._panel_for(range(1, 18)))
        first = out.row(0, named=True)
        assert first["at_risk"] is True
        assert first["recur_same_season"] is True
        assert first["recur_games_to"] == 2, "weeks 2 and 3 were the exposure"
        assert first["recur_k3"] is True and first["recur_k6"] is True

    def test_recurrence_outside_horizon_is_not_flagged(self):
        eps = episodes(
            [week(1, "impaired", group="hamstring")]
            + [week(w, "available") for w in range(2, 12)]
            + [week(12, "impaired", group="hamstring"), week(13, "available")]
        )
        out = attach_recurrence(eps, self._panel_for(range(1, 18)))
        first = out.row(0, named=True)
        assert first["recur_games_to"] == 10
        assert first["recur_k3"] is False and first["recur_k6"] is False
        assert first["recur_k12"] is True

    def test_a_different_body_group_is_not_a_recurrence(self):
        eps = episodes([
            week(1, "impaired", group="hamstring"),
            week(2, "available"),
            week(3, "impaired", group="shoulder"),
            week(4, "available"),
        ])
        out = attach_recurrence(eps, self._panel_for(range(1, 18)))
        assert out.row(0, named=True)["recur_same_season"] is False

    def test_bye_pauses_the_clock(self):
        """Exposure is games available, so a bye must not consume the window."""
        eps = episodes([
            week(1, "impaired", group="calf"),
            week(2, "available"),
            week(3, "available", played=False),
            week(4, "impaired", group="calf"),
            week(5, "available"),
        ])
        # week 3 was a bye, so it is absent from the played panel
        out = attach_recurrence(eps, self._panel_for([1, 2, 4, 5, 6, 7]))
        assert out.row(0, named=True)["recur_games_to"] == 1

    def test_unresolved_episode_is_not_at_risk(self):
        """An injury that never resolved cannot recur, and must not count as a clean outcome."""
        eps = episodes([week(17, "impaired", group="knee", reserve=True)])
        out = attach_recurrence(eps, self._panel_for([17]))
        e = out.row(0, named=True)
        assert e["at_risk"] is False
        assert e["recur_k6"] is None and e["recur_same_season"] is None

    def test_next_season_recurrence_flagged_separately(self):
        rows = [
            week(1, "impaired", group="groin", season=2023),
            week(2, "available", season=2023),
            week(1, "impaired", group="groin", season=2024),
            week(2, "available", season=2024),
        ]
        eps = build_episodes(panel(rows))
        played = pl.concat([
            self._panel_for(range(1, 18), season=2023),
            self._panel_for(range(1, 18), season=2024),
        ])
        out = attach_recurrence(eps, played).sort("season")
        first = out.row(0, named=True)
        assert first["recur_next_season"] is True
        assert first["recur_same_season"] is False, "cross-season is not a same-season recurrence"


class TestNonBodyGroups:
    def test_reserve_week_does_not_adopt_an_illness_label(self):
        """Reserve makes a week impaired regardless of the report, so a stray
        'Illness' row must not become the spell's body part."""
        eps = episodes([
            week(1, "impaired", group="illness", reserve=True),
            week(2, "impaired", group="illness", reserve=True),
            week(3, "available"),
        ])
        assert eps.height == 1
        assert eps.row(0, named=True)["body_group"] == "unknown"

    def test_a_real_part_seen_later_still_labels_the_spell(self):
        eps = episodes([
            week(1, "impaired", group=None, reserve=True),
            week(2, "impaired", group="knee"),
            week(3, "available"),
        ])
        assert eps.row(0, named=True)["body_group"] == "knee"


class TestFillInteriorWeeks:
    """`rosters_weekly` omits bye weeks; filling them is what keeps long absences whole."""

    def _schedules(self, *, bye_week=14, team="KC", season=2023, weeks=19):
        rows = []
        for w in range(1, weeks):
            if w == bye_week:
                continue
            rows.append({"season": season, "week": w, "game_type": "REG",
                         "home_team": team, "away_team": "OPP"})
        return pl.DataFrame(rows)

    def _panel(self, weeks, *, team="KC", season=2023):
        return pl.DataFrame([
            {"season": season, "week": w, "gsis_id": "P1", "team": team, "game_type": "REG",
             "on_practice_squad": False, "on_reserve": True, "designated": False,
             "active": False, "team_played": True}
            for w in weeks
        ])

    def test_missing_bye_week_is_filled_as_bye(self):
        from medstaff.episodes.build import _fill_interior_weeks

        panel = self._panel([w for w in range(1, 19) if w != 14])
        out = _fill_interior_weeks(panel, self._schedules()).sort("week")
        assert out.height == 18, "the missing week is materialised"
        wk14 = out.filter(pl.col("week") == 14).row(0, named=True)
        assert wk14["week_state"] == "bye"
        assert wk14["team"] == "KC", "filled week inherits the club it was on"

    def test_missing_week_the_club_played_is_off_roster(self):
        from medstaff.episodes.build import _fill_interior_weeks

        # week 9 missing, and the club DID play it -> a genuine absence, not a bye
        panel = self._panel([w for w in range(1, 19) if w not in (9, 14)])
        out = _fill_interior_weeks(panel, self._schedules()).sort("week")
        states = dict(zip(out["week"].to_list(), out["week_state"].to_list()))
        assert states[9] == "off_roster"
        assert states[14] == "bye"

    def test_real_rows_keep_a_null_state_for_the_normal_rules(self):
        from medstaff.episodes.build import _fill_interior_weeks

        panel = self._panel([w for w in range(1, 19) if w != 14])
        out = _fill_interior_weeks(panel, self._schedules())
        real = out.filter(pl.col("week") != 14)
        assert real["week_state"].null_count() == real.height

    def test_no_trailing_or_leading_weeks_are_invented(self):
        """Only *interior* gaps are filled — a player who joined in week 10 has no weeks 1-9."""
        from medstaff.episodes.build import _fill_interior_weeks

        panel = self._panel([10, 11, 12, 13, 15, 16])
        out = _fill_interior_weeks(panel, self._schedules())
        assert out["week"].min() == 10 and out["week"].max() == 16


class TestSanityChecks:
    """Guards written *after* the bye-week defect survived four stages undetected.

    Each asserts the check fires on the shape the broken pipeline actually produced — a guard
    that has never been shown to catch anything is decoration.
    """

    def _load(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "medstaff_episodes", ROOT / "scripts" / "medstaff_episodes.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _panel(self, weeks):
        return pl.DataFrame([{"season": 2023, "gsis_id": "P1", "week": w} for w in weeks])

    def _episodes(self, *, max_games, off_roster_share=0.0, n=100):
        rows = []
        for i in range(n):
            rows.append({
                "games_missed": max_games if i == 0 else 2,
                "censor_reason": "off_roster" if i < off_roster_share * n else None,
            })
        return pl.DataFrame(rows)

    def test_truncated_absences_are_caught(self):
        """The actual symptom: longest spell in five seasons was 13 games."""
        mod = self._load()
        out = mod.sanity_checks(self._panel(range(1, 19)),
                                self._episodes(max_games=13))
        assert any("truncated" in i for i in out["issues"])

    def test_healthy_data_is_clear(self):
        mod = self._load()
        out = mod.sanity_checks(self._panel(range(1, 19)),
                                self._episodes(max_games=17))
        assert out["issues"] == []

    def test_implausible_off_roster_rate_is_caught(self):
        """21% of spells ended as 'player left the roster' and nobody noticed."""
        mod = self._load()
        out = mod.sanity_checks(self._panel(range(1, 19)),
                                self._episodes(max_games=17, off_roster_share=0.21))
        assert any("off_roster" in i for i in out["issues"])

    def test_interior_panel_holes_are_caught(self):
        """The root cause itself — a missing week the builder cannot interpret."""
        mod = self._load()
        out = mod.sanity_checks(self._panel([w for w in range(1, 19) if w != 14]),
                                self._episodes(max_games=17))
        assert out["player_seasons_with_holes"] == 1
        assert any("interior missing weeks" in i for i in out["issues"])
