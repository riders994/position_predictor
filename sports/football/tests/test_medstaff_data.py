"""Stage-1 tests: the body-part normaliser, position groups, and team canonicalisation.

The normaliser carries most of the risk in this stage. It turns 170-odd club-typed free-text
strings into a taxonomy that every later stage groups by, so a mis-mapping here does not fail
loudly — it silently moves injuries between categories and changes every downstream count. The
cases below are all drawn from strings that actually appear in the 2021-2025 data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medstaff.data.positions import (  # noqa: E402
    GROUP_ORDER, position_group, side_of_ball,
)
from medstaff.data.taxonomy import (  # noqa: E402
    FOCAL_GROUPS, GROUPS, body_part_group, is_non_injury, is_severe, normalize,
)
from medstaff.data.teams import canonical_team  # noqa: E402


class TestNormalize:
    def test_strips_laterality_prefix(self):
        assert normalize("right Shoulder") == "shoulder"
        assert normalize("left Knee") == "knee"

    def test_strips_bracketed_clause(self):
        assert normalize("Ankle [Not Injury Related - Personal, Thursday Only]") == "ankle"

    def test_collapses_case_and_whitespace(self):
        assert normalize("  Lower   LEG  ") == "lower leg"

    def test_null_and_blank(self):
        assert normalize(None) == ""
        assert normalize("   ") == ""


class TestBodyPartGroup:
    @pytest.mark.parametrize("raw,expected", [
        ("Knee", "knee"),
        ("Ankle", "ankle"),
        ("Back", "back"),
        ("Hip", "hip"),
        ("Concussion", "concussion"),
        ("Hamstring", "soft_tissue_lower"),
        ("Quadricep", "soft_tissue_lower"),
        ("Shoulder", "shoulder"),
        ("Thumb", "arm_hand"),
        ("Ribs", "torso"),
        ("Rib", "torso"),
        ("Achilles", "lower_leg"),
        ("Calf", "lower_leg"),
        ("Toe", "foot"),
        ("Neck", "neck"),
        ("Eye", "head_face"),
        ("Illness", "illness"),
    ])
    def test_common_strings(self, raw, expected):
        assert body_part_group(raw) == expected

    def test_hip_flexor_is_soft_tissue_not_hip(self):
        """Longest-match-wins is what keeps this right, independent of rule ordering."""
        assert body_part_group("Hip Flexor") == "soft_tissue_lower"
        assert body_part_group("hip") == "hip"

    def test_case_and_laterality_variants_agree(self):
        for variant in ("Shoulder", "shoulder", "right Shoulder", "Left Shoulder"):
            assert body_part_group(variant) == "shoulder"

    def test_multi_part_takes_the_first_listed(self):
        """The primary injury is listed first, so earliest match wins — not rule order."""
        assert body_part_group("Foot/Wrist/Hip") == "foot"
        assert body_part_group("Wrist/Foot") == "arm_hand"

    def test_free_text_sentence(self):
        raw = ("Andrew has cleared concussion protocol and does not have a game status. ")
        assert body_part_group(raw) == "concussion"

    def test_illness_word_boundary_does_not_swallow_achilles(self):
        """A naive substring check for 'ill' would classify Achilles as an illness."""
        assert body_part_group("Achilles") == "lower_leg"

    def test_unknown_string_falls_through_to_other(self):
        assert body_part_group("Raym") == "other"

    def test_null_and_blank_return_none(self):
        assert body_part_group(None) is None
        assert body_part_group("  ") is None

    def test_grouping_is_total(self):
        """Every string lands somewhere in the taxonomy — nothing escapes."""
        samples = ["Knee", "Raym", "Jury Duty", "Foot/Wrist/Hip", "", None, "Lips", "Glute"]
        for raw in samples:
            group = body_part_group(raw)
            assert group is None or group in GROUPS

    def test_focal_groups_are_real_groups(self):
        assert set(FOCAL_GROUPS) <= set(GROUPS)


class TestNonInjury:
    @pytest.mark.parametrize("raw", [
        "Not injury related - resting player",
        "Not injury related - personal matter",
        "Not Injury Related",
        "Not Injury Related - Returning From Suspension",
        "Returning from Suspension",
        "Jury Duty",
        "Not injury related - travel",
        "Inactive",
    ])
    def test_excluded_not_grouped(self, raw):
        assert is_non_injury(raw)
        assert body_part_group(raw) == "non_injury"

    def test_body_part_with_non_injury_clause_is_excluded(self):
        """The ankle is real but the absence was personal — exclude, don't count it."""
        raw = "Ankle [Not Injury Related - Personal, Thursday Only]"
        assert body_part_group(raw) == "non_injury"

    def test_real_injuries_are_not_flagged(self):
        for raw in ("Knee", "Hamstring", "Concussion", "Wrist"):
            assert not is_non_injury(raw)


class TestSeverity:
    @pytest.mark.parametrize("raw", ["Torn ACL", "Achilles", "ruptured Achilles",
                                     "Fractured Fibula", "Broken Hand"])
    def test_severe_words_detected(self, raw):
        assert is_severe(raw)

    def test_bare_body_part_is_not_severe(self):
        """Most rows are a bare body part, so severity mostly comes from practice status."""
        assert not is_severe("Knee")
        assert not is_severe(None)


class TestPositionGroups:
    @pytest.mark.parametrize("position,expected", [
        ("QB", "QB"), ("RB", "RB"), ("FB", "RB"), ("WR", "WR_TE"), ("TE", "WR_TE"),
        ("T", "OL"), ("G", "OL"), ("C", "OL"), ("DE", "DL"), ("NT", "DL"),
        ("OLB", "LB"), ("CB", "DB"), ("FS", "DB"), ("K", "ST"),
    ])
    def test_mapping(self, position, expected):
        assert position_group(position) == expected

    def test_unknown_falls_to_st_not_dropped(self):
        """A dropped player is missing exposure, which reads downstream as an absence."""
        assert position_group("XYZ") == "ST"
        assert position_group(None) == "ST"

    def test_sides_partition_the_groups(self):
        sides = {g: side_of_ball(g) for g in GROUP_ORDER}
        assert sides["QB"] == "OFF" and sides["DL"] == "DEF" and sides["ST"] == "ST"
        assert set(sides.values()) == {"OFF", "DEF", "ST"}


class TestCanonicalTeam:
    @pytest.mark.parametrize("raw,expected", [
        ("SD", "LAC"), ("STL", "LA"), ("OAK", "LV"), ("JAC", "JAX"), ("LAR", "LA"),
        ("KC", "KC"), ("kc", "KC"), (" NE ", "NE"),
    ])
    def test_aliases_and_passthrough(self, raw, expected):
        assert canonical_team(raw) == expected

    def test_null_and_blank(self):
        assert canonical_team(None) is None
        assert canonical_team("") is None
