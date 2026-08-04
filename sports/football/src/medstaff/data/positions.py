"""Position -> position-group mapping.

Eight groups, chosen so that the offense/defense split used by the signature analysis
(plan §4.8) cuts cleanly across them. Raw nflverse ``position`` spellings vary by dataset and
by era, so the map is deliberately permissive and everything unrecognised falls to ``ST``
(specialists and the long tail) rather than being dropped — a dropped player is missing
exposure, which reads downstream as an absence.
"""

from __future__ import annotations

POSITION_GROUPS: dict[str, str] = {
    # offense
    "QB": "QB",
    "RB": "RB", "FB": "RB", "HB": "RB",
    "WR": "WR_TE", "TE": "WR_TE",
    "T": "OL", "OT": "OL", "G": "OL", "OG": "OL", "C": "OL", "OL": "OL", "LS": "OL",
    # defense
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
    # specialists
    "K": "ST", "P": "ST", "PK": "ST",
}

OFFENSE_GROUPS = ("QB", "RB", "WR_TE", "OL")
DEFENSE_GROUPS = ("DL", "LB", "DB")
GROUP_ORDER = (*OFFENSE_GROUPS, *DEFENSE_GROUPS, "ST")


def position_group(position: str | None) -> str:
    """Map a raw position code to one of :data:`GROUP_ORDER`; unknown -> ``"ST"``."""
    if position is None:
        return "ST"
    return POSITION_GROUPS.get(str(position).strip().upper(), "ST")


def side_of_ball(group: str | None) -> str:
    """``"OFF"`` / ``"DEF"`` / ``"ST"`` — the primary split for the signature analysis."""
    if group in OFFENSE_GROUPS:
        return "OFF"
    if group in DEFENSE_GROUPS:
        return "DEF"
    return "ST"


def attach_position_group(frame, *, source: str = "position"):
    """Attach ``position_group`` and ``side`` to a polars frame."""
    import polars as pl

    return frame.with_columns(
        pl.col(source).map_elements(position_group, return_dtype=pl.String)
        .alias("position_group"),
    ).with_columns(
        pl.col("position_group").map_elements(side_of_ball, return_dtype=pl.String).alias("side")
    )


__all__ = [
    "DEFENSE_GROUPS", "GROUP_ORDER", "OFFENSE_GROUPS", "POSITION_GROUPS",
    "attach_position_group", "position_group", "side_of_ball",
]
