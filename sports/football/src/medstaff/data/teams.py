"""Canonical team codes.

The sample spans 2012-2025, which contains three relocations and a handful of code variants
that nflverse spells inconsistently across datasets (``injuries`` vs ``rosters_weekly`` vs
``schedules``). A team that silently splits into two codes looks like two franchises with half
the exposure each, which would quietly halve every per-team count in the project — so
canonicalisation is applied on **both** sides of every join.
"""

from __future__ import annotations

# Relocations are folded onto the *current* code so a franchise is one unit across the window.
# This is a deliberate choice: the analysis grades an organisation over a window, and the
# building changing city does not make it a different organisation. The stadium *surface*
# change that comes with a relocation is carried as a covariate instead.
TEAM_ALIASES = {
    "SD": "LAC",    # San Diego -> Los Angeles Chargers (2017)
    "SDG": "LAC",
    "STL": "LA",    # St. Louis -> Los Angeles Rams (2016)
    "LAR": "LA",    # nflverse uses LA for the Rams; some feeds emit LAR
    "RAM": "LA",
    "OAK": "LV",    # Oakland -> Las Vegas Raiders (2020)
    "RAI": "LV",
    "JAC": "JAX",   # spelling variant only
    "WAS": "WAS",
    "WSH": "WAS",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
}


def canonical_team(code: str | None) -> str | None:
    """Map a team code to its canonical form; ``None``/blank passes through as ``None``."""
    if code is None:
        return None
    text = str(code).strip().upper()
    if not text:
        return None
    return TEAM_ALIASES.get(text, text)


def canonicalize(frame, *columns: str):
    """Return ``frame`` with each named team column canonicalised (polars or pandas)."""
    present = [c for c in columns if c in frame.columns]
    if not present:
        return frame
    if hasattr(frame, "with_columns"):  # polars
        import polars as pl

        return frame.with_columns(
            [pl.col(c).map_elements(canonical_team, return_dtype=pl.String) for c in present]
        )
    out = frame.copy()
    for col in present:
        out[col] = out[col].map(canonical_team)
    return out


__all__ = ["TEAM_ALIASES", "canonical_team", "canonicalize"]
