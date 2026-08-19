"""Normalise team codes to a stable **franchise** identity across nflverse sources.

Two problems make a naive ``breakout_team != draft_team`` comparison wrong:

1. **Different abbreviation styles.** ``load_draft_picks`` carries Pro-Football-Reference codes
   (``GNB``, ``NWE``, ``SFO``, ``TAM``) while weekly stats carry nflverse codes (``GB``, ``NE``,
   ``SF``, ``TB``). Left uncorrected, Aaron Rodgers reads as having left Green Bay.

2. **Franchise relocations.** A team moving cities is not a QB changing situations — Drew Brees
   in 2004 was on the same Chargers team that drafted him, even though the franchise is now coded
   ``LAC``. Relocations are therefore collapsed onto one canonical code.

The relocation collapses below are only safe because this project's cohort starts at the 1999
entry season (``labels.cohort``): pre-1999 reuses of the same abbreviation by *genuinely different*
franchises — ``HOU`` for the Oilers before 1997 vs the Texans from 2002, ``STL`` for the Cardinals
before 1988 vs the Rams from 1995, ``CLE``/``BAL`` across the 1996 Browns move — fall outside it.
"""

from __future__ import annotations

# Pro-Football-Reference style -> nflverse style. Codes that already agree are omitted.
_PFR_TO_NFLVERSE = {
    "GNB": "GB", "KAN": "KC", "LVR": "LV", "NOR": "NO", "NWE": "NE",
    "SDG": "SD", "SFO": "SF", "TAM": "TB", "LAR": "LA", "RAM": "LA",
    "RAI": "OAK", "PHO": "ARI", "CRD": "ARI", "OTI": "TEN", "HTX": "HOU",
    "CLT": "IND", "RAV": "BAL", "NYG": "NYG", "NYJ": "NYJ",
}

# Same franchise, different city/era -> one canonical code. See module docstring for why this
# is safe given the 1999+ cohort.
_FRANCHISE = {
    "SD": "LAC", "SDG": "LAC",           # Chargers: San Diego -> Los Angeles (2017)
    "STL": "LA", "LAR": "LA", "RAM": "LA",  # Rams: St. Louis -> Los Angeles (2016)
    "OAK": "LV", "LVR": "LV", "RAI": "LV",  # Raiders: Oakland -> Las Vegas (2020)
    "JAC": "JAX",                        # Jaguars: nflverse used both spellings over time
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",  # occasional alt spellings
}


def canonical_team(code):
    """Return the stable franchise code for a team abbreviation from either source.

    ``None``/NaN passes through unchanged so callers can distinguish "no team" (undrafted, or a
    season with no team of record) from a real mismatch.
    """
    if code is None or code != code:  # NaN
        return code
    code = str(code).strip().upper()
    code = _PFR_TO_NFLVERSE.get(code, code)
    return _FRANCHISE.get(code, code)


def canonicalize(series):
    """Vectorised :func:`canonical_team` over a pandas Series."""
    return series.map(canonical_team)
