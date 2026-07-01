"""Phase 2 — fantasy-roster **archetype composition** (and the join to success labels).

For each fantasy team we pull its NBA roster from Fantrax, name-match the players to the Phase-1
archetype membership for the league's season, and summarize the roster as an **archetype
composition**: soft shares (mean of the rostered players' membership vectors) + hard counts per
archetype. Joining that to the ``max_pf`` success labels (graded **M2 > M3 > M1**) gives the
Phase-2 modeling table: *which archetype mixes build a winning 9-cat roster*.

Roster source = the latest-period snapshot per team (end-of-season for a finished league); its
``period_date`` also fixes the season for the archetype join.
"""
from __future__ import annotations

import re
import unicodedata

from ..utils.io import DATA_PROCESSED, read_parquet, write_parquet

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def _norm(name) -> str:
    """Normalise a player name for cross-dataset matching.

    Fantrax and ESPN disagree on **diacritics** ("fonts") — e.g. Jokić / Dončić / Šengün / Jović —
    and on punctuation/suffixes. So: NFKD-decompose and strip combining marks (ASCII-fold the
    accents), lowercase, drop ``.'/-`` punctuation and generational suffixes, collapse whitespace.
    Folding *both* sides means it matches whichever dataset keeps the accent.
    """
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))   # strip accents/diacritics
    s = s.lower().replace(".", " ").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def season_from_date(d) -> int:
    """NBA season-ENDING year for a date in the season (Oct–Dec -> year+1, else year)."""
    return d.year + 1 if d.month >= 10 else d.year


# ---------------------------------------------------------------- roster fetch (network)

def _league_rosters(league_id, *, retries=3, backoff=30):
    """Pull every team's latest-period roster for one league (retry/backoff for Fantrax throttles)."""
    import time

    from fantraxapi import FantraxAPI
    rows = []
    for attempt in range(retries):
        try:
            api = FantraxAPI(league_id)
            rows = []
            for team in api.teams:
                roster = api.team_roster(team.id)
                season = season_from_date(roster.period_date)
                for rr in roster.rows:
                    if rr.player is None:
                        continue
                    rows.append({"league_id": league_id, "season": season,
                                 "team_id": team.id, "team_name": team.name,
                                 "fantrax_player_id": rr.player.id, "player_name": rr.player.name,
                                 "fantrax_pos": getattr(rr.player, "pos_short_name", None)})
            return rows
        except Exception:  # noqa: BLE001 - retry transient Fantrax throttles
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    return rows


def fetch_rosters(config, *, write: bool = True):
    """Pull all configured leagues' rosters into one tidy table (one row per rostered player)."""
    import pandas as pd
    league_ids = config.get("fantasy.league_ids", []) or []
    rows = []
    for lid in league_ids:
        rows.extend(_league_rosters(lid))
    df = pd.DataFrame(rows)
    if write and not df.empty:
        write_parquet(df, DATA_PROCESSED / "fantrax_rosters.parquet")
    return df


# ---------------------------------------------------------------- composition (pure)

def team_composition(rosters, membership, aliases=None):
    """Per fantasy team: archetype soft-shares + hard counts, name-matched within the league season.

    ``aliases`` is an optional ``{fantrax_name: espn_name}`` escape hatch for residual mismatches
    that survive normalization (nicknames, "Cam"/"Cameron", reordered names). Returns
    ``(composition_df, match_rate, unmatched_df)``; ``unmatched_df`` lists the roster players that
    didn't resolve, so systematic name problems can be found and fixed (alias or upstream).
    """
    import pandas as pd

    pcols = [c for c in membership.columns if c.startswith("p") and c[1:].isdigit()]
    arch_name = (membership.drop_duplicates("arch").set_index("arch")["arch_name"].to_dict())
    mem = membership.copy()
    mem["_key"] = list(zip(mem["season"], mem["player_name"].map(_norm)))
    lut = mem.drop_duplicates("_key").set_index("_key")

    remap = {_norm(k): _norm(v) for k, v in (aliases or {}).items()}
    r = rosters.copy()
    r["_nn"] = r["player_name"].map(_norm).replace(remap)
    r["_key"] = list(zip(r["season"], r["_nn"]))
    r["_matched"] = r["_key"].isin(lut.index)
    match_rate = float(r["_matched"].mean()) if len(r) else 0.0
    unmatched = (r.loc[~r["_matched"], ["league_id", "season", "team_name", "player_name"]]
                 .drop_duplicates().reset_index(drop=True))

    out = []
    for (lid, tid), g in r.groupby(["league_id", "team_id"]):
        gm = g[g["_matched"]]
        rec = {"league_id": lid, "team_id": tid, "team_name": g["team_name"].iloc[0],
               "season": int(g["season"].iloc[0]),
               "n_roster": len(g), "n_matched": len(gm),
               "match_rate": round(len(gm) / len(g), 3) if len(g) else 0.0}
        if len(gm):
            probs = lut.loc[gm["_key"]][pcols]
            shares = probs.mean()                                  # soft archetype shares
            hard = probs.values.argmax(axis=1)                    # hard label per matched player
            for j, pc in enumerate(pcols):
                nm = arch_name.get(j, f"A{j}")
                rec[f"comp_{nm}"] = round(float(shares[pc]), 3)
                rec[f"n_{nm}"] = int((hard == j).sum())
        out.append(rec)
    return pd.DataFrame(out), match_rate, unmatched


def build_phase2_table(config, *, write: bool = True):
    """Join roster composition (cached) to the max_pf success labels -> the Phase-2 modeling table.

    Returns ``(table, match_rate, unmatched_df)``.
    """
    rosters = read_parquet(DATA_PROCESSED / "fantrax_rosters.parquet")
    membership = read_parquet(DATA_PROCESSED / "nba_archetype_membership.parquet")
    success = read_parquet(DATA_PROCESSED / "fantrax_success.parquet")

    aliases = config.get("fantasy.name_aliases", {}) or {}
    comp, match_rate, unmatched = team_composition(rosters, membership, aliases=aliases)
    keep = ["league_id", "team_id", "name", "actual_pf", "m1_pf", "m2_pf", "m3_pf",
            "primary_pf", "lineup_gap"]
    tbl = comp.merge(success[[c for c in keep if c in success.columns]],
                     on=["league_id", "team_id"], how="left")
    if write and not tbl.empty:
        write_parquet(tbl, DATA_PROCESSED / "phase2_composition.parquet")
    return tbl, match_rate, unmatched
