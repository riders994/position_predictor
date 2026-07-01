"""Phase-3 age source — fetch player **birthdates** from nba_api and derive per-season age.

Model B swaps the years-of-experience proxy for **true age**. nba_api has reliable birthdates but keys on
its own player ids, so we map our ESPN ``athlete_id`` -> nba_api id by normalized name (99%+ unambiguous),
fetch ``BIRTHDATE`` once per player (cached), and compute age at each season. ESPN rosters' own
``date_of_birth`` (latest-season only) is used to validate and to backfill the few name-ambiguous or
unmatched players. Age is taken at **Feb 1 of the season's ending year** (mid-season), matching our
"season = ending year" convention.
"""
from __future__ import annotations

import re
import time
import unicodedata
from datetime import date

from ..utils.io import DATA_INTERIM, DATA_RAW, ensure_dir, read_parquet, write_parquet

BIO_CACHE = DATA_INTERIM / "nba_player_bio.parquet"          # nba_id -> birthdate (network cache)
AGES_PATH = DATA_INTERIM / "nba_player_ages.parquet"         # athlete_id, season -> age (derived)


def _norm(name):
    """Fold accents, drop generational suffixes, keep letters — for cross-source name matching."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s.lower())
    return re.sub(r"[^a-z ]", "", s).strip()


def _nba_name_index():
    """Normalized name -> list of nba_api player ids (list flags ambiguous duplicate names)."""
    from nba_api.stats.static import players

    lut = {}
    for p in players.get_players():
        lut.setdefault(_norm(p["full_name"]), []).append(p["id"])
    return lut


def map_players(need):
    """Map unique (athlete_id, player_name) to an **unambiguous** nba_api id by normalized name.

    Returns ``(mapped_df[athlete_id, player_name, nba_id], unmatched, ambiguous)`` name lists.
    """
    lut = _nba_name_index()
    rows, unmatched, ambiguous = [], [], []
    for aid, name in need[["athlete_id", "player_name"]].drop_duplicates().itertuples(index=False):
        ids = lut.get(_norm(name), [])
        if len(ids) == 1:
            rows.append({"athlete_id": aid, "player_name": name, "nba_id": ids[0]})
        elif not ids:
            unmatched.append(name)
        else:
            ambiguous.append(name)
    import pandas as pd
    return pd.DataFrame(rows), unmatched, ambiguous


def fetch_birthdates(nba_ids, *, sleep=0.3, retries=2, refresh=False):
    """Fetch ``BIRTHDATE`` per nba_api id, caching to ``BIO_CACHE`` (only fetches ids not already cached).

    Resilient: short sleep between calls + retry on transient errors; failures are skipped (left absent
    so a later run can retry them). Returns the full cached ``DataFrame[nba_id, birthdate]``.
    """
    import pandas as pd
    from nba_api.stats.endpoints import commonplayerinfo

    have = read_parquet(BIO_CACHE) if (BIO_CACHE.exists() and not refresh) else pd.DataFrame(
        columns=["nba_id", "birthdate"])
    cached = set(have["nba_id"].tolist())
    todo = [i for i in dict.fromkeys(nba_ids) if i not in cached]
    new = []
    for k, nid in enumerate(todo, 1):
        for attempt in range(retries + 1):
            try:
                df = commonplayerinfo.CommonPlayerInfo(player_id=nid, timeout=30).get_data_frames()[0]
                new.append({"nba_id": nid, "birthdate": str(df["BIRTHDATE"].iloc[0])[:10]})
                break
            except Exception:
                if attempt == retries:
                    break
                time.sleep(sleep * (attempt + 2))
        time.sleep(sleep)
        if k % 100 == 0:
            print(f"[bio] fetched {k}/{len(todo)} new birthdates")
    out = pd.concat([have, pd.DataFrame(new)], ignore_index=True) if new else have
    if new:
        ensure_dir(DATA_INTERIM)
        write_parquet(out, BIO_CACHE)
    return out


def _espn_dob():
    """ESPN rosters' own date_of_birth (latest-season only) -> {athlete_id: 'YYYY-MM-DD'} fallback."""
    r = read_parquet(DATA_RAW / "rosters.parquet")
    import pandas as pd
    r = r.dropna(subset=["date_of_birth"]).copy()
    r["athlete_id"] = pd.to_numeric(r["athlete_id"], errors="coerce").astype("Int64")
    r = r.drop_duplicates("athlete_id")
    return {int(a): str(d)[:10] for a, d in zip(r["athlete_id"], r["date_of_birth"]) if pd.notna(a)}


def _age_on(birthdate, season):
    """Age in years at Feb 1 of the season's ending year (mid-season)."""
    try:
        b = date.fromisoformat(str(birthdate)[:10])
    except (ValueError, TypeError):
        return None
    ref = date(int(season), 2, 1)
    return round((ref - b).days / 365.25, 1)


def build_player_ages(membership=None, *, write=True, **fetch_kw):
    """Derive per-(athlete_id, season) age from nba_api birthdates (+ ESPN fallback). Returns the table."""
    import numpy as np
    import pandas as pd

    if membership is None:
        membership = read_parquet(DATA_INTERIM.parent / "processed" / "nba_archetype_membership.parquet")
    need = membership[["athlete_id", "player_name"]].drop_duplicates()
    mapped, unmatched, ambiguous = map_players(need)
    bio = fetch_birthdates(mapped["nba_id"].tolist(), **fetch_kw)

    dob = mapped.merge(bio, on="nba_id", how="left").set_index("athlete_id")["birthdate"].to_dict()
    espn = _espn_dob()
    dob = {int(a): (dob.get(a) or espn.get(int(a))) for a in membership["athlete_id"].unique()}
    # backfill unmatched/ambiguous purely from ESPN where possible
    for a in membership["athlete_id"].unique():
        if not dob.get(int(a)):
            dob[int(a)] = espn.get(int(a))

    m = membership[["athlete_id", "season"]].drop_duplicates().copy()
    m["birthdate"] = m["athlete_id"].map(lambda a: dob.get(int(a)))
    m["age"] = [
        _age_on(b, s) for b, s in zip(m["birthdate"], m["season"])]
    m["age"] = m["age"].astype("float64")
    coverage = float(m["age"].notna().mean())
    print(f"[bio] age coverage {coverage:.1%} · matched {len(mapped)} nba_api · "
          f"{len(unmatched)} unmatched, {len(ambiguous)} ambiguous (ESPN-backfilled where possible)")
    m = m[["athlete_id", "season", "age"]].replace({np.nan: None})
    m["age"] = pd.to_numeric(m["age"], errors="coerce")
    if write:
        ensure_dir(DATA_INTERIM)
        write_parquet(m, AGES_PATH)
    return m, {"coverage": coverage, "unmatched": unmatched, "ambiguous": ambiguous,
               "n_mapped": len(mapped)}
