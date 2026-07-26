"""High-school layer: ESPN recruiting profiles for QB prospects.

This is the project's substitute for high-school box scores, which do not exist in any free,
reproducible bulk source. What a recruiting profile *does* give is arguably better structured
anyway: a national scouting grade, ranks at three geographic scopes, camp-measured athleticism,
the high school itself, and the recruitment's own shape (how many schools, how long it took).

Two things here are worth more than the rest:

**ESPN types QB recruits as ``QB-PP`` (pocket passer) or ``QB-DT`` (dual threat).** That is a
high-school archetype label assigned by scouts at the time, with no hindsight — exactly the kind
of feature this project is meant to test. Some future NFL QBs are also listed as ``ATH``, which
is its own signal: the player was athletic enough that his position was an open question.

**The ``schools`` block records the recruitment, not just its outcome** — every visit with a date
and status. Commitment timing and churn are behavioural signals available before a snap was
played.

Access
------
The ESPN core API needs no key, and the list endpoint embeds the complete recruit record
(grade, attributes, high school, schools) in each item — so a class costs a handful of paged
requests rather than one request per prospect.

Caveat: **camp measurables are unreliable.** ESPN's ``threeConeDrill`` in particular carries
obvious garbage (values like 99.0 where a real time is ~7 seconds). :func:`parse_recruit` keeps
raw values and :func:`clean_measurables` nulls out-of-range ones rather than dropping the column,
so the missingness is explicit and modellable.
"""

from __future__ import annotations

import time

BASE = ("http://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
        "/recruiting")

# ESPN position codes that mean "this prospect might become an NFL quarterback". ATH is included
# deliberately: athletes recruited without a settled position are a live source of late-blooming
# QBs, and excluding them would bias the cohort toward players scouts had already made up their
# minds about.
QB_POSITIONS = ("QB", "QB-PP", "QB-DT", "ATH")

# Plausible ranges for camp-measured athleticism. Anything outside is a data-entry artifact, not
# an extreme athlete — a 99-second three-cone is not a slow player.
MEASURABLE_RANGES = {
    "forty_yd": (4.2, 6.0),
    "three_cone": (6.4, 8.5),
    "shuttle_20yd": (3.8, 5.2),
    "vertical_jump": (18.0, 46.0),
}

# ESPN attribute name -> our column name.
_ATTRS = {
    "rank": "rank_national",
    "positionRank": "rank_position",
    "stateRank": "rank_state",
    "regionRank": "rank_region",
    "fortyYrdDash": "forty_yd",
    "threeConeDrill": "three_cone",
    "twentyYrdShuttle": "shuttle_20yd",
    "verticalJump": "vertical_jump",
}


def parse_recruit(item: dict) -> dict:
    """Flatten one ESPN recruiting item into a single record.

    Keeps raw measurable values; see :func:`clean_measurables` for range validation.
    """
    ath = item.get("athlete") or {}
    hs = ath.get("highSchool") or {}
    hs_addr = hs.get("address") or {}
    home = ath.get("hometown") or {}
    pos = (ath.get("position") or {}).get("abbreviation")

    rec = {
        "espn_recruit_id": ath.get("id"),
        "espn_athlete_id": ath.get("alternateId"),
        "recruit_name": ath.get("fullName") or ath.get("displayName"),
        "first_name": ath.get("firstName"),
        "last_name": ath.get("lastName"),
        "recruit_class": item.get("recruitingClass"),
        "recruit_position": pos,
        # The scouts' own high-school archetype call, before any college or NFL evidence.
        "dual_threat": None if pos is None else (1 if pos == "QB-DT" else 0),
        "recruited_as_athlete": None if pos is None else int(pos == "ATH"),
        "espn_grade": item.get("grade"),
        "recruit_status": (item.get("status") or {}).get("description"),
        "hs_height_in": ath.get("height"),
        "hs_weight_lb": ath.get("weight"),
        "hs_name": hs.get("properName") or hs.get("name"),
        "hs_city": hs_addr.get("city") or home.get("city"),
        "hs_state": hs_addr.get("stateAbbreviation") or home.get("stateAbbreviation"),
    }
    for attr in item.get("attributes") or []:
        col = _ATTRS.get(attr.get("name"))
        if col:
            rec[col] = attr.get("value")

    # Recruitment shape: how many programs were seriously involved, and when it ended.
    schools = item.get("schools") or []
    rec["n_schools_involved"] = len(schools)
    visits = sorted(s.get("visit") for s in schools if s.get("visit"))
    rec["first_visit"] = visits[0] if visits else None
    rec["last_visit"] = visits[-1] if visits else None
    rec["n_official_visits"] = len(visits)
    return rec


def clean_measurables(df):
    """Null camp measurables that fall outside physically plausible ranges.

    Returns a copy with an added ``n_measurables`` count so a model can condition on how much
    athletic testing a prospect actually has on record — itself informative, since heavily
    scouted prospects get measured more.
    """
    out = df.copy()
    for col, (lo, hi) in MEASURABLE_RANGES.items():
        if col in out.columns:
            bad = out[col].notna() & ~out[col].between(lo, hi)
            out.loc[bad, col] = None
    present = [c for c in MEASURABLE_RANGES if c in out.columns]
    out["n_measurables"] = out[present].notna().sum(axis=1) if present else 0
    return out


def fetch_recruiting_class(year: int, *, page_size: int = 1000, positions=QB_POSITIONS,
                           session=None, pause: float = 0.2, max_pages: int = 40):
    """Fetch one recruiting class, keeping only ``positions``.

    The list endpoint embeds each recruit's full record, so this pages through the class rather
    than issuing a request per prospect. ``positions=None`` keeps everything.
    """
    import requests

    sess = session or requests.Session()
    rows, page = [], 1
    while page <= max_pages:
        resp = sess.get(f"{BASE}/{year}/athletes",
                        params={"limit": page_size, "page": page}, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("items") or []
        if not items:
            break
        for item in items:
            pos = ((item.get("athlete") or {}).get("position") or {}).get("abbreviation")
            if positions is None or pos in positions:
                rows.append(parse_recruit(item))
        if page >= int(payload.get("pageCount") or 1):
            break
        page += 1
        time.sleep(pause)
    return rows


def fetch_qb_recruits(years, *, session=None, pause: float = 0.2, progress=None):
    """Fetch QB-ish recruits across ``years`` and return a cleaned DataFrame.

    ``progress`` is an optional callable taking ``(year, n_rows)`` for logging.
    """
    import pandas as pd

    frames = []
    for year in years:
        rows = fetch_recruiting_class(year, session=session, pause=pause)
        if progress:
            progress(year, len(rows))
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return clean_measurables(df)
