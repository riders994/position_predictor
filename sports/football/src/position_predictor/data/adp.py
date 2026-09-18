"""Historical ADP (Average Draft Position) — the *other* market signal for the postseason report.

ECR (expert consensus rank) is already wired (:mod:`position_predictor.data.benchmark`), but
``db_fpecr`` carries no ADP. ADP is what fantasy drafters actually did, so it's the natural
companion to ECR when grading a season after the fact. We pull it from the free
**FantasyFootballCalculator** API, which serves a preseason PPR ADP snapshot per year:

    https://fantasyfootballcalculator.com/api/v1/adp/ppr?year=Y&teams=12&position=all

FFC has occasional per-season holes (2025 once returned nothing; it serves a 249-player board as of
2026-09-15), so when it returns no data we **fall back to the FantasyPros consensus ADP board** for that year; the
report notes which source was used. Neither source carries ``gsis_id``; we join to our universe by
**normalised name + position** (reusing the keeper matcher), so a handful of names may not map —
the caller reports the match count. The result is cached to ``data/external/adp_<stem>.parquet``
(a regenerable, gitignored artifact).
"""

from __future__ import annotations

FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr?year={year}&teams={teams}&position=all"
FANTASYPROS_URL = "https://www.fantasypros.com/nfl/adp/overall.php?year={year}"
FFC_BOARD_URL = ("https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?year={year}&teams=12"
                 "&position=all")
# FFC's path segment for each of our scoring formats. Half-PPR boards start in 2018.
FFC_FORMATS = {"ppr": "ppr", "half_ppr": "half-ppr", "standard": "standard"}
FFC_BOARD_COLS = ["ffc_id", "name", "position", "team", "adp", "stdev", "high", "low",
                  "times_drafted", "total_drafts", "start_date", "end_date"]


def fetch_ffc_board(season: int, *, scoring: str = "ppr"):
    """The full FFC preseason board for ``season`` in ``scoring``, with each player's draft spread.

    Unlike :func:`fetch_ffc_adp` this keeps every position (K and DST included — they take real
    picks) and the spread of each player's real draft slot across the site's mock drafts:
    ``stdev``, ``high``, ``low``, plus the board's ``total_drafts`` and date window. That spread
    is what calibrates the draft simulator's room (:mod:`eval.draftsim`).

    FFC serves **12-team boards only**: a ``teams=10`` request returns the 12-team board byte for
    byte (checked on 2026, the one season that answers both), so no ``teams`` argument is offered.
    Raises on a network/parse failure or an empty board.
    """
    import json
    import urllib.request

    import pandas as pd

    from ..scoring import normalize_scoring

    fmt = FFC_FORMATS[normalize_scoring(scoring)]
    url = FFC_BOARD_URL.format(fmt=fmt, year=int(season))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — fixed trusted host
        payload = json.load(resp)
    players = payload.get("players") or []
    if not players:
        raise RuntimeError(f"FFC returned no {fmt} board for {season}")
    meta = payload.get("meta") or {}
    df = pd.DataFrame(players).rename(columns={"player_id": "ffc_id"})
    df["total_drafts"] = meta.get("total_drafts")
    df["start_date"] = meta.get("start_date")
    df["end_date"] = meta.get("end_date")
    return df.reindex(columns=FFC_BOARD_COLS)


def load_ffc_board(season: int, *, scoring: str = "ppr", refresh: bool = False):
    """:func:`fetch_ffc_board`, cached to ``data/external/ffc_board_<scoring>_<season>.parquet``.

    A past season's board never changes, so the cache is read unless ``refresh`` is set.
    """
    import pandas as pd

    from ..scoring import normalize_scoring
    from ..utils.io import DATA_EXTERNAL, ensure_dir

    scoring = normalize_scoring(scoring)
    path = DATA_EXTERNAL / f"ffc_board_{scoring}_{int(season)}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch_ffc_board(season, scoring=scoring)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return df


def fetch_ffc_adp(season: int, *, teams: int = 12):
    """Fetch the FantasyFootballCalculator preseason PPR ADP for ``season``.

    Returns a DataFrame ``[name, position, team, adp, times_drafted]`` (one row per ranked player).
    Raises on a network/parse failure or an empty payload so the caller can fall back to "no ADP".
    """
    import json
    import urllib.request

    import pandas as pd

    url = FFC_URL.format(year=int(season), teams=int(teams))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — fixed trusted host
        payload = json.load(resp)
    players = payload.get("players") or []
    if not players:
        raise RuntimeError(f"FFC returned no ADP players for {season}")
    rows = [{"name": p.get("name"), "position": p.get("position"), "team": p.get("team"),
             "adp": p.get("adp"), "times_drafted": p.get("times_drafted")} for p in players]
    return pd.DataFrame(rows)


def fetch_fantasypros_adp(season: int):
    """Fetch the FantasyPros consensus overall ADP board for ``season`` (FFC fallback).

    FFC occasionally has no data for a given year (e.g. 2025), so we fall back to the
    FantasyPros ``adp/overall`` board, which covers the same seasons. FantasyPros gates its
    JSON API, so we parse the HTML table; its structure is stable and regular (one ``<tr>``
    per player, with ``fp-player-name``/``fp-id`` attributes and the consensus ``AVG`` column).
    We avoid an HTML-parser dependency by extracting the regular rows directly.

    Returns the same shape as :func:`fetch_ffc_adp` — ``[name, position, team, adp,
    times_drafted]`` (``times_drafted`` is unavailable here, left null). Raises on a
    network/parse failure or an empty table so the caller can report "no ADP".
    """
    import re
    import urllib.request

    import pandas as pd

    url = FANTASYPROS_URL.format(year=int(season))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — fixed trusted host
        html = resp.read().decode("utf-8", "replace")

    body = re.search(r"<tbody>(.*?)</tbody>", html, re.S)
    rows = re.findall(r"<tr>(.*?)</tr>", body.group(1), re.S) if body else []
    out = []
    for r in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
        if len(cells) < 6:
            continue
        name = re.search(r'fp-player-name="([^"]+)"', cells[1])
        smalls = re.findall(r"<small>(.*?)</small>", cells[1])
        position = re.sub(r"\d+$", "", cells[2].strip())          # "WR1" -> "WR"
        try:
            adp = float(re.sub(r"<.*?>", "", cells[5]).strip())   # consensus AVG column
        except ValueError:
            continue
        if not name:
            continue
        out.append({"name": name.group(1), "position": position,
                    "team": smalls[0].strip() if smalls else None,
                    "adp": adp, "times_drafted": None})
    if not out:
        raise RuntimeError(f"FantasyPros returned no ADP players for {season}")
    return pd.DataFrame(out)


def build_adp_benchmark(config, season: int, name_id_map: dict, *, write: bool = True,
                        teams: int = 12):
    """Preseason ADP for the configured position, mapped to our ``player_id`` (``gsis_id``).

    ``name_id_map`` maps a normalised player name to ``player_id`` for the position's universe in
    ``season`` (built by the caller from our season-Y rows). Returns
    ``[player_id, season, adp, adp_pos_rank]`` (dense positional rank, lower ADP = earlier pick) and
    a ``match`` summary dict ``{ranked, matched}``. Best-effort: unmatched FFC names are dropped.
    """
    from ..eval.keeper import _norm
    from ..utils.io import DATA_EXTERNAL, ensure_dir
    from ..utils.naming import artifact_stem

    position = config.require("experiment.position").upper()
    stem = artifact_stem(config)

    source = "FFC"
    try:
        raw = fetch_ffc_adp(season, teams=teams)
    except Exception:
        # FFC has occasional per-season holes (e.g. 2025); fall back to the FantasyPros
        # consensus board, which covers those seasons. Surfaced via the match summary.
        raw = fetch_fantasypros_adp(season)
        source = "FantasyPros"
    pos = raw[raw["position"].str.upper() == position].copy()
    pos["player_id"] = pos["name"].map(lambda n: name_id_map.get(_norm(n)))
    matched = pos.dropna(subset=["player_id", "adp"]).copy()
    matched["season"] = int(season)
    matched["adp"] = matched["adp"].astype(float)
    matched["adp_pos_rank"] = matched["adp"].rank(method="dense").astype(int)
    out = matched[["player_id", "season", "adp", "adp_pos_rank"]].sort_values(
        "adp_pos_rank").reset_index(drop=True)

    if write:
        path = ensure_dir(DATA_EXTERNAL) / f"adp_{stem}.parquet"
        # keep other seasons already cached; replace this one
        if path.exists():
            import pandas as pd
            prior = pd.read_parquet(path)
            prior = prior[prior["season"] != int(season)]
            out_all = pd.concat([prior, out], ignore_index=True)
        else:
            out_all = out
        out_all.to_parquet(path, index=False)

    return out, {"ranked": int(len(pos)), "matched": int(len(out)), "source": source}
