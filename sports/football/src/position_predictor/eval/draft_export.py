"""Live-draft export: one self-contained JSON per league for the draft tool on the home site.

The tool runs the draft policies of :mod:`eval.draftsim` in the browser (ported to JS and pinned
by ``tests/fixtures/draft_golden.json``) over this file. It carries everything a policy reads:
the league shape, every player the board *or the market* knows about, the model's value and the
market's ADP for each, and the policy the tool should run.

**Which policy.** ``market_window`` (Entry 106): the market picks the position and the round,
the model picks the player inside that round. Replayed over past drafts it ties ADP drafting,
where the model's own VORP board loses 5-9 points a week. Adding the roster-aware insurance bench
to it (``market_window_depth``) was measured and **loses** 0.8-2.3 more, so the bench is market
too.

**The market is load-bearing here**, not a benchmark: ``AdpPolicy`` picks the position from ADP.
So every player the market ranks is exported, including rookies and veterans the model doesn't
score (``value = null``). The model's numbers are untouched; the market never moves a value.

ADP source, in order: FFC's live board for the league's format (the 2QB board for any league
that starts two QBs), if it ranks at least :data:`MIN_ADP_COVERAGE` of the draft; else the
league's own ECR board, ranked over the modeled positions, with the fall prior's spread. FFC
serves 12-team boards only, so every league reads 12-team order (the backtest's stated
assumption for 10-team rooms; 14 teams returns the same numbers).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .draftsim import SPREAD_A, SPREAD_B, adp_spread
from .keeper import _norm
from .league import MODELED_POS

EXPORT_SCHEMA_VERSION = 1
# The ADP board must rank this share of the draft's picks (the backtest's MIN_BOARD_COVERAGE).
MIN_ADP_COVERAGE = 0.85
# Market-only players (ranked, not on the model board) are exported down to this multiple of the
# draft's picks; the room never reaches deeper.
MARKET_DEPTH_FACTOR = 1.5
# What the backtest covered: 1QB, managed lineups, PPR/half-PPR, 10- and 12-team rooms.
BACKTESTED_TEAMS = (10, 12)
BACKTESTED_SCORING = ("ppr", "half_ppr")


def _num(x, digits: int = 2):
    """JSON-safe float: ``None`` for missing/NaN/inf."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if math.isfinite(f) else None


def bye_weeks(schedules, season: int) -> dict:
    """``{team: bye week}`` from the regular-season schedule (a team's one week without a game)."""
    reg = schedules[(schedules["season"] == int(season)) & (schedules["game_type"] == "REG")]
    if reg.empty:
        return {}
    weeks = set(int(w) for w in reg["week"].unique())
    played: dict = {}
    for side in ("home_team", "away_team"):
        for team, wk in zip(reg[side], reg["week"]):
            played.setdefault(team, set()).add(int(wk))
    out = {}
    for team, wks in played.items():
        missing = sorted(weeks - wks)
        if len(missing) == 1:
            out[team] = missing[0]
    return out


def ecr_adp_board(ecr_df):
    """The league's ECR board as an ADP stand-in: overall rank over the modeled positions.

    Returns ``[name, position, adp, stdev, fp_id]``. The spread is the fall prior's, which was fit
    on real draft slots, not on expert rank — a stated approximation for when no ADP board exists.
    """
    board = ecr_df[ecr_df["pos"].isin(MODELED_POS)].sort_values("ecr").reset_index(drop=True)
    out = board.rename(columns={"player": "name", "pos": "position"})[["name", "position"]].copy()
    out["adp"] = range(1, len(out) + 1)
    out["stdev"] = adp_spread(out["adp"].to_numpy())
    out["fp_id"] = board["id"] if "id" in board else None
    out["team"], out["bye"], out["ffc_id"] = None, None, None
    return out


def choose_adp_board(league, ffc_board, ecr_df, *, ffc_label: str):
    """``(board, adp_source, warnings)`` — FFC when it covers the draft, else ECR rank."""
    picks = league.teams * league.roster_size
    warnings = []
    if ffc_board is not None and len(ffc_board):
        modeled = ffc_board[ffc_board["position"].isin(MODELED_POS)]
        if len(modeled) >= MIN_ADP_COVERAGE * picks:
            return modeled.reset_index(drop=True), ffc_label, warnings
        warnings.append(f"{ffc_label} ranks only {len(modeled)} players for {picks} picks "
                        f"(a rolling in-season window?); ADP falls back to ECR rank")
    else:
        warnings.append(f"{ffc_label} unavailable; ADP falls back to ECR rank")
    if ecr_df is None or len(ecr_df) == 0:
        warnings.append("no ECR board either: market_adp is empty and the policy can't run")
        return None, None, warnings
    return ecr_adp_board(ecr_df), "ecr_rank", warnings


def _fp_to_gsis(ids):
    from ..data.benchmark import _to_int_id

    if ids is None:
        return {}
    ids = ids[["fantasypros_id", "gsis_id"]].dropna()
    ids = ids.assign(fantasypros_id=_to_int_id(ids["fantasypros_id"])).dropna()
    return dict(zip(ids["fantasypros_id"].astype(int), ids["gsis_id"]))


def build_draft_export(lb, *, draft_season: int, feature_season: int, adp_board, adp_source,
                       ecr_df, rosters, schedules, ids=None, warnings=(), generated_at=None,
                       ffc_meta=None) -> dict:
    """Assemble the export for one :class:`~eval.redraft.LeagueBoard`. Pure: no I/O.

    ``adp_board`` has ``[name, position, team, adp, stdev]`` (+ ``ffc_id``/``fp_id``/``bye`` when
    known), ``rosters`` the draft season's nflverse rosters, ``ids`` the nflverse id crosswalk.
    """
    import pandas as pd

    from ..data.benchmark import _to_int_id
    from .draft_backtest import match_board_ids

    lg = lb.league
    picks = lg.teams * lg.roster_size
    warnings = list(warnings)
    fp_gsis = _fp_to_gsis(ids)
    season_ro = rosters[rosters["season"] == int(draft_season)]
    team_of = dict(zip(season_ro["player_id"], season_ro["team"]))
    byes = bye_weeks(schedules, draft_season)

    # -- the model board (model rows + the market's rookies) ---------------------------------
    board = lb.board.copy()
    value_col = getattr(lb, "value_col", "proj_ppg")
    if value_col not in board:
        value_col = "proj_ppg"

    # -- market lookups by gsis id and by (name, position) -----------------------------------
    ecr_by_id, ecr_by_name = {}, {}
    if ecr_df is not None and len(ecr_df):
        e = ecr_df[ecr_df["pos"].isin(MODELED_POS)]
        fp = _to_int_id(e["id"]) if "id" in e else pd.Series([None] * len(e), index=e.index)
        for name, pos, val, f in zip(e["player"], e["pos"], e["ecr"], fp):
            gsis = fp_gsis.get(int(f)) if pd.notna(f) else None
            if gsis:
                ecr_by_id[gsis] = float(val)
            ecr_by_name[(_norm(name), pos)] = float(val)

    adp_rows = []
    if adp_board is not None and len(adp_board):
        a = adp_board.copy().reset_index(drop=True)
        if "fp_id" in a and a["fp_id"].notna().any():
            fp = _to_int_id(a["fp_id"])
            a["player_id"] = [fp_gsis.get(int(f)) if pd.notna(f) else None for f in fp]
        else:
            a["season"] = int(draft_season)
            a["player_id"] = match_board_ids(a, rosters)["player_id"]
        adp_rows = a.to_dict("records")
        for r in adp_rows:        # unmatched ids come back as NaN, which is truthy
            if not isinstance(r.get("player_id"), str):
                r["player_id"] = None
    adp_by_id = {r["player_id"]: r for r in adp_rows if r.get("player_id")}
    adp_by_name = {(_norm(r["name"]), r["position"]): r for r in adp_rows}

    players, used = [], set()

    def market_for(gsis, name, pos):
        r = adp_by_id.get(gsis) if gsis else None
        if r is None:
            r = adp_by_name.get((_norm(name), pos))
        return r

    for row in board.to_dict("records"):
        pos, name = row["position"], row["player_name"]
        gsis = row.get("player_id") if isinstance(row.get("player_id"), str) else None
        m = market_for(gsis, name, pos)
        if gsis is None and m is not None and m.get("player_id"):
            gsis = m["player_id"]
        if m is not None:
            used.add(id(m))
        team = team_of.get(gsis) or (m or {}).get("team")
        ecr = row.get("market_ecr")
        if _num(ecr) is None:
            ecr = ecr_by_id.get(gsis) if gsis else None
            ecr = ecr if ecr is not None else ecr_by_name.get((_norm(name), pos))
        players.append({
            "id": gsis or f"name:{_norm(name)}|{pos}", "gsis_id": gsis, "name": name, "pos": pos,
            "team": team, "bye": byes.get(team, _num((m or {}).get("bye"), 0)),
            "source": row.get("source", "model"),
            "value": _num(row.get(value_col)), "proj_ppg": _num(row.get("proj_ppg")),
            "vorp": _num(row.get("vorp")), "pos_rank": _num(row.get("proj_pos_rank"), 0),
            "board_rank": int(row["proj_overall_rank"]),
            "bench_insurance_median": _num(row.get("bench_insurance")),
            "market_ecr": _num(ecr), "market_adp": _num((m or {}).get("adp")),
            "market_adp_sd": _num((m or {}).get("stdev")),
        })

    # -- market-only players: ranked by the market, absent from the model board --------------
    extra = sorted((r for r in adp_rows if id(r) not in used and _num(r.get("adp")) is not None
                    and float(r["adp"]) <= MARKET_DEPTH_FACTOR * picks),
                   key=lambda r: float(r["adp"]))
    for r in extra:
        gsis = r.get("player_id")
        pos, name = r["position"], r["name"]
        team = team_of.get(gsis) or r.get("team")
        ecr = ecr_by_id.get(gsis) if gsis else None
        ecr = ecr if ecr is not None else ecr_by_name.get((_norm(name), pos))
        ext = (f"ffc:{int(r['ffc_id'])}" if _num(r.get("ffc_id")) is not None
               else f"name:{_norm(name)}|{pos}")
        players.append({
            "id": gsis or ext, "gsis_id": gsis, "name": name, "pos": pos, "team": team,
            "bye": byes.get(team, _num(r.get("bye"), 0)), "source": "market_only",
            "value": None, "proj_ppg": None, "vorp": None, "pos_rank": None, "board_rank": None,
            "bench_insurance_median": None, "market_ecr": _num(ecr),
            "market_adp": _num(r["adp"]), "market_adp_sd": _num(r.get("stdev")),
        })

    # A player must appear once; a duplicate id would make two pool rows of one person.
    seen, dupes = set(), []
    for p in players:
        if p["id"] in seen:
            dupes.append(p["id"])
        seen.add(p["id"])
    if dupes:
        raise ValueError(f"{lg.name}: duplicate player ids in the export: {dupes[:5]}")

    ranked_board = sum(1 for p in players if p["source"] != "market_only"
                       and p["market_adp"] is not None)
    on_board = sum(1 for p in players if p["source"] != "market_only")
    if adp_source and on_board and ranked_board < 0.9 * on_board:
        warnings.append(f"only {ranked_board}/{on_board} board players matched an ADP row")

    backtested = (not lg.is_superflex and not lg.bestball and lg.scoring in BACKTESTED_SCORING
                  and lg.teams in BACKTESTED_TEAMS)
    if not backtested:
        warnings.append(f"market_window was backtested in 1QB, {'/'.join(BACKTESTED_SCORING)}, "
                        f"{'/'.join(map(str, BACKTESTED_TEAMS))}-team leagues only; this league "
                        f"({lg.label}) is outside that")
    if lb.filter_note:
        warnings.append(lb.filter_note)

    scrape = None
    if ecr_df is not None and len(ecr_df) and "scrape_date" in ecr_df:
        scrape = str(pd.to_datetime(ecr_df["scrape_date"]).max().date())

    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "draft_season": int(draft_season), "feature_season": int(feature_season),
        "league": {"name": lg.name, "label": lg.label, "scoring": lg.scoring, "teams": lg.teams,
                   "roster_size": lg.roster_size, "rounds": lg.roster_size,
                   "starters": dict(lg.starters), "flex_positions": list(lg.flex_positions),
                   "bestball": lg.bestball, "qb_starters": lg.qb_starters,
                   "is_superflex": lg.is_superflex, "ecr_type": lb.ecr_type},
        "value_col": value_col,
        "replacement": {k: _num(v) for k, v in lb.replacement.items()},
        "market": {"adp_source": adp_source, "adp_teams": 12, "ecr_type": lb.ecr_type,
                   "ecr_scrape_date": scrape, "ffc": ffc_meta or None,
                   "fall_prior": {"a": SPREAD_A, "b": SPREAD_B}},
        "policy": {"name": "market_window", "window": lg.teams, "bench": "market",
                   "position_caps": {"QB": lg.qb_starters + 1,
                                     "TE": lg.starters.get("TE", 0) + 1,
                                     "default": lg.roster_size},
                   "backtested": backtested},
        "pool_semantics": ("players with market_adp null queue after every ranked player, in "
                           "array order; value null = the model can't score him (market "
                           "choices may take him, model choices skip him); board_rank is the "
                           "model's VORP order"),
        "warnings": warnings,
        "players": players,
    }


def export_league(lb, *, draft_season: int, feature_season: int, rosters, schedules, ids=None,
                  fetch_ffc=None) -> dict:
    """Fetch the live ADP board for ``lb``'s format and build its export (network: FFC only)."""
    from ..data.adp import FFC_2QB_FORMAT, fetch_ffc_board

    lg = lb.league
    fmt = FFC_2QB_FORMAT if lg.is_superflex else None
    label = f"ffc_{fmt or lg.scoring}"
    fetch = fetch_ffc_board if fetch_ffc is None else fetch_ffc
    warnings, ffc, meta = [], None, None
    try:
        ffc = fetch(draft_season, scoring=lg.scoring, fmt=fmt)
        meta = {k: (str(ffc[k].iloc[0]) if k in ffc and len(ffc) else None)
                for k in ("total_drafts", "start_date", "end_date")}
    except Exception as exc:  # noqa: BLE001 — FFC down: fall back to ECR rank, with a warning
        warnings.append(f"{label} fetch failed ({type(exc).__name__})")
    if lg.is_superflex and lg.scoring != "ppr":
        warnings.append(f"FFC's 2QB board is PPR-scored; this league is {lg.scoring}")
    board, source, more = choose_adp_board(lg, ffc, lb.market_board, ffc_label=label)
    return build_draft_export(lb, draft_season=draft_season, feature_season=feature_season,
                              adp_board=board, adp_source=source, ecr_df=lb.market_board,
                              rosters=rosters, schedules=schedules, ids=ids,
                              warnings=warnings + more,
                              ffc_meta=meta if source == label else None)
