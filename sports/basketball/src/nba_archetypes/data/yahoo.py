"""Phase 2 **augmentation** — the user's personal Yahoo 9-cat redraft history.

The Fantrax dynasty leagues give only 2 team-config -> success examples; the user's long-running
Yahoo NBA redraft league ("H2H Cat One", **exactly** the same 9 categories) adds ~8 finished seasons
≈ ~90 more. This mirrors the Fantrax path (`fantrax.py` + `eval/compose.py`) but with the two
"do it properly" choices the user picked:

- **Roster "configuration" = season-long average.** Redraft rosters churn (waivers/trades), so a
  single snapshot is misleading. We pull each team's roster for *every regular-season week* and weight
  each rostered player's archetype membership by **weeks-on-roster** — a season-long hold counts more
  than a one-week streamer.
- **Success label = category-win rate.** From each week's ``stat_winners`` (per-category match winner)
  we count the share of the 9 categories a team won across the regular season — finer-grained and
  less playoff-luck-driven than final rank (which we keep as a reference column).

Two Yahoo gotchas handled here (see memory): ``Game.league_ids()`` is **not** sport-filtered, so we
drive off explicit ``fantasy_yahoo.league_keys`` (never a sport guess); and Yahoo labels NBA seasons
by **start year** while our archetype membership is keyed by **end year**, so seasons get **+1**.

The full pull is ~1800+ roster calls and Yahoo rate-limits, so each league is cached to
``fantasy_yahoo.cache_dir`` and the fetch is resilient: a failed/throttled league doesn't lose the
finished ones (rerun resumes from cache).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..eval.compose import _norm  # reuse the diacritic/punct-folding name match (the "fonts" issue)
from ..utils.io import (DATA_PROCESSED, MANIFEST_DIR, ensure_dir, read_parquet, resolve,
                        sha256_file, write_parquet)


# ---------------------------------------------------------------- auth + league helpers

def _oauth(token_file):
    """OAuth2 session from the Yahoo token JSON (auto-refreshes in place when stale)."""
    from yahoo_oauth import OAuth2
    sc = OAuth2(None, None, from_file=str(Path(token_file).expanduser()))
    if not sc.token_is_valid():
        sc.refresh_access_token()
    return sc


def _end_season(settings) -> int:
    """Archetype season (END year) for a Yahoo league whose ``season`` is the START year."""
    return int(settings["season"]) + 1


def _regular_weeks(settings) -> list[int]:
    """Regular-season week numbers (``start_week`` .. just before ``playoff_start_week``)."""
    start = int(settings.get("start_week", 1))
    playoff = int(settings.get("playoff_start_week") or settings.get("end_week"))
    return list(range(start, playoff))


def _retry(fn, *, retries=4, backoff=5):
    """Call ``fn`` with linear backoff (Yahoo rate-limits heavy week-by-week pulls)."""
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retry transient Yahoo throttles
            last = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last


def _matchup_team_keys(matchup) -> list[str]:
    """The two ``team_key``s in a matchup (needed to split tied categories 0.5/0.5)."""
    keys = []
    for k, v in matchup["0"]["teams"].items():
        if k == "count":
            continue
        flat = {}
        for d in v["team"][0]:
            if isinstance(d, dict):
                flat.update(d)
        if flat.get("team_key"):
            keys.append(flat["team_key"])
    return keys


# ---------------------------------------------------------------- per-league fetch (network)

def fetch_team_weeks(sc, league_key, *, max_weeks=None, sleep=0.4):
    """Every team's roster for every regular-season week (one row per player-week)."""
    import yahoo_fantasy_api as yfa
    lg = yfa.League(sc, league_key)
    settings = lg.settings()
    season = _end_season(settings)
    weeks = _regular_weeks(settings)
    if max_weeks:
        weeks = weeks[:max_weeks]
    rows = []
    for tk, tinfo in lg.teams().items():
        tm = lg.to_team(tk)
        for wk in weeks:
            roster = _retry(lambda tm=tm, wk=wk: tm.roster(week=wk))
            for p in roster:
                rows.append({"league_key": league_key, "season": season, "team_key": tk,
                             "team_name": tinfo.get("name"), "week": wk,
                             "yahoo_player_id": p.get("player_id"), "player_name": p.get("name")})
            time.sleep(sleep)
    return rows


def tally_category_wins(matchups, team_keys):
    """Sum each team's category wins/plays over regular-season ``matchups`` (pure; tie = 0.5 each).

    ``matchups`` is an iterable of Yahoo ``matchup`` dicts. Playoff/consolation matchups are skipped.
    Returns ``(won, played)`` dicts. By construction ``sum(won) == sum(played)/2`` so the per-team
    win rate ``won/played`` averages to ~0.5 across a league.
    """
    won = {tk: 0.0 for tk in team_keys}
    played = {tk: 0 for tk in team_keys}
    for mu in matchups:
        if int(mu.get("is_playoffs", 0)) or int(mu.get("is_consolation", 0)):
            continue
        pair = _matchup_team_keys(mu)
        for sw in mu.get("stat_winners", []):
            w = sw.get("stat_winner", {})
            for tk in pair:
                played[tk] = played.get(tk, 0) + 1
            tied = str(w.get("is_tied", "0")) in ("1", "true", "True")
            winner = w.get("winner_team_key")
            if tied or not winner:
                for tk in pair:
                    won[tk] = won.get(tk, 0.0) + 0.5
            else:
                won[winner] = won.get(winner, 0.0) + 1
    return won, played


def fetch_labels(sc, league_key, *, max_weeks=None, sleep=0.4):
    """Per-team success labels: regular-season **category-win rate** + final standings (reference).

    Category-win rate = (categories won) / (categories played) over regular-season matchups, where a
    tied category gives each team 0.5. Symmetric by construction (the league mean is ~0.5).
    """
    import yahoo_fantasy_api as yfa
    lg = yfa.League(sc, league_key)
    settings = lg.settings()
    season = _end_season(settings)
    weeks = _regular_weeks(settings)
    if max_weeks:
        weeks = weeks[:max_weeks]
    teams = lg.teams()
    names = {tk: t.get("name") for tk, t in teams.items()}

    matchups = []
    for wk in weeks:
        raw = _retry(lambda wk=wk: lg.matchups(week=wk))
        wk_matchups = raw["fantasy_content"]["league"][1]["scoreboard"]["0"]["matchups"]
        matchups.extend(mv["matchup"] for mk, mv in wk_matchups.items() if mk != "count")
        time.sleep(sleep)
    won, played = tally_category_wins(matchups, list(teams.keys()))

    standings = {}
    for t in _retry(lambda: lg.standings()):
        tk = t.get("team_key")
        if tk is None:  # some seasons omit team_key in standings — fall back to name
            tk = next((k for k, n in names.items() if n == t.get("name")), None)
        if tk is not None:
            o = t.get("outcome_totals", {})
            standings[tk] = {"final_rank": int(t["rank"]) if t.get("rank") else None,
                             "playoff_seed": int(t["playoff_seed"]) if t.get("playoff_seed") else None,
                             "reg_wins": int(o.get("wins", 0)), "reg_losses": int(o.get("losses", 0)),
                             "reg_ties": int(o.get("ties", 0))}

    rows = []
    for tk in teams:
        p = played.get(tk, 0)
        rec = {"league_key": league_key, "season": season, "team_key": tk, "team_name": names[tk],
               "cats_won": round(won.get(tk, 0.0), 1), "cats_played": p,
               "cat_win_rate": round(won.get(tk, 0.0) / p, 4) if p else None}
        rec.update(standings.get(tk, {}))
        rows.append(rec)
    return rows


def fetch_yahoo(config, *, max_weeks=None, sleep=0.4, write: bool = True):
    """Fetch (cached, per league) season-long roster-weeks + labels for all configured leagues.

    Each league's two tables are cached under ``fantasy_yahoo.cache_dir`` so a rerun resumes past any
    rate-limit/failure. Writes the combined ``yahoo_team_weeks.parquet`` + ``yahoo_labels.parquet``.
    """
    import pandas as pd

    sc = _oauth(config.require("fantasy_yahoo.token_file"))
    league_keys = config.get("fantasy_yahoo.league_keys", []) or []
    cache_dir = resolve(config.get("fantasy_yahoo.cache_dir", ".cache/yahoo"))
    ensure_dir(cache_dir)

    weeks_frames, label_frames, failed = [], [], []
    for lk in league_keys:
        wpath = cache_dir / f"{lk}_team_weeks.parquet"
        lpath = cache_dir / f"{lk}_labels.parquet"
        try:
            if wpath.exists() and lpath.exists():
                tw, lb = read_parquet(wpath), read_parquet(lpath)
            else:
                tw = pd.DataFrame(fetch_team_weeks(sc, lk, max_weeks=max_weeks, sleep=sleep))
                lb = pd.DataFrame(fetch_labels(sc, lk, max_weeks=max_weeks, sleep=sleep))
                write_parquet(tw, wpath)
                write_parquet(lb, lpath)
            weeks_frames.append(tw)
            label_frames.append(lb)
            print(f"[yahoo] {lk}: {len(tw)} player-weeks, {len(lb)} teams "
                  f"(season {int(lb['season'].iloc[0]) if len(lb) else '—'})")
        except Exception as exc:  # noqa: BLE001 - surface, don't abort the batch
            failed.append((lk, f"{type(exc).__name__}: {exc}"[:160]))
    for lk, msg in failed:
        print(f"[yahoo] WARNING: league {lk} failed — {msg}")

    team_weeks = pd.concat(weeks_frames, ignore_index=True) if weeks_frames else pd.DataFrame()
    labels = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    if write and not team_weeks.empty:
        write_parquet(team_weeks, DATA_PROCESSED / "yahoo_team_weeks.parquet")
        write_parquet(labels, DATA_PROCESSED / "yahoo_labels.parquet")
        _write_manifest(team_weeks, labels, league_keys, failed)
    return team_weeks, labels


# ---------------------------------------------------------------- composition (pure)

def season_long_composition(team_weeks, membership, aliases=None):
    """Per team: **weeks-weighted** archetype soft-shares + hard counts, matched within the season.

    Each rostered player's archetype membership is weighted by how many regular-season weeks they were
    on the roster, so the composition reflects the roster a manager actually *held* all season rather
    than any one snapshot. Returns ``(composition_df, match_rate, unmatched_df)`` where ``match_rate``
    and the unmatched diagnostic are **weighted by player-weeks** (a season-long unmatched star matters
    more than a one-week streamer).
    """
    import pandas as pd

    pcols = [c for c in membership.columns if c.startswith("p") and c[1:].isdigit()]
    arch_name = membership.drop_duplicates("arch").set_index("arch")["arch_name"].to_dict()
    mem = membership.copy()
    mem["_key"] = list(zip(mem["season"], mem["player_name"].map(_norm)))
    lut = mem.drop_duplicates("_key").set_index("_key")

    remap = {_norm(k): _norm(v) for k, v in (aliases or {}).items()}
    r = team_weeks.copy()
    r["_nn"] = r["player_name"].map(_norm).replace(remap)
    r["_key"] = list(zip(r["season"], r["_nn"]))
    # collapse player-weeks -> one row per (team, player) carrying the weeks-on-roster weight
    w = (r.groupby(["league_key", "team_key", "team_name", "season", "_key", "player_name"],
                   dropna=False).size().reset_index(name="weeks"))
    w["_matched"] = w["_key"].isin(lut.index)
    total_w = float(w["weeks"].sum())
    match_rate = float(w.loc[w["_matched"], "weeks"].sum() / total_w) if total_w else 0.0
    unmatched = (w.loc[~w["_matched"], ["league_key", "season", "team_name", "player_name", "weeks"]]
                 .sort_values("weeks", ascending=False).reset_index(drop=True))

    out = []
    for (lk, tk), g in w.groupby(["league_key", "team_key"]):
        gm = g[g["_matched"]]
        gw = float(g["weeks"].sum())
        rec = {"league_key": lk, "team_key": tk, "team_name": g["team_name"].iloc[0],
               "season": int(g["season"].iloc[0]),
               "n_player_weeks": int(gw), "matched_player_weeks": int(gm["weeks"].sum()),
               "match_rate": round(gm["weeks"].sum() / gw, 3) if gw else 0.0}
        if len(gm):
            probs = lut.loc[gm["_key"]][pcols].to_numpy()
            wt = gm["weeks"].to_numpy()
            shares = (probs * wt[:, None]).sum(axis=0) / wt.sum()   # weeks-weighted mean membership
            hard = probs.argmax(axis=1)
            for j, _ in enumerate(pcols):
                nm = arch_name.get(j, f"A{j}")
                rec[f"comp_{nm}"] = round(float(shares[j]), 3)
                rec[f"n_{nm}"] = int((hard == j).sum())          # distinct players (not weighted)
        out.append(rec)
    return pd.DataFrame(out), match_rate, unmatched


def build_yahoo_phase2_table(config, *, write: bool = True):
    """Join weeks-weighted composition (cached) to the category-win-rate labels.

    Returns ``(table, match_rate, unmatched_df)``; writes ``phase2_yahoo_composition.parquet``.
    """
    team_weeks = read_parquet(DATA_PROCESSED / "yahoo_team_weeks.parquet")
    membership = read_parquet(DATA_PROCESSED / "nba_archetype_membership.parquet")
    labels = read_parquet(DATA_PROCESSED / "yahoo_labels.parquet")

    aliases = config.get("fantasy.name_aliases", {}) or {}   # share the Fantrax nickname aliases
    comp, match_rate, unmatched = season_long_composition(team_weeks, membership, aliases=aliases)
    keep = ["league_key", "team_key", "cat_win_rate", "cats_won", "cats_played",
            "final_rank", "playoff_seed", "reg_wins", "reg_losses", "reg_ties"]
    tbl = comp.merge(labels[[c for c in keep if c in labels.columns]],
                     on=["league_key", "team_key"], how="left")
    if write and not tbl.empty:
        write_parquet(tbl, DATA_PROCESSED / "phase2_yahoo_composition.parquet")
    return tbl, match_rate, unmatched


def _write_manifest(team_weeks, labels, league_keys, failed) -> None:
    ensure_dir(MANIFEST_DIR)
    path = DATA_PROCESSED / "yahoo_team_weeks.parquet"
    manifest = {
        "name": "yahoo_phase2",
        "source": "yahoo_fantasy_api",
        "note": "season-long roster-weeks + category-win-rate labels (Phase-2 augmentation examples)",
        "league_keys": list(league_keys),
        "leagues_succeeded": sorted(labels["league_key"].unique().tolist()) if len(labels) else [],
        "leagues_failed": [lk for lk, _ in failed],
        "seasons": sorted(int(s) for s in labels["season"].unique()) if len(labels) else [],
        "n_player_weeks": int(team_weeks.shape[0]),
        "n_team_seasons": int(labels.shape[0]),
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }
    with open(MANIFEST_DIR / "yahoo_phase2.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
