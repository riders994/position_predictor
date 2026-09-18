"""Draft backtest — does opportunity-cost drafting win real past leagues?

Entry 102 measured opportunity cost in the model's own currency: projections were treated as the
truth. This replays **past drafts** instead and scores every team on what actually happened.

For each season *Y* with a preseason FantasyFootballCalculator board and a leak-safe projection:

1. **The room** drafts off that season's real FFC board (:mod:`data.adp`), each opponent reading a
   player at ``market_rank + k * sd * z``. ``k`` is calibrated per board so simulated drafts
   reproduce the board's *observed* draft-slot spread (:func:`eval.draftsim.calibrate_noise`).
2. **My team** drafts from one slot with one of three policies — draft like the market, walk the
   model's VORP board, or look ahead — using the model's preseason projection for *Y*, trained only
   on labels before *Y* (:func:`eval.projection.project_position` with ``feature_season=Y-1``).
3. **Every team** is scored on season-*Y* actual weekly points under one lineup rule for all teams
   (:func:`team_points_per_week`). The default, ``managed``, starts each week the active players
   with the best season points per game — the redraft leagues this is for set lineups before
   kickoff. ``bestball`` (the best lineup after the fact) is a variant only: it pays a roster for
   every bench spike week, which rewards the depth a market drafter builds and a starters-focused
   board doesn't (the first 2023 run's ADP team got 19.6 QB points a week from Mahomes + Fields).

Every draw of the room is shared by all policies and all slots, so policy comparisons are paired.

League shapes: 1QB / 2RB / 2WR / 1TE / 1FLEX (RB/WR/TE), 14 drafted players (a 15–16 round draft
less K and DST), 10 or 12 teams, half or full PPR. FFC publishes 12-team boards only — its
``teams=10`` request returns the 12-team board byte-for-byte — so a 10-team room drafts in the
12-team market order with the same pick-number spread. That is an assumption, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .draftsim import (POS_INDEX, AdpPolicy, BoardPolicy, DraftSim, LookaheadPolicy,
                       MarketWindowPolicy, Pool, calibrate_noise, lineup_value)
from .league import MODELED_POS, league_from_dict

# Seasons with a preseason FFC board of that format and a leak-safe projection. 2020 is excluded
# from the pipeline (data.exclude_seasons) and a 2021 board would need 2020 feature rows.
BACKTEST_SEASONS = {
    "ppr": (2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025),
    "half_ppr": (2018, 2019, 2022, 2023, 2024, 2025),
}
BACKTEST_TEAMS = (10, 12)
BACKTEST_STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
BACKTEST_ROSTER_SIZE = 14
# A board must rank this share of the draft's picks, or the room runs out of market opinions and
# drafts unranked players in arbitrary order (FFC 2012 ranks 92 offensive players).
MIN_BOARD_COVERAGE = 0.85
# Model-only players (unranked by the market) enter the pool down to this multiple of the draft.
MODEL_DEPTH_FACTOR = 1.5
POLICY_NAMES = ("adp", "vorp_board", "vorp_board_ranked", "vorp_board_ranked_depth",
                "market_window", "lookahead_ranked")
# Paired comparisons reported by :func:`summarize` (each only when both policies were run).
COMPARISONS = (("vorp_board", "adp"), ("vorp_board_ranked", "vorp_board"),
               ("vorp_board_ranked", "adp"), ("vorp_board_ranked_depth", "vorp_board_ranked"),
               ("vorp_board_ranked_depth", "adp"), ("market_window", "adp"),
               ("market_window", "vorp_board_ranked"), ("lookahead_ranked", "vorp_board_ranked"),
               ("lookahead_ranked", "adp"), ("lookahead", "vorp_board"), ("lookahead", "adp"))
# FFC team codes that differ from nflverse's.
_TEAM_ALIASES = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS"}


def backtest_league(teams: int, scoring: str):
    return league_from_dict({"name": f"backtest_{teams}_{scoring}", "scoring": scoring,
                             "teams": int(teams), "starters": dict(BACKTEST_STARTERS),
                             "flex_positions": ["RB", "WR", "TE"],
                             "roster_size": BACKTEST_ROSTER_SIZE})


# -- projections -----------------------------------------------------------------------------------
def projection_cache_path(scoring: str, position: str):
    from ..utils.io import DATA_PROCESSED
    return DATA_PROCESSED / "draftsim" / f"proj_{scoring}_{position.lower()}.parquet"


def build_projection_cache(scoring: str, position: str, seasons=None):
    """Leak-safe preseason projections for every backtest season of one position; cached."""
    import pandas as pd

    from ..utils.config import Config
    from ..utils.io import ensure_dir
    from .projection import ensure_features, project_position

    seasons = seasons or BACKTEST_SEASONS[scoring]
    cfg = Config.load(f"config/football_{position.lower()}.yaml").with_overrides(
        {"target.scoring": scoring})
    ensure_features(cfg)
    frames = []
    for y in seasons:
        p = project_position(cfg, feature_season=int(y) - 1)
        if p.empty:
            continue
        if int(p["proj_season"].iloc[0]) != int(y):
            raise RuntimeError(f"{position} {scoring}: projected {p['proj_season'].iloc[0]}, "
                               f"expected {y}")
        frames.append(p.assign(scoring=scoring))
    out = pd.concat(frames, ignore_index=True)
    path = projection_cache_path(scoring, position)
    ensure_dir(path.parent)
    out.to_parquet(path, index=False)
    return out


def load_projections(scoring: str, season: int, *, build_missing: bool = True):
    import pandas as pd

    frames = []
    for pos in MODELED_POS:
        path = projection_cache_path(scoring, pos)
        df = (pd.read_parquet(path) if path.exists()
              else build_projection_cache(scoring, pos) if build_missing else None)
        if df is None:
            raise FileNotFoundError(path)
        frames.append(df[df["proj_season"] == int(season)])
    return pd.concat(frames, ignore_index=True)


# -- the market board ------------------------------------------------------------------------------
def match_board_ids(board, rosters):
    """Attach nflverse ``player_id`` to FFC board rows by name, in three passes.

    1. season + position + normalised name (``full_name``, or ``football_name`` + ``last_name``);
    2. season + name, when that name maps to exactly one player that season;
    3. season + position + team + surname, when unique — catches nicknames ("Hollywood Brown",
       "Robbie Chosen", "Kenny Gainwell") without a hand-kept alias list.

    Adds ``player_id`` (``None`` when unmatched) and ``match_pass`` (1/2/3, or 0).
    """
    import pandas as pd

    from .keeper import _norm

    ro = rosters[rosters["player_id"].notna()].copy()
    ro["position"] = ro["position"].replace({"FB": "RB"})
    nick = (ro["football_name"].fillna("") + " " + ro["last_name"].fillna("")).map(_norm)
    cand = pd.concat([ro.assign(k=ro["full_name"].map(_norm)), ro.assign(k=nick)])
    cand = cand[["season", "position", "team", "k", "player_id"]].drop_duplicates()
    cand["surname"] = cand["k"].str.split().str[-1]

    def unique_id(keys):
        g = cand.groupby(keys)["player_id"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
        return g.dropna().to_dict()

    out = board.copy()
    out["k"] = out["name"].map(_norm)
    out["surname"] = out["k"].str.split().str[-1]
    out["team_nfl"] = out["team"].map(lambda t: _TEAM_ALIASES.get(t, t))
    passes = [(["season", "position", "k"], ["season", "position", "k"]),
              (["season", "k"], ["season", "k"]),
              (["season", "position", "team", "surname"], ["season", "position", "team_nfl",
                                                          "surname"])]
    out["player_id"], out["match_pass"] = None, 0
    for n, (cand_keys, board_keys) in enumerate(passes, start=1):
        lookup = unique_id(cand_keys)
        todo = out["player_id"].isna()
        hits = [lookup.get(tuple(r)) for r in out.loc[todo, board_keys].itertuples(index=False)]
        out.loc[todo, "player_id"] = hits
        out.loc[todo & out["player_id"].notna(), "match_pass"] = n
    # One nflverse player can't be two board rows; keep the earlier-drafted.
    dup = out["player_id"].notna() & out.sort_values("adp").duplicated("player_id").reindex(out.index)
    out.loc[dup, ["player_id", "match_pass"]] = [None, 0]
    return out.drop(columns=["k", "surname", "team_nfl"])


# -- the pool --------------------------------------------------------------------------------------
def build_pool(board, proj, league, *, model_depth_factor: float = MODEL_DEPTH_FACTOR):
    """Merge the market board and the model board into one draft pool.

    Returns ``(frame, pool)``: ``frame`` is one row per draftable player (``player_id``, ``name``,
    ``position``, ``market_rank``, ``stdev``, ``proj_ppg``, ``vorp``); ``pool`` the
    :class:`~eval.draftsim.Pool` over the same rows. Market players the model can't score keep
    ``proj_ppg = NaN``; model players the market doesn't rank keep ``market_rank = NaN``.
    Unmatched market rows stay in (the room drafts them) under a synthetic ``ffc:`` id.
    """
    import pandas as pd

    from .keeper import build_board

    model, _, _ = build_board(proj, teams=league.teams, roster=league.starters,
                              flex_positions=league.flex_positions)
    total = league.teams * league.roster_size
    model_keep = model[model["proj_overall_rank"] <= int(total * model_depth_factor)]

    mk = board[board["position"].isin(MODELED_POS)].sort_values("adp").copy()
    mk["market_rank"] = np.arange(1, len(mk) + 1, dtype=float)   # re-ranked without K/DST
    mk["player_id"] = mk["player_id"].where(mk["player_id"].notna(),
                                            "ffc:" + mk["ffc_id"].astype(str))
    mk = mk[["player_id", "name", "position", "market_rank", "stdev"]]

    ids = pd.unique(pd.concat([mk["player_id"], model_keep["player_id"]], ignore_index=True))
    frame = pd.DataFrame({"player_id": ids})
    frame = frame.merge(mk, on="player_id", how="left")
    frame = frame.merge(model[["player_id", "player_name", "position", "proj_ppg", "vorp"]]
                        .rename(columns={"position": "model_position"}),
                        on="player_id", how="left")
    # The model's position wherever it scores the player, so value and lineup slot agree.
    frame["position"] = frame["model_position"].fillna(frame["position"])
    frame["name"] = frame["name"].fillna(frame["player_name"])
    frame = frame.drop(columns=["model_position", "player_name"]).reset_index(drop=True)

    scored = frame[frame["proj_ppg"].notna()].sort_values(["vorp", "proj_ppg"],
                                                          ascending=False)
    pool = Pool(position=frame["position"].map(POS_INDEX).to_numpy(),
                adp=frame["market_rank"].to_numpy(dtype=float),
                value=frame["proj_ppg"].to_numpy(dtype=float),
                board_order=scored.index.to_numpy(), player_id=frame["player_id"].to_numpy())
    return frame, pool


def ranked_only_pool(pool: Pool) -> Pool:
    """``pool`` with every player the market doesn't rank made undraftable for *my* team.

    A player absent from a late-August market board is overwhelmingly one who is retired, unsigned,
    suspended or hurt — facts the model can't see (2023: Tom Brady, retired, projected 15.5 PPG;
    36 of the model board's top 168 were unranked and averaged 4.4 actual points a week against
    10.1 for ranked players). The room already never takes them. This uses the market to decide
    *who is draftable*, not what anyone is worth: every value stays the model's.
    """
    ranked = np.isfinite(pool.adp)
    return Pool(position=pool.position, adp=pool.adp,
                value=np.where(ranked, pool.value, np.nan),
                board_order=pool.board_order[ranked[pool.board_order]],
                player_id=pool.player_id)


def make_policy(name: str, spec, draw: int, slot: int):
    """``(policy, pool_variant)`` for a policy name in :data:`POLICY_NAMES`."""
    if name == "adp":
        return AdpPolicy(), "all"
    if name == "vorp_board":
        return BoardPolicy(), "all"
    if name == "vorp_board_ranked":
        return BoardPolicy(), "ranked"
    if name == "vorp_board_ranked_depth":
        return BoardPolicy(bench="insurance"), "ranked"
    if name == "market_window":
        return MarketWindowPolicy(), "all"
    if name in ("lookahead", "lookahead_ranked"):
        rng = np.random.default_rng([spec.seed, spec.season, spec.teams, draw, slot, 1])
        policy = LookaheadPolicy(n_plan_draws=spec.plan_draws, per_pos=spec.per_pos, rng=rng)
        return policy, ("ranked" if name.endswith("_ranked") else "all")
    raise ValueError(f"unknown policy {name!r}; expected one of {POLICY_NAMES}")


# -- actual scoring --------------------------------------------------------------------------------
LINEUP_RULES = ("managed", "bestball")


def weekly_matrix(player_ids, weekly_pts, season: int, *, with_played: bool = False):
    """``(len(player_ids), weeks)`` actual points for ``season``'s regular-season weeks (0 = no game).

    With ``with_played`` also returns the boolean ``played`` matrix — a 0-point game is still a
    game, which a managed lineup needs to tell apart from a bye.
    """
    wk = weekly_pts[weekly_pts["season"] == int(season)]
    weeks = np.sort(wk["week"].unique())
    index = {pid: i for i, pid in enumerate(player_ids)}
    col = {w: j for j, w in enumerate(weeks)}
    m = np.zeros((len(player_ids), len(weeks)))
    played = np.zeros(m.shape, dtype=bool)
    rows = wk[wk["player_id"].isin(index)]
    r, c = rows["player_id"].map(index).to_numpy(), rows["week"].map(col).to_numpy()
    m[r, c] = rows["points"].to_numpy()
    played[r, c] = True
    return (m, played) if with_played else m


def select_lineup(keys, positions, league) -> list[int]:
    """Indices of the best starting lineup ranked by ``keys``: dedicated slots, then flex."""
    order = np.argsort(-np.asarray(keys, dtype=float), kind="stable")
    chosen: list[int] = []
    for p in MODELED_POS:
        k = league.starters.get(p, 0)
        chosen += [int(j) for j in order if positions[j] == POS_INDEX[p]][:k]
    taken = set(chosen)
    flex = [int(j) for j in order
            if j not in taken and MODELED_POS[positions[j]] in league.flex_positions]
    return chosen + flex[: league.starters.get("FLEX", 0)]


def team_points_per_week(roster, points: np.ndarray, positions: np.ndarray, league, *,
                         played=None, lineup: str = "bestball") -> float:
    """Mean weekly points a roster actually scores under a lineup rule.

    * ``bestball`` — the best lineup after the fact each week (dedicated slots, then flex).
    * ``managed`` — each week, among players who played, start those with the best season points
      per game: a manager who knows who is active and who is good, but not which week will pop.
      That hindsight about quality is shared by every team, so it favours no policy. Needs
      ``played`` (see :func:`weekly_matrix`).
    """
    if lineup not in LINEUP_RULES:
        raise ValueError(f"lineup must be one of {LINEUP_RULES}; got {lineup!r}")
    roster = np.asarray(roster, dtype=int)
    if points.shape[1] == 0 or len(roster) == 0:
        return 0.0
    sub, pos = points[roster], positions[roster]
    weeks = sub.shape[1]
    if lineup == "bestball":
        total = 0.0
        for w in range(weeks):
            col = sub[:, w]
            total += lineup_value({p: col[pos == i].tolist() for i, p in enumerate(MODELED_POS)},
                                  league)
        return total / weeks
    if played is None:
        raise ValueError("managed lineups need the played matrix")
    active = played[roster]
    games = active.sum(axis=1)
    ppg = np.where(games > 0, sub.sum(axis=1) / np.maximum(games, 1), -np.inf)
    total = 0.0
    for w in range(weeks):
        keys = np.where(active[:, w], ppg, -np.inf)
        starters = [j for j in select_lineup(keys, pos, league) if active[j, w]]
        total += float(sub[starters, w].sum())
    return total / weeks


# -- one season ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SeasonSpec:
    scoring: str
    season: int
    teams: int
    draws: int = 10
    plan_draws: int = 8
    per_pos: int = 3
    seed: int = 1729
    lineup: str = "managed"


def run_season(spec: SeasonSpec, *, board, proj, weekly_pts, slots=None, policies=POLICY_NAMES):
    """Replay ``spec.draws`` drafts of one season from every slot under each policy.

    Returns ``(rows, info)``: one row per (draw, slot, policy) with my actual points per week, the
    league's mean, my finish rank and my projected lineup; ``info`` carries the calibrated noise
    scale, coverage and a skip reason if the board is too thin to simulate.
    """
    league = backtest_league(spec.teams, spec.scoring)
    frame, pool = build_pool(board, proj, league)
    ranked = int(np.isfinite(pool.adp).sum())
    picks = league.teams * league.roster_size
    info = {"scoring": spec.scoring, "season": spec.season, "teams": spec.teams,
            "lineup": spec.lineup, "ranked": ranked, "picks": picks, "pool": len(pool),
            "scored_share_top_picks": float(np.isfinite(pool.value[np.isfinite(pool.adp)
                                                                   & (pool.adp <= picks)]).mean())}
    if ranked < MIN_BOARD_COVERAGE * picks:
        info["skipped"] = f"board ranks {ranked} players for {picks} picks"
        return [], info

    # FFC boards are 12-team, so calibrate on the 12-team shape and reuse the scale for 10 teams.
    cal = DraftSim(backtest_league(12, spec.scoring), pool)
    k, _ = calibrate_noise(cal, frame["stdev"].to_numpy(dtype=float), n_drafts=30, seed=spec.seed)
    info["noise_scale"] = k

    # Both pools share positions and market ranks, so one room draw serves every policy.
    sims = {"all": DraftSim(league, pool, noise_scale=k),
            "ranked": DraftSim(league, ranked_only_pool(pool), noise_scale=k)}
    points, played = weekly_matrix(frame["player_id"].to_numpy(), weekly_pts, spec.season,
                                   with_played=True)
    slots = range(league.teams) if slots is None else [s - 1 for s in slots]
    rng = np.random.default_rng([spec.seed, spec.season, spec.teams])
    rows = []
    for d in range(spec.draws):
        read = sims["all"].room(rng)
        for slot in slots:
            for name in policies:
                policy, variant = make_policy(name, spec, d, slot)
                sim = sims[variant]
                state = sim.run(sim.new_state(), read, me=slot, policy=policy)
                scores = np.array([team_points_per_week(r, points, pool.position, league,
                                                        played=played, lineup=spec.lineup)
                                   for r in state.rosters])
                mine = state.rosters[slot]
                rows.append({"scoring": spec.scoring, "season": spec.season, "teams": spec.teams,
                             "lineup": spec.lineup, "draw": d, "slot": slot + 1, "policy": name,
                             "points_pw": float(scores[slot]),
                             "league_mean_pw": float(scores.mean()),
                             "rank": int(1 + (scores > scores[slot]).sum()),
                             "proj_lineup": sim.my_value(state, slot),
                             "rookies_or_unscored": int((~pool.scored[mine]).sum()),
                             "first_picks": "|".join(str(frame.at[i, "name"]) for i in mine[:3])})
    return rows, info


def load_season_inputs(scoring: str, season: int, *, weekly=None, rosters=None, refresh=False):
    """Board (matched), projections and weekly actual points for one season."""
    import pandas as pd

    from ..data.adp import load_ffc_board
    from ..utils.io import DATA_RAW
    from .bestball import weekly_fantasy_points

    if rosters is None:
        rosters = pd.read_parquet(DATA_RAW / "rosters.parquet",
                                  columns=["season", "team", "position", "full_name",
                                           "football_name", "last_name", "player_id"])
    if weekly is None:
        weekly = pd.read_parquet(DATA_RAW / "weekly.parquet",
                                 columns=["player_id", "season", "week", "season_type",
                                          "position", "position_group", "fantasy_points",
                                          "receptions"])
    board = load_ffc_board(season, scoring=scoring, refresh=refresh)
    board = match_board_ids(board.assign(season=int(season)),
                            rosters[rosters["season"] == int(season)])
    proj = load_projections(scoring, season)
    pts = weekly_fantasy_points(weekly[weekly["season"] == int(season)], scoring=scoring)
    return board, proj, pts


# -- summary ---------------------------------------------------------------------------------------
def summarize(results):
    """Per (scoring, teams): policy means and paired gains, with season-clustered standard errors.

    Draws within a season share one board and one set of outcomes, so they are not independent;
    each season contributes one mean and the standard error is taken across seasons.
    """
    import pandas as pd

    r = results.copy()
    if "lineup" not in r:
        r["lineup"] = "managed"
    r["vs_league"] = r["points_pw"] - r["league_mean_pw"]
    league_key = ["lineup", "scoring", "teams"]
    key = [*league_key, "season", "draw", "slot"]
    wide = r.pivot_table(index=key, columns="policy", values="points_pw").reset_index()
    pairs = [(a, b) for a, b in COMPARISONS if a in wide and b in wide]
    for a, b in pairs:
        wide[f"{a}_minus_{b}"] = wide[a] - wide[b]
    per_season = wide.groupby([*league_key, "season"]).mean(numeric_only=True)

    def mean_se(s):
        return pd.Series({"mean": s.mean(), "se": s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1
                          else np.nan, "seasons_won": int((s > 0).sum()), "seasons": len(s)})

    gains = []
    for a, b in pairs:
        col = f"{a}_minus_{b}"
        g = per_season[col].groupby(level=league_key).apply(mean_se).unstack()
        gains.append(g.assign(comparison=f"{a} − {b}").reset_index())
    gains = pd.concat(gains, ignore_index=True) if gains else pd.DataFrame()

    policy = (r.groupby([*league_key, "policy"])
               .agg(points_pw=("points_pw", "mean"), vs_league=("vs_league", "mean"),
                    win_rate=("rank", lambda s: float((s == 1).mean())),
                    top3_rate=("rank", lambda s: float((s <= 3).mean())),
                    mean_rank=("rank", "mean"), proj_lineup=("proj_lineup", "mean"))
               .reset_index())
    by_slot = (wide.groupby([*league_key, "slot"])
                   [[f"{a}_minus_{b}" for a, b in pairs]].mean().reset_index())
    by_season = per_season[[f"{a}_minus_{b}" for a, b in pairs]].reset_index()
    return {"gains": gains, "policy": policy, "by_slot": by_slot, "by_season": by_season}


# -- report ----------------------------------------------------------------------------------------
POLICY_LABELS = {
    "adp": "ADP drafter (market order, fills open slots)",
    "vorp_board": "VORP board, whole projection pool",
    "vorp_board_ranked": "VORP board, market-ranked players only",
    "vorp_board_ranked_depth": "VORP board, market-ranked only, insurance bench",
    "market_window": "Market picks position + round; model picks the player within one round",
    "lookahead_ranked": "Lookahead (opportunity cost), market-ranked only",
    "lookahead": "Lookahead (opportunity cost), whole pool",
}


def _md_table(df, *, floats: int = 2) -> list[str]:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("—" if np.isnan(v) else f"{v:.{floats}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return out


def render_markdown(summary: dict, infos, *, draws: int, lookahead_note: str = "") -> str:
    """The committed backtest report. ``infos`` is the list of :func:`run_season` info dicts."""
    import pandas as pd

    lines = [
        "# Draft backtest — does the model's board win real past drafts?", "",
        "_Every past season with a preseason FantasyFootballCalculator board and a leak-safe "
        "projection is re-drafted from every slot. The room drafts off that season's real board "
        "with noise calibrated to its observed draft-slot spread; my team drafts by one policy; "
        "**every team is scored on actual weekly points** under the row's lineup rule "
        "(`managed`: each week start the active players with the best season points per game; "
        "`bestball`: the best lineup after the fact). "
        f"{draws} room draws per season, shared by every policy, so comparisons are paired._", "",
        "_League: 1QB / 2RB / 2WR / 1TE / 1FLEX (RB/WR/TE), 14 drafted players. FFC publishes "
        "12-team boards only, so the 10-team room drafts in 12-team market order — an assumption. "
        "Standard errors are across seasons (draws inside a season share one outcome)._", "",
    ]
    if lookahead_note:
        lines += [f"_{lookahead_note}_", ""]
    lines += ["## Policies", ""]
    lines += [f"- `{k}` — {v}" for k, v in POLICY_LABELS.items()
              if k in set(summary["policy"]["policy"])]

    gains = summary["gains"].copy()
    if not gains.empty:
        gains["seasons won"] = (gains["seasons_won"].astype(int).astype(str) + "/"
                                + gains["seasons"].astype(int).astype(str))
        gains = gains[["lineup", "scoring", "teams", "comparison", "mean", "se", "seasons won"]]
        lines += ["", "## Paired gains — actual points per week", ""]
        lines += _md_table(gains.rename(columns={"mean": "gain / week", "se": "± se"}))

    pol = summary["policy"].copy()
    pol["win_rate"] = (100 * pol["win_rate"]).round(1)
    pol["top3_rate"] = (100 * pol["top3_rate"]).round(1)
    lines += ["", "## Policy outcomes", "",
              "_`vs league` is my points per week minus the league's mean team. `win %` = "
              "finished first in actual points; `proj lineup` is the model's own projected "
              "starting lineup — the gap between it and actual points is the model's optimism._",
              ""]
    lines += _md_table(pol.rename(columns={"points_pw": "pts / week", "vs_league": "vs league",
                                           "win_rate": "win %", "top3_rate": "top-3 %",
                                           "mean_rank": "mean finish",
                                           "proj_lineup": "proj lineup"}))

    by_season = summary["by_season"]
    if not by_season.empty:
        lines += ["", "## By season", ""]
        lines += _md_table(by_season.rename(columns=lambda c: c.replace("_minus_", " − ")))

    by_slot = summary["by_slot"]
    if not by_slot.empty:
        lines += ["", "## By draft slot (mean over seasons and draws)", ""]
        lines += _md_table(by_slot.rename(columns=lambda c: c.replace("_minus_", " − ")))

    info = pd.DataFrame(infos)
    if not info.empty:
        keep = [c for c in ["scoring", "season", "teams", "ranked", "picks", "pool",
                            "scored_share_top_picks", "noise_scale", "skipped"] if c in info]
        info = info[keep].drop_duplicates(["scoring", "season", "teams"]).sort_values(
            ["scoring", "teams", "season"])
        lines += ["", "## Seasons simulated", "",
                  "_`ranked` = offensive players on the FFC board; `scored share` = share of the "
                  "market's first `picks` players the model can project (the rest are mostly "
                  "rookies); `noise` = calibrated room noise scale._", ""]
        lines += _md_table(info.rename(columns={"scored_share_top_picks": "scored share",
                                                "noise_scale": "noise"}).fillna(""))
    return "\n".join(lines) + "\n"
