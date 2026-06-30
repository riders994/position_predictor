"""Phase-2 **simulation** — bootstrap fantasy teams to recover the composition→success signal.

The real-team model hit a four-way null because ``cat_win_rate`` is dominated by in-season management,
streaming, injuries and schedule luck, which swamp the draft-composition signal. This simulator
removes that noise: it drafts teams from a **leak-safe prior-season value** prior with **strategy
variation** (consensus + punt builds), then scores each team by simulating head-to-head **9-cat**
outcomes from the players' *actual* season production. That isolates "which archetype mixes cover the
categories" and lets us sample the composition space — including the punt builds where the signal lives
— at arbitrary scale. (PROJECT_PLAN §4 "augment with simulated fields seeded by real settings".)

Pipeline: ``player_value_table`` (per-season 9-cat z-scores, impact-based percentages) →
``attach_prior`` (season N-1 z-vector = ex-ante draft prior; inner-join keeps **returning players**
only, per the v1 scope) → ``simulate_draft`` (snake draft, each team picks by its strategy's weighting
of the *prior* z-vector, softmax temperature for realism) → ``team_outcomes`` (team totals from
*actual* season-N stats) → ``cat_win_rates`` (round-robin) → ``simulated_composition`` (archetype
shares by ``athlete_id``). The output table feeds the same ``phase2_model`` machinery.

Honesty: a fully-simulated model partly bakes in the archetype→category mapping, so its *directional*
conclusions are reality-checked against the 88 real Yahoo teams — simulate to learn the mechanism, real
teams as the check. The prior-season prior is leak-safe (uses only pre-season-N info); real mock/ADP is
the v2 realism upgrade.
"""
from __future__ import annotations

import numpy as np

from ..utils.io import DATA_INTERIM, DATA_PROCESSED, read_parquet, write_parquet

# 9-cat z-vector, all oriented higher = better (TOV negated; FG/FT as volume-weighted impact).
ZCOLS = ["z_pts", "z_reb", "z_ast", "z_stl", "z_blk", "z_fg3m", "z_ft", "z_fg", "z_tov"]
CAT_LABEL = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "FT%", "FG%", "TO"]

# Draft strategies = which categories a drafter ignores (punts). Diversity here is what spreads teams
# across the composition space; consensus-only drafting yields all-balanced rosters and no contrast.
PUNTS = {
    "balanced": [], "punt_ft": ["z_ft"], "punt_fg": ["z_fg"], "punt_ast": ["z_ast"],
    "punt_blk": ["z_blk"], "punt_stl": ["z_stl"], "punt_fg3m": ["z_fg3m"], "punt_tov": ["z_tov"],
    "punt_pts": ["z_pts"], "punt_reb": ["z_reb"], "punt_ft_ast": ["z_ft", "z_ast"],
    "punt_blk_stl": ["z_blk", "z_stl"], "punt_fg3m_ft": ["z_fg3m", "z_ft"],
}


def player_value_table(player_seasons):
    """Per-season standardized 9-cat value for eligible players (impact-based FG%/FT%, TOV negated)."""
    import pandas as pd

    df = player_seasons[player_seasons["eligible"].astype(bool)].copy()
    out = []
    for _, g in df.groupby("season"):
        g = g.copy()
        fg_lg = g["fgm_pg"].sum() / g["fga_pg"].sum()
        ft_lg = g["ftm_pg"].sum() / g["fta_pg"].sum()
        raw = {
            "z_pts": g["pts_pg"], "z_reb": g["reb_pg"], "z_ast": g["ast_pg"], "z_stl": g["stl_pg"],
            "z_blk": g["blk_pg"], "z_fg3m": g["fg3m_pg"], "z_tov": -g["tov_pg"],
            "z_ft": g["fta_pg"] * (g["ft_pct"] - ft_lg),   # FT% impact = volume * deviation
            "z_fg": g["fga_pg"] * (g["fg_pct"] - fg_lg),   # FG% impact
        }
        for k, v in raw.items():
            sd = v.std(ddof=0)
            g[k] = (v - v.mean()) / sd if sd else 0.0
        g["value"] = g[ZCOLS].sum(axis=1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def attach_prior(values):
    """Join each player-season to its **season N-1** z-vector (the ex-ante draft prior).

    Inner join → only **returning players** are draftable (v1 scope; rookies are a separate model).
    """
    prev = values[["athlete_id", "season", *ZCOLS, "value"]].copy()
    prev["season"] = prev["season"] + 1
    prev = prev.rename(columns={**{c: f"prior_{c}" for c in ZCOLS}, "value": "prior_value"})
    return values.merge(prev, on=["athlete_id", "season"], how="inner")


def strategy_weights(name):
    """Category weight vector for a draft strategy (punted cats -> 0)."""
    w = np.ones(len(ZCOLS))
    for c in PUNTS[name]:
        w[ZCOLS.index(c)] = 0.0
    return w


def simulate_draft(pool, n_teams, n_rounds, team_strats, team_temps, rng):
    """Snake draft: each team picks by its strategy's weighting of players' **prior** z-vectors.

    ``team_temps`` is the per-team softmax temperature over the top available — **low** = rigid
    best-available (auto-draft follows the ranking), **higher** = a manager reaching/varying. This is
    how the auto-draft-vs-strategize behavior mix enters. Returns ``{team_idx: [athlete_id, ...]}``.
    """
    prior = pool[[f"prior_{c}" for c in ZCOLS]].to_numpy(dtype=float)
    ids = pool["athlete_id"].to_numpy()
    available = np.ones(len(pool), dtype=bool)
    weights = [strategy_weights(s) for s in team_strats]
    picks = {t: [] for t in range(n_teams)}
    order = list(range(n_teams))
    for rnd in range(n_rounds):
        for t in (order if rnd % 2 == 0 else order[::-1]):
            score = np.where(available, prior @ weights[t], -np.inf)
            top = np.argsort(score)[::-1][:15]
            top = top[np.isfinite(score[top])]
            s = score[top]
            p = np.exp((s - s.max()) / team_temps[t])
            choice = top[rng.choice(len(top), p=p / p.sum())]
            picks[t].append(ids[choice])
            available[choice] = False
    return picks


def team_outcomes(picks, pool, n_teams):
    """Team 9-cat vectors (higher = better) from **actual** season totals (per-game * games played)."""
    by_id = pool.set_index("athlete_id")
    vecs = np.zeros((n_teams, 9))
    for t, ids in picks.items():
        g = by_id.loc[ids]
        gp = g["gp"].to_numpy(dtype=float)

        def tot(col, gp=gp, g=g):
            return float((g[col].to_numpy(dtype=float) * gp).sum())

        fga, ftm, fta = tot("fga_pg"), tot("ftm_pg"), tot("fta_pg")
        fgm = tot("fgm_pg")
        vecs[t] = [tot("pts_pg"), tot("reb_pg"), tot("ast_pg"), tot("stl_pg"), tot("blk_pg"),
                   tot("fg3m_pg"), (ftm / fta if fta else 0.0), (fgm / fga if fga else 0.0),
                   -tot("tov_pg")]
    return vecs


def team_coverage(picks, pool, n_teams):
    """Per-team 9-cat **coverage** = roster-summed standardized contributions, in two variants.

    ``actual`` sums each player's **season-N** z-vector (``ZCOLS``); ``prior`` sums the **season-N-1**
    z-vector (``prior_*``) — the latter is leak-safe (knowable at draft) and is what the optimizer can
    target. Coverage is the *category* representation the shootout shows carries the signal (archetype
    shares discard it). Rosters are equal-sized, so sum vs mean is a constant rescale (same model).
    Returns ``(actual[n_teams, 9], prior[n_teams, 9])`` aligned to ``ZCOLS``.
    """
    by_id = pool.set_index("athlete_id")
    acols, pcols = ZCOLS, [f"prior_{c}" for c in ZCOLS]
    actual, prior = np.zeros((n_teams, 9)), np.zeros((n_teams, 9))
    for t, ids in picks.items():
        g = by_id.loc[ids]
        actual[t] = g[acols].to_numpy(dtype=float).sum(axis=0)
        prior[t] = g[pcols].to_numpy(dtype=float).sum(axis=0)
    return actual, prior


def cat_win_rates(vecs):
    """Round-robin category-win rate per team (ties = 0.5); league mean is 0.5 by construction."""
    n = len(vecs)
    wins = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            wins[i] += (vecs[i] > vecs[j]).sum() + 0.5 * (vecs[i] == vecs[j]).sum()
    return wins / (9 * (n - 1))


def _assign_behaviors(n_teams, autodraft_frac, autodraft_temp, manager_temp, rng):
    """A realistic mix per league: some teams **auto-draft** (rigid consensus), the rest **manage**.

    Auto-drafters take ``balanced`` at low temperature (follow the default ranking); managers sample a
    strategy (incl. punt builds) at a higher temperature. Returns ``(strategies, temps, behaviors)``.
    """
    manager_strats = list(PUNTS)
    strats, temps, behaviors = [], [], []
    for _ in range(n_teams):
        if rng.random() < autodraft_frac:
            strats.append("balanced")
            temps.append(autodraft_temp)
            behaviors.append("autodraft")
        else:
            strats.append(str(rng.choice(manager_strats)))
            temps.append(manager_temp)
            behaviors.append("manager")
    return strats, temps, behaviors


def simulate_leagues(values_prior, *, seasons, n_leagues, n_teams=12, n_rounds=13, seed=1729,
                     autodraft_frac=0.3, autodraft_temp=0.04, manager_temp=0.12):
    """Run ``n_leagues`` drafts per season with a **mix of auto-draft and managed teams**.

    ``autodraft_frac`` ≈ share of teams that auto-draft (rigid best-available); the rest pick a
    strategy (incl. punts). Returns tidy rosters + labels frames (labels carry ``strategy`` +
    ``behavior``).
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    roster_rows, label_rows = [], []
    for season in seasons:
        pool = values_prior[values_prior["season"] == season]
        if len(pool) < n_teams * n_rounds:
            continue
        for lg in range(n_leagues):
            strats, temps, behaviors = _assign_behaviors(n_teams, autodraft_frac, autodraft_temp,
                                                          manager_temp, rng)
            picks = simulate_draft(pool, n_teams, n_rounds, strats, temps, rng)
            rates = cat_win_rates(team_outcomes(picks, pool, n_teams))
            cov_act, cov_pri = team_coverage(picks, pool, n_teams)
            sim_id = f"{season}_L{lg:03d}"
            for t in range(n_teams):
                row = {"sim_league_id": sim_id, "season": int(season), "team_id": t,
                       "strategy": strats[t], "behavior": behaviors[t],
                       "sim_cat_win_rate": round(float(rates[t]), 4)}
                for j, c in enumerate(ZCOLS):
                    row[f"cov_act_{c[2:]}"] = round(float(cov_act[t, j]), 4)
                    row[f"cov_pri_{c[2:]}"] = round(float(cov_pri[t, j]), 4)
                label_rows.append(row)
                for aid in picks[t]:
                    roster_rows.append({"sim_league_id": sim_id, "season": int(season),
                                        "team_id": t, "athlete_id": aid})
    return pd.DataFrame(roster_rows), pd.DataFrame(label_rows)


def simulated_composition(rosters, membership):
    """Archetype soft shares + hard counts per simulated team, joined by ``athlete_id`` (no name match)."""
    import pandas as pd

    pcols = [c for c in membership.columns if c.startswith("p") and c[1:].isdigit()]
    arch_name = membership.drop_duplicates("arch").set_index("arch")["arch_name"].to_dict()
    lut = membership.drop_duplicates(["season", "athlete_id"]).set_index(["season", "athlete_id"])

    r = rosters.copy()
    r["_key"] = list(zip(r["season"], r["athlete_id"]))
    r["_matched"] = r["_key"].isin(lut.index)
    out = []
    for (sid, tid), g in r.groupby(["sim_league_id", "team_id"]):
        gm = g[g["_matched"]]
        rec = {"sim_league_id": sid, "team_id": tid, "season": int(g["season"].iloc[0]),
               "n_roster": len(g), "n_matched": len(gm)}
        if len(gm):
            probs = lut.loc[gm["_key"]][pcols].to_numpy()
            shares = probs.mean(axis=0)
            hard = probs.argmax(axis=1)
            for j, _ in enumerate(pcols):
                nm = arch_name.get(j, f"A{j}")
                rec[f"comp_{nm}"] = round(float(shares[j]), 3)
                rec[f"n_{nm}"] = int((hard == j).sum())
        out.append(rec)
    return pd.DataFrame(out)


def strategy_leaderboard(labels):
    """Mean ``sim_cat_win_rate`` per draft **strategy** with a z-stat vs the 0.5 league mean.

    The signal lives in the **punt builds**: if category coverage matters, some punt strategies should
    clear 0.5 by many standard errors while others fall below. ``z = (mean - 0.5)/(std/sqrt(n))``.
    Returns rows sorted best-first.
    """
    rows = []
    for strat, g in labels.groupby("strategy"):
        n = len(g)
        mean = float(g["sim_cat_win_rate"].mean())
        sd = float(g["sim_cat_win_rate"].std(ddof=1)) if n > 1 else 0.0
        z = (mean - 0.5) / (sd / np.sqrt(n)) if sd and n > 1 else 0.0
        rows.append({"strategy": strat, "n": n, "mean_cat_win_rate": round(mean, 4),
                     "z_vs_0.5": round(z, 2)})
    rows.sort(key=lambda r: r["mean_cat_win_rate"], reverse=True)
    return rows


def build_simulated_table(*, seasons, n_leagues=60, n_teams=12, n_rounds=13, seed=1729,
                          write: bool = True):
    """End-to-end: value -> prior -> simulate -> compose -> join labels. Returns the modeling table."""
    ps = read_parquet(DATA_INTERIM / "nba_player_seasons.parquet")
    membership = read_parquet(DATA_PROCESSED / "nba_archetype_membership.parquet")
    values_prior = attach_prior(player_value_table(ps))

    rosters, labels = simulate_leagues(values_prior, seasons=seasons, n_leagues=n_leagues,
                                       n_teams=n_teams, n_rounds=n_rounds, seed=seed)
    comp = simulated_composition(rosters, membership)
    tbl = comp.merge(labels, on=["sim_league_id", "team_id", "season"], how="left")
    if write and not tbl.empty:
        write_parquet(tbl, DATA_PROCESSED / "phase2_simulated_composition.parquet")
    return tbl, rosters, labels
