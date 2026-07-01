"""Phase-2 **roster optimizer** — build the draft roster that maximizes projected 9-cat coverage.

PROJECT_PLAN §4 deliverable #3. The simulation shootout established that **category coverage** (not
archetype shares) is what wins 9-cat, and that the binding constraint is **draft-time projection**: a
player's prior-season z-profile is only a noisy estimate of next season's. So the optimizer works on the
*projected* (prior-season) coverage and targets the quantity that actually decides matchups — the number
of **categories won**, not raw total z.

Objective. In a round-robin/H2H league a team wins category ``c`` roughly in proportion to how far its
coverage sits above the field. Modeling the field's per-category team coverage as ``N(mu_c, sigma_c)``,
the expected win rate in ``c`` is ``Phi((cov_c - mu_c)/sigma_c)`` and the roster's value is the mean of
that over the **contested** categories (a punt build drops its punted categories from the mean). The
Gaussian-CDF shape is what makes the objective *punt-aware*: once a category is locked (``cov_c`` far
above the field) extra investment barely moves ``Phi``, so the greedy builder reallocates to winnable
categories instead of piling on — the 9-cat draft logic a linear "sum of z" objective misses.

Honesty. The projection ceiling is low (prior coverage tracks next-season success only weakly), so the
optimizer's realistic edge is **modest** — the real test is the in-sim check (``evaluate_in_sim``): does a
team drafting by this objective, in a snake draft against the manager/auto field, actually win more
categories? That end-to-end lift, not the objective's standalone calibration, is the proof.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .simulate import (
    PUNTS, ZCOLS, _assign_behaviors, cat_win_rates, simulate_draft, team_outcomes)

# 9-cat suffixes aligned to ZCOLS (z_pts -> pts, ...), matching the sim's cov_* / label columns.
CATS = [c[2:] for c in ZCOLS]


def _punt_cats(punts):
    """Resolve ``punts`` (a strategy name like ``"punt_ft"``, or an iterable of cat suffixes/strategy
    names) to the set of conceded category suffixes."""
    if not punts:
        return set()
    items = [punts] if isinstance(punts, str) else list(punts)
    out = set()
    for p in items:
        if p in PUNTS:
            out |= {c[2:] for c in PUNTS[p]}
        elif p in CATS:
            out.add(p)
    return out


def _contested_mask(punts):
    """Boolean mask over ``CATS`` of the categories a build actively contests (not punted)."""
    pc = _punt_cats(punts)
    return np.array([c not in pc for c in CATS])


def field_coverage_stats(table, kind="cov_act"):
    """Per-category field mean/sd of team coverage — the opponent distribution the optimizer beats.

    ``kind='cov_act'`` uses realized coverage (the true field the roster is scored against). Returns
    ``(mu[9], sigma[9])`` aligned to ``CATS``.
    """
    arr = table[[f"{kind}_{c}" for c in CATS]].to_numpy(dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


def expected_category_wins(cov, mu, sigma, punts=()):
    """Per-category expected win prob ``Phi((cov-mu)/sigma)`` plus contested/overall means.

    Returns ``(per_cat[9], mean_contested, mean_overall)``. ``mean_contested`` is the objective the
    optimizer maximizes; ``mean_overall`` is the honest expected cat-win-rate across all 9 (punted
    categories included at their low natural win prob).
    """
    per_cat = norm.cdf((np.asarray(cov, dtype=float) - mu) / sigma)
    contested = _contested_mask(punts)
    return per_cat, float(per_cat[contested].mean()), float(per_cat.mean())


def coverage_picker(pool, mu, sigma, punts=()):
    """A stateful draft policy: greedily add the available player that most raises contested-cat wins.

    Returns ``pick(available_bool_array) -> pool-row index``, tracking its own running coverage. Uses the
    players' **prior** (projected) z-vectors — the draft-time information. Plugs into
    ``simulate_draft(..., pickers={t: coverage_picker(...)})``.
    """
    prior = pool[[f"prior_{c}" for c in ZCOLS]].to_numpy(dtype=float)
    contested = _contested_mask(punts)
    state = {"cov": np.zeros(len(CATS))}

    def pick(available):
        cand = state["cov"][None, :] + prior                      # coverage if we add each player
        val = norm.cdf((cand - mu) / sigma)[:, contested].mean(axis=1)
        val = np.where(available, val, -np.inf)
        idx = int(np.argmax(val))
        state["cov"] = cand[idx]
        return idx

    return pick


def optimize_roster(pool, mu, sigma, *, n_rounds, punts=(), taken=()):
    """Greedy best roster of ``n_rounds`` players from ``pool`` for a chosen (punt) build.

    ``pool`` needs ``athlete_id`` + ``prior_*`` z columns (a season slice of ``attach_prior`` output).
    ``taken`` = athlete_ids already drafted (excluded). Returns a dict with the picked ``athlete_ids``,
    the projected coverage vector, and the expected per-category / contested / overall win rates.
    """
    ids = pool["athlete_id"].to_numpy()
    available = ~np.isin(ids, np.asarray(list(taken)))
    pick = coverage_picker(pool, mu, sigma, punts)
    chosen = []
    for _ in range(n_rounds):
        if not available.any():
            break
        idx = pick(available)
        available[idx] = False
        chosen.append(int(ids[idx]))
    proj = pool[pool["athlete_id"].isin(chosen)][[f"prior_{c}" for c in ZCOLS]].to_numpy().sum(axis=0)
    per_cat, contested_mean, overall = expected_category_wins(proj, mu, sigma, punts)
    return {"athlete_ids": chosen,
            "proj_coverage": {c: round(float(proj[j]), 3) for j, c in enumerate(CATS)},
            "exp_win_by_cat": {c: round(float(per_cat[j]), 3) for j, c in enumerate(CATS)},
            "exp_contested": round(contested_mean, 4), "exp_overall": round(overall, 4)}


def evaluate_in_sim(values_prior, mu, sigma, *, seasons, n_leagues, punts, n_teams=12, n_rounds=13,
                    seed=1729, autodraft_frac=0.3, autodraft_temp=0.04, manager_temp=0.12):
    """In-sim proof: seat one **optimizer** team (given punt build) in each league vs the manager/auto
    field, and compare realized ``cat_win_rate``.

    Returns ``(opt_rates, field_rates, summary)``; summary has the optimizer mean, field mean, lift, and
    the share of leagues the optimizer finishes top-of-field. The optimizer takes a random snake slot
    each league; opponents draft with the usual behavior mix.
    """
    rng = np.random.default_rng(seed)
    opt_rates, field_rates, wins = [], [], 0
    for season in seasons:
        pool = values_prior[values_prior["season"] == season]
        if len(pool) < n_teams * n_rounds:
            continue
        for _ in range(n_leagues):
            strats, temps, _ = _assign_behaviors(n_teams, autodraft_frac, autodraft_temp,
                                                 manager_temp, rng)
            opt_slot = int(rng.integers(n_teams))
            pickers = {opt_slot: coverage_picker(pool, mu, sigma, punts)}
            picks = simulate_draft(pool, n_teams, n_rounds, strats, temps, rng, pickers=pickers)
            rates = cat_win_rates(team_outcomes(picks, pool, n_teams))
            opt_rates.append(float(rates[opt_slot]))
            field = [rates[t] for t in range(n_teams) if t != opt_slot]
            field_rates.extend(float(r) for r in field)
            wins += int(rates[opt_slot] >= max(field))
    opt_rates, field_rates = np.array(opt_rates), np.array(field_rates)
    n = len(opt_rates)
    summary = {"punts": _punt_cats(punts) or "balanced", "n_leagues": n,
               "opt_mean": round(float(opt_rates.mean()), 4) if n else float("nan"),
               "field_mean": round(float(field_rates.mean()), 4) if n else float("nan"),
               "lift": round(float(opt_rates.mean() - field_rates.mean()), 4) if n else float("nan"),
               "top_of_field_rate": round(wins / n, 3) if n else float("nan")}
    return opt_rates, field_rates, summary
