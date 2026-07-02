"""Phase-2 draft intel — which archetypes convert roster VALUE into 9-cat wins (and which are traps).

Round-by-round *archetype selection weights* were tested and **lose in-sim** (balancing across archetypes
sacrifices the value/coverage concentration that actually wins — see REPORT_archetype_value.md). What
survives is this **descriptive** read, a companion to the coverage optimizer (the draft engine), not a
replacement:

Regress simulated ``cat_win_rate`` on each team's **value-weighted archetype exposure** — ``Σ value·p``
over the roster (magnitude kept; *not* normalized shares, since normalizing to proportions was the
Phase-2 null). The per-archetype standardized Ridge coefficient reads as "how efficiently value invested
in this archetype converts to category wins." **Prior-value** (draft-time, leak-safe) coefficients are the
actionable guide; **actual-value** is the post-hoc ceiling. The gap between them is the projection
ceiling from Phase 3 — so treat these as *intel*, not a mechanical edge.
"""
from __future__ import annotations

import numpy as np

from ..utils.io import DATA_INTERIM, DATA_PROCESSED, read_parquet, write_parquet
from .simulate import attach_prior, build_simulated_table, player_value_table


def _prob_cols(membership):
    return sorted((c for c in membership.columns if c.startswith("p") and c[1:].isdigit()),
                  key=lambda c: int(c[1:]))


def value_weighted_exposure(rosters, values, membership):
    """Per fantasy team, the value-weighted archetype **exposure** ``Σ max(value,0)·p`` for both
    prior-season (draft-time) and actual-season value. Returns a tidy frame keyed by team with
    ``expp_<arch>`` / ``expa_<arch>`` columns (arch names) — magnitude kept, not normalized."""
    import pandas as pd

    pcols = _prob_cols(membership)
    names = membership.drop_duplicates("arch").set_index("arch")["arch_name"].to_dict()
    vlut = values.set_index(["athlete_id", "season"])[["value", "prior_value"]]
    mlut = membership.set_index(["athlete_id", "season"])[pcols]
    r = (rosters.merge(vlut, left_on=["athlete_id", "season"], right_index=True, how="left")
                .merge(mlut, left_on=["athlete_id", "season"], right_index=True, how="left")
                .dropna(subset=pcols))
    P = r[pcols].to_numpy(dtype=float)
    out = r[["sim_league_id", "team_id", "season"]].copy()
    frames = [out]
    for weight, prefix in [("prior_value", "expp_"), ("value", "expa_")]:
        w = np.clip(r[weight].to_numpy(dtype=float), 0, None)
        cols = [f"{prefix}{names.get(j, f'A{j}')}" for j in range(len(pcols))]
        frames.append(pd.DataFrame(P * w[:, None], columns=cols, index=r.index))
    tidy = pd.concat(frames, axis=1)
    expcols = [c for c in tidy.columns if c.startswith(("expp_", "expa_"))]
    return tidy.groupby(["sim_league_id", "team_id", "season"], as_index=False)[expcols].sum()


def build_exposure_table(*, seasons, n_leagues=60, seed=1729, write=True):
    """Simulate leagues, compute value-weighted archetype exposure per team, join ``sim_cat_win_rate``."""
    tbl, rosters, _ = build_simulated_table(seasons=seasons, n_leagues=n_leagues, seed=seed, write=False)
    values = attach_prior(player_value_table(read_parquet(DATA_INTERIM / "nba_player_seasons.parquet")))
    membership = read_parquet(DATA_PROCESSED / "nba_archetype_membership.parquet")
    exp = value_weighted_exposure(rosters, values, membership)
    out = tbl[["sim_league_id", "team_id", "sim_cat_win_rate"]].merge(
        exp, on=["sim_league_id", "team_id"], how="inner")
    if write and not out.empty:
        write_parquet(out, DATA_PROCESSED / "phase2_archetype_value_exposure.parquet")
    return out


def exposure_coefficients(table, *, kind="prior", target="sim_cat_win_rate", n_boot=300, seed=1729):
    """Standardized Ridge coefficient per archetype (value→wins efficiency) + season-bootstrap sign
    stability + top-vs-bottom-quartile raw exposure contrast.

    ``kind='prior'`` uses the draft-time (prior-value) exposure; ``'actual'`` the post-hoc ceiling.
    Returns rows sorted by coefficient (best first): archetype, coef_std, sign_stability, top_minus_bottom.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    prefix = "expp_" if kind == "prior" else "expa_"
    cols = [c for c in table.columns if c.startswith(prefix)]
    df = table[table[target].notna()]
    X = df[cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    groups = df["season"].to_numpy()
    alphas = np.logspace(-3, 3, 25)

    def _fit(Xi, yi):
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(
            Xi, yi).named_steps["ridgecv"].coef_

    coef = _fit(X, y)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    signs = np.zeros((n_boot, len(cols)))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        signs[b] = np.sign(_fit(X[idx], y[idx]))
    stability = (np.sign(coef)[None, :] == signs).mean(axis=0)

    q_hi, q_lo = np.quantile(y, 0.75), np.quantile(y, 0.25)
    hi, lo = df[y >= q_hi][cols].mean(), df[y <= q_lo][cols].mean()
    rows = [{"archetype": c[len(prefix):], "coef_std": round(float(coef[j]), 4),
             "sign_stability": round(float(stability[j]), 3),
             "top_minus_bottom": round(float(hi[c] - lo[c]), 2)} for j, c in enumerate(cols)]
    rows.sort(key=lambda r: r["coef_std"], reverse=True)
    return rows


def classify(prior_rows, actual_rows):
    """Label each archetype from the prior (draft-time) + actual coefficients: PRIORITIZE / solid /
    TRAP / avoid. A 'trap' = pays off on actual value but not at draft time (empty-value archetype)."""
    act = {r["archetype"]: r for r in actual_rows}
    out = []
    for r in prior_rows:
        a = act.get(r["archetype"], {})
        p, stab = r["coef_std"], r["sign_stability"]
        if p <= 0 and stab >= 0.9:
            label = "avoid"                               # stably negative draft-time weight
        elif p <= 0.002 and a.get("coef_std", 0) > 0.015:
            label = "trap"                                # good on actual value, wash at draft time
        elif p >= 0.012 and stab >= 0.95:
            label = "PRIORITIZE"
        else:
            label = "solid"
        out.append({**r, "actual_coef": a.get("coef_std"), "label": label})
    return out
