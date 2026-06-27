"""Phase 1 — archetype discovery via soft Gaussian-mixture clustering.

Fits a GMM on the ``*_z`` style features for **eligible E2+E3** player-seasons (the modern game),
then **assigns** soft membership to *all* eligible seasons (incl. E1) so earlier years carry
archetypes for Phases 2/3 (PROJECT_PLAN §2.2). Style space is continuous (low silhouette), so the
output is soft membership (a probability vector per player-season), not just a hard label; a hard
``arch`` (argmax) + provisional names are provided for interpretation. Names are consolidated in the
soft->hard EDA step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..utils.io import DATA_PROCESSED, REPORTS_DIR, ensure_dir, read_parquet, write_parquet

IDENTITY = ["athlete_id", "season", "player_name", "position", "team", "era", "min_pg", "gp"]


def feature_cols(block_map) -> list[str]:
    return [c for cols in block_map.values() for c in cols]


def _load():
    df = read_parquet(DATA_PROCESSED / "nba_archetype_features.parquet")
    block_map = json.load(open(DATA_PROCESSED / "nba_archetype_feature_blocks.json"))
    return df, block_map, feature_cols(block_map)


def select_k(config, k_range=None):
    """BIC + silhouette over a k range on the fit pool (model-selection diagnostics)."""
    import pandas as pd
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture

    df, _, zcols = _load()
    cov = config.get("archetypes.covariance_type", "full")
    n_init = int(config.get("archetypes.n_init", 5))
    seed = int(config.get("reproducibility.random_seed", 1729))
    lo, hi = k_range or config.get("archetypes.k_range", [8, 16])
    X = df[df.eligible & df.era_use_for_archetypes][zcols].fillna(0.0).to_numpy()
    rows = []
    for k in range(int(lo), int(hi) + 1):
        gm = GaussianMixture(k, covariance_type=cov, random_state=seed, n_init=n_init,
                             max_iter=400).fit(X)
        lab = gm.predict(X)
        rows.append({"k": k, "bic": round(gm.bic(X), 0),
                     "silhouette": round(float(silhouette_score(X, lab)), 3)})
    return pd.DataFrame(rows)


@dataclass
class ArchetypeResult:
    membership: object        # per-eligible-season: identity + arch + name + probs + entropy
    profiles: object          # arch x feature mean-z (over the fit pool) + size
    names: dict
    k: int
    bic: float


def discover_archetypes(config, *, k=None, write: bool = True) -> ArchetypeResult:
    import numpy as np
    from sklearn.mixture import GaussianMixture

    df, block_map, zcols = _load()
    cov = config.get("archetypes.covariance_type", "full")
    n_init = int(config.get("archetypes.n_init", 5))
    seed = int(config.get("reproducibility.random_seed", 1729))
    k = int(k or config.get("archetypes.k", 12))
    names = {int(i): n for i, n in (config.get("archetypes.names", {}) or {}).items()}

    fit = df[df.eligible & df.era_use_for_archetypes]
    assign = df[df.eligible].copy()                    # all eligible eras get assigned
    Xf = fit[zcols].fillna(0.0).to_numpy()
    Xa = assign[zcols].fillna(0.0).to_numpy()

    gm = GaussianMixture(k, covariance_type=cov, random_state=seed, n_init=n_init,
                         max_iter=400).fit(Xf)
    probs = gm.predict_proba(Xa)
    hard = probs.argmax(axis=1)

    mem = assign[IDENTITY].copy()
    mem["arch"] = hard
    mem["arch_name"] = [names.get(int(a), f"A{a}") for a in hard]
    mem["top_prob"] = probs.max(axis=1).round(3)
    mem["entropy"] = (-(probs * np.log(probs + 1e-12)).sum(axis=1)).round(3)  # blend-iness
    for c in range(k):
        mem[f"p{c}"] = probs[:, c].round(3)
    mem = mem.reset_index(drop=True)

    # cluster profiles: mean z over the FIT pool (defines what each archetype IS)
    fit2 = fit.copy()
    fit2["arch"] = gm.predict(Xf)
    prof = fit2.groupby("arch")[zcols].mean().round(2)
    prof.insert(0, "size", fit2.groupby("arch").size())
    prof.insert(1, "name", [names.get(int(a), f"A{a}") for a in prof.index])

    if write:
        ensure_dir(DATA_PROCESSED)
        write_parquet(mem, DATA_PROCESSED / "nba_archetype_membership.parquet")
        ensure_dir(REPORTS_DIR / "results")
        prof.to_csv(REPORTS_DIR / "results" / "archetype_profiles.csv")
        (REPORTS_DIR / "REPORT_archetypes.md").write_text(
            render_report(ArchetypeResult(mem, prof, names, k, gm.bic(Xf)), zcols, fit2))
    return ArchetypeResult(mem, prof, names, k, gm.bic(Xf))


def _signature(prof_row, zcols, n=3):
    """Top +/- distinguishing features for an archetype (drops the 'size'/'name' cols)."""
    s = prof_row[zcols].astype(float).sort_values()
    hi = ", ".join(f"{c[:-2]} +{s[c]:.1f}" for c in s.index[-n:][::-1])
    lo = ", ".join(f"{c[:-2]} {s[c]:.1f}" for c in s.index[:2])
    return hi, lo


def render_report(result: ArchetypeResult, zcols, fit_assigned) -> str:
    L = [f"# NBA Archetypes (Phase 1) — k={result.k} soft GMM", ""]
    L.append("_Soft Gaussian-mixture on z-scored play-STYLE features (per-36 rates + shot profile + "
             "tendencies), fit on eligible **E2+E3** (modern game) and assigned to all eligible "
             "2013+ seasons. Style space is continuous, so membership is **soft** (probabilities); "
             "the hard label is argmax. **Names are provisional** (seed-tied) pending soft→hard "
             "consolidation._")
    L.append("")
    L.append(f"Fit pool: **{int(result.profiles['size'].sum())}** player-seasons · "
             f"BIC {result.bic:.0f}. Assigned (all eligible): **{len(result.membership)}**.")
    L.append("")
    L.append("## Archetypes")
    L.append("")
    L.append("| # | archetype | n | signature (top + / −, season-z) | exemplars (most minutes) |")
    L.append("|---|---|---|---|---|")
    for a in result.profiles.index:
        row = result.profiles.loc[a]
        hi, lo = _signature(row, zcols)
        ex = ", ".join(fit_assigned[fit_assigned.arch == a]
                       .sort_values("min_pg", ascending=False)
                       .drop_duplicates("player_name").player_name.head(3))
        L.append(f"| {a} | **{row['name']}** | {int(row['size'])} | +{hi} · −{lo} | {ex} |")
    L.append("")
    L.append("## Notes")
    L.append("- Membership table (`data/processed/nba_archetype_membership.parquet`) carries the full "
             "probability vector `p0..pK-1` + `entropy` (blend-iness) per player-season — the soft "
             "input for Phase 2 (composition) and Phase 3 (the predictor target).")
    L.append("- Continuous style space → low silhouette is expected; soft membership is the design, "
             "not a defect. Next: cross-season stability + soft→hard consolidation/naming.")
    return "\n".join(L) + "\n"
