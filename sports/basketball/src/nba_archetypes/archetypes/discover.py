"""Phase 1 — archetype discovery via soft Gaussian-mixture clustering.

Fits a GMM on the ``*_z`` style features for **eligible E2+E3** player-seasons (the modern game),
then **assigns** soft membership to *all* eligible seasons (incl. E1) so earlier years carry
archetypes for Phases 2/3 (PROJECT_PLAN §2.2). Style space is continuous (low silhouette), so the
output is **soft membership** (a probability vector per player-season) — the representation of record,
consumed directly by Phase 2 (soft shares) and Phase 3 (soft features + calibrated soft target). A hard
``arch`` (argmax) + names are an interpretive view only. The originally-planned soft→hard *consolidation*
into a discrete taxonomy was evaluated and **decided against**: the space is genuinely continuous (~half
of player-seasons are blends), so collapsing to hard buckets would discard signal the pipeline uses.
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

    from sklearn.decomposition import PCA

    df, _, zcols = _load()
    cov = config.get("archetypes.covariance_type", "full")
    n_init = int(config.get("archetypes.n_init", 5))
    seed = int(config.get("reproducibility.random_seed", 1729))
    pca_var = config.get("archetypes.pca_variance", 0.90)
    lo, hi = k_range or config.get("archetypes.k_range", [8, 16])
    X = df[df.eligible & df.era_use_for_archetypes][zcols].fillna(0.0).to_numpy()
    Z = PCA(n_components=pca_var, whiten=True, random_state=seed).fit_transform(X)
    rows = []
    for k in range(int(lo), int(hi) + 1):
        gm = GaussianMixture(k, covariance_type=cov, random_state=seed, n_init=n_init,
                             max_iter=400).fit(Z)
        lab = gm.predict(Z)
        rows.append({"k": k, "bic": round(gm.bic(Z), 0),
                     "silhouette": round(float(silhouette_score(Z, lab)), 3)})
    return pd.DataFrame(rows)


@dataclass
class ArchetypeResult:
    membership: object        # per-eligible-season: identity + arch + name + probs + entropy
    profiles: object          # arch x feature mean-z (over the fit pool) + size
    names: dict
    k: int
    bic: float
    softness: dict            # median_top_prob, pct_blends (genuinely-soft check)
    stability: dict           # YoY persistence: overall, n_pairs, per_archetype


def archetype_stability(mem) -> dict:
    """Year-over-year hard-archetype persistence (validation + the Phase-3 must-beat baseline)."""
    m = mem.sort_values(["athlete_id", "season"]).copy()
    m["next_arch"] = m.groupby("athlete_id")["arch"].shift(-1)
    m["next_season"] = m.groupby("athlete_id")["season"].shift(-1)
    pairs = m[(m["next_season"] == m["season"] + 1) & m["next_arch"].notna()]
    if pairs.empty:
        return {"overall": float("nan"), "n_pairs": 0, "per_archetype": {}}
    per = pairs.groupby("arch_name").apply(
        lambda g: (g["arch"] == g["next_arch"]).mean(), include_groups=False)
    return {"overall": round(float((pairs["arch"] == pairs["next_arch"]).mean()), 3),
            "n_pairs": int(len(pairs)),
            "per_archetype": per.round(3).sort_values().to_dict()}


def discover_archetypes(config, *, k=None, write: bool = True) -> ArchetypeResult:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    df, block_map, zcols = _load()
    cov = config.get("archetypes.covariance_type", "full")
    n_init = int(config.get("archetypes.n_init", 5))
    seed = int(config.get("reproducibility.random_seed", 1729))
    pca_var = config.get("archetypes.pca_variance", 0.90)
    k = int(k or config.get("archetypes.k", 12))
    names = {int(i): n for i, n in (config.get("archetypes.names", {}) or {}).items()}

    fit = df[df.eligible & df.era_use_for_archetypes]
    assign = df[df.eligible].copy()                    # all eligible eras get assigned
    Xf = fit[zcols].fillna(0.0).to_numpy()
    Xa = assign[zcols].fillna(0.0).to_numpy()

    # PCA-whiten (decorrelate collinear style features) -> genuinely soft, more stable membership
    pca = PCA(n_components=pca_var, whiten=True, random_state=seed).fit(Xf)
    Zf, Za = pca.transform(Xf), pca.transform(Xa)
    gm = GaussianMixture(k, covariance_type=cov, random_state=seed, n_init=n_init,
                         max_iter=400).fit(Zf)
    probs = gm.predict_proba(Za)
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
    fit2["arch"] = gm.predict(Zf)
    prof = fit2.groupby("arch")[zcols].mean().round(2)
    prof.insert(0, "size", fit2.groupby("arch").size())
    prof.insert(1, "name", [names.get(int(a), f"A{a}") for a in prof.index])

    softness = {"median_top_prob": round(float(mem["top_prob"].median()), 3),
                "pct_blends": round(float((mem["top_prob"] < 0.8).mean()), 3)}
    result = ArchetypeResult(mem, prof, names, k, gm.bic(Zf), softness,
                             archetype_stability(mem))

    if write:
        ensure_dir(DATA_PROCESSED)
        write_parquet(mem, DATA_PROCESSED / "nba_archetype_membership.parquet")
        ensure_dir(REPORTS_DIR / "results")
        prof.to_csv(REPORTS_DIR / "results" / "archetype_profiles.csv")
        (REPORTS_DIR / "REPORT_archetypes.md").write_text(render_report(result, zcols, fit2))
    return result


def _signature(prof_row, zcols, n=3):
    """Top +/- distinguishing features for an archetype (drops the 'size'/'name' cols)."""
    s = prof_row[zcols].astype(float).sort_values()
    hi = ", ".join(f"{c[:-2]} +{s[c]:.1f}" for c in s.index[-n:][::-1])
    lo = ", ".join(f"{c[:-2]} {s[c]:.1f}" for c in s.index[:2])
    return hi, lo


def render_report(result: ArchetypeResult, zcols, fit_assigned) -> str:
    L = [f"# NBA Archetypes (Phase 1) — k={result.k} soft GMM (PCA-whitened)", ""]
    L.append("_Soft Gaussian-mixture on **PCA-whitened** z-scored play-STYLE features (per-36 rates "
             "+ shot profile + tendencies), fit on eligible **E2+E3** (modern game) and assigned to "
             "all eligible 2013+ seasons. Whitening decorrelates the collinear style features so "
             "membership is genuinely **soft** — and soft membership is what Phases 2/3 consume. The "
             "hard label is argmax over the vector, an interpretive shorthand only; a discrete soft→hard "
             "consolidation was **evaluated and decided against** (continuous space, ~half blends)._")
    L.append("")
    sf, st = result.softness, result.stability
    L.append(f"Fit pool: **{int(result.profiles['size'].sum())}** player-seasons · "
             f"BIC {result.bic:.0f}. Assigned (all eligible): **{len(result.membership)}**.")
    L.append(f"Softness: median top_prob **{sf['median_top_prob']}**, "
             f"**{sf['pct_blends']:.0%}** of player-seasons are blends (top_prob < 0.8).")
    L.append(f"Stability: **{st['overall']:.0%}** keep their archetype year-over-year "
             f"(N={st['n_pairs']} consecutive pairs; vs ~{1/result.k:.0%} random).")
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
    L.append("## Cross-season stability (YoY persistence)")
    L.append("")
    L.append("Share of players who keep an archetype the next season — validation, and the "
             "must-beat baseline for the Phase-3 predictor. Distinctive roles are stickiest; the "
             "low-signal middle churns most — expected for a continuous style space (players drift "
             "across soft boundaries), not a defect to consolidate away.")
    L.append("")
    per = st["per_archetype"]
    if per:
        L.append("| archetype | YoY persistence |")
        L.append("|---|---|")
        for name in sorted(per, key=per.get, reverse=True):
            L.append(f"| {name} | {per[name]:.2f} |")
    L.append("")
    L.append("## Notes")
    L.append("- Membership table (`data/processed/nba_archetype_membership.parquet`) carries the full "
             "probability vector `p0..pK-1` + `entropy` (blend-iness) per player-season — the soft "
             "input for Phase 2 (composition) and Phase 3 (the predictor target).")
    L.append("- PCA-whitening decorrelates the collinear style features → genuinely soft membership "
             "and higher YoY stability than a raw full-cov GMM.")
    L.append("- **Soft→hard consolidation: evaluated and decided against.** The style space is "
             "continuous (low silhouette, ~half of player-seasons are blends) and every consumer "
             "already uses the soft vector (Phase 2 soft shares; Phase 3 soft features + calibrated "
             "soft output), so collapsing to a discrete hard taxonomy would discard information the "
             "pipeline uses. Hard labels/names remain only as interpretive shorthand.")
    return "\n".join(L) + "\n"
