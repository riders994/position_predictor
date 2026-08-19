"""Stage 5: college QB archetypes, discovered rather than asserted.

The project needs a *kind* of quarterback, not a ranking of quarterbacks. That distinction drives
every choice in this module.

Style, not quality
------------------
The obvious thing to cluster on is EPA per dropback, success rate and touchdown rate. Doing so
produces clusters ordered from good to bad, which is not a taxonomy — it is a leaderboard with
four bins, and stage 6 already has a model for quality. So the clustering features are restricted
to four **style rates** that describe how a quarterback plays rather than how well:

``rush_share``            mobility as a share of offensive workload
``rush_yds_per_att``      whether the running is effective or merely frequent
``rush_td_share``         how the offense scores with him — goal-line role
``yards_per_completion``  depth of target — the closest proxy available without air yards

Quality columns are held out of the fit **on purpose** and then used to profile the result. If the
clusters were secretly a quality ranking, EPA would separate them cleanly; if they are styles, EPA
should vary *within* each of them. That is a check the taxonomy can fail, and it is the check that
removed ``completion_pct`` from this list. Measured as the share of (within-season) EPA variance
falling *between* archetypes at k=4:

===============================================  =====
Clustering on quality stats (a control)           0.56
The four style rates **plus** ``completion_pct``  0.31
The four style rates above                        0.14
Mobility alone                                    0.03
===============================================  =====

Completion percentage is an accuracy measure, but it is also the most quality-loaded rate a
quarterback has, and including it doubled the leakage. It is far more useful as a profile column
and as a direct feature in stage 6, where quality is supposed to live. The taxonomy that remains
explains 14% of quality — not zero, because real styles do differ in average quality, but small
enough that the clusters describe *kinds* rather than *ranks*.

Portability
-----------
All four features are portable-tier (§2.5): they are computed natively and near-identically by
both cfbfastR and CFBD, so an archetype fitted on 2004–2021 can be assigned to a 2025 prospect.
Notably absent is anything per-game — CFBD exposes no games-played for a player season, so a
per-game feature would have made the taxonomy un-assignable to exactly the prospects it exists to
judge.

Era
---
College offense changed enormously across the span (2004 completion rates run 5 points above the
middle of the era). Features are therefore **z-scored within season**, so an archetype means
"mobile relative to his contemporaries", not "played after the spread took over". Without this the
first split found is simply a date.

Unit
----
Clustering is on QB-*seasons*, which is where a style exists; a career then takes the archetype of
its **final** season, matching the ``final_*`` preference established in §4.2 (a truncated career
still has a real final season). ``archetype_stability`` records how much of the career was spent
in that archetype, which is itself a candidate feature — a quarterback who changed style is a
different prospect from one who never did.
"""

from __future__ import annotations

CLUSTER_FEATURES = ("rush_share", "rush_yds_per_att", "rush_td_share", "yards_per_completion")

# Columns deliberately excluded from the fit and used only to profile the result. If a "style"
# taxonomy turns out to separate these cleanly, it is a quality ranking wearing a disguise.
PROFILE_FEATURES = ("completion_pct", "yards_per_attempt", "pass_epa_per_db", "pass_success_rate",
                    "td_rate", "int_rate", "sack_rate", "dropbacks", "games")

# Rate features on thin denominators are the classic source of artefact clusters — a quarterback
# with 60 dropbacks has a completion percentage dominated by noise, and k-means will happily give
# those rows a cluster of their own. The taxonomy is therefore **fitted** on seasons above this
# floor and **assigned** to everything, so no season is lost and none of them define a centroid.
FIT_MIN_DROPBACKS = 150

RANDOM_STATE = 17


def add_style_features(qb_seasons):
    """Add the derived style rates the clustering needs."""
    import polars as pl

    return qb_seasons.with_columns(
        # Depth of target proxy. Yards per *attempt* mixes depth with accuracy; dividing by
        # completions removes the accuracy term and leaves how far the ball travels when caught.
        yards_per_completion=pl.when(pl.col("completions") > 0)
        .then(pl.col("pass_yds") / pl.col("completions"))
        .otherwise(None),
        # Goal-line role: of the touchdowns this quarterback was responsible for, how many he ran
        # in himself. A ratio of his own outputs, so it says how the offense uses him near the
        # end zone without saying how good the offense is.
        rush_td_share=pl.when((pl.col("pass_td") + pl.col("rush_td")) > 0)
        .then(pl.col("rush_td") / (pl.col("pass_td") + pl.col("rush_td")))
        .otherwise(None),
    )


def standardize_within_season(df, features=CLUSTER_FEATURES):
    """Z-score each feature against its own season.

    Era drift in college football is large enough that raw features cluster on date. Standardising
    within season removes the level shift while preserving where a quarterback sat relative to the
    quarterbacks he actually played against.
    """
    import polars as pl

    # A season with one quarterback, or with no spread in a feature, has an undefined z-score.
    # Dividing anyway yields NaN, which is *not* null — it slides through a null check and lands
    # in the clustering as a crash or, worse, as a coordinate.
    return df.with_columns([
        pl.when(pl.col(f).std().over("season") > 0)
        .then((pl.col(f) - pl.col(f).mean().over("season")) / pl.col(f).std().over("season"))
        .otherwise(None)
        .alias(f"z_{f}")
        for f in features
    ])


def fit_archetypes(qb_seasons, *, k, features=CLUSTER_FEATURES,
                   fit_min_dropbacks=FIT_MIN_DROPBACKS, random_state=RANDOM_STATE):
    """Fit k-means on the high-volume seasons and assign every season to a cluster.

    Returns ``(assigned, model)`` where ``assigned`` carries ``archetype`` and
    ``fitted_on`` (1 where the row helped define the centroids).
    """
    import numpy as np
    import polars as pl
    from sklearn.cluster import KMeans

    df = standardize_within_season(add_style_features(qb_seasons), features)
    zcols = [f"z_{f}" for f in features]

    # NaN and null are different things in polars and only one of them is caught by a null check.
    # An input feature can still be NaN here (0/0 upstream), and k-means will not survive it.
    usable = pl.all_horizontal(
        [pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in zcols]
    )
    df = df.with_columns(
        fitted_on=(usable & (pl.col("dropbacks") >= fit_min_dropbacks)).cast(pl.Int8),
        _usable=usable,
    )

    train = df.filter(pl.col("fitted_on") == 1).select(zcols).to_numpy()
    model = KMeans(n_clusters=k, n_init=25, random_state=random_state).fit(train)

    scored = df.filter(pl.col("_usable"))
    labels = model.predict(scored.select(zcols).to_numpy())
    scored = scored.with_columns(archetype=pl.Series(np.asarray(labels, dtype=np.int32)))

    out = df.join(
        scored.select(["season", "team", "player", "archetype"]),
        on=["season", "team", "player"], how="left",
    ).drop("_usable")

    names = name_archetypes(out, features=features)
    return out.with_columns(
        archetype_name=pl.col("archetype").replace_strict(
            {k: v[0] for k, v in names.items()}, default=None, return_dtype=pl.String),
        archetype_label=pl.col("archetype").replace_strict(
            {k: v[1] for k, v in names.items()}, default=None, return_dtype=pl.String),
    ), model


def choose_k(qb_seasons, *, candidates=range(2, 11), features=CLUSTER_FEATURES,
             fit_min_dropbacks=FIT_MIN_DROPBACKS, random_state=RANDOM_STATE, sample=1500):
    """Score candidate k by silhouette and by how unbalanced the smallest cluster is.

    Silhouette alone will happily recommend k=2 on data with no gaps, which is the usual outcome
    for continuous style space — so the smallest-cluster share is reported alongside it. A
    taxonomy whose smallest archetype holds 2% of seasons is not describing a type of
    quarterback, it is describing outliers.
    """
    import numpy as np
    import polars as pl
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    df = standardize_within_season(add_style_features(qb_seasons), features)
    zcols = [f"z_{f}" for f in features]
    train = df.filter(
        pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in zcols])
        & (pl.col("dropbacks") >= fit_min_dropbacks)
    ).select(zcols).to_numpy()

    rng = np.random.default_rng(random_state)
    idx = (rng.choice(len(train), sample, replace=False) if len(train) > sample
           else np.arange(len(train)))

    rows = []
    for k in candidates:
        model = KMeans(n_clusters=k, n_init=25, random_state=random_state).fit(train)
        counts = np.bincount(model.labels_, minlength=k)
        rows.append({
            "k": int(k),
            "silhouette": round(float(silhouette_score(train[idx], model.labels_[idx])), 4),
            "inertia": round(float(model.inertia_), 1),
            "smallest_share": round(float(counts.min() / counts.sum()), 3),
        })
    return rows


# The two axes the taxonomy actually turns on. Names are derived from where a centroid sits on
# them rather than from the k-means label integer, because label integers are an artefact of the
# seed: refit the model and cluster 2 becomes cluster 0, silently renaming every archetype in
# every downstream artefact. Position-derived names survive a refit.
_MOBILITY_AXIS = "z_rush_share"
_DEPTH_AXIS = "z_yards_per_completion"

_QUADRANT_NAMES = {
    (False, False): ("pocket_quick", "Quick-Game Pocket Passer"),
    (False, True): ("pocket_downfield", "Downfield Pocket Passer"),
    (True, False): ("runner_quick", "Short-Game Runner"),
    (True, True): ("runner_downfield", "Downfield Dual-Threat"),
}


def name_archetypes(assigned, *, features=CLUSTER_FEATURES):
    """Map cluster ids to stable names derived from centroid position.

    At k=4 the clusters fall into one quadrant each of mobility x depth — a 2x2 the algorithm was
    never told to look for, which is the strongest evidence available that these axes are real. If
    a refit puts two centroids in the same quadrant, the more extreme one keeps the plain name and
    the other is suffixed, so a name is never silently reused for a different thing.

    Returns a dict of ``{archetype_id: (slug, label)}``.
    """
    import polars as pl

    centroids = (
        assigned.filter(pl.col("archetype").is_not_null())
        .group_by("archetype")
        .agg(mob=pl.col(_MOBILITY_AXIS).mean(), depth=pl.col(_DEPTH_AXIS).mean())
        .sort("archetype")
    )

    by_quadrant: dict[tuple[bool, bool], list[tuple[float, int]]] = {}
    for row in centroids.to_dicts():
        quad = (row["mob"] > 0, row["depth"] > 0)
        # Distance from the origin: within a quadrant, the more extreme centroid is the more
        # typical example of that name.
        extremity = abs(row["mob"]) + abs(row["depth"])
        by_quadrant.setdefault(quad, []).append((-extremity, int(row["archetype"])))

    names = {}
    for quad, members in by_quadrant.items():
        slug, label = _QUADRANT_NAMES[quad]
        for rank, (_, cid) in enumerate(sorted(members)):
            names[cid] = (slug, label) if rank == 0 else (f"{slug}_{rank + 1}", f"{label} ({rank + 1})")
    return names


def profile_archetypes(assigned, *, features=CLUSTER_FEATURES, profile=PROFILE_FEATURES):
    """Summarise each archetype on its clustering features and on the held-out quality columns."""
    import polars as pl

    cols = [f for f in (*features, *profile) if f in assigned.columns]
    return (
        assigned.filter(pl.col("archetype").is_not_null())
        .group_by("archetype")
        .agg(
            n=pl.len(),
            **{c: pl.col(c).median().round(3) for c in cols},
        )
        .sort("archetype")
    )


def quality_leakage(assigned, columns=("pass_epa_per_db", "pass_success_rate", "td_rate")):
    """Share of within-season quality variance that falls *between* archetypes (eta squared).

    The taxonomy claims to describe style rather than rank. This is the measurement that claim
    can fail: near 0 means quality varies freely inside every archetype (styles), near 1 means the
    archetypes *are* the quality ordering (a leaderboard). Quality is z-scored within season first,
    so era drift is not counted as a difference between archetypes.
    """
    import numpy as np
    import polars as pl

    df = assigned.filter(pl.col("archetype").is_not_null() & (pl.col("fitted_on") == 1))
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        z = df.with_columns(
            _z=(pl.col(col) - pl.col(col).mean().over("season"))
            / pl.col(col).std().over("season")
        ).select(["archetype", "_z"]).drop_nulls()
        y, g = z["_z"].to_numpy(), z["archetype"].to_numpy()
        grand = y.mean()
        between = sum(len(y[g == c]) * (y[g == c].mean() - grand) ** 2 for c in np.unique(g))
        total = float(((y - grand) ** 2).sum())
        rows.append({"held_out_column": col, "n": int(len(y)),
                     "eta_squared": round(between / total, 3) if total else None})
    return rows


def career_archetypes(assigned):
    """Collapse per-season archetypes into one row per college career.

    ``archetype`` is the **final** season's, matching the project's preference for ``final_*``
    features: a career clipped by the 2004 coverage floor still has a genuine last season, and it
    is the season NFL evaluators weighted most.
    """
    import polars as pl

    df = assigned.filter(pl.col("archetype").is_not_null()).sort(["player", "season"])
    return (
        df.group_by("player")
        .agg(
            archetype=pl.col("archetype").last(),
            archetype_name=pl.col("archetype_name").last(),
            archetype_label=pl.col("archetype_label").last(),
            archetype_first=pl.col("archetype").first(),
            archetype_modal=pl.col("archetype").mode().first(),
            n_archetype_seasons=pl.len(),
            archetype_stability=(
                pl.col("archetype").eq(pl.col("archetype").last()).mean()
            ),
        )
        .with_columns(
            archetype_changed=(pl.col("archetype") != pl.col("archetype_first")).cast(pl.Int8),
        )
    )
