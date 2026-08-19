"""Descriptive analysis of the late-breakout QB cohort.

Answers the questions this project opens with, before any modelling: which QBs actually did
this, what draft situations they came from, and how deep the trough was before they emerged.
Every function returns a plain DataFrame so the report writer and any notebook share one
implementation.
"""

from __future__ import annotations

# Draft-capital buckets. Round alone is too coarse at the top (pick 1 and pick 32 are very
# different investments) and too fine at the bottom, where round 4-7 all mean "no commitment".
DRAFT_BUCKETS = [
    ("R1 top-10", lambda r, p: r == 1 and p <= 10),
    ("R1 11-32", lambda r, p: r == 1 and p > 10),
    ("R2", lambda r, p: r == 2),
    ("R3", lambda r, p: r == 3),
    ("R4-7", lambda r, p: r >= 4),
]


def draft_bucket(row):
    """Bucket a career row by draft capital; undrafted QBs get their own bucket."""
    rnd, pick = row.get("draft_round"), row.get("draft_pick")
    if rnd is None or rnd != rnd:  # NaN -> undrafted
        return "Undrafted"
    for name, test in DRAFT_BUCKETS:
        if test(int(rnd), int(pick) if pick == pick else 999):
            return name
    return "R4-7"


def label_comparison(careers, labels=("late", "late_qb1", "late_sustained")):
    """Compare the candidate breakout definitions on the same cohort.

    Reports, per definition, how many QBs it can label at all and how many it calls late. The
    definitions differ mostly in how easily a mediocre season counts as a breakout, so the
    ``defined`` column matters as much as the ``late`` column — a stricter bar labels fewer
    careers but labels them more meaningfully.
    """
    import pandas as pd

    rows = []
    for lbl in labels:
        if lbl not in careers.columns:
            continue
        defined = careers[lbl].notna()
        rows.append({
            "definition": lbl,
            "n_defined": int(defined.sum()),
            "n_late": int((careers[lbl] == 1).sum()),
            "pct_late": round(100 * (careers[lbl] == 1).sum() / max(defined.sum(), 1), 1),
        })
    return pd.DataFrame(rows)


def cohort_crosstab(careers, *, late_col="late_sustained", reloc_col="sustained_relocated"):
    """The four-cell view: developed late vs needed a new building.

    Rows are ``late`` (breakout in NFL year 4+), columns are ``relocated`` (breakout came with a
    franchise other than the drafting one). Undrafted QBs have no drafting franchise and so
    appear only in the margin.
    """
    import pandas as pd

    df = careers[careers[late_col].notna()]
    ct = pd.crosstab(
        df[late_col].map({0.0: "on-time", 1.0: "late"}),
        df[reloc_col].map({0.0: "same franchise", 1.0: "relocated"}).fillna("undrafted"),
        dropna=False,
    )
    ct.index.name = "breakout timing"
    ct.columns.name = "franchise"
    return ct


def draft_situation(careers, *, late_col="late_sustained"):
    """Base rates by draft capital: which draft situations produce late breakouts.

    Three outcomes per bucket — never broke out, broke out on time, broke out late — plus the
    late share *among those who broke out at all*. That last column is the one worth reading:
    it separates "this bucket produces good QBs" from "this bucket produces good QBs slowly".
    Censored careers (too recent to have had the chance) are excluded so recent draft classes
    do not depress the never-rate.
    """
    import pandas as pd

    df = careers[~careers["censored"]].copy()
    df["bucket"] = df.apply(draft_bucket, axis=1)
    df["outcome"] = df[late_col].map({0.0: "on_time", 1.0: "late"}).fillna("never")

    out = (
        pd.crosstab(df["bucket"], df["outcome"])
        .reindex(columns=["never", "on_time", "late"], fill_value=0)
    )
    out["n"] = out.sum(axis=1)
    broke = out["on_time"] + out["late"]
    out["pct_ever_broke_out"] = (100 * broke / out["n"]).round(1)
    out["pct_late_given_breakout"] = (100 * out["late"] / broke.replace(0, pd.NA)).round(1)
    order = ["R1 top-10", "R1 11-32", "R2", "R3", "R4-7", "Undrafted"]
    return out.reindex([b for b in order if b in out.index])


def trough_profile(careers, ranked, *, late_col="late_sustained", through_year: int = 3):
    """How bad were they before? Early-career production, split by eventual outcome.

    The premise of a late breakout is that the league had seen enough to move on, so the early
    years should look genuinely poor — not merely unlucky. This summarises each QB's first
    ``through_year`` NFL seasons (the rookie-contract window the league judges them on) and
    compares the groups.

    ``mean_early_ppg_rank`` treats an unranked season (fewer than the games cutoff) as rank 40,
    slightly worse than the worst real rank, so "did not play enough to be ranked" counts as the
    bad outcome it is rather than dropping out of the average.
    """
    import numpy as np

    UNRANKED = 40.0
    r = ranked.merge(careers[["player_id", "entry_season", late_col, "censored"]],
                     on="player_id", how="inner")
    r["nfl_year"] = r["season"] - r["entry_season"] + 1
    early = r[(r["nfl_year"] >= 1) & (r["nfl_year"] <= through_year)].copy()
    early["rank_filled"] = early["ppg_rank"].fillna(UNRANKED)

    per_player = (
        early.groupby("player_id")
        .agg(early_seasons=("season", "nunique"),
             early_games=("games", "sum"),
             mean_early_ppg=("ppg", "mean"),
             mean_early_ppg_rank=("rank_filled", "mean"),
             best_early_ppg_rank=("rank_filled", "min"))
        .reset_index()
    )
    per_player = per_player.merge(
        careers[["player_id", late_col, "censored"]], on="player_id", how="left"
    )
    per_player = per_player[~per_player["censored"]]
    per_player["outcome"] = (
        per_player[late_col].map({0.0: "on_time", 1.0: "late"}).fillna("never")
    )

    summary = (
        per_player.groupby("outcome")
        .agg(n=("player_id", "size"),
             early_games=("early_games", "median"),
             mean_early_ppg=("mean_early_ppg", "median"),
             mean_early_rank=("mean_early_ppg_rank", "median"),
             best_early_rank=("best_early_ppg_rank", "median"))
        .round(2)
        .reindex(["never", "on_time", "late"])
    )
    return summary.replace({np.nan: None})


def timing_distribution(careers, *, late_col="late_sustained", year_col="sustained_nfl_year"):
    """When the breakout arrives, as a distribution over NFL year."""

    df = careers[careers[year_col].notna()]
    dist = (
        df[year_col].astype(int).value_counts().sort_index()
        .rename("n_qbs").rename_axis("nfl_year").reset_index()
    )
    dist["cumulative_pct"] = (100 * dist["n_qbs"].cumsum() / dist["n_qbs"].sum()).round(1)
    return dist


def pending_watchlist(careers, *, late_col="late_sustained"):
    """QBs whose late-breakout window is still open — the live version of the question.

    Two groups belong here: careers too recent to have reached the late threshold at all, and
    careers where a breakout season has happened but has not yet had time to prove it sustains.
    Sam Darnold's 2024 is the current example — one top-20 season with the window still running.
    They are excluded from every base rate above, and they are exactly who a working model would
    be scoring.
    """
    import pandas as pd

    sustained_pending = (
        careers["sustained_censored"].fillna(False).astype(bool)
        if "sustained_censored" in careers.columns
        else pd.Series(False, index=careers.index)
    )
    df = careers[(careers["censored"] | sustained_pending) & (careers["seasons_elapsed"] >= 2)]
    df = df.copy()
    # A QB already holding a top-20 season is a live candidate; one who has never posted one is
    # merely young. Rank the former first — that is the interesting end of the list.
    df["pending_kind"] = df["breakout_season"].notna().map(
        {True: "breakout pending sustain", False: "window still open"}
    )
    cols = ["player_name", "pending_kind", "entry_season", "draft_round", "draft_pick",
            "draft_franchise", "draft_college", "seasons_elapsed", "breakout_season",
            "breakout_nfl_year", "best_ppg_rank"]
    cols = [c for c in cols if c in df.columns]
    return (
        df[cols]
        .sort_values(
            ["breakout_season", "best_ppg_rank"], ascending=[False, True], na_position="last"
        )
        .reset_index(drop=True)
    )


def late_roster(careers, *, late_col="late_sustained"):
    """The named list of late breakouts — the cohort the whole project is built to predict."""
    cols = [
        "player_name", "entry_season", "draft_round", "draft_pick", "draft_franchise",
        "draft_college", "sustained_season", "sustained_nfl_year", "sustained_rank",
        "peak_season", "peak_nfl_year", "peak_rank", "sustained_relocated",
    ]
    df = careers[careers[late_col] == 1]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("sustained_season", ascending=False).reset_index(drop=True)
