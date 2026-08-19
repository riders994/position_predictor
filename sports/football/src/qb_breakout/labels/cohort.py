"""Build the QB breakout cohort: per-season fantasy ranks -> career breakout events.

The label this project predicts is a **career event**, not a season outcome: *did this QB ever
post a top-N fantasy season, and if so, when and for whom?* Two flags then split the cohort:

``late``       breakout arrived in NFL year 4 or later — the rookie contract expired before the
               player was any good, so the league had effectively written them off.
``relocated``  breakout came with a team other than the one that drafted them — the change was
               situational (new building) rather than purely developmental.

Crossing the two gives the four cells the analysis is built around (see
``docs/QB_BREAKOUT_PLAN.md`` §3):

===================  ==========================  ==============================
                     ``relocated=False``         ``relocated=True``
===================  ==========================  ==============================
``late=False``       on-time franchise QB        early bloomer, traded up-front
``late=True``        slow burn, same building    **the Darnold/Geno/Baker cell**
===================  ==========================  ==============================

Design decisions
----------------
**Top-20, not top-12.** Superflex is the relevant format and QB15-ish is where draft-day value
actually lives, so a "breakout" is a top-20 PPG season among eligible QBs (``BREAKOUT_RANK``).
Rank 12 is carried alongside as ``is_qb1`` for reference.

**PPG, not totals**, ranked among QBs clearing the games cutoff — this reuses the games-played
eligibility bar the sibling project derived analytically for QB (``g* = 7``, see
``config/football_qb.yaml``) so ranks mean the same thing in both projects.

**2020 is kept.** ``position_predictor`` drops the COVID season from *supervised* use, but a
breakout is a career milestone: dropping 2020 would silently erase the first top-20 season of
anyone who broke out that year and mislabel them as never having broken out. Rows carry
``season`` so any downstream fit can still exclude it.

**Entry year, not draft year, indexes a career.** Drafted players enter the season they were
drafted; undrafted players (Romo, Keenum, Warner) enter at their first NFL season, which is the
only entry signal that exists for them.
"""

from __future__ import annotations

from .teams import canonicalize

# A breakout season has to clear two different bars, because "became good" and "stayed useful"
# are different claims and one number cannot carry both.
#
# BREAKOUT_RANK is the **quality** bar the breakout season itself must clear: QB15, where
# genuine draft-day value starts. SUSTAIN_RANK is the **relevance** bar the surrounding window
# must hold: QB20, still a startable superflex asset.
#
# Collapsing them is what breaks the label. At a single top-20 bar, one ordinary rookie season
# counts (Mayfield's 2018 ranks exactly 20th). At a single top-15 bar, the archetypes disappear
# instead — Mayfield's good run is 17/4/19 and Geno's is 9/21/16, so neither has two top-15
# seasons in any three-year window even though both were plainly valuable throughout.
BREAKOUT_RANK = 15
SUSTAIN_RANK = 20
# Reference tier carried alongside the primary label (traditional 1-QB league starter).
QB1_RANK = 12
# "Late" = the breakout arrived in NFL year 4+, i.e. after the rookie contract's cheap years.
LATE_YEAR_THRESHOLD = 4
# Games-played eligibility bar for a season to be *rankable*, from the sibling project's
# analytically-derived QB cutoff (config/football_qb.yaml: eligibility.chosen_games_played).
MIN_GAMES = 7
# A "start"-like appearance: enough dropbacks that the week reflects the player quarterbacking
# the team rather than mop-up or a wildcat snap. Used for descriptive context, not eligibility.
START_ATTEMPTS = 10
# Careers beginning before this are left-censored: nflverse weekly stats start in 1999, so we
# cannot see whether a 1996 entrant broke out in year 1-3, nor index their career correctly.
FIRST_ENTRY_SEASON = 1999
# A breakout "sticks" if the QB is top-20 in at least SUSTAIN_NEED of the SUSTAIN_WINDOW seasons
# starting with it — enough to distinguish a tier change from a one-season blip.
SUSTAIN_WINDOW = 3
SUSTAIN_NEED = 2


def build_qb_seasons(weekly, *, regular_season_only: bool = True):
    """Aggregate weekly rows into one row per QB ``(player_id, season)``.

    Mirrors ``position_predictor.data.build.aggregate_player_seasons`` (games = number of
    regular-season weeks the player appears) so ranks are comparable across the two projects,
    but keeps only QB rows and adds the passing-volume columns this project reasons about.

    Parameters
    ----------
    weekly : DataFrame
        nflverse weekly player stats (pandas or polars; polars is converted).
    regular_season_only : bool
        Drop postseason rows. Fantasy seasons are regular-season only.

    Returns
    -------
    DataFrame with ``player_id, season, team, games, starts, attempts, ppr_points, ppg`` plus
    identity columns.
    """
    import numpy as np
    import pandas as pd

    df = weekly.to_pandas() if hasattr(weekly, "to_pandas") else weekly
    if regular_season_only and "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    df = df[df["position"] == "QB"] if "position" in df.columns else df

    if df.empty:
        return pd.DataFrame(
            columns=["player_id", "season", "team", "games", "starts", "attempts",
                     "ppr_points", "ppg"]
        )

    sum_cols = [
        c for c in ("fantasy_points_ppr", "attempts", "completions", "passing_yards",
                    "passing_tds", "interceptions", "carries", "rushing_yards", "rushing_tds",
                    "sacks", "passing_epa")
        if c in df.columns
    ]
    grouped = df.groupby(["player_id", "season"], as_index=False)
    out = grouped[sum_cols].sum(numeric_only=True)

    games = grouped.size().rename(columns={"size": "games"})
    out = out.merge(games, on=["player_id", "season"], how="left")

    # Start-like weeks: a real look at quarterbacking the team, not a kneel-down cameo.
    if "attempts" in df.columns:
        starts = (
            df.assign(_is_start=(df["attempts"] >= START_ATTEMPTS).astype(int))
            .groupby(["player_id", "season"], as_index=False)["_is_start"]
            .sum()
            .rename(columns={"_is_start": "starts"})
        )
        out = out.merge(starts, on=["player_id", "season"], how="left")
    else:
        out["starts"] = np.nan

    # Team of record for the season = the team the QB last appeared for. A midseason trade is
    # rare at QB and the destination is the situation that matters for `relocated`.
    ident_cols = [c for c in ("player_display_name", "player_name", "recent_team")
                  if c in df.columns]
    if ident_cols:
        sort_cols = ["player_id", "season"] + (["week"] if "week" in df.columns else [])
        ident = (
            df.sort_values(sort_cols)
            .groupby(["player_id", "season"], as_index=False)[ident_cols]
            .last()
        )
        out = out.merge(ident, on=["player_id", "season"], how="left")
    out = out.rename(columns={"recent_team": "team"})

    out["ppr_points"] = out.get("fantasy_points_ppr", np.nan)
    out["ppg"] = np.where(out["games"] > 0, out["ppr_points"] / out["games"], np.nan)
    return out.sort_values(["season", "ppg"], ascending=[True, False]).reset_index(drop=True)


def rank_qb_seasons(seasons, *, min_games: int = MIN_GAMES,
                    breakout_rank: int = BREAKOUT_RANK,
                    sustain_rank: int = SUSTAIN_RANK):
    """Rank each season's eligible QBs by PPG and flag breakout-tier seasons.

    Only seasons clearing ``min_games`` are ranked; the rest get ``ppg_rank = NaN`` and cannot
    trigger a breakout. That is deliberate — a four-game hot streak is not a breakout, and the
    cutoff is the same bar the sibling project uses to call a QB season rankable.

    Adds ``eligible``, ``ppg_rank``, ``is_breakout`` (rank <= 20) and ``is_qb1`` (rank <= 12).
    """
    import numpy as np

    df = seasons.copy()
    df["eligible"] = df["games"] >= min_games
    df["ppg_rank"] = np.nan
    elig = df["eligible"]
    df.loc[elig, "ppg_rank"] = (
        df.loc[elig].groupby("season")["ppg"].rank(method="min", ascending=False)
    )
    # Quality bar (can trigger a breakout) vs relevance bar (keeps one alive across the window).
    df["is_breakout"] = df["ppg_rank"].le(breakout_rank).fillna(False)
    df["is_startable"] = df["ppg_rank"].le(sustain_rank).fillna(False)
    df["is_qb1"] = df["ppg_rank"].le(QB1_RANK).fillna(False)
    return df


def first_sustained_breakout(ranked, *, latest_season=None, window: int = SUSTAIN_WINDOW,
                             need: int = SUSTAIN_NEED):
    """First season that clears the quality bar *and* holds the relevance bar around it.

    A breakout is triggered by an ``is_breakout`` season (top-15 — genuine draft-day value) and
    confirmed by ``is_startable`` (top-20 — still a usable superflex asset) in ``need`` of the
    ``window`` seasons starting there.

    Both halves are load-bearing, and using one number for both fails in opposite directions. A
    single top-20 trigger is too weak: one ordinary rookie season clears it, and Mayfield's 2018
    ranks exactly 20th before he goes 27th/24th/28th. Requiring *two* top-15 seasons is too
    strong: Mayfield's actual good run is 17/4/19 and Geno Smith's is 9/21/16, so neither has two
    top-15 seasons in any three-year window despite both being plainly valuable throughout.
    Trigger high, confirm lower.

    Seasons the QB missed entirely count against the window (a benched or injured QB is not
    sustaining a tier), so absence is read from the season index rather than from row presence.
    A candidate whose window runs past the last completed season and has not yet cleared ``need``
    is reported as censored rather than rejected — the window simply has not closed on them.

    Returns ``player_id, sustained_season, sustained_rank, sustained_team, sustained_censored``.
    """
    import numpy as np
    import pandas as pd

    latest = int(latest_season if latest_season is not None else ranked["season"].max())
    rows = []
    # Fall back to the trigger flag when no separate relevance flag is present, so callers
    # holding an older ranked frame still get the single-bar behaviour rather than an error.
    sustain_col = "is_startable" if "is_startable" in ranked.columns else "is_breakout"
    for pid, grp in ranked.groupby("player_id"):
        trigger = dict(zip(grp["season"], grp["is_breakout"]))
        hold = dict(zip(grp["season"], grp[sustain_col]))
        info = grp.set_index("season")[["ppg_rank", "team"]].to_dict("index")
        cands = sorted(s for s, h in trigger.items() if h)
        censored = False
        for s in cands:
            span = range(s, s + window)
            got = sum(1 for y in span if hold.get(y, False))
            if got >= need:
                rows.append({"player_id": pid, "sustained_season": s,
                             "sustained_rank": info[s]["ppg_rank"],
                             "sustained_team": info[s]["team"], "sustained_censored": False})
                break
            # Could still clear it once the unplayed remainder of the window is played out.
            if got + sum(1 for y in span if y > latest) >= need:
                censored = True
        else:
            rows.append({"player_id": pid, "sustained_season": np.nan, "sustained_rank": np.nan,
                         "sustained_team": None, "sustained_censored": censored})
    return pd.DataFrame(rows)


def resolve_entry(player_ids, draft, rosters):
    """Determine each QB's NFL entry season, preferring the most trustworthy available signal.

    Getting this right matters more than it looks: ``breakout_nfl_year`` — and therefore the
    entire ``late`` flag — is measured from it. Falling back to "first season we observe" is
    wrong for anyone already in the league when the data window opens (Vinny Testaverde, the
    1987 #1 overall pick, first appears in 1999 weekly stats and would read as a 1999 rookie).

    Precedence:

    1. ``draft_picks.season`` — the draft class, exact when the ``gsis_id`` join lands.
    2. ``rosters.entry_year`` / ``rookie_year`` — nflverse's own entry field, which covers
       undrafted players and older draftees whose ``gsis_id`` is missing upstream.
    3. First observed season, but **only** if it is after the data window opens. A QB whose
       first observed season is the window's first season is indistinguishable from a veteran
       already in the league, so they get ``NaN`` here and are dropped by the caller.

    Returns a DataFrame ``player_id, entry_season, entry_source``.
    """
    import numpy as np
    import pandas as pd

    ids = pd.DataFrame({"player_id": pd.unique(pd.Series(player_ids))})

    d = draft.dropna(subset=["player_id"])[["player_id", "draft_season"]].drop_duplicates(
        subset=["player_id"], keep="first"
    )
    out = ids.merge(d, on="player_id", how="left")

    if rosters is not None and len(rosters):
        r = rosters.to_pandas() if hasattr(rosters, "to_pandas") else rosters
        cols = [c for c in ("entry_year", "rookie_year") if c in r.columns]
        if cols:
            ry = (
                r[["player_id", *cols]]
                .dropna(subset=["player_id"])
                .groupby("player_id", as_index=False)
                .min()  # earliest non-null value across the player's roster rows
            )
            ry["roster_entry"] = ry[cols].min(axis=1)
            out = out.merge(ry[["player_id", "roster_entry"]], on="player_id", how="left")
    if "roster_entry" not in out.columns:
        out["roster_entry"] = np.nan

    out["entry_season"] = out["draft_season"].fillna(out["roster_entry"])
    out["entry_source"] = np.where(
        out["draft_season"].notna(), "draft",
        np.where(out["roster_entry"].notna(), "roster", "unresolved"),
    )
    return out[["player_id", "entry_season", "entry_source"]]


def build_qb_careers(
    ranked,
    draft_picks,
    rosters=None,
    *,
    late_threshold: int = LATE_YEAR_THRESHOLD,
    first_entry_season: int = FIRST_ENTRY_SEASON,
    latest_season: int | None = None,
):
    """Collapse ranked seasons into one row per QB career with the breakout event and flags.

    Parameters
    ----------
    ranked : DataFrame
        Output of :func:`rank_qb_seasons`.
    draft_picks : DataFrame
        nflverse ``load_draft_picks`` — supplies draft season/round/pick/team and the
        ``cfb_player_id`` bridge to college data. Joined on ``gsis_id`` -> ``player_id``.
    late_threshold : int
        NFL year index at/after which a breakout counts as ``late``.
    first_entry_season : int
        Drop careers that began before this season. Weekly stats start in 1999, so an earlier
        entrant's first NFL years are unobserved: we would neither see an early breakout nor be
        able to trust ``breakout_nfl_year``. Left-censoring is excluded, not imputed.
    latest_season : int | None
        Most recent completed season (defaults to the max season present). Used to set
        ``censored``: a QB who entered too recently to have had a year-``late_threshold`` season
        yet cannot be labelled "never broke out late" — the event may simply not have had time
        to happen.

    Returns
    -------
    One row per ``player_id``: entry and draft context, career shape (seasons, best rank), and
    the breakout event (``breakout_season``, ``breakout_nfl_year``, ``breakout_team``) with the
    ``ever_breakout`` / ``late`` / ``relocated`` flags. ``late`` and ``relocated`` are NaN for
    QBs who never broke out — they are properties of an event that did not happen, and coding
    them as False would quietly merge "never good" into "good on time".
    """
    import numpy as np
    import pandas as pd

    df = ranked
    draft = draft_picks.to_pandas() if hasattr(draft_picks, "to_pandas") else draft_picks
    draft = draft[draft["position"] == "QB"] if "position" in draft.columns else draft
    draft = (
        draft.rename(columns={
            "gsis_id": "player_id", "season": "draft_season", "team": "draft_team",
            "round": "draft_round", "pick": "draft_pick", "college": "draft_college",
        })
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id"], keep="first")
    )
    keep = ["player_id", "draft_season", "draft_round", "draft_pick", "draft_team",
            "draft_college", "cfb_player_id", "pfr_player_id", "age"]
    # Reindex rather than intersect: downstream flags read every one of these columns, so a
    # source frame missing one (or an empty frame with no rows to infer from) must yield an
    # all-null column instead of a KeyError deep in the flag logic.
    draft = draft.reindex(columns=[c for c in keep if c in draft.columns]).reindex(columns=keep)
    draft = draft.rename(columns={"age": "draft_age"})

    name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
    agg = (
        df.groupby("player_id")
        .agg(
            player_name=(name_col, "last"),
            first_season=("season", "min"),
            last_season=("season", "max"),
            n_seasons=("season", "nunique"),
            n_eligible_seasons=("eligible", "sum"),
            career_games=("games", "sum"),
            career_starts=("starts", "sum"),
            best_ppg_rank=("ppg_rank", "min"),
            n_breakout_seasons=("is_breakout", "sum"),
            n_qb1_seasons=("is_qb1", "sum"),
        )
        .reset_index()
    )
    agg = agg.merge(draft, on="player_id", how="left")

    # Entry season drives the whole `late` flag — resolve it from the best signal available
    # rather than assuming the first season we happen to observe is a rookie year.
    entry = resolve_entry(agg["player_id"], draft, rosters)
    agg = agg.merge(entry, on="player_id", how="left")
    agg["undrafted"] = agg["draft_season"].isna()
    # Last resort: trust "first observed season" only when it is strictly inside the window,
    # where a first appearance really does imply a debut.
    fallback = agg["entry_season"].isna() & (agg["first_season"] > first_entry_season)
    agg.loc[fallback, "entry_season"] = agg.loc[fallback, "first_season"]
    agg.loc[fallback, "entry_source"] = "first_season"
    # Anything still unresolved is a veteran who predates the data window — drop, don't guess.
    agg = agg[agg["entry_season"].notna()].copy()
    agg["entry_season"] = agg["entry_season"].astype(int)

    # First breakout season -> its NFL year index and the team it happened for.
    hits = df[df["is_breakout"]].sort_values(["player_id", "season"])
    first_hit = (
        hits.groupby("player_id")
        .agg(breakout_season=("season", "first"),
             breakout_team=("team", "first"),
             breakout_ppg=("ppg", "first"),
             breakout_rank=("ppg_rank", "first"))
        .reset_index()
    )
    agg = agg.merge(first_hit, on="player_id", how="left")

    # First QB1-tier (top-12) season — a stricter alternative breakout bar (PLAN §3.2).
    qb1_hits = df[df["is_qb1"]].sort_values(["player_id", "season"])
    first_qb1 = (
        qb1_hits.groupby("player_id")
        .agg(qb1_season=("season", "first"), qb1_team=("team", "first"))
        .reset_index()
    )
    agg = agg.merge(first_qb1, on="player_id", how="left")
    agg = agg.merge(
        first_sustained_breakout(df, latest_season=latest_season), on="player_id", how="left"
    )

    # Left-censoring: careers that started before weekly stats exist cannot be indexed or
    # labelled correctly, so they leave the cohort entirely rather than carry a wrong label.
    agg = agg[agg["entry_season"] >= first_entry_season].copy()

    # --- Career peak, alongside first breakout -------------------------------------------
    # "First top-20 season" turns out to measure first *competence*, not the late peak this
    # project is about: Baker Mayfield's 2018 rookie year ranks exactly 20th and Ryan
    # Tannehill's 2014 ranks 10th, so both read as on-time breakouts even though their careers
    # are the archetypal late second act. Peak timing — when the best season actually arrived —
    # separates them correctly, so both labels are carried and compared (PLAN §3.2).
    peaks = ranked[ranked["ppg_rank"].notna()].sort_values(["player_id", "ppg_rank", "season"])
    peak = (
        peaks.groupby("player_id")
        .agg(peak_season=("season", "first"),
             peak_team=("team", "first"),
             peak_ppg=("ppg", "first"),
             peak_rank=("ppg_rank", "first"))
        .reset_index()
    )
    agg = agg.merge(peak, on="player_id", how="left")

    agg["ever_breakout"] = agg["breakout_season"].notna()
    # NFL year index is 1-based: a rookie-season breakout is year 1.
    agg["breakout_nfl_year"] = agg["breakout_season"] - agg["entry_season"] + 1
    agg["late"] = np.where(
        agg["ever_breakout"], agg["breakout_nfl_year"] >= late_threshold, np.nan
    )

    # Compare *franchises*, not raw abbreviations: the two sources use different code styles
    # (GNB vs GB) and a franchise relocating cities is not the QB changing situations.
    draft_fr = canonicalize(agg["draft_team"])
    breakout_fr = canonicalize(agg["breakout_team"])
    agg["draft_franchise"] = draft_fr
    agg["breakout_franchise"] = breakout_fr
    # Undrafted QBs have no drafting team, so `relocated` is undefined rather than False —
    # "left the building that drafted him" has no meaning if nobody drafted him.
    agg["relocated"] = np.where(
        agg["ever_breakout"] & draft_fr.notna(), breakout_fr != draft_fr, np.nan
    )

    peak_fr = canonicalize(agg["peak_team"])
    agg["peak_franchise"] = peak_fr
    agg["peak_nfl_year"] = agg["peak_season"] - agg["entry_season"] + 1
    agg["late_peak"] = np.where(
        agg["ever_breakout"], agg["peak_nfl_year"] >= late_threshold, np.nan
    )
    agg["peak_relocated"] = np.where(
        agg["ever_breakout"] & draft_fr.notna(), peak_fr != draft_fr, np.nan
    )
    # A "second act": competent early, then the real peak arrives later and usually elsewhere.
    # This is the Baker/Tannehill/Foles pattern that first-breakout timing alone misses.
    agg["second_act"] = np.where(
        agg["ever_breakout"], (agg["late"] == 0) & (agg["late_peak"] == 1), np.nan
    )

    # The two alternative breakout bars, indexed and flagged the same way so the three
    # definitions can be compared head-to-head on the same cohort.
    agg["qb1_nfl_year"] = agg["qb1_season"] - agg["entry_season"] + 1
    agg["late_qb1"] = np.where(
        agg["qb1_season"].notna(), agg["qb1_nfl_year"] >= late_threshold, np.nan
    )
    agg["qb1_relocated"] = np.where(
        agg["qb1_season"].notna() & draft_fr.notna(),
        canonicalize(agg["qb1_team"]) != draft_fr, np.nan,
    )
    agg["sustained_nfl_year"] = agg["sustained_season"] - agg["entry_season"] + 1
    agg["late_sustained"] = np.where(
        agg["sustained_season"].notna(), agg["sustained_nfl_year"] >= late_threshold, np.nan
    )
    agg["sustained_relocated"] = np.where(
        agg["sustained_season"].notna() & draft_fr.notna(),
        canonicalize(agg["sustained_team"]) != draft_fr, np.nan,
    )

    # Right-censoring: a QB who entered too recently to have reached year `late_threshold`
    # cannot yet be a negative for the late label — the window hasn't closed on them.
    latest = int(latest_season if latest_season is not None else ranked["season"].max())
    agg["seasons_elapsed"] = latest - agg["entry_season"] + 1
    agg["censored"] = (~agg["ever_breakout"]) & (agg["seasons_elapsed"] < late_threshold)

    agg["cell"] = pd.Series(
        np.select(
            [
                agg["censored"],
                ~agg["ever_breakout"],
                agg["undrafted"] & (agg["late"] == 1),
                agg["undrafted"] & (agg["late"] == 0),
                (agg["late"] == 1) & (agg["relocated"] == 1),
                (agg["late"] == 1) & (agg["relocated"] == 0),
                (agg["late"] == 0) & (agg["relocated"] == 1),
                (agg["late"] == 0) & (agg["relocated"] == 0),
            ],
            [
                "censored", "never",
                "late_undrafted", "ontime_undrafted",
                "late_relocated", "late_same_team",
                "ontime_relocated", "ontime_same_team",
            ],
            default="unclassified",
        ),
        index=agg.index,
    )
    return agg.sort_values(["entry_season", "player_name"]).reset_index(drop=True)
