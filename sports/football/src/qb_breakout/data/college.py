"""College layer: per-season QB production derived from cfbfastR play-by-play.

This is the project's primary pre-NFL evidence (see ``docs/QB_BREAKOUT_PLAN.md`` §2.2). The
high-school recruiting layer reaches back only to the 2006 class; cfbfastR play-by-play starts in
2002 and so reaches QBs entering the NFL from about 2005, which recovers the entire 2001–2005
cluster of late breakouts that recruiting cuts off.

Why play-by-play rather than a season-stats table:

- **EPA and success rate per dropback** are the efficiency measures that actually travel to the
  NFL. Raw yardage is a function of scheme and pace.
- **Sacks separate cleanly from rushing.** NCAA box scores charge sack yardage against rushing,
  so official college rushing totals understate a mobile QB badly — Baker Mayfield's 2015 reads
  as 405 rushing yards officially, but he gained 604 on actual rush plays and lost the difference
  on 39 sacks. Splitting them is the difference between measuring mobility and measuring
  pass protection.
- **Opponent identity is on every play**, so strength of competition can be derived later rather
  than approximated by conference.

Identity caveat: cfbfastR play-by-play names players but carries **no player ID**, so a QB-season
is keyed on ``(season, team, name)``. That is reliable within a team-season and is the reason the
NFL join (``link.py``) is conservative rather than fuzzy.
"""

from __future__ import annotations

# cfbfastR play-by-play, public and unauthenticated. 405 columns per season, ~57 MB; only the
# columns below are read so a season aggregates without materialising the whole frame.
PBP_URL = ("https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/pbp/parquet"
           "/play_by_play_{season}.parquet")

# Earliest season with a usable schema. 2002 and 2003 exist but are a thinner, older build (366
# columns vs 405) missing `completion`, `pass_td`, `rush_td`, `EPA_success` and the team-name
# fields, so a QB-season cannot be assembled from them — the practical floor is 2004, one year
# later than sportsdataverse's documented 2003.
FIRST_PBP_SEASON = 2004

# cfbfastR-data publishes through 2021 and no further; sportsdataverse's loader reads the same
# repo, so this is a hard ceiling on the source, not a fetch bug. It costs nothing for *fitting*
# — a QB whose last college season is 2022+ enters the NFL in 2023+ and is right-censored for the
# label regardless — but it does mean this layer cannot score today's prospects. Closing that gap
# needs CFBD (free API key) or an ESPN-based loader; see docs/QB_BREAKOUT_PLAN.md §2.2.
LAST_PBP_SEASON = 2021

# Flag columns whose dtype drifts across seasons: 2009 ships `rush` as Float64 where every other
# season has Boolean. Cast on read so aggregation does not fail on one season out of twenty.
_BOOL_COLUMNS = ("pass_attempt", "completion", "pass_td", "int", "sack", "rush", "rush_td",
                 "EPA_success", "scrimmage_play")

PBP_COLUMNS = [
    "season", "week", "game_id", "start.pos_team.name", "def_pos_team",
    "passer_player_name", "rusher_player_name",
    "pass_attempt", "completion", "yds_receiving", "pass_td", "int",
    "sack", "yds_sacked", "rush", "yds_rushed", "rush_td",
    "EPA", "EPA_success", "scrimmage_play",
]

# A QB-season needs enough volume to mean anything. Below this a "passer" is a trick play, a
# holder, or a wildcat back — not someone whose efficiency is measurable.
MIN_DROPBACKS = 50


def pbp_url(season: int) -> str:
    return PBP_URL.format(season=season)


def load_pbp_season(season: int, *, cache_dir=None, columns=None):
    """Load one season of college play-by-play, caching the raw parquet if ``cache_dir`` given."""
    import polars as pl

    cols = columns or PBP_COLUMNS
    if cache_dir is not None:
        from pathlib import Path

        path = Path(cache_dir) / f"cfb_pbp_{season}.parquet"
        if not path.exists():
            import requests

            path.parent.mkdir(parents=True, exist_ok=True)
            resp = requests.get(pbp_url(season), timeout=600)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        return _normalise_dtypes(pl.read_parquet(path, columns=cols))
    return _normalise_dtypes(pl.read_parquet(pbp_url(season), columns=cols))


def _normalise_dtypes(df):
    """Cast the flag columns to Boolean regardless of how a given season typed them."""
    import polars as pl

    casts = [
        pl.col(c).cast(pl.Boolean, strict=False).alias(c)
        for c in _BOOL_COLUMNS
        if c in df.columns and df.schema[c] != pl.Boolean
    ]
    return df.with_columns(casts) if casts else df


def aggregate_qb_seasons(pbp, *, min_dropbacks: int = MIN_DROPBACKS):
    """Aggregate one season of play-by-play into one row per QB-season.

    Passing and rushing are aggregated separately — by ``passer_player_name`` and
    ``rusher_player_name`` — then joined on ``(season, team, player)``. A QB's own scrambles and
    designed runs therefore land on his row, while sacks stay on the passing side where they
    belong.

    Returns volume, efficiency (EPA per dropback, success rate) and rate stats. Rows below
    ``min_dropbacks`` are dropped as non-quarterbacks.
    """
    import polars as pl

    df = pbp.rename({"start.pos_team.name": "team"})

    passing = (
        df.filter(pl.col("passer_player_name").is_not_null())
        .group_by(["season", "team", "passer_player_name"])
        .agg(
            games=pl.col("game_id").n_unique(),
            dropbacks=pl.len(),
            attempts=pl.col("pass_attempt").sum(),
            completions=pl.col("completion").sum(),
            pass_yds=pl.col("yds_receiving").sum(),
            pass_td=pl.col("pass_td").sum(),
            interceptions=pl.col("int").sum(),
            sacks=pl.col("sack").sum(),
            sack_yds=pl.col("yds_sacked").sum(),
            pass_epa=pl.col("EPA").sum(),
            pass_epa_per_db=pl.col("EPA").mean(),
            pass_success_rate=pl.col("EPA_success").mean(),
            n_opponents=pl.col("def_pos_team").n_unique(),
        )
        .rename({"passer_player_name": "player"})
    )

    rushing = (
        df.filter(pl.col("rusher_player_name").is_not_null() & pl.col("rush"))
        .group_by(["season", "team", "rusher_player_name"])
        .agg(
            rush_att=pl.len(),
            rush_yds=pl.col("yds_rushed").sum(),
            rush_td=pl.col("rush_td").sum(),
            rush_epa_per_att=pl.col("EPA").mean(),
            rush_success_rate=pl.col("EPA_success").mean(),
        )
        .rename({"rusher_player_name": "player"})
    )

    # An empty rushing frame carries null-typed key columns, which polars refuses to join against
    # a string key. Align the key dtypes to the passing side so a QB (or a filtered slice) with no
    # rushing attempts still aggregates instead of raising.
    if rushing.height == 0:
        rushing = rushing.with_columns([
            pl.col(k).cast(passing.schema[k]) for k in ("season", "team", "player")
        ])

    out = passing.join(rushing, on=["season", "team", "player"], how="left")
    out = out.filter(pl.col("dropbacks") >= min_dropbacks)

    zero = {"rush_att": 0, "rush_yds": 0.0, "rush_td": 0}
    out = out.with_columns([pl.col(c).fill_null(v) for c, v in zero.items()])

    return out.with_columns(
        completion_pct=pl.col("completions") / pl.col("attempts"),
        yards_per_attempt=pl.col("pass_yds") / pl.col("attempts"),
        td_rate=pl.col("pass_td") / pl.col("attempts"),
        int_rate=pl.col("interceptions") / pl.col("attempts"),
        sack_rate=pl.col("sacks") / pl.col("dropbacks"),
        # Sacks are pass-protection and pocket-processing failures, so they belong in the
        # denominator of passing efficiency, not in rushing.
        adj_yards_per_dropback=(
            (pl.col("pass_yds") - pl.col("sack_yds")) / pl.col("dropbacks")
        ),
        rush_yds_per_att=pl.when(pl.col("rush_att") > 0)
        .then(pl.col("rush_yds") / pl.col("rush_att"))
        .otherwise(None),
        # Mobility as a share of total offensive workload — the cleanest single proxy for the
        # dual-threat / pocket-passer axis that does not depend on a scout's label.
        rush_share=pl.col("rush_att") / (pl.col("dropbacks") + pl.col("rush_att")),
        pass_yds_per_game=pl.col("pass_yds") / pl.col("games"),
    ).sort(["season", "pass_epa"], descending=[False, True])


def build_college_qb_seasons(seasons, *, cache_dir=None, progress=None,
                             min_dropbacks: int = MIN_DROPBACKS):
    """Aggregate several seasons of play-by-play into one QB-season table.

    Each season is loaded, reduced, and released before the next, so peak memory stays at roughly
    one season rather than the whole span.
    """
    import polars as pl

    frames = []
    for season in seasons:
        pbp = load_pbp_season(season, cache_dir=cache_dir)
        agg = aggregate_qb_seasons(pbp, min_dropbacks=min_dropbacks)
        if progress:
            progress(season, agg.height)
        frames.append(agg)
        del pbp
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def add_career_features(qb_seasons):
    """Collapse QB-seasons into one pre-NFL career row per player.

    The model predicts a career outcome from a career profile, so per-season rows have to become
    one row. What matters beyond career totals is the **shape** of the college career, which is
    where the late-breakout hypothesis actually lives:

    - ``n_college_seasons`` / ``first_season`` / ``last_season`` — how long it took to play.
    - ``breakout_season_idx`` — which college season was their best by EPA. A QB whose best year
      was his last is a different prospect from one who peaked as a sophomore.
    - ``final_*`` — the last college season, which is what NFL scouts weighted most heavily.
    - ``epa_trend`` — improvement across college seasons, the closest pre-NFL analogue to the
      "develops late" hypothesis this project is testing.
    - ``n_teams`` — transfers.

    **Truncation matters and is flagged, not hidden.** Play-by-play starts in 2004, so a QB whose
    college career began earlier has it clipped: Aaron Rodgers shows one college season and 274
    attempts because only 2004 is in range, not because he was a one-year starter. Any model
    reading ``career_*`` or ``n_college_seasons`` would badly misread those players, so
    ``college_career_truncated`` marks them. The ``final_*`` block is unaffected — a final season
    is a final season whether or not the earlier ones are visible — which is why the feature set
    leans on it.
    """
    import polars as pl

    df = qb_seasons.sort(["player", "season"])

    career = df.group_by("player").agg(
        n_college_seasons=pl.len(),
        first_season=pl.col("season").min(),
        last_season=pl.col("season").max(),
        n_teams=pl.col("team").n_unique(),
        teams=pl.col("team").unique(),
        career_dropbacks=pl.col("dropbacks").sum(),
        career_attempts=pl.col("attempts").sum(),
        career_pass_yds=pl.col("pass_yds").sum(),
        career_pass_td=pl.col("pass_td").sum(),
        career_int=pl.col("interceptions").sum(),
        career_rush_yds=pl.col("rush_yds").sum(),
        career_rush_td=pl.col("rush_td").sum(),
        career_games=pl.col("games").sum(),
        best_epa_per_db=pl.col("pass_epa_per_db").max(),
        mean_epa_per_db=pl.col("pass_epa_per_db").mean(),
        mean_success_rate=pl.col("pass_success_rate").mean(),
        mean_rush_share=pl.col("rush_share").mean(),
        # Final college season — the sample scouts weight most.
        final_season=pl.col("season").last(),
        final_team=pl.col("team").last(),
        final_epa_per_db=pl.col("pass_epa_per_db").last(),
        final_success_rate=pl.col("pass_success_rate").last(),
        final_completion_pct=pl.col("completion_pct").last(),
        final_yards_per_attempt=pl.col("yards_per_attempt").last(),
        final_adj_yards_per_db=pl.col("adj_yards_per_dropback").last(),
        final_rush_share=pl.col("rush_share").last(),
        final_dropbacks=pl.col("dropbacks").last(),
        first_epa_per_db=pl.col("pass_epa_per_db").first(),
        _epa_by_season=pl.col("pass_epa_per_db"),
    )

    return career.with_columns(
        career_completion_pct=pl.col("career_pass_yds").is_not_null().cast(pl.Float64) * 0,
        # Which college season was the best one, 1-indexed.
        breakout_season_idx=pl.col("_epa_by_season").list.arg_max() + 1,
        peaked_in_final_season=(
            pl.col("_epa_by_season").list.arg_max() + 1 == pl.col("n_college_seasons")
        ).cast(pl.Int8),
        # Improvement from first to last measured season: the pre-NFL "develops late" signal.
        epa_trend=pl.col("final_epa_per_db") - pl.col("first_epa_per_db"),
        transferred=(pl.col("n_teams") > 1).cast(pl.Int8),
        # A career that starts in the first covered season is almost certainly clipped: we cannot
        # tell a true freshman starter from a senior whose earlier years are off-camera.
        college_career_truncated=(pl.col("first_season") <= FIRST_PBP_SEASON).cast(pl.Int8),
    ).drop(["_epa_by_season", "career_completion_pct"])
