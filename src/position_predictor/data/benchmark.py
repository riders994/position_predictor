"""Market benchmark acquisition — preseason ADP/ECR (PROJECT_PLAN §2.3, §7.4).

The benchmark is the crowd's **preseason** ranking of returning players for each test season,
used purely as a baseline to beat (never as a feature). We source historical FantasyPros expert
consensus rank (ECR) from the open DynastyProcess archive and join it to our universe via the
nflverse ID crosswalk (``fantasypros_id`` → ``gsis_id``).

For each season *Y* we take the **latest preseason** redraft-overall scrape (a snapshot just
before kickoff, so it is a fair "prior knowledge" predictor of season *Y*), restrict to the
position, map to ``gsis_id``, and emit a small, committed reference table in ``data/external/``.
The experiment scores ``−ecr`` with the same ranking metrics as the models, on the same eligible
universe — the headline test of whether our stats-derived model beats the market.
"""

from __future__ import annotations

ECR_URL = "https://github.com/dynastyprocess/data/raw/master/files/db_fpecr.parquet"
# redraft-overall consensus rank (single-QB redraft leagues), the right preseason board for PPR.
REDRAFT_OVERALL = "ro"
PRESEASON_START = "08-01"   # earliest scrape date treated as "preseason" for season Y
PRESEASON_END = "09-15"     # latest (just after Week 1 kickoff windows)
EXTERNAL_NAME = "market_{sport}_{position}.parquet"


def _to_int_id(series):
    import pandas as pd
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def latest_preseason_by_season(ecr_df, *, start=PRESEASON_START, end=PRESEASON_END,
                               min_players=30):
    """Pick, per season, the single latest preseason scrape (≥ ``min_players`` ranked).

    Returns the ECR rows of that scrape with a ``season`` column. Scrapes outside the
    ``[start, end]`` preseason window or with too few players are ignored.
    """
    import pandas as pd

    df = ecr_df.copy()
    df["scrape_date"] = pd.to_datetime(df["scrape_date"], errors="coerce")
    df = df.dropna(subset=["scrape_date", "ecr"])
    df["season"] = df["scrape_date"].dt.year
    out = []
    for season, g in df.groupby("season"):
        lo = pd.Timestamp(f"{season}-{start}")
        hi = pd.Timestamp(f"{season}-{end}")
        window = g[(g["scrape_date"] >= lo) & (g["scrape_date"] <= hi)]
        if window.empty:
            continue
        latest = window["scrape_date"].max()
        snap = window[window["scrape_date"] == latest]
        if len(snap) >= min_players:
            out.append(snap.assign(season=int(season)))
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


def build_market_benchmark(config, *, ecr_url=ECR_URL, write: bool = True):
    """Build the per-season preseason ECR benchmark for the configured position.

    Returns a tidy DataFrame ``[player_id, season, market_ecr, market_rank, scrape_date]`` where
    ``player_id`` is ``gsis_id`` (joins to our dataset), ``market_ecr`` is the raw consensus rank
    (lower = better) and ``market_rank`` is the dense positional rank within the season. When
    ``write`` also persists it to ``data/external/`` (a committed reference file).
    """
    import pandas as pd

    from ..utils.io import DATA_EXTERNAL, DATA_RAW, ensure_dir, read_parquet

    sport = config.get("experiment.sport", "sport")
    position = config.require("experiment.position")

    ecr = pd.read_parquet(ecr_url, columns=["player", "id", "pos", "ecr", "ecr_type",
                                            "scrape_date"])
    ecr = ecr[(ecr["pos"] == position) & (ecr["ecr_type"] == REDRAFT_OVERALL)]
    snaps = latest_preseason_by_season(ecr)
    if snaps.empty:
        raise RuntimeError("no preseason ECR scrapes found for the configured window/position")

    ids = read_parquet(DATA_RAW / "ids.parquet")[["fantasypros_id", "gsis_id"]].dropna()
    ids = ids.assign(fantasypros_id=_to_int_id(ids["fantasypros_id"])).dropna(
        subset=["fantasypros_id"])
    xwalk = dict(zip(ids["fantasypros_id"], ids["gsis_id"]))

    snaps = snaps.assign(fp_id=_to_int_id(snaps["id"]))
    snaps["player_id"] = snaps["fp_id"].map(xwalk)
    out = snaps.dropna(subset=["player_id"])[
        ["player_id", "season", "ecr", "scrape_date"]].rename(columns={"ecr": "market_ecr"})
    # one row per (player, season): keep the best (lowest) ECR if duplicated
    out = out.sort_values("market_ecr").drop_duplicates(["player_id", "season"])
    out["market_rank"] = out.groupby("season")["market_ecr"].rank(method="dense").astype(int)
    out = out.sort_values(["season", "market_rank"]).reset_index(drop=True)

    if write:
        path = ensure_dir(DATA_EXTERNAL) / EXTERNAL_NAME.format(
            sport=sport, position=position).lower()
        out.to_parquet(path, index=False)
    return out
