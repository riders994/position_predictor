"""Market benchmark acquisition — preseason ADP/ECR (PROJECT_PLAN §2.3, §7.4).

The benchmark is the crowd's **preseason** ranking of returning players for each test season,
used purely as a baseline to beat (never as a feature). We source historical FantasyPros expert
consensus rank (ECR) from the open DynastyProcess archive and join it to our universe via the
nflverse ID crosswalk (``fantasypros_id`` → ``gsis_id``).

For each season *Y* we take the **latest preseason** scrape (a snapshot just before kickoff, so
it is a fair "prior knowledge" predictor of season *Y*), restrict to the position, map to
``gsis_id``, and emit a small, committed reference table in ``data/external/``. The experiment
scores ``−ecr`` with the same ranking metrics as the models, on the same eligible universe — the
headline test of whether our stats-derived model beats the market.

**Which board.** FantasyPros publishes a separate consensus for each roster shape, and a league
that starts two QBs is priced off a different one. :func:`ecr_type_for_league` picks it; the
constants below record what each is and how much it actually moves.
"""

from __future__ import annotations

ECR_URL = "https://github.com/dynastyprocess/data/raw/master/files/db_fpecr.parquet"
# redraft-overall consensus rank (single-QB redraft leagues) — FantasyPros ppr-cheatsheets.php.
REDRAFT_OVERALL = "ro"
# redraft superflex — FantasyPros ppr-superflex-cheatsheets.php. The right board for a league that
# can start two QBs: within a position the two charts agree almost perfectly (2026 rank Spearman
# QB .991, RB .999, WR .997, TE .997), but *cross-position* placement moves hard — the median QB
# sits 162nd overall on `ro` and 76th on `rsf`. Anything comparing positions needs this one.
REDRAFT_SUPERFLEX = "rsf"
# Both pages are full PPR; FantasyPros publishes no half-PPR consensus board, so a half-PPR league
# is still benchmarked against a PPR chart (documented mismatch, not an error).
ECR_TYPES = (REDRAFT_OVERALL, REDRAFT_SUPERFLEX)
ECR_TYPE_LABELS = {REDRAFT_OVERALL: "1QB redraft-overall", REDRAFT_SUPERFLEX: "superflex"}
DEFAULT_ECR_TYPE = REDRAFT_OVERALL
PRESEASON_START = "08-01"   # earliest scrape date treated as "preseason" for season Y
PRESEASON_END = "09-15"     # latest (just after Week 1 kickoff windows)
EXTERNAL_NAME = "market_{sport}_{position}.parquet"
# Non-default boards get their own cache file so `rsf` never clobbers the committed `ro` tables.
EXTERNAL_NAME_TYPED = "market_{sport}_{position}_{ecr_type}.parquet"


def resolve_ecr_type(ecr_type) -> str:
    """Validate an ECR board name, naming the offending value (a typo would silently empty the
    filter and raise a confusing "no preseason scrapes" error much later)."""
    value = str(ecr_type or DEFAULT_ECR_TYPE).lower()
    if value not in ECR_TYPES:
        raise ValueError(f"unknown ecr_type {ecr_type!r}; expected one of {list(ECR_TYPES)}")
    return value


def ecr_type_for_league(league) -> str:
    """The market board that matches ``league``'s QB shape.

    A league that can start two QBs (true 2QB or superflex) is priced off the superflex board;
    everything else off redraft-overall. See :meth:`eval.league.LeagueConfig.is_superflex`.
    """
    return REDRAFT_SUPERFLEX if getattr(league, "is_superflex", False) else REDRAFT_OVERALL


def benchmark_path(sport: str, position: str, ecr_type: str = DEFAULT_ECR_TYPE):
    """Cache path for one (sport, position, board) benchmark table.

    The default board keeps the historical un-suffixed name so the committed `ro` files and every
    reader of them stay valid.
    """
    from ..utils.io import DATA_EXTERNAL

    ecr_type = resolve_ecr_type(ecr_type)
    name = (EXTERNAL_NAME.format(sport=sport, position=position) if ecr_type == DEFAULT_ECR_TYPE
            else EXTERNAL_NAME_TYPED.format(sport=sport, position=position, ecr_type=ecr_type))
    return DATA_EXTERNAL / name.lower()


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


def build_market_benchmark(config, *, ecr_url=ECR_URL, ecr_type=None, write: bool = True):
    """Build the per-season preseason ECR benchmark for the configured position.

    Returns a tidy DataFrame ``[player_id, season, market_ecr, market_rank, scrape_date]`` where
    ``player_id`` is ``gsis_id`` (joins to our dataset), ``market_ecr`` is the raw consensus rank
    (lower = better) and ``market_rank`` is the dense positional rank within the season. When
    ``write`` also persists it to ``data/external/`` (a committed reference file).

    ``ecr_type`` picks the market board (see :data:`ECR_TYPES`); it defaults to the config's
    ``benchmark.ecr_type`` and then to redraft-overall. Use :func:`ecr_type_for_league` to derive
    it from a league. Each board caches to its own file (:func:`benchmark_path`).
    """
    import pandas as pd

    from ..utils.io import DATA_EXTERNAL, DATA_RAW, ensure_dir, read_parquet

    sport = config.get("experiment.sport", "sport")
    position = config.require("experiment.position")
    ecr_type = resolve_ecr_type(ecr_type or config.get("benchmark.ecr_type", DEFAULT_ECR_TYPE))

    ecr = pd.read_parquet(ecr_url, columns=["player", "id", "pos", "ecr", "ecr_type",
                                            "scrape_date"])
    ecr = ecr[(ecr["pos"] == position) & (ecr["ecr_type"] == ecr_type)]
    snaps = latest_preseason_by_season(ecr)
    if snaps.empty:
        raise RuntimeError(f"no preseason {ecr_type} ECR scrapes found for the configured "
                           f"window/position")

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
        ensure_dir(DATA_EXTERNAL)
        out.to_parquet(benchmark_path(sport, position, ecr_type), index=False)
    return out
