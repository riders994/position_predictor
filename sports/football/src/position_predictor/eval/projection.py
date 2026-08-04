"""Forward projection — run a trained per-position model on the upcoming season.

The walk-forward experiment scores past folds; this projects the **next** season for real use
(keeper/draft tools). It fits the configured era ensemble on all labeled history and predicts the
latest feature season's rows — the *censored* season-N rows whose N+1 outcome isn't known yet are
exactly the live-board inputs (PROJECT_PLAN: "remain available as prediction inputs"). Output is
each player's projected next-season PPR PPG and within-position rank.
"""
from __future__ import annotations

TARGET = "target"


def project_position(config, *, model=None, combine=None, feature_season=None,
                     write: bool = False):
    """Project a season for one position; return a tidy projection frame.

    Columns: ``player_id, player_name, position, proj_ppg, proj_pos_rank, feature_season,
    proj_season``. The model defaults to ``projection.model`` (else ``ridge`` — stable, strong
    top-of-board precision), trained over all eras via the standard :class:`EraEnsemble`.

    ``feature_season`` picks the **board** season (the rows scored); it defaults to the latest
    season present (the live, censored season → upcoming-year board). Passing an earlier season
    reconstructs the *preseason* projection that was knowable before that season's outcome — the
    training set is restricted to rows whose label was known by then (``season < feature_season``),
    so a postseason backtest never trains on the answer. In live mode this filter is a no-op (the
    latest season is censored, so every labeled row already precedes it).
    """
    import json

    import pandas as pd

    from ..eras import load_eras
    from ..models.era_ensemble import EraEnsemble
    from ..utils.io import DATA_PROCESSED, REPORTS_DIR, ensure_dir, read_parquet
    from ..utils.naming import artifact_stem

    position = config.require("experiment.position")
    stem = artifact_stem(config)
    seed = int(config.get("reproducibility.random_seed", 1729))
    horizon = int(config.get("target.predict_horizon", 1))
    model = model or config.get("projection.model", "ridge")
    combine = combine or config.get("projection.combine",
                                    config.get("era_modeling.combine", "val_weighted"))

    eras = load_eras(config)
    df = read_parquet(DATA_PROCESSED / f"{stem}_features.parquet").rename(
        columns={"target_ppg_next": TARGET})
    block_columns = json.load(open(DATA_PROCESSED / f"{stem}_feature_blocks.json"))

    board_season = int(feature_season) if feature_season is not None else int(df["season"].max())
    # labeled history known before the board season's outcome (era ensemble routes by era)
    train = df[df[TARGET].notna() & (df["season"] < board_season)]
    board = df[df["season"] == board_season].copy()  # the feature season being scored
    if train.empty or board.empty:
        return pd.DataFrame()

    from ..features.build import offseason_degenerate
    if offseason_degenerate(df, block_columns, board_season):
        import warnings
        warnings.warn(
            f"offseason block is all-zero for board season {board_season} — the N+1 roster join "
            f"is missing; this {position.upper()} board lacks offseason signal. Refresh next "
            f"season's rosters/draft (e.g. `make redraft`) to populate it.", stacklevel=2)

    ens = EraEnsemble(model, eras, block_columns, combine=combine,
                      target_col=TARGET, seed=seed).fit(train)
    board["proj_ppg"] = ens.predict(board)
    board = board.sort_values("proj_ppg", ascending=False).reset_index(drop=True)
    board["proj_pos_rank"] = board["proj_ppg"].rank(ascending=False, method="min").astype(int)

    name_col = "player_display_name" if "player_display_name" in board.columns else "player_name"
    out = board[["player_id", name_col, "proj_ppg", "proj_pos_rank"]].rename(
        columns={name_col: "player_name"})
    out.insert(1, "position", position.upper())
    out["feature_season"] = board_season
    out["proj_season"] = board_season + horizon
    out["proj_ppg"] = out["proj_ppg"].round(2)

    if write:
        ensure_dir(REPORTS_DIR)
        out.to_csv(REPORTS_DIR / f"projections_{stem}.csv", index=False)
    return out


def project_positions(configs):
    """Project several positions and stack them into one board. ``configs`` is an iterable of
    loaded ``Config`` objects (one per position)."""
    import pandas as pd

    frames = [project_position(c) for c in configs]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
