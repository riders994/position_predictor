"""Handcuff selection — which backups to draft, driven by starter injury risk.

A *handcuff* is the backup who inherits a starter's workload if the starter gets hurt. This tool
ranks handcuffs by **contingent upside**: how much projected PPG the backup gains when the starter
misses time, weighted by how likely / how long that absence is.

Injury risk is **measured, not assumed.** The project's §7.4 availability model (a Poisson games
count GBM) does *not* beat a naive prior-games baseline in backtests, so the risk signal is chosen
by a leak-safe backtest among candidates (the availability model vs transparent durability
baselines); whichever best discriminates durable from fragile (clears-cutoff AUC) drives the board.

Scope: **RB only** — the canonical handcuff, where a workhorse back's injury concentrates touches
onto one clear backup. Model-only: ECR/ADP stay benchmarks, never inputs (PROJECT_PLAN no-blend).

**Scoring matters here, and not only in magnitude.** The board is built from projections, so the
configured format (``target.scoring``) decides both the contingent-upside numbers *and* which
player is the handcuff — the handcuff is whichever backup projects highest, and halving reception
value reorders pass-catching backs against early-down ones. Roster shape is irrelevant by
construction: this compares a starter to his own backup, so replacement level never enters.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..scoring import scoring_of

GAMES_TARGET = "games_next"


def _season_length(season: int) -> int:
    """Regular-season games: 17 from 2021 on, else 16."""
    return 17 if int(season) >= 2021 else 16


# --- risk signals: each maps (train, score) -> predicted next-season games for the score rows ----

def _predict_availability_model(train, score, *, feature_cols, seed, horizon=1):
    """§7.4 availability model: a Poisson-objective GBM over the full feature set."""
    from ..models.zoo import make_availability_estimator
    tr = train[train[GAMES_TARGET].notna()]
    est = make_availability_estimator(seed=seed)
    est.fit(tr[feature_cols], tr[GAMES_TARGET])
    return est.predict(score[feature_cols])


def _predict_prior_games(train, score, *, horizon=1, **_):
    """Naive baseline: next season's games ≈ this season's games."""
    return score["games"].to_numpy(dtype=float)


def _predict_durability(train, score, *, horizon=1, **_):
    """Transparent durability index: 3-yr availability rate × next season's length."""
    import numpy as np
    rate = score["avail_rate_3yr"].fillna(score["availability_rate"]).to_numpy(dtype=float)
    length = np.array([_season_length(int(s) + horizon) for s in score["season"]], dtype=float)
    return length * rate


SIGNALS = {
    "availability_model": _predict_availability_model,
    "durability_3yr": _predict_durability,
    "prior_games": _predict_prior_games,
}

_BACKTEST_METRICS = ("games_mae", "games_rmse", "clears_auc", "clears_pr_auc")


def backtest_risk_signals(history, feature_cols, *, cutoff, horizon, seed,
                          test_seasons=None, min_train=30, min_test=5):
    """Walk-forward backtest of every risk signal on labeled history; return (table, winner).

    For each signal and each held-out feature season, train on strictly earlier labeled rows and
    score next-season games. The winner maximises mean **clears-cutoff AUC** (separating durable
    from fragile — what handcuff ranking needs), tie-broken by lower games MAE.
    """
    import numpy as np
    import pandas as pd

    labeled = history[history[GAMES_TARGET].notna()]
    seasons = sorted(int(s) for s in labeled["season"].unique())
    if test_seasons is None:
        test_seasons = seasons[-5:]

    rows = []
    for name, fn in SIGNALS.items():
        per = []
        for fs in test_seasons:
            train = labeled[labeled["season"] < fs]
            score = labeled[labeled["season"] == fs]
            if len(train) < min_train or len(score) < min_test:
                continue
            pred = fn(train, score, feature_cols=feature_cols, seed=seed, horizon=horizon)
            from .metrics import availability_metrics
            per.append(availability_metrics(score[GAMES_TARGET], pred, cutoff=cutoff))
        if per:
            agg = {m: float(np.nanmean([p[m] for p in per])) for m in _BACKTEST_METRICS}
            rows.append({"signal": name, **agg, "n_folds": len(per)})

    table = pd.DataFrame(rows)
    if table.empty:
        return table, "prior_games"
    table = table.sort_values(["clears_auc", "games_mae"],
                              ascending=[False, True]).reset_index(drop=True)
    return table, str(table.iloc[0]["signal"])


def project_risk(df, feature_cols, board, *, board_season, signal, horizon, seed):
    """Fit the chosen risk signal on labeled history < board_season; score the board rows.

    Returns a frame keyed on ``player_id`` with predicted next-season games and the implied
    ``exp_games_missed`` / ``miss_share`` (share of the season the player is projected to miss).
    """
    import numpy as np
    import pandas as pd

    labeled = df[df[GAMES_TARGET].notna() & (df["season"] < board_season)]
    pred = SIGNALS[signal](labeled, board, feature_cols=feature_cols, seed=seed, horizon=horizon)
    length = _season_length(int(board_season) + horizon)
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, length)
    return pd.DataFrame({
        "player_id": board["player_id"].to_numpy(),
        "pred_games_next": np.round(pred, 1),
        "exp_games_missed": np.round(length - pred, 1),
        "miss_share": np.round((length - pred) / length, 3),
    })


def build_handcuff_board(rb_proj, features_board, risk, *, max_starter_rank=36, min_handcuff_ppg=0.0):
    """Per team: starter = top projected RB, handcuff = next. Score the backup's contingent upside.

    ``contingent_upside`` = (starter − handcuff projected PPG) × the starter's projected miss share
    — the PPG the backup gains, in expectation, from the starter's injury risk. Only teams whose
    starter is a real fantasy starter (``proj_pos_rank <= max_starter_rank``) are included. Ranked
    by contingent upside (the handcuff-specific signal). Returns a tidy board.
    """
    import pandas as pd

    team = features_board[["player_id", "recent_team"]].rename(columns={"recent_team": "team"})
    proj = (rb_proj.merge(team, on="player_id", how="left")
                   .merge(risk, on="player_id", how="left")
                   .dropna(subset=["team"]))

    rows = []
    for tm, g in proj.groupby("team"):
        g = g.sort_values("proj_ppg", ascending=False)
        if len(g) < 2:
            continue
        starter, backup = g.iloc[0], g.iloc[1]
        if int(starter["proj_pos_rank"]) > max_starter_rank:
            continue
        if float(backup["proj_ppg"]) < min_handcuff_ppg:
            continue
        miss_share = float(starter.get("miss_share") or 0.0)
        gap = max(float(starter["proj_ppg"]) - float(backup["proj_ppg"]), 0.0)
        contingent = gap * miss_share
        rows.append({
            "team": tm,
            "starter": starter["player_name"],
            "starter_pos_rank": int(starter["proj_pos_rank"]),
            "starter_proj_ppg": round(float(starter["proj_ppg"]), 2),
            "starter_exp_games_missed": round(float(starter.get("exp_games_missed") or 0.0), 1),
            "starter_miss_share": round(miss_share, 3),
            "handcuff": backup["player_name"],
            "handcuff_pos_rank": int(backup["proj_pos_rank"]),
            "handcuff_proj_ppg": round(float(backup["proj_ppg"]), 2),
            "contingent_upside": round(contingent, 2),
            "handcuff_value": round(float(backup["proj_ppg"]) + contingent, 2),
        })

    board = pd.DataFrame(rows)
    if board.empty:
        return board
    board = board.sort_values("contingent_upside", ascending=False).reset_index(drop=True)
    board.insert(0, "rank", range(1, len(board) + 1))
    return board


@dataclass
class HandcuffResult:
    board: object
    backtest: object
    signal: str
    winner: str
    season: int
    cutoff: int
    scoring: str = "ppr"


def _board_with_risk(config, *, draft_season, signal, seed):
    """Shared setup for the handcuff tools: project the position and attach the injury-risk signal.

    Returns ``(proj, features_board, risk, backtest, chosen, winner, board_season, horizon,
    cutoff)``. ``draft_season`` defaults to the upcoming season; passing an earlier year
    reconstructs that draft leak-safely (risk + projection train only on strictly earlier labeled
    rows). The risk feature set drops the offseason block — it is opportunity context (not
    durability) and degenerate on a live board's N+1 horizon, which wrecks the tree count model.
    """
    import json

    from .experiment import _all_feature_columns
    from .projection import project_position
    from ..utils.io import DATA_PROCESSED, read_parquet
    from ..utils.naming import artifact_stem

    stem = artifact_stem(config)
    seed = int(seed if seed is not None else config.get("reproducibility.random_seed", 1729))
    horizon = int(config.get("target.predict_horizon", 1))
    cutoff = int(config.get("eligibility.chosen_games_played", 4))

    df = read_parquet(DATA_PROCESSED / f"{stem}_features.parquet")
    block_columns = json.load(open(DATA_PROCESSED / f"{stem}_feature_blocks.json"))
    feature_cols = [c for c in _all_feature_columns(block_columns, df.columns)
                    if c not in set(block_columns.get("offseason", []))]

    board_season = (int(df["season"].max()) if draft_season is None
                    else int(draft_season) - horizon)
    features_board = df[df["season"] == board_season].copy()

    history = df[df["season"] < board_season]
    backtest, winner = backtest_risk_signals(history, feature_cols, cutoff=cutoff,
                                             horizon=horizon, seed=seed)
    chosen = signal or winner
    risk = project_risk(df, feature_cols, features_board, board_season=board_season,
                        signal=chosen, horizon=horizon, seed=seed)
    proj = project_position(config, feature_season=board_season)
    return (proj, features_board, risk, backtest, chosen, winner, board_season, horizon, cutoff)


def run_handcuff(config, *, draft_season=None, signal=None, seed=None, max_starter_rank=36):
    """Full RB handcuff workflow (starter→backup contingent-upside board). See module docstring."""
    (proj, features_board, risk, backtest, chosen, winner,
     board_season, horizon, cutoff) = _board_with_risk(
        config, draft_season=draft_season, signal=signal, seed=seed)
    board = build_handcuff_board(proj, features_board, risk, max_starter_rank=max_starter_rank)
    return HandcuffResult(board=board, backtest=backtest, signal=chosen, winner=winner,
                          season=board_season + horizon, cutoff=cutoff,
                          scoring=scoring_of(config))


@dataclass
class InjuryRiskResult:
    risk_list: object
    backtest: object
    signal: str
    winner: str
    season: int
    cutoff: int
    n_starters: int
    scoring: str = "ppr"


def build_injury_risk_list(proj, risk, *, top_starters=32):
    """Rank projected-starter QBs by injury/availability risk and tier them.

    For pass-catchers a backup rarely inherits standalone value, so for QB the useful deliverable is
    just *who is likely to miss time* → draft a backup. Returns the projected top-``top_starters``
    QBs ranked by expected games missed, with **relative** tiers (High/Moderate/Lower terciles of
    the starter cohort). NB the QB games model regresses toward a backup-heavy pool mean, so the
    *ranking* is trustworthy (clears-AUC ~0.90) but the absolute games number is not — hence tiers.
    """
    m = proj.merge(risk, on="player_id", how="left")
    starters = m[m["proj_pos_rank"] <= top_starters].copy()
    starters = starters.sort_values("exp_games_missed", ascending=False).reset_index(drop=True)
    n = len(starters)
    if n == 0:
        return starters
    starters.insert(0, "risk_rank", range(1, n + 1))

    # Quartile tiers (rank is 0-based here). High = top 25% only, so the actionable
    # "draft a backup" flag stays high-precision — within the starter cohort the model's
    # discrimination is softer than its headline AUC, so a loose cut flags durable QBs too.
    def _tier(i):
        if i < n * 0.25:
            return "High"
        return "Lower" if i >= n * 0.75 else "Moderate"

    starters["risk_tier"] = [_tier(i) for i in range(n)]
    starters["draft_backup"] = starters["risk_tier"] == "High"
    cols = ["risk_rank", "player_name", "proj_pos_rank", "proj_ppg", "pred_games_next",
            "exp_games_missed", "risk_tier", "draft_backup"]
    return starters[cols]


def run_injury_risk(config, *, draft_season=None, signal=None, seed=None, top_starters=32):
    """Injury-risk list for a position (used for QB). See :func:`build_injury_risk_list`."""
    (proj, _features_board, risk, backtest, chosen, winner,
     board_season, horizon, cutoff) = _board_with_risk(
        config, draft_season=draft_season, signal=signal, seed=seed)
    risk_list = build_injury_risk_list(proj, risk, top_starters=top_starters)
    return InjuryRiskResult(risk_list=risk_list, backtest=backtest, signal=chosen, winner=winner,
                            season=board_season + horizon, cutoff=cutoff,
                            n_starters=len(risk_list), scoring=scoring_of(config))


def render_markdown(result: HandcuffResult, *, top: int | None = None) -> str:
    """Render a handcuff board to markdown (the same content the CSV carries, ranked)."""
    lines = [f"# Handcuff Board — {result.season} (RB)", ""]
    lines.append(f"_Scoring: **{result.scoring.replace('_', ' ')}**. Model-only (ECR/ADP are "
                 f"benchmarks, never inputs). A handcuff's **contingent upside** = "
                 f"(starter − backup projected PPG) × the starter's projected miss share._")
    lines.append("")

    lines.append("## Risk signal (leak-safe backtest)")
    lines.append("")
    lines.append(f"Chosen: **`{result.signal}`** "
                 f"(best clears-cutoff AUC at g\\*={result.cutoff}; "
                 f"`winner`=`{result.winner}`).")
    lines.append("")
    bt = result.backtest
    if bt is not None and not bt.empty:
        lines.append("| signal | games MAE | clears AUC | folds |")
        lines.append("|---|---|---|---|")
        for _, r in bt.iterrows():
            lines.append(f"| {r['signal']} | {r['games_mae']:.2f} | "
                         f"{r['clears_auc']:.3f} | {int(r['n_folds'])} |")
        lines.append("")

    lines.append("## Top handcuffs to draft")
    lines.append("")
    board = result.board
    if board is None or board.empty:
        lines.append("_No handcuff candidates (need ≥2 projected RBs per team)._")
        return "\n".join(lines) + "\n"
    show = board if top is None else board.head(top)
    lines.append("| # | handcuff (RB rank) | starter (RB rank) | starter exp. games missed | "
                 "contingent upside | handcuff value |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in show.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['handcuff']} ({int(r['handcuff_pos_rank'])}) | "
            f"{r['starter']} ({int(r['starter_pos_rank'])}) | "
            f"{r['starter_exp_games_missed']:.1f} | {r['contingent_upside']:.2f} | "
            f"{r['handcuff_value']:.2f} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_injury_markdown(result: InjuryRiskResult, *, position: str = "QB") -> str:
    """Render a position injury-risk list to markdown (the same content the CSV carries)."""
    lines = [f"# {position} Injury-Risk List — {result.season}", ""]
    lines.append(f"_Scoring: **{result.scoring.replace('_', ' ')}**. Projected {position} starters "
                 f"ranked by injury/availability risk — **draft a backup** for the High tier. "
                 f"Model-only (ECR/ADP are benchmarks)._")
    lines.append("")
    lines.append("> **Read the tiers, not the raw number.** The QB games model regresses toward a "
                 "backup-heavy pool mean, so it ranks risk well (clears-cutoff AUC ~0.90) but "
                 "under-predicts absolute games for everyone. `risk_tier` is **relative to the "
                 "projected-starter cohort** (High = riskiest quartile).")
    lines.append(">")
    lines.append("> **Caveat:** the model reads rushing/workload as injury exposure, so durable "
                 "high-usage QBs (e.g. Josh Allen, Lamar Jackson) can be flagged riskier than their "
                 "track record warrants — they are outliers who sustain that load. Treat a long "
                 "clean availability history as a discount on the model's ranking.")
    lines.append("")

    lines.append("## Risk signal (leak-safe backtest)")
    lines.append("")
    lines.append(f"Chosen: **`{result.signal}`** (best clears-cutoff AUC at g\\*={result.cutoff}; "
                 f"`winner`=`{result.winner}`).")
    lines.append("")
    bt = result.backtest
    if bt is not None and not bt.empty:
        lines.append("| signal | games MAE | clears AUC | folds |")
        lines.append("|---|---|---|---|")
        for _, r in bt.iterrows():
            lines.append(f"| {r['signal']} | {r['games_mae']:.2f} | "
                         f"{r['clears_auc']:.3f} | {int(r['n_folds'])} |")
        lines.append("")

    lines.append(f"## {position}s most likely to miss time (draft a backup)")
    lines.append("")
    rl = result.risk_list
    if rl is None or rl.empty:
        lines.append(f"_No projected {position} starters found._")
        return "\n".join(lines) + "\n"
    lines.append(f"| # | {position} ({position} rank) | proj PPG | risk tier | draft a backup? |")
    lines.append("|---|---|---|---|---|")
    for _, r in rl.iterrows():
        flag = "**yes**" if bool(r["draft_backup"]) else "—"
        lines.append(f"| {int(r['risk_rank'])} | {r['player_name']} "
                     f"({int(r['proj_pos_rank'])}) | {r['proj_ppg']:.1f} | "
                     f"{r['risk_tier']} | {flag} |")
    lines.append("")
    return "\n".join(lines) + "\n"
