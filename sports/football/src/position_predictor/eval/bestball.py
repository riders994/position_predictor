"""Best ball valuation — pricing week-to-week upside, not just the season mean.

In a managed league you set a lineup before kickoff, so a player is worth his *expected* points
and a bad week costs you the slot. In **best ball** the platform scores your best lineup after
the fact: a dud week costs nothing (someone else fills the slot) while a spike week is captured
in full. Upside is therefore worth real points, and two players with the same projected PPG are
not worth the same.

How that is measured here
-------------------------
**Realized best-ball value** of a player-week is ``max(0, his points - the weekly threshold)``,
where the threshold is the score of that week's *marginal starter* at his position — the
``teams x slots``-th best weekly score league-wide. Above the threshold he'd have been in the
lineup and his surplus is real; below it he'd have been benched and the surplus is zero, not
negative. Averaged over the season this is a weekly VORP with the downside truncated, which is
exactly what best-ball scoring does. It is computed directly from weekly box scores — no roster
simulation and no distributional assumption.

**The board adjustment** is ``bestball_ppg = proj_ppg + lambda * sigma``, where ``sigma`` is the
player's projected week-to-week standard deviation (empirical-Bayes shrunk toward the position
mean, since one 6-game season says little about volatility).

``lambda`` is **fit from history, not assumed** (:func:`fit_lambdas`): for each past season we
sweep lambda and keep the value whose ranking of ``ppg + lambda*sigma`` best correlates with the
*next* season's realized best-ball value.

The measured answer: **lambda is zero.** Fit over 2012-2024 (13 walk-forward season pairs) the
Spearman-vs-lambda curve is flat to ~0.003 at every position — QB gains exactly nothing, and
RB/WR/TE gain +0.003 with p = 0.10 / 0.13 / 0.19 and 8-9 winning seasons out of 13, i.e. noise.
Re-running with two other upside measures (rate of ceiling weeks above 1.5x the starter
threshold, and weekly skew) reproduces the null, several of them optimising at exactly zero.

The mechanism is visible in the data: within a position-season, weekly sigma is 0.90 rank-
correlated with weekly mean at RB/WR/TE (0.45 at QB). "Upside" is almost entirely a restatement
of "good", so once a player's projected mean is known his volatility carries no extra information
about next year's best-ball value. **The practical conclusion for a best-ball draft is to rank by
projected PPG and not pay up for spike-week reputations.**

So :data:`DEFAULT_LAMBDAS` is all zeros and a best-ball board equals the mean board unless a
lambda is passed explicitly. The machinery stays because the finding is worth re-testing when the
projection model changes, and because ``sigma`` is still reported as context.

Caveat worth stating: lambda is calibrated on *observed* season-N mean and sigma, while the board
applies it to *projected* mean and sigma. Projections are already shrunk toward the mean, so the
fitted lambda is best read as an upper bound on the tilt that upside deserves — which makes a
fitted zero a stronger null, not a weaker one.
"""

from __future__ import annotations

from ..scoring import DEFAULT_SCORING, reception_points
from .league import MODELED_POS

# Volatility prior strength: a player-season is worth its own sigma only once it has enough weeks
# behind it. 8 games ~= half a season, the point at which a sample sigma stops being mostly noise.
SIGMA_PRIOR_GAMES = 8.0
# Minimum weeks before a season contributes a sigma observation at all.
MIN_GAMES_FOR_SIGMA = 4
# Recent seasons weigh more; a player's volatility drifts with role.
SIGMA_SEASON_WEIGHTS = (1.0, 0.6, 0.35)
# Grid swept when fitting lambda. Upside can only ever be worth a fraction of a sigma per game.
LAMBDA_GRID = tuple(round(0.05 * i, 2) for i in range(0, 21))   # 0.00 .. 1.00

# Shipped default: zero at every position — see the module docstring for the calibration that
# produced it (flat to ~0.003 Spearman, p >= 0.10, reproduced across three upside measures).
# Override per run with `--bestball-lambda WR=0.3`, or re-fit with scripts/bestball_calibrate.py.
DEFAULT_LAMBDAS = dict.fromkeys(MODELED_POS, 0.0)


def weekly_fantasy_points(weekly, *, scoring: str = DEFAULT_SCORING, regular_season_only=True):
    """Per player-week fantasy points under ``scoring``, restricted to modeled positions.

    Uses the same exact identity as the season build (:mod:`position_predictor.scoring`), so a
    player's weekly points always sum to his season total.
    """
    df = weekly
    if regular_season_only and "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    grp = "position_group" if "position_group" in df.columns else "position"
    df = df[df[grp].isin(MODELED_POS)]
    out = df[["player_id", "season", "week"]].copy()
    out["position"] = df[grp].values
    out["points"] = df["fantasy_points"] + reception_points(scoring) * df["receptions"].fillna(0)
    return out.dropna(subset=["points"]).reset_index(drop=True)


def starters_per_position(league):
    """Started slots league-wide per position, flex spread across its eligible positions.

    The weekly threshold needs a *position* count, but a flex slot isn't owned by one position.
    Splitting it evenly across the eligible positions is the neutral choice — the alternative
    (assigning it week by week to whoever scored more) would bake the outcome into the threshold.
    """
    counts = {p: float(league.teams * league.starters.get(p, 0)) for p in MODELED_POS}
    flex_total = league.teams * league.starters.get("FLEX", 0)
    if flex_total and league.flex_positions:
        share = flex_total / len(league.flex_positions)
        for p in league.flex_positions:
            counts[p] += share
    return counts


def weekly_thresholds(weekly_pts, league):
    """The marginal starter's score for each ``(season, week, position)``.

    Returns a frame with ``season, week, position, threshold``.
    """
    import pandas as pd

    counts = starters_per_position(league)
    rows = []
    for (season, week, pos), grp in weekly_pts.groupby(["season", "week", "position"],
                                                       sort=False):
        n = int(round(counts.get(pos, 0)))
        if n <= 0:
            continue
        scores = grp["points"].sort_values(ascending=False).to_numpy()
        # The n-th best score is the last starter; anyone below him sat that week.
        threshold = float(scores[n - 1]) if n <= len(scores) else float(scores[-1])
        rows.append({"season": season, "week": week, "position": pos, "threshold": threshold})
    return pd.DataFrame(rows)


def realized_bestball_ppg(weekly_pts, league, *, min_weeks=1):
    """Per ``(player_id, season)`` realized best-ball value per league week.

    Value is averaged over the season's *league weeks*, not the player's games played: missing a
    week contributes zero, which is the real cost of an injury to a best-ball roster (you cannot
    replace him).
    """
    import pandas as pd

    thresholds = weekly_thresholds(weekly_pts, league)
    if thresholds.empty:
        return pd.DataFrame(columns=["player_id", "season", "position", "bestball_ppg_actual"])
    df = weekly_pts.merge(thresholds, on=["season", "week", "position"], how="inner")
    df["surplus"] = (df["points"] - df["threshold"]).clip(lower=0.0)
    weeks_per_season = df.groupby("season")["week"].nunique().rename("league_weeks")
    agg = (df.groupby(["player_id", "season", "position"], as_index=False)
             .agg(surplus_total=("surplus", "sum"), weeks_played=("week", "nunique")))
    agg = agg.merge(weeks_per_season, on="season", how="left")
    agg = agg[agg["weeks_played"] >= min_weeks]
    agg["bestball_ppg_actual"] = agg["surplus_total"] / agg["league_weeks"]
    return agg[["player_id", "season", "position", "bestball_ppg_actual", "weeks_played"]]


def season_volatility(weekly_pts, *, min_games=MIN_GAMES_FOR_SIGMA):
    """Per ``(player_id, season)`` weekly mean, standard deviation and games played."""
    agg = (weekly_pts.groupby(["player_id", "season", "position"], as_index=False)
                     .agg(week_mean=("points", "mean"), week_sd=("points", "std"),
                          games=("week", "nunique")))
    return agg[agg["games"] >= min_games].reset_index(drop=True)


def projected_sigma(volatility, *, board_season, prior_games=SIGMA_PRIOR_GAMES,
                    weights=SIGMA_SEASON_WEIGHTS):
    """Each player's projected weekly sigma for ``board_season``.

    A recency-weighted average of his recent seasons' sample sigma, then shrunk toward the
    position's mean sigma with weight ``n / (n + prior_games)`` — one short season is mostly
    noise, three full ones are mostly signal. Returns ``player_id, position, sigma, sigma_games``.
    """
    import numpy as np
    import pandas as pd

    hist = volatility[volatility["season"] < board_season].copy()
    if hist.empty:
        return pd.DataFrame(columns=["player_id", "position", "sigma", "sigma_games"])
    hist["age"] = board_season - hist["season"]
    hist = hist[hist["age"] <= len(weights)]
    if hist.empty:
        return pd.DataFrame(columns=["player_id", "position", "sigma", "sigma_games"])
    hist["w"] = hist["age"].map({i + 1: w for i, w in enumerate(weights)}).astype(float)
    hist["w_games"] = hist["w"] * hist["games"]
    hist["w_sd"] = hist["w_games"] * hist["week_sd"].fillna(0.0)

    per_player = (hist.groupby(["player_id", "position"], as_index=False)
                      .agg(w_sd=("w_sd", "sum"), w_games=("w_games", "sum"),
                           sigma_games=("games", "sum")))
    per_player["raw_sigma"] = np.where(per_player["w_games"] > 0,
                                       per_player["w_sd"] / per_player["w_games"], np.nan)
    pos_mean = (per_player.groupby("position")["raw_sigma"].mean().rename("pos_sigma"))
    per_player = per_player.merge(pos_mean, on="position", how="left")
    shrink = per_player["sigma_games"] / (per_player["sigma_games"] + prior_games)
    per_player["sigma"] = (shrink * per_player["raw_sigma"].fillna(per_player["pos_sigma"])
                           + (1 - shrink) * per_player["pos_sigma"]).round(3)
    return per_player[["player_id", "position", "sigma", "sigma_games"]]


def apply_bestball(proj, sigma, lambdas):
    """Attach ``sigma`` and ``bestball_ppg = proj_ppg + lambda_pos * sigma`` to a projection frame.

    Players without a sigma estimate (no qualifying history) fall back to their position's mean
    sigma, so they are neither rewarded nor punished for being unmeasured.
    """
    out = proj.merge(sigma[["player_id", "sigma"]], on="player_id", how="left")
    pos_mean = out.groupby("position")["sigma"].transform("mean")
    out["sigma"] = out["sigma"].fillna(pos_mean).fillna(0.0).round(3)
    lam = out["position"].map(lambdas).fillna(0.0)
    out["bestball_lambda"] = lam
    out["bestball_ppg"] = (out["proj_ppg"] + lam * out["sigma"]).round(2)
    return out


def fit_lambdas(weekly_pts, league, *, seasons=None, grid=LAMBDA_GRID, min_players=20):
    """Fit lambda per position by walk-forward rank correlation. Returns ``(lambdas, detail)``.

    For each season *N* we rank players by ``week_mean + lambda * sigma`` (both observed in *N*,
    sigma shrunk from *N* and earlier) and score that ranking against season *N+1*'s realized
    best-ball value. The lambda with the best mean Spearman across seasons wins; ``detail`` keeps
    the whole curve so the report can show how flat or peaked it is.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    vol = season_volatility(weekly_pts)
    actual = realized_bestball_ppg(weekly_pts, league)
    all_seasons = sorted(set(vol["season"]) & set(actual["season"] - 1))
    seasons = sorted(set(seasons) & set(all_seasons)) if seasons is not None else all_seasons

    records = []
    for season in seasons:
        sig = projected_sigma(vol, board_season=season + 1)
        base = vol[vol["season"] == season][["player_id", "position", "week_mean"]]
        nxt = actual[actual["season"] == season + 1][["player_id", "bestball_ppg_actual"]]
        frame = base.merge(sig[["player_id", "sigma"]], on="player_id", how="left")
        frame["sigma"] = frame["sigma"].fillna(
            frame.groupby("position")["sigma"].transform("mean")).fillna(0.0)
        frame = frame.merge(nxt, on="player_id", how="inner")
        for pos, grp in frame.groupby("position"):
            if len(grp) < min_players:
                continue
            for lam in grid:
                score = grp["week_mean"] + lam * grp["sigma"]
                rho = stats.spearmanr(score, grp["bestball_ppg_actual"]).statistic
                if not np.isnan(rho):
                    records.append({"season": season, "position": pos, "lambda": lam,
                                    "spearman": float(rho), "n": len(grp)})

    detail = pd.DataFrame(records)
    if detail.empty:
        return {p: 0.0 for p in MODELED_POS}, detail
    curve = detail.groupby(["position", "lambda"], as_index=False)["spearman"].mean()
    best = curve.loc[curve.groupby("position")["spearman"].idxmax()]
    lambdas = {row["position"]: float(row["lambda"]) for _, row in best.iterrows()}
    return {p: lambdas.get(p, 0.0) for p in MODELED_POS}, detail
