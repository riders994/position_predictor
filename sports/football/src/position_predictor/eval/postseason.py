"""Postseason report — grade the model, ECR, and ADP against a completed season's finish.

Use-case #3 (after keeper + redraft). For a finished season **Y** this assembles, per player:
the model's **preseason** projection for Y, the **ECR** and **ADP** going into Y, and the **actual**
Y finish — then a summary of how well each source called the season.

Faithfulness matters: the model's "preseason" prediction is reconstructed with
:func:`eval.projection.project_position` boarding the **Y−1** feature rows and training only on
labels known before Y (its ``feature_season`` arg), so the backtest never trains on the answer.
ECR comes from the committed ``data/external/market_<stem>.parquet``; ADP from
:func:`data.adp.build_adp_benchmark`; actuals are the season-Y ``ppg``/``games`` already in the
processed features table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..data.availability import (AvailabilityReport, check_season_available,
                                 datasets_needing_refresh)
from ..data.build import build_dataset
from ..eval.keeper import _norm
from ..eval.metrics import ranking_metrics
from ..eval.projection import project_position
from ..eval.redraft import _override_season
from ..features.build import build_features

# How far back to auto-probe for the latest fully-published completed season.
_AUTODETECT_LOOKBACK = 3
# Columns of the per-player board, in display order.
BOARD_COLS = ["position", "player_name", "player_id", "model_proj_ppg", "model_rank",
              "ecr", "ecr_rank", "adp", "adp_rank", "actual_ppg", "actual_rank"]


@dataclass
class PostseasonResult:
    season: int
    availability: AvailabilityReport
    board: object = None               # combined per-player DataFrame
    summary: list = field(default_factory=list)   # rank-accuracy rows (position × source)
    highlights: dict = field(default_factory=dict)
    adp_match: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.availability.ok


def _resolve_season(season, base_cfg):
    """Return ``(season, availability_report)``. If ``season`` is None, probe back from last year
    to the most recent fully-published completed season."""
    if season is not None:
        return int(season), check_season_available(int(season))
    start = datetime.now().year - 1
    rep = None
    for cand in range(start, start - _AUTODETECT_LOOKBACK, -1):
        rep = check_season_available(cand)
        if rep.ok:
            return cand, rep
    return start, rep  # none ready — return the first probe for messaging


def _rank_accuracy(actual_ppg, actual_rank, score, pred_rank, k_tiers):
    """ranking_metrics(actual, score) on covered∩actual rows + mean |pred_rank − actual_rank|."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"y": actual_ppg, "s": score, "pr": pred_rank, "ar": actual_rank}).dropna(
        subset=["y", "s"])
    if len(df) < 2:
        return None
    m = ranking_metrics(df["y"].to_numpy(), df["s"].to_numpy(), k_tiers=k_tiers)
    both = df.dropna(subset=["pr", "ar"])
    m["mean_abs_rank_err"] = (float(np.abs(both["pr"] - both["ar"]).mean())
                              if len(both) else float("nan"))
    m["n"] = int(len(df))
    return m


def _position_report(cfg, season: int):
    """Build the per-player table + per-source rank accuracy + highlights for one position."""
    import pandas as pd

    from ..data.adp import build_adp_benchmark
    from ..utils.io import DATA_EXTERNAL, DATA_PROCESSED, read_parquet

    sport = cfg.get("experiment.sport", "sport")
    position = cfg.require("experiment.position").upper()
    stem = f"{sport}_{position}".lower()
    g_star = int(cfg.get("eligibility.chosen_games_played", 4))
    k_tiers = tuple(cfg.get("metrics.precision_at_k_tiers", [12, 24, 36]))

    # Materialise data through Y, then reconstruct the preseason (Y-1) projection (leak-safe).
    cfg2 = _override_season(cfg, season)
    build_dataset(cfg2, write=True)
    build_features(cfg2, write=True)
    model = project_position(cfg2, feature_season=season - 1, write=False).rename(
        columns={"proj_ppg": "model_proj_ppg", "proj_pos_rank": "model_rank"})

    feats = read_parquet(DATA_PROCESSED / f"{stem}_features.parquet")
    yr = feats[feats["season"] == season].copy()
    name_col = "player_display_name" if "player_display_name" in yr.columns else "player_name"
    yr = yr[["player_id", name_col, "ppg", "games"]].rename(
        columns={name_col: "player_name", "ppg": "actual_ppg"})
    elig = yr[yr["games"] >= g_star].copy()
    elig["actual_rank"] = elig["actual_ppg"].rank(ascending=False, method="min").astype(int)
    actual = yr.merge(elig[["player_id", "actual_rank"]], on="player_id", how="left")

    # ECR (committed) and ADP (fetched + name-matched to this season's universe).
    mpath = DATA_EXTERNAL / f"market_{stem}.parquet"
    if mpath.exists():
        ecr = read_parquet(mpath)
        ecr = ecr[ecr["season"] == season][["player_id", "market_ecr", "market_rank"]].rename(
            columns={"market_ecr": "ecr", "market_rank": "ecr_rank"})
    else:
        ecr = pd.DataFrame(columns=["player_id", "ecr", "ecr_rank"])

    name_id_map = {_norm(n): pid for n, pid in zip(actual["player_name"], actual["player_id"])}
    try:
        adp, match = build_adp_benchmark(cfg, season, name_id_map, write=True)
        adp = adp[["player_id", "adp", "adp_pos_rank"]].rename(columns={"adp_pos_rank": "adp_rank"})
    except Exception as exc:  # noqa: BLE001 — ADP source down → report without ADP
        adp = pd.DataFrame(columns=["player_id", "adp", "adp_rank"])
        match = {"ranked": 0, "matched": 0, "error": f"{type(exc).__name__}: {exc}"}

    board = actual.merge(model[["player_id", "model_proj_ppg", "model_rank"]],
                         on="player_id", how="outer")
    board = board.merge(ecr, on="player_id", how="outer").merge(adp, on="player_id", how="outer")
    # coalesce names (model carries names for veterans who didn't play Y)
    board = board.merge(model[["player_id", "player_name"]].rename(
        columns={"player_name": "_mname"}), on="player_id", how="left")
    board["player_name"] = board["player_name"].fillna(board["_mname"])
    board["position"] = position
    board = board.drop(columns="_mname")[BOARD_COLS]

    summary = []
    for src, score, pr in [("model", board["model_proj_ppg"], board["model_rank"]),
                           ("ecr", -board["ecr"], board["ecr_rank"]),
                           ("adp", -board["adp"], board["adp_rank"])]:
        m = _rank_accuracy(board["actual_ppg"], board["actual_rank"], score, pr, k_tiers)
        if m is not None:
            summary.append({"position": position, "source": src, **m})

    highlights = _highlights(board, k_tiers[0])
    return board, summary, highlights, match


def _highlights(board, tier1, n=5):
    """Biggest model hits / busts and best ADP values / reaches (small, narrative lists)."""
    b = board.copy()
    hits = b[(b["model_rank"] <= tier1) & (b["actual_rank"] <= tier1)].sort_values(
        "actual_ppg", ascending=False).head(n)
    bust = b[b["model_rank"] <= tier1].copy()
    bust["miss"] = bust["actual_rank"].fillna(999) - bust["model_rank"]
    busts = bust.sort_values("miss", ascending=False).head(n)
    val = b.dropna(subset=["adp_rank", "actual_rank"]).copy()
    val["value"] = val["adp_rank"] - val["actual_rank"]      # finished better than drafted
    values = val.sort_values("value", ascending=False).head(n)
    reaches = val.sort_values("value", ascending=True).head(n)

    def _slim(df, *extra):
        cols = ["player_name", "model_rank", "ecr_rank", "adp_rank", "actual_rank", *extra]
        return df[[c for c in cols if c in df.columns]].to_dict("records")
    return {"hits": _slim(hits), "busts": _slim(busts, "miss"),
            "values": _slim(values, "value"), "reaches": _slim(reaches, "value")}


def build_postseason_report(configs, *, season=None, refresh: bool = True) -> PostseasonResult:
    """Run the postseason report across the position configs. See module docstring."""
    import pandas as pd

    from ..data.fetch import fetch_all

    configs = list(configs)
    if not configs:
        raise ValueError("build_postseason_report needs at least one position config")

    base = configs[0]
    season, report = _resolve_season(season, base)
    result = PostseasonResult(season=season, availability=report)
    if not report.ok:
        return result

    if refresh:
        earliest = int(base.require("data.earliest_season"))
        stale = datasets_needing_refresh(season)
        if stale:
            fetch_all(list(range(earliest, season + 1)), datasets=stale, overwrite=True)

    boards, summaries = [], []
    for cfg in configs:
        board, summary, highlights, match = _position_report(cfg, season)
        boards.append(board)
        summaries.extend(summary)
        pos = cfg.require("experiment.position").upper()
        result.highlights[pos] = highlights
        result.adp_match[pos] = match

    result.board = pd.concat(boards, ignore_index=True) if boards else pd.DataFrame()
    result.summary = summaries
    return result


def _fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def render_markdown(result: PostseasonResult, *, top: int = 24) -> str:
    """Render the postseason report as Markdown (per-position accuracy + highlights + top board)."""
    import pandas as pd
    from datetime import date

    Y = result.season
    lines = [f"# Postseason report — {Y}", "",
             f"_Generated {date.today().isoformat()}._ Model preseason projection (trained on "
             f"labels < {Y}) vs preseason **ECR** and **ADP**, graded against actual {Y} finish "
             "(PPR PPG). Ranking metrics are computed on each source's covered ∩ actual-eligible "
             "players.", ""]
    summ = pd.DataFrame(result.summary)
    board = result.board if result.board is not None else pd.DataFrame()

    for pos in board["position"].drop_duplicates() if not board.empty else []:
        lines += [f"## {pos}", ""]
        s = summ[summ["position"] == pos] if not summ.empty else pd.DataFrame()
        if not s.empty:
            prec = sorted(int(c.rsplit("_", 1)[1]) for c in s.columns
                          if c.startswith("precision_at_") and c.rsplit("_", 1)[1].isdigit())
            head = ["source", "n", "Spearman"] + [f"P@{k}" for k in prec] + ["mean rank err"]
            lines += ["| " + " | ".join(head) + " |",
                      "|" + "|".join("---" for _ in head) + "|"]
            label = {"model": "**model**", "ecr": "ECR", "adp": "ADP"}
            for _, r in s.iterrows():
                cells = [label.get(r["source"], r["source"]), str(int(r["n"])),
                         _fmt(r.get("spearman"), 3)] \
                    + [_fmt(r.get(f"precision_at_{k}")) for k in prec] \
                    + [_fmt(r.get("mean_abs_rank_err"), 1)]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        m = result.adp_match.get(pos, {})
        if m:
            lines.append(f"_ADP coverage: matched {m.get('matched', 0)}/{m.get('ranked', 0)} "
                         f"FFC names to {pos}." + (f" ({m['error']})" if m.get("error") else "") + "_")
            lines.append("")
        lines += _highlight_block(result.highlights.get(pos, {}))

        sub = board[board["position"] == pos].sort_values(
            "actual_rank", na_position="last").head(top)
        lines += ["", f"### {pos} — top {top} by actual finish", "",
                  "| actual | player | model | ECR | ADP | actual PPG |",
                  "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(f"| {_rk(r['actual_rank'])} | {r['player_name']} | {_rk(r['model_rank'])} "
                         f"| {_rk(r['ecr_rank'])} | {_rk(r['adp_rank'])} | "
                         f"{_fmt(r['actual_ppg'])} |")
        lines.append("")
    return "\n".join(lines)


def _rk(x):
    try:
        return str(int(x))
    except (TypeError, ValueError):
        return "—"


def _highlight_block(h):
    if not h:
        return []
    out = []
    titles = {"hits": "Model hits (called top tier, finished top tier)",
              "busts": "Model busts (ranked top tier, finished worse)",
              "values": "Best ADP values (finished better than drafted)",
              "reaches": "Biggest ADP reaches (drafted earlier than they finished)"}
    for key in ("hits", "busts", "values", "reaches"):
        rows = h.get(key) or []
        if not rows:
            continue
        names = ", ".join(
            f"{r['player_name']} (act {_rk(r.get('actual_rank'))}"
            + (f", mdl {_rk(r.get('model_rank'))}" if key in ("hits", "busts") else
               f", adp {_rk(r.get('adp_rank'))}") + ")"
            for r in rows)
        out.append(f"- **{titles[key]}:** {names}")
    return out + [""]
