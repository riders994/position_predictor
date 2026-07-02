"""Tests for the Phase-3 next-season archetype predictor (leak-safe table + walk-forward)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.predict.build import (  # noqa: E402
    build_predict_table, feature_columns, target_prob_columns)
from nba_archetypes.predict.model import walk_forward  # noqa: E402


def _synthetic(n_players=80, seasons=range(2014, 2022), n_arch=12, seed=0):
    """Membership + features frames where each player mostly keeps a latent archetype (sticky) but drifts
    predictably with experience — so persistence is strong and a model can add calibrated signal."""
    rng = np.random.default_rng(seed)
    zcols = [f"f{i}_z" for i in range(6)]
    mem_rows, feat_rows = [], []
    for pid in range(n_players):
        arch = int(rng.integers(n_arch))
        for k, s in enumerate(seasons):
            if rng.random() < 0.15 + 0.02 * k:            # drift chance rises with experience
                arch = int(rng.integers(n_arch))
            probs = np.full(n_arch, 0.02)
            probs[arch] = 1.0
            probs = probs / probs.sum()
            mem_rows.append({"athlete_id": pid, "season": s, "player_name": f"P{pid}", "arch": arch,
                             "arch_name": f"A{arch}", "top_prob": round(float(probs[arch]), 3),
                             "entropy": 0.2, **{f"p{j}": round(float(probs[j]), 3) for j in range(n_arch)}})
            feat_rows.append({"athlete_id": pid, "season": s,
                              **{c: float(rng.normal(arch - 6, 1.0)) for c in zcols}})
    return pd.DataFrame(mem_rows), pd.DataFrame(feat_rows)


def test_build_table_is_leak_safe_and_paired():
    mem, feat = _synthetic()
    t = build_predict_table(mem, feat, write=False)
    assert (t["target_arch"] >= 0).all()
    # target is the NEXT season's arch — reconstruct and compare
    lut = mem.set_index(["athlete_id", "season"])["arch"]
    for _, r in t.sample(20, random_state=1).iterrows():
        assert r["target_arch"] == lut[(r["athlete_id"], r["season"] + 1)]


def test_build_table_yoe_and_tenure_monotone():
    mem, feat = _synthetic()
    t = build_predict_table(mem, feat, write=False)
    # yoe starts at 0 for a player's first predictable season and increases by 1 per season
    for pid, g in t.sort_values("season").groupby("athlete_id"):
        assert g["yoe"].iloc[0] == g["season"].iloc[0] - mem[mem.athlete_id == pid].season.min()
        assert list(g["yoe"]) == sorted(g["yoe"])
    assert (t["arch_tenure"] >= 1).all()


def test_feature_columns_kind_switch():
    mem, feat = _synthetic()
    t = build_predict_table(mem, feat, write=False)
    yoe_cols = feature_columns(t, kind="yoe")
    assert "yoe" in yoe_cols and "age" not in yoe_cols
    t["age"] = 25.0
    age_cols = feature_columns(t, kind="age")
    assert "age" in age_cols and "yoe" not in age_cols
    assert len(target_prob_columns(t)) == 12


def test_walk_forward_reports_baselines_on_same_rows():
    mem, feat = _synthetic(seed=2)
    t = build_predict_table(mem, feat, write=False)
    res = walk_forward(t, kind="yoe")
    assert res["n_eval"] > 0
    for name in ("model", "persistence", "marginal"):
        assert 0 <= res[name]["accuracy"] <= 1
        assert res[name]["log_loss"] >= 0
    # sticky data -> persistence must crush the marginal (most-common-class) floor
    assert res["persistence"]["accuracy"] > res["marginal"]["accuracy"]
    # model's soft distribution should beat marginal on log-loss
    assert res["model"]["log_loss"] < res["marginal"]["log_loss"]
