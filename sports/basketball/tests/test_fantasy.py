"""No-network test for the Phase-2 success-label shaping (max_pf.run is monkeypatched)."""
import sys
from pathlib import Path

import max_pf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.data.fantrax import fetch_success_labels  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402


def test_success_labels_shape_and_lineup_gap(monkeypatch):
    TS = max_pf.TeamSeason
    fake = [
        TS(team_id="t1", name="Alpha", periods=20,
           actual_pf=120.0, m1_pf=150.0, m2_pf=148.0, m3_pf=145.0),
        TS(team_id="t2", name="Beta", periods=20,
           actual_pf=140.0, m1_pf=145.0, m2_pf=143.0, m3_pf=141.0),
    ]
    monkeypatch.setattr(max_pf, "run", lambda login, **kw: fake)

    cfg = Config.load(ROOT / "config" / "nba_archetypes.yaml")  # two league_ids
    df = fetch_success_labels(cfg, write=False)

    assert len(df) == 2 * len(cfg.get("fantasy.league_ids"))     # teams x leagues
    assert {"league_id", "team_id", "name", "m1_pf", "lineup_gap"} <= set(df.columns)
    alpha = df[df.name == "Alpha"].iloc[0]
    assert alpha["lineup_gap"] == 30.0                           # m1_pf - actual_pf = 150 - 120
    assert set(df["league_id"]) == set(cfg.get("fantasy.league_ids"))
