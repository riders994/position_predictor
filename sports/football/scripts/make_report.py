"""Stage 9 — build the results report (PROJECT_PLAN §3 ``[9] report``, §8).

Renders ``reports/REPORT_<sport>_<position>.md`` + summary figures from the Stage-8 result tables.
Run ``make experiment`` (and optionally ``make benchmark``) first.

Usage:
    uv run python scripts/make_report.py --config config/football_rb.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.eval.report import build_report  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the results report (stage 9).")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    print(f"[report] {cfg}")
    text = build_report(cfg, write=True)
    sport = cfg.get("experiment.sport", "sport")
    position = cfg.require("experiment.position")
    stem = f"{sport}_{position}".lower()
    print(text)
    print(f"\n[report] wrote reports/REPORT_{stem}.md + reports/figures/report_{stem}_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
