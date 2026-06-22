"""Cross-version progress report.

Renders ``reports/versions/<sport>_<position>/PROGRESS_<stem>.md`` from the committed
per-version snapshots, tracking ranking quality vs the compute / data volume each
version took. Snapshots are written by ``make report`` when the config carries an
``experiment.version``; run this after a couple of versions exist.

Usage:
    uv run python scripts/make_progress.py --config config/football_rb.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.eval.progress import build_progress  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the cross-version progress report.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    print(f"[progress] {cfg}")
    text = build_progress(cfg, write=True)
    if text is None:
        print("[progress] no version snapshots found under reports/versions/ — nothing to do.")
        return 0
    sport = cfg.get("experiment.sport", "sport")
    position = cfg.require("experiment.position")
    stem = f"{sport}_{position}".lower()
    print(text)
    print(f"\n[progress] wrote reports/versions/{stem}/PROGRESS_{stem}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
