# Football (NFL) — fantasy-rank modeling project

Predict an NFL player's **next-season PPR points-per-game** and the rank it implies, modeled
**separately per position** (RB, WR, QB, TE). A
standalone project: its own code, data, and pipeline. Design rationale in
[`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md); full decision trail in
[`docs/PROMPT_LOG.md`](docs/PROMPT_LOG.md).

## What it does

Uses a player's NFL history through season *N* to predict **season *N+1* PPR PPG**, ranks players
by that prediction, and measures the ranking vs naive baselines and the ADP/ECR market — over
10/20/30-year training windows (recency study). Returning players only; the games-eligibility
cutoff `g*` is derived per position (RB = 4, WR = 7, QB = 7). Market data (ECR from FantasyPros,
ADP from FantasyFootballCalculator) is a **benchmark only — never blended** into the model.

## Structure

```
src/position_predictor/   code: fetch · build · features · eligibility · eda · models · eval
src/qb_breakout/          second project: late-breakout QBs from pre-NFL evidence (see below)
scripts/                  stage entrypoints + serving tools (keeper/redraft/postseason)
config/                   per-position experiment configs (football_{rb,wr,qb}.yaml)
docs/                     PROJECT_PLAN · QB_BREAKOUT_PLAN · data_dictionary · PROMPT_LOG
notebooks/rb/             EDA, eligibility-cutoff, feature analysis
reports/                  committed REPORT_*.md + versions/<stem>/<v>/ (figures/results git-ignored)
data/                     git-ignored cache (committed: raw/_manifests/ + external/ reference)
tests/
```

Paths resolve relative to this project root, so commands work from anywhere via `-C` /
`--config`.

## Run it

```bash
export PATH="$HOME/.local/bin:$PATH"
uv sync                                   # shared env (from the repo root pyproject)

# full pipeline for one position (CONFIG defaults to football RB)
make -C sports/football fetch build features eligibility experiment benchmark report

# serving tools (write under sports/football/reports/)
make -C sports/football project    CONFIG=config/football_rb.yaml   # next-season projection board
make -C sports/football redraft    SEASON=2026                      # new-season draft board (QB/RB/WR/TE)
make -C sports/football postseason SEASON=2025                      # grade model/ECR/ADP vs actuals
make -C sports/football handcuff   SEASON=2026                      # RB backups to draft by starter injury risk
make -C sports/football handcuff   CONFIG=config/football_qb.yaml   # QB injury-risk list (who to draft a backup for)
uv run python sports/football/scripts/keeper.py \
    --input sports/football/examples/keepers_example.csv --teams 12 --format sf
```

(Or `cd sports/football` and drop the `-C`/path prefixes.)

**Full command reference** — every stage and tool, with flags and outputs:
[`docs/USAGE.md`](docs/USAGE.md).

## Headline

Era-ensemble ranks returning players at **Spearman ≈ 0.74–0.75** (RB/WR) on the full eligible
universe, beating every baseline. Versus the market on the rows it ranks, the model trails slightly
on overall rank but **matches/edges it on top-tier precision (P@12)** with no market input.
## Second project — late-breakout QBs

A separate question in the same tree: **which QBs break out after the league has written them
off, and was it visible before they ever took an NFL snap?** Models here are restricted to
**pre-NFL evidence** (high-school recruiting profile + college production) by design — NFL data
builds the label only.

```bash
make -C sports/football fetch                                  # once, shared caches
uv run python sports/football/scripts/qb_breakout_cohort.py    # cohort + labels + report
```

A breakout is a **sustained top-20 PPR PPG season** (superflex-relevant); *late* means NFL year
4+. Of 331 QBs entering since 1999, 69 ever broke out and **17 did it late**. The result that
motivates the constraint: through three NFL seasons, future late breakouts look far more like
busts than like on-time breakouts — early NFL production does not separate them.
Write-up: [`reports/REPORT_qb_breakout_cohort.md`](reports/REPORT_qb_breakout_cohort.md);
design: [`docs/QB_BREAKOUT_PLAN.md`](docs/QB_BREAKOUT_PLAN.md).

Per-position write-ups: [`reports/REPORT_football_rb.md`](reports/REPORT_football_rb.md),
[`reports/REPORT_football_wr.md`](reports/REPORT_football_wr.md),
[`reports/REPORT_football_qb.md`](reports/REPORT_football_qb.md); cross-version progress under
[`reports/versions/`](reports/versions/).
