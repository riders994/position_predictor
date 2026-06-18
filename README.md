# position_predictor

Predict a pro sports player's **next-season fantasy rank** from their past data, run as a
**reproducible experiment**. One independent process per **sport**; each **position** is
modeled **separately** for accuracy.

**First model:** NFL — Running Back (RB).

---

## What it does

For a position, it uses a player's NFL history through season *N* to predict their
**season *N+1* PPR points-per-game (PPG)**, then ranks players by that prediction and
measures how well the predicted ranking matches reality — versus naive baselines and the
ADP/expert-consensus market benchmark.

Key design choices (full rationale in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)):

- **Target:** next-season **PPR PPG** → derived rank (returning players only).
- **Eligibility cutoff** (games played / snap share) for the ranking universe is
  **derived analytically per model**, not hardcoded — a first-class pipeline step.
- **Recency-bias study:** every model is run over **10-, 20-, and 30-year** training
  windows and compared.
- **Data:** [`nfl_data_py`](https://github.com/nflverse/nfl_data_py) / nflverse.
- **Market data (ADP/ECR):** benchmark to beat, not a feature.

## Project structure

```
position_predictor/
├── README.md
├── pyproject.toml / uv.lock        # uv-managed, pinned deps
├── .python-version
├── Makefile                        # reproducible pipeline targets
├── config/                         # per-(sport,position) experiment configs (YAML)
│   └── football_rb.yaml
├── data/                           # git-ignored (except external/ + manifests)
│   ├── raw/  interim/  processed/  external/
├── docs/
│   ├── PROJECT_PLAN.md             # canonical design doc
│   ├── PROMPT_LOG.md               # full prompt history (reproducibility)
│   └── data_dictionary.md          # feature & field definitions
├── notebooks/football/rb/          # EDA, cutoff analysis, feature analysis, results
├── src/position_predictor/         # shared library (fetch, features, models, eval)
├── scripts/                        # stage entrypoints (fetch → build → ... → report)
├── reports/                        # figures + results tables
└── tests/
```

## Pipeline stages

`fetch → build dataset → target+eligibility → features → EDA → eligibility cutoff →
feature selection → modeling (walk-forward × 10/20/30yr) → report`

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) §3.

## Getting started

> Tooling: [uv](https://docs.astral.sh/uv/) + Jupyter. (Not yet implemented — scaffold stage.)

```bash
uv sync                 # create env from pyproject.toml / uv.lock
make fetch              # pull nflverse data → data/raw
make experiment         # run the RB experiment end-to-end
```

### Use the models — keeper-league assistant

Project next season and get a prioritized keeper list (predicted rank + value vs the pick you'd
pay). Input is a CSV of `player,pick`:

```bash
# next-season projection board for one position → reports/projections_football_rb.csv
make project CONFIG=config/football_rb.yaml

# keeper priorities for your league (RB/WR/QB; TE/K/DST shown as unscored)
uv run python scripts/keeper.py --input examples/keepers_example.csv --teams 12 --format sf
```

`--format` is `1qb` / `sf` (superflex) / `2qb`; `--teams` is 8–16. Priority = **surplus** =
`pick paid − projected board slot`, where the board is a value-over-replacement ranking from the
models (ECR stays a benchmark, never blended).

## Status

✅ **RB v1 end-to-end complete.** Pipeline: fetch → build (+ target/eligibility & `status_next`)
→ era-aware features → eligibility cutoff (`g* = 4`) → EDA (`make eda`) → walk-forward experiment
(`make experiment`) → market benchmark (`make benchmark`) → report (`make report`).

**Headline (test seasons 2020–2024):** the era ensemble ranks returning RBs at **Spearman ≈ 0.74**
on the full eligible universe, beating every must-beat baseline (persistence 0.69, linear 0.72).
Against the **market** (FantasyPros preseason ECR) on the rows it ranks, the model does **not** win
on overall rank (market 0.73 vs model 0.69) but **matches/edges it on top-12 precision** — with no
market information. Full write-up in [`reports/REPORT_football_rb.md`](reports/REPORT_football_rb.md).

**WR added** (`config/football_wr.yaml`): the shared pipeline generalized with only the
`offseason` block needing position-parameterizing. WR eligibility re-derives to **g\* = 7 games**
(WR scoring is noisier than RB); WR best-model Spearman ≈ 0.75, and the tree models **beat the
market on Precision@12** (0.58 vs 0.57). NGS earns its place for WR (unlike RB).

**Next:** QB (passing features). TE skipped (too few fantasy-relevant TEs/season).
See [`PROMPT_LOG.md`](docs/PROMPT_LOG.md) for the full decision trail.

## Reproducibility

Pinned deps + interpreter, config-driven experiments, cached/manifested raw data, fixed
seeds, notebooks that import library code only, results saved with their config, and a
complete [`PROMPT_LOG.md`](docs/PROMPT_LOG.md). See PROJECT_PLAN §10.
