# Football — usage

How to run every stage and tool in the football project. All commands assume the shared env is
synced (`uv sync` from the repo root) and `uv` is on PATH (`export PATH="$HOME/.local/bin:$PATH"`).

**Where to run.** Paths below are relative to the football project root. Either `cd sports/football`
and run as shown, or stay at the repo root and prefix: `make -C sports/football <target>` /
`uv run python sports/football/scripts/<script>.py …`. Config paths resolve against the project
root regardless of your shell's cwd (e.g. `--config config/football_rb.yaml` always works), because
`Config.load` resolves relative to `PROJECT_ROOT`.

**Config selects the model.** Every pipeline script takes `--config`; the position (RB/WR/QB) and
all experiment settings come from that YAML. The `Makefile` defaults `CONFIG` to
`config/football_rb.yaml`; override per invocation: `make build CONFIG=config/football_wr.yaml`.

---

## Quick start

```bash
uv sync                                   # shared env (repo root)
cd sports/football

# one position, end to end (RB by default)
make fetch build features eligibility benchmark experiment report

# or another position
make fetch build features eligibility benchmark experiment report CONFIG=config/football_qb.yaml
```

`make help` lists every target. `make test` / `make lint` run pytest / ruff.

---

## Pipeline stages (in order)

Each `make` target wraps `uv run python scripts/<script>.py --config $(CONFIG)`. Run scripts
directly if you need their extra flags.

| Stage | `make` | Script | Output |
|---|---|---|---|
| 1 fetch | `make fetch` | `fetch_data.py` | `data/raw/*.parquet` + `data/raw/_manifests/` |
| 2 build | `make build` | `build_dataset.py` | `data/interim/football_<pos>_player_seasons.parquet` |
| 4 features | `make features` | `build_features.py` | `data/processed/football_<pos>_features.parquet` + `_feature_blocks.json` |
| 5 eligibility | `make eligibility` | `run_eligibility.py` | `reports/results/eligibility_*` + `reports/figures/eligibility_*.png` |
| 6 EDA | `make eda` | `run_eda.py` | `reports/results/eda_*` + figures |
| market | `make benchmark` | `fetch_benchmark.py` | `data/external/market_football_<pos>[_<board>].parquet` (committed) |
| 8 experiment | `make experiment` | `run_experiment.py` | `reports/results/experiment_*` + `summary.json` |
| 9 report | `make report` | `make_report.py` | `reports/REPORT_football_<pos>.md` + figures + `reports/versions/<stem>/<version>/` snapshot |
| 9 progress | `make progress` | `make_progress.py` | `reports/versions/<stem>/PROGRESS_<stem>.md` |

### Stage details & extra flags

**`fetch_data.py`** — pull nflverse + Sleeper data into the cache (idempotent: skips datasets
already cached).
```bash
uv run python scripts/fetch_data.py --config config/football_rb.yaml [options]
  --datasets seasonal weekly ...   # fetch only a subset (default: all non-large)
  --include-large                  # also pull play-by-play (heavy)
  --overwrite                      # re-fetch even if cached (use to pick up a newly-published season)
  --dry-run                        # show the fetch plan without contacting nflverse
```
Raw nflverse caches (`weekly`, `seasonal`, `rosters`, …) are position-agnostic NFL data, so you fetch
once and reuse across RB/WR/QB.

**`build_dataset.py` / `build_features.py` / `run_eligibility.py` / `run_eda.py` /
`fetch_benchmark.py` / `make_report.py` / `make_progress.py`** — `--config` only.

**`run_experiment.py`** — walk-forward CV × windows × models.
```bash
uv run python scripts/run_experiment.py --config config/football_rb.yaml [options]
  --fast                  # quick smoke run (fewer models/windows)
  --models ridge xgboost  # override the candidate model list
  --windows 10 20         # override the history windows (years)
  --top-weighted          # enable the (off-by-default) top-weighted training probe
```

> Order matters: `features` needs `build`; `eligibility`/`eda`/`experiment` need `features`;
> `report` needs `experiment` (+ `benchmark` for the market comparison); `progress` reads the
> committed `reports/versions/` snapshots.

---

## Serving tools (use the trained models)

These fit the configured era ensemble on labeled history and act on it. Model projections are
model-only — **ECR/ADP are benchmarks, never blended in.**

**`project.py`** — next-season projection board for one position.
```bash
make project CONFIG=config/football_rb.yaml          # or:
uv run python scripts/project.py --config config/football_rb.yaml [--model ridge] [--top 30]
# → reports/projections_football_<pos>.csv
```

**`keeper.py`** — keeper-league priority list from your draft picks (cross-position VORP board;
surplus = pick paid − projected board slot).
```bash
uv run python scripts/keeper.py --input examples/keepers_example.csv [options]
  --league config/leagues/my_2qb.yaml   # scoring + teams + slots (recommended)
  --teams 12               # league size (8–16); only without --league
  --format 1qb|sf|2qb      # QB format shorthand; only without --league
  --configs ...            # per-position configs (default QB/RB/WR/TE)
  --out reports/keeper_board[_<league>].csv
# → reports/keeper_board[_<league>].{md,csv}
```
Input CSV columns: `player,pick` (optional `position`). Covers QB/RB/WR/TE; K/DST and unmatched
names are listed as unscored.

**Pass `--league`.** It sets both halves of the valuation — the scoring format decides the
projections (a real retrain against that format's target) and the roster shape decides replacement
level. Without it you get full PPR and a 12-team 1QB/2RB/2WR/1TE/1FLEX shape, which is the right
answer for exactly one league. `--teams`/`--format` are rejected alongside `--league` rather than
silently half-overriding it. The format's artifacts are built automatically on first use.

How much it matters, on the same keeper list (`examples/keepers_example.csv`):

| player | PPR 1QB surplus | 10-team 2QB half-PPR surplus |
|---|---|---|
| Bo Nix | +56 keep | **+89 keep** |
| Jayden Daniels | −2 **don't keep** | **+45 keep** |
| Brian Thomas Jr. | +10 **keep** | **−5 don't keep** |
| Sam LaPorta | −14 don't keep | −24 don't keep |

Two of eight recommendations flip. The QBs gain because 20 QBs start instead of 12 (replacement QB
15.5 → 13.7 PPG); the receivers lose because half-PPR takes a point off every reception.

**`redraft.py`** — new-season draft boards, **one per league**. Checks nflverse has published the
just-completed season, refreshes stale caches, projects once per *(scoring format, market board)*,
then values each league with the keeper tool's VORP engine. Returning players within each top-N
(rookie count estimated from the market board that matches the league — see below).

**The overall board is a pick order — one row per pick, so you can draft down it.** "Top N" means
N *board slots*, some of which the market expects to be rookies the model can't score. Each league
keeps its own depth minus its own rookie count, and those same rookies are then spliced back in at
their market slot as `source == "market_rookie"` rows carrying no projection (italic in the `.md`,
with `market_ecr` in the CSV). Both the count and the depth are per league even when several
leagues share a pipeline pass, and only rookies the market ranks *inside the draft* count — one
ranked past the final pick takes no slot, so subtracting him would leave the board short.

No model number is market-derived: the market decides how many rookies a position's top-N holds
and where each falls in the order, nothing else. The per-position tables below the board stay
returning-players-only, as labelled.
```bash
make redraft SEASON=2026                              # or:
uv run python scripts/redraft.py [options]
  --season 2026            # draft season (default: current calendar year)
  --no-refresh             # use the cache as-is (skip the fetch step)
  --leagues config/leagues/my_2qb.yaml ...   # default: every shipped league
  --top-qb 20 --top-rb 50 ...                # override the league-derived board depth
  --bestball-lambda WR=0.3                   # override the fitted upside weight (default 0)
  --top 60                 # rows in each report's overall-board section
  --configs ...            --out-dir reports/
# → reports/redraft_<season>_<league>.{md,csv} for each league
```
Hard-stops (exit 1) if the feature season isn't published yet.

**Which market board.** FantasyPros publishes a separate consensus per roster shape, and the one a
league is read against follows its QB slots: a league that can start two QBs (true 2QB *or* a
QB-eligible flex) uses redraft **superflex** (`rsf`), everything else redraft-**overall** (`ro`).
`ecr_type_for_league` picks it and each report's header names the board it used.

Within a position the two charts agree almost perfectly (2026 rank Spearman QB .991, RB .999,
WR .997, TE .997), so the rookie counts barely move — as of the 2026-08-28 scrape they are
identical. The charts diverge **across** positions: the median QB sits 162nd overall on `ro` and
76th on `rsf`. Any comparison that places positions against each other needs the league's own
board; comparing within a position can use either.

Two caveats this cannot fix. Superflex is not true 2QB (superflex may punt the second QB into the
flex, a 2QB league must fill both slots), so genuine 2QB demand runs slightly ahead of `rsf`. And
every FantasyPros consensus page is full PPR — there is no half-PPR chart — so a half-PPR league
is benchmarked against a PPR board. Both are documented mismatches, not errors.

### League configs

A board is a projection seen through a league, so scoring and roster shape live in
`config/leagues/*.yaml`:

```yaml
name: my_2qb
label: "10-team 2QB half-PPR"
scoring: half_ppr                # ppr | half_ppr | standard
teams: 10
starters: {QB: 2, RB: 2, WR: 2, TE: 1, FLEX: 1}
flex_positions: [RB, WR, TE]     # TE-eligible is the default; say [RB, WR] to opt out
roster_size: 16                  # sets board depth: teams x roster_size picks
bestball: false
```

Shipped: `ppr_1qb` (12-team 1QB PPR — the historical default, unchanged), `suz_1qb` (14-team 1QB
PPR, 1RB + 3FLEX), `sar_1qb` (14-team 1QB PPR, 2RB + 1FLEX — the same depth as `suz_1qb` with a
conventional lineup, so the pair isolates what the flex structure alone does), `my_2qb` (10-team
2QB half-PPR), `underdog_bestball` (12-team half-PPR, 3WR, 18 rounds). All five run a TE-eligible
flex.

Note that TE-flex eligibility is currently **inert** at these projections: every team
already starts a dedicated TE, so the flex contest is TE13+ against RB25+/WR25+, and the
TE pool falls off far faster (TE13 ~9.8 PPG vs the marginal flex RB/WR ~11.5-12.5). No TE
wins a flex slot, so no replacement level moves. It would only bite in a league with no
dedicated TE slot, or if the TE pool got much deeper.

Two things follow from the config:

* **Scoring is a retrain, not a rescale.** Season PPG is the model's training target, so each
  format gets its own dataset → features → fit, with artifacts namespaced (`football_rb` for PPR,
  `football_rb_half_ppr` otherwise). The run is keyed on *format*, so leagues sharing one are
  nearly free; the three shipped leagues cost two passes (~35 s total).
* **Roster shape sets replacement level**, which is what makes positions comparable. A 10-team
  2QB league starts 20 QBs instead of 12, which lifts the top QB from board slot ~17 to ~10.

**`bestball_calibrate.py`** — re-fit the best-ball upside weight against history.
```bash
make bestball-calibrate                               # or:
uv run python scripts/bestball_calibrate.py --league config/leagues/underdog_bestball.yaml
# → reports/results/bestball_lambda_<league>.json + the lambda curve as CSV
```
Current answer is a **null result**: over 13 walk-forward season pairs (2012–2024) no position's
volatility term beats λ=0 at p<0.05 (gains ≤ +0.003 Spearman). Weekly σ is ~0.90 rank-correlated
with weekly mean at RB/WR/TE, so upside is mostly a restatement of quality. Best-ball boards
therefore rank by projected PPG; `sigma` is reported as context only.

**`postseason.py`** — after a season completes, grade model vs ECR vs ADP vs the actual finish.
This is the only comparison that measures *accuracy*; correlating a board against a chart measures
conformity to it. Grading is per-position, where the board choice barely matters, so it reads the
default `ro` tables.
```bash
make postseason SEASON=2025                           # or:
uv run python scripts/postseason.py [options]
  --season 2025            # completed season (default: latest published)
  --no-refresh             # use the cache as-is
  --top 24                 # rows per position in the board table
  --configs ...            --out reports/postseason_<season>.md
# → reports/postseason_<season>.{md,csv}
```

**`handcuff.py`** — which backups to draft, driven by measured starter injury risk. RB mode ranks
handcuffs by *contingent upside* (PPG the backup gains if the starter misses time × how likely
that is); QB/other mode emits a projected-starter injury-risk list instead.
```bash
make handcuff                                         # or:
uv run python scripts/handcuff.py [options]
  --config config/football_qb.yaml   # QB injury-risk list instead of the RB board
  --season 2026                      # draft season (default: upcoming)
  --scoring half_ppr                 # board in another format (default: PPR)
  --league config/leagues/my_2qb.yaml  # take the scoring from a league instead
  --signal ...                       # override the backtest-chosen risk signal
  --top 25 / --top-starters 32
# → reports/handcuff_<pos>_<season>[_<scoring>].{md,csv}
```

**Scoring changes who the handcuff is, not just the numbers.** The handcuff is whichever backup
projects highest on that team, so halving reception value reorders pass-catching backs against
early-down ones. On the 2026 board, PPR → half-PPR changes the *identified* handcuff for 2 of 27
starters (James Cook: Ty Johnson → Ray Davis; Derrick Henry: Justice Hill → Keaton Mitchell),
moves 19 of 25 handcuff ranks, and drops mean contingent upside from 1.29 to 1.14 PPG.

Handcuff takes a *scoring format*, not a league: it compares a starter to his own backup, so
roster shape and replacement level never enter. PPR keeps the historical filename; other formats
get a suffix so boards don't clobber each other.

---

## Notes

- **Regenerable vs committed.** `data/` (cache) and `reports/{figures,results}` + serving CSV/MD
  outputs are git-ignored. Committed: `data/raw/_manifests/`, `data/external/` market references,
  `reports/REPORT_*.md`, and `reports/versions/`.
- **Notebooks** (EDA) need the extra: `uv sync --extra dev --extra notebooks`, then
  `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/rb/*.ipynb`.
- **New season not modeling yet?** nflverse publishes box scores (`seasonal`/`weekly`) some months
  after the season ends; `redraft`/`postseason` will hard-stop until then. Re-run with refresh once
  it lands (or `fetch_data.py --overwrite` to force-refresh the trailing season).
