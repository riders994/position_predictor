# position_predictor

A collection of **independent, per-sport** fantasy-rank modeling projects. Each sport predicts a
player's **next-season fantasy rank** from prior-season data — but the way fantasy scoring works and
the way each sport shapes its data differ so much that **every sport is its own standalone modeling
project**, not a shared engine. They live side by side under `sports/<sport>/` and share only the
dev environment and lint/test config.

## Layout

```
position_predictor/
├── README.md                  # this index
├── pyproject.toml / uv.lock   # ONE shared dev env (deps) + ruff + pytest config
├── .python-version
├── .gitignore
└── sports/
    └── football/              # a complete, standalone project (NFL RB/WR/QB)
        ├── README.md          # how to run it, headline results
        ├── Makefile           # its pipeline targets
        ├── src/position_predictor/   # its own code (fetch, features, models, eval)
        ├── scripts/  tests/  config/  notebooks/  examples/
        ├── docs/              # PROJECT_PLAN, data_dictionary, PROMPT_LOG (football)
        ├── reports/           # committed write-ups + versions/ (generated artifacts git-ignored)
        └── data/              # git-ignored cache (committed: manifests + external reference)
```

A new sport is a **new project**: add `sports/<sport>/` (start fresh, or copy `sports/football/` as
a template) and let it diverge — its own data sources, scoring, features, and models. Nothing in
`sports/football/` is meant to be imported by another sport.

## Working on a sport

One shared uv env serves every sport (uv finds the root `pyproject.toml` by walking up):

```bash
uv sync                                   # create the shared env
make -C sports/football help              # see that sport's pipeline targets
uv run pytest                             # runs each sport's tests (testpaths in pyproject)
uv run ruff check .
```

Then dive into [`sports/football/README.md`](sports/football/README.md) for that project.
