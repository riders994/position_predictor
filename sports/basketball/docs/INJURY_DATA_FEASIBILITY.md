# NBA injury data — source feasibility

> **Question:** can the football `medstaff` analysis
> (`sports/football/docs/MEDSTAFF_PLAN.md`) be ported to basketball?
> **Verdict: not today.** No accessible source provides NBA body-part injury *history* covering
> the window such an analysis needs. Two paths exist if it is ever wanted; both cost something.
> Scoping only — no fetch code, no cached data, no package was added.

The football analysis rests on two feeds: a league-mandated weekly injury report **carrying a body
part**, and weekly roster status. Its most interesting result — the cross-position-group signature
— is *entirely* a body-part analysis. So the deciding question is not "is there injury data" but
**"is there body-part data, with history, at volume."**

---

## 1. What was probed, and what came back

| source | body part? | history? | status |
|---|---|---|---|
| `nba_api` (274 endpoints) | — | — | **zero injury endpoints** |
| Box score `COMMENT` field | **no** | yes (per game) | 4 values only |
| **ESPN** `espn_nba_injuries()` | **yes, rich** | **no** | live snapshot only |
| ESPN core `athletes/{id}/injuries` | — | — | **HTTP 404** for every athlete |
| NBA official injury report (PDF) | yes | 2017+ | **HTTP 403** |
| Pro Sports Transactions | yes | ~1999+ | **HTTP 403, Cloudflare** |
| Kaggle / GitHub PST derivatives | yes | **ends ~2020** | stale |

### ESPN is structurally excellent and has no memory

`espn_nba_injuries()` returns a live league-wide snapshot — 148 records when probed. Each carries
more than the NFL injury report ever did:

```
details.type      Knee 26 · Ankle 17 · Hamstring 9 · Foot 8 · Back 7 · Achilles 5 · Calf 5 · Finger 5 …
details.detail    Surgery 36 · Sprain 20 · Strain 15 · Soreness 7 · Bruise 7 · Fracture 3
details.side      Left / Right        details.returnDate   expected return
status            Day-To-Day 141 · Out 7      + shortComment / longComment free text
```

Two things the NFL data never had: an explicit **diagnosis** field (`detail`), and
**`type = "Rest"` as its own category — 28 of 148 records, 19%**. That last one is a genuine
load-management separator, which is the single biggest reason to want this data.

**But there is no per-athlete history.** `sports.core.api.espn.com/.../athletes/{id}/injuries`
returns **404 for every athlete tested, including players who appear in the current snapshot**.
The sibling routes on the same athlete resource do work — `/eventlog`, `/statisticslog` and
`/awards` all return 200 — so this is not an auth or id problem: the resource does not exist.
The snapshot is all ESPN exposes.

### Box scores have availability but no diagnosis

`BoxScoreTraditionalV2.COMMENT` is the only per-game availability signal in `nba_api`, and its
entire vocabulary is four strings:

```
DNP - Coach's Decision · DNP - Injury/Illness · DND - Injury/Illness · DNP - League Suspension
```

No body part, no severity, no side. Worse for the analysis: **92% of sampled DNP rows were
"Coach's Decision"**, which in the modern NBA blends genuine rest, load management and
injury-adjacent precaution with no way to separate them.

### The two historical sources are both walled off

- **NBA official injury report** (PDF, ~4×/day since Dec 2017): HTTP **403** on
  `ak-static.cms.nba.com`, unchanged under a browser User-Agent. `nba.com/robots.txt` is readable
  and disallows `/api/*` among others.
- **Pro Sports Transactions**, the standard research source: HTTP **403** returning a Cloudflare
  *"Just a moment…"* interstitial — **even on `/robots.txt`**. That is a bot-protection challenge,
  i.e. an access control rather than a crawl directive.

**Neither was circumvented and neither should be.** A 403 behind a challenge page is an answer.

### Derived open datasets stop right before the window we need

Community datasets exist and are real, but they are PST scrapes frozen in time — the principal one
covers **2010–2020**, and the football analysis' comparable window is **2021–2025**. They end
exactly where we would need to begin. Licence is also unclear, being derived from a source whose
own terms are not readable behind Cloudflare.

- [NBA Injuries 2010–2020 (Kaggle)](https://www.kaggle.com/datasets/ghopkins/nba-injuries-2010-2018)
- [Active NBA Players' 10-Year Injury History (Kaggle)](https://www.kaggle.com/datasets/buyuknacar/active-nba-players-10-year-injury-history)
- [elap733/NBA-Injuries-Analysis](https://github.com/elap733/NBA-Injuries-Analysis) — extracts
  body-part keywords from PST injury notes and groups them into regions, i.e. the same taxonomy
  problem solved in `medstaff/data/taxonomy.py`, against the same free text
- [alexmjn/NBA-Injuries](https://github.com/alexmjn/NBA-Injuries) — 2010–2018
- [wyattowalsh/nbadb](https://github.com/wyattowalsh/nbadb) — broad NBA database, `stats.nba.com`
  derived, so it inherits the no-injury-endpoint limitation above

---

## 2. The load-management verdict

Even granting body parts, the football analysis would not transfer cleanly.

Football's end-of-season rest problem was solvable because the *body-part text itself* said
`"Not injury related - coach's decision"` — 2,634 such rows, all excludable, including 56 that
carried an injury-looking designation (see `MEDSTAFF_PLAN.md` §2.9). The NBA has no equivalent in
the per-game record: rest and injury collapse into `DNP - Coach's Decision`.

ESPN's `type = "Rest"` helps, but it labels only **disclosed** rest, and it exists solely in the
snapshot that has no history. A basketball availability board built today would substantially be
measuring **how openly a club discloses rest**, which is precisely the confounder the football
project spent three stages controlling.

---

## 3. Recommendation

**Do not build the basketball port now.** The blocking constraint is history, not method, and no
amount of pipeline work fixes it.

Two paths if it is ever wanted, in cost order:

1. **Prospective accumulation — cheap, slow.** ESPN's snapshot is structurally the best injury
   feed encountered in either sport. Archiving `espn_nba_injuries()` daily to
   `data/raw/espn_injuries/<date>.parquet` is a small scheduled job with no licence ambiguity.
   **Cost: ~3 seasons before a 3-year window exists, ~5 before a 5-year one.** If the basketball
   analysis is wanted *eventually*, starting the archive now is the only action with a deadline —
   every day not archiving is a day permanently missing.
2. **A paid feed — fast, costed.** sportsdata.io, Sportradar and Rotowire all sell NBA injury
   history with body-part granularity. **Not evaluated here**: pricing and licence terms need a
   budget decision, not a technical one. Scope that only if the analysis is actually wanted.

**Not recommended:** reconstructing history from the frozen 2010–2020 community datasets. They end
before the usable window, have unclear licence, and would answer a question about a different era
of the sport — the load-management era began in earnest *after* they stop.

---

## 4. What was deliberately not done

- No fetch code, no dataset registration, no package, no cached data.
- No attempt to route around the Cloudflare challenge on Pro Sports Transactions, or the 403 on
  the NBA CMS. Both were probed once with a standard and then a browser User-Agent; both refused;
  that is the finding.
- No evaluation of paid APIs beyond naming them — that is a budget conversation.

Related: `sports/football/docs/MEDSTAFF_PLAN.md` (the analysis being considered for porting) and
its §2.10, which records the parallel finding that the *NFL* source has no diagnosis-level
granularity either.
