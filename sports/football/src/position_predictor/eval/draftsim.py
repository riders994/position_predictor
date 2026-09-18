"""Draft simulation — what a pick costs, given who the market lets fall to my next pick.

The redraft board ranks by VORP: projected PPG over the league-wide replacement player. That value
is **pick-independent** — it never asks what will still be on the board when I pick again. Entry
102 measured the gap: against a room that drafts like the model, drafting by opportunity cost
matches VORP; against a room that drafts like the market, it wins +0.8–2.5 lineup PPG, because
VORP spends early picks on players the market would have let fall.

So availability needs a model of the *room*, and the room is the market. The market is used here
for that one job — predicting which players opponents take — and never to move a projection (the
"two independent opinions" rule is intact: every value this module assigns to a player is the
model's).

The fall prior
--------------
FantasyFootballCalculator publishes, per player, the standard deviation of his real draft slot
across thousands of mock drafts. Pooled over 12-team boards 2012–2025, that spread grows
sub-linearly with ADP::

    sd(adp) = a * adp ** b,     a ≈ 0.42, b ≈ 0.68   (sd/adp ≈ 0.10 past round 1)

stable across seasons and identical for PPR and half-PPR. An opponent's read of a player is
``adp + k * sd(adp) * z``; he takes the best read available, within soft position caps.
Taking the best of what's left compresses the realized spread below the per-read spread, so ``k``
is **calibrated** (:func:`calibrate_noise`) until simulated drafts reproduce the observed spread.

Policies for my team
--------------------
* :class:`AdpPolicy` — draft like the market (the baseline a typical manager achieves).
* :class:`BoardPolicy` — walk the model's VORP board, taking the first player who improves my
  starting lineup. This is how the shipped board is meant to be used.
* :class:`LookaheadPolicy` — fork each candidate, roll the rest of the draft out under *planning*
  draws of the room (never the draws being evaluated), keep the candidate with the best mean final
  lineup. One-step policy improvement over :class:`BoardPolicy`.

Rookies the model cannot score carry ``value = NaN``: opponents and the ADP drafter take them, the
model policies cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .league import MODELED_POS

POS_INDEX = {p: i for i, p in enumerate(MODELED_POS)}

# Fall prior, fit on FFC 12-team PPR + half-PPR boards 2012-2025 (see module docstring; refit with
# :func:`fit_spread`). Players the market doesn't rank sit past the last ADP with the widest spread.
SPREAD_A = 0.42
SPREAD_B = 0.68


def adp_spread(adp, *, a: float = SPREAD_A, b: float = SPREAD_B):
    """Expected standard deviation of a player's draft slot at market ADP ``adp``."""
    return a * np.power(np.maximum(np.asarray(adp, dtype=float), 1.0), b)


def fit_spread(adp, stdev) -> tuple[float, float]:
    """Least-squares fit of ``stdev = a * adp ** b`` in log space; returns ``(a, b)``."""
    adp, stdev = np.asarray(adp, dtype=float), np.asarray(stdev, dtype=float)
    ok = (adp > 0) & (stdev > 0) & np.isfinite(adp) & np.isfinite(stdev)
    if ok.sum() < 3:
        raise ValueError("fit_spread needs at least 3 players with positive adp and stdev")
    b, log_a = np.polyfit(np.log(adp[ok]), np.log(stdev[ok]), 1)
    return float(np.exp(log_a)), float(b)


def snake_order(teams: int, rounds: int) -> np.ndarray:
    """Team index on the clock for every pick of a snake draft."""
    order: list[int] = []
    for r in range(rounds):
        order.extend(range(teams) if r % 2 == 0 else range(teams - 1, -1, -1))
    return np.asarray(order, dtype=int)


def position_caps(league) -> np.ndarray:
    """Soft per-team roster caps: nobody drafts a third QB in 1QB or a third TE with one TE slot."""
    caps = dict.fromkeys(MODELED_POS, league.roster_size)
    caps["QB"] = league.qb_starters + 1
    caps["TE"] = league.starters.get("TE", 0) + 1
    return np.asarray([caps[p] for p in MODELED_POS], dtype=int)


def lineup_value(by_pos: dict, league) -> float:
    """Best starting lineup's total from per-position value lists (dedicated slots, then flex)."""
    total, leftovers = 0.0, []
    for p in MODELED_POS:
        vals = sorted(by_pos.get(p, ()), reverse=True)
        k = league.starters.get(p, 0)
        total += sum(vals[:k])
        if p in league.flex_positions:
            leftovers.extend(vals[k:])
    leftovers.sort(reverse=True)
    return total + sum(leftovers[: league.starters.get("FLEX", 0)])


@dataclass
class Pool:
    """The draftable players of one draft, as parallel arrays.

    ``adp`` is the market's ADP (``inf`` where the market doesn't rank the player — the room never
    takes him while anyone ranked is left). ``value`` is the model's projection in the league's
    currency (``NaN`` = the model can't score him). ``board_order`` is the model's VORP pick order
    over scored players.
    """

    position: np.ndarray
    adp: np.ndarray
    value: np.ndarray
    board_order: np.ndarray
    player_id: np.ndarray | None = None

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=int)
        self.adp = np.asarray(self.adp, dtype=float)
        self.value = np.asarray(self.value, dtype=float)
        self.board_order = np.asarray(self.board_order, dtype=int)
        self.scored = np.isfinite(self.value)
        ranked = np.isfinite(self.adp)
        top = float(self.adp[ranked].max()) if ranked.any() else 1.0
        # Unranked players queue behind every ranked one, in no particular order among themselves.
        self._adp_filled = np.where(ranked, self.adp, top + 1.0 + np.arange(len(self.adp)))
        self.adp_order = np.argsort(self._adp_filled, kind="stable")

    def __len__(self) -> int:
        return len(self.position)


@dataclass
class DraftState:
    avail: np.ndarray
    counts: np.ndarray                     # (teams, 4) players drafted per position
    rosters: list = field(default_factory=list)   # per team: list of pool indices
    values: list = field(default_factory=list)    # per team: {pos: [model values]}

    def copy(self) -> DraftState:
        return DraftState(self.avail.copy(), self.counts.copy(),
                          [list(r) for r in self.rosters],
                          [{p: list(v) for p, v in d.items()} for d in self.values])


class DraftSim:
    """A snake draft of ``pool`` in ``league`` against a market-drafting room.

    ``noise_scale`` is ``k`` in the opponent read ``adp + k * sd(adp) * z``. Room draws are made
    per simulation via :meth:`room`, so two policies can be evaluated on identical rooms.
    """

    def __init__(self, league, pool: Pool, *, noise_scale: float = 1.0, rounds=None,
                 spread=(SPREAD_A, SPREAD_B)):
        self.league, self.pool = league, pool
        self.teams = int(league.teams)
        self.rounds = int(rounds or league.roster_size)
        self.order = snake_order(self.teams, self.rounds)
        self.caps = position_caps(league)
        self.noise_scale = float(noise_scale)
        self._sd = adp_spread(pool._adp_filled, a=spread[0], b=spread[1])

    # -- the room --------------------------------------------------------------------------------
    def room(self, rng) -> np.ndarray:
        """One draw of every team's read of every player: ``(teams, n)``, lower = taken earlier."""
        z = rng.standard_normal((self.teams, len(self.pool)))
        return self.pool._adp_filled[None, :] + self.noise_scale * self._sd[None, :] * z

    def new_state(self) -> DraftState:
        return DraftState(np.ones(len(self.pool), dtype=bool), np.zeros((self.teams, 4), dtype=int),
                          [[] for _ in range(self.teams)],
                          [{p: [] for p in MODELED_POS} for _ in range(self.teams)])

    def allowed(self, state: DraftState, team: int) -> np.ndarray:
        return state.avail & (state.counts[team][self.pool.position] < self.caps[self.pool.position])

    def take(self, state: DraftState, team: int, i: int) -> None:
        state.avail[i] = False
        pos = int(self.pool.position[i])
        state.counts[team][pos] += 1
        state.rosters[team].append(int(i))
        if self.pool.scored[i]:
            state.values[team][MODELED_POS[pos]].append(float(self.pool.value[i]))

    def room_pick(self, state: DraftState, team: int, read: np.ndarray) -> int:
        ok = self.allowed(state, team)
        if not ok.any():
            ok = state.avail
        return int(np.where(ok, read[team], np.inf).argmin())

    def run(self, state: DraftState, read: np.ndarray, *, me=None, policy=None, start: int = 0,
            stop=None) -> DraftState:
        """Play picks ``[start, stop)``; team ``me`` uses ``policy``, everyone else the room."""
        stop = len(self.order) if stop is None else stop
        for k in range(start, stop):
            t = int(self.order[k])
            i = policy.pick(self, state, t, k) if (t == me and policy is not None) \
                else self.room_pick(state, t, read)
            self.take(state, t, i)
        return state

    def my_value(self, state: DraftState, team: int) -> float:
        return lineup_value(state.values[team], self.league)


# -- my policies -----------------------------------------------------------------------------------
def open_positions(sim: DraftSim, state: DraftState, team: int) -> np.ndarray:
    """Positions whose next player would still start for ``team``: an unfilled dedicated slot, or
    any flex-eligible position while a flex slot is open. Counts players rather than values, so it
    works for players the model can't score."""
    league, counts = sim.league, state.counts[team]
    need = np.zeros(len(MODELED_POS), dtype=bool)
    in_flex = 0
    for p in MODELED_POS:
        i, k = POS_INDEX[p], league.starters.get(p, 0)
        if counts[i] < k:
            need[i] = True
        elif p in league.flex_positions:
            in_flex += counts[i] - k
    if in_flex < league.starters.get("FLEX", 0):
        for p in league.flex_positions:
            need[POS_INDEX[p]] = True
    return need


class AdpPolicy:
    """Draft like a disciplined market manager: best ADP at a position that still starts, then best
    ADP within the caps once the lineup is full.

    Without the need rule a pure ADP drafter can finish with no tight end, which no real manager
    does (it happened in the first 2023 smoke run).
    """

    name = "adp"

    def pick(self, sim: DraftSim, state: DraftState, team: int, k: int) -> int:
        ok = sim.allowed(state, team)
        need = open_positions(sim, state, team)[sim.pool.position]
        order = sim.pool.adp_order
        for mask in (ok & need, ok):
            cand = order[mask[order]]
            if len(cand):
                return int(cand[0])
        return int(np.flatnonzero(state.avail)[0])


class BoardPolicy:
    """Walk the VORP board; take the first player who improves my starting lineup.

    Once nobody available improves the lineup, ``bench`` decides the pick. ``"board"`` takes the
    next board player — cheap, and what rollouts use, since their objective is the starting lineup.
    ``"insurance"`` takes, among the top ``scan`` board players, the one who best covers an
    absence: the mean lineup gain over each rostered player being out. Weekly scoring pays for
    depth, which a starters-only objective can't see (the first 2023 smoke run spent bench picks on
    a backup QB and two unsigned backs).
    """

    def __init__(self, *, bench: str = "board", scan: int = 25):
        if bench not in ("board", "insurance"):
            raise ValueError(f"bench must be 'board' or 'insurance'; got {bench!r}")
        self.bench, self.scan = bench, int(scan)
        self.name = "vorp_board" if bench == "board" else "vorp_board_depth"

    def pick(self, sim: DraftSim, state: DraftState, team: int, k: int) -> int:
        ok = sim.allowed(state, team)
        mine = state.values[team]
        current = lineup_value(mine, sim.league)
        fallback = None
        for i in sim.pool.board_order:
            if not ok[i]:
                continue
            if fallback is None:
                fallback = int(i)
            pos = MODELED_POS[sim.pool.position[i]]
            mine[pos].append(float(sim.pool.value[i]))
            gain = lineup_value(mine, sim.league) - current
            mine[pos].pop()
            if gain > 1e-9:
                return int(i)
        if fallback is None:
            return AdpPolicy().pick(sim, state, team, k)
        if self.bench == "insurance":
            return self._insurance_pick(sim, mine, ok, fallback)
        return fallback

    def _insurance_pick(self, sim: DraftSim, mine: dict, ok: np.ndarray, fallback: int) -> int:
        rostered = [(p, v) for p, vals in mine.items() for v in vals]
        if not rostered:
            return fallback
        absences = []
        for p_out, v_out in rostered:
            lineup = {p: list(v) for p, v in mine.items()}
            lineup[p_out].remove(v_out)
            absences.append((lineup, lineup_value(lineup, sim.league)))
        best, best_gain, seen = fallback, -np.inf, 0
        for i in sim.pool.board_order:
            if not ok[i]:
                continue
            pos, v = MODELED_POS[sim.pool.position[i]], float(sim.pool.value[i])
            gain = 0.0
            for lineup, base in absences:
                lineup[pos].append(v)
                gain += lineup_value(lineup, sim.league) - base
                lineup[pos].pop()
            if gain > best_gain + 1e-9:
                best, best_gain = int(i), gain
            seen += 1
            if seen >= self.scan:
                break
        return best


class MarketWindowPolicy:
    """Let the market pick the position and the round; let the model pick the player.

    :class:`AdpPolicy`'s choice fixes a position and a market rank *r*. Among available players at
    that position whom the market ranks no later than ``r + window`` (default: one round, ``teams``
    picks), take the one the model projects highest; if the model scores none of them (a rookie
    run), take the market's pick. This separates the model's *within-position* judgement from its
    cross-position VORP allocation, and bounds how far past the market it may reach — the reaches
    are where the backtest found the model's errors concentrated.
    """

    name = "market_window"

    def __init__(self, *, window=None):
        self.window = window

    def pick(self, sim: DraftSim, state: DraftState, team: int, k: int) -> int:
        base = AdpPolicy().pick(sim, state, team, k)
        pool = sim.pool
        window = sim.teams if self.window is None else self.window
        ok = (sim.allowed(state, team) & pool.scored & (pool.position == pool.position[base])
              & (pool._adp_filled <= pool._adp_filled[base] + window))
        if not ok.any():
            return base
        cand = np.flatnonzero(ok)
        return int(cand[np.argmax(pool.value[cand])])


class LookaheadPolicy:
    """Opportunity-cost drafting: fork each candidate and roll the draft out.

    For my first ``n_plan_picks`` picks, candidates are the top ``per_pos`` scored players
    available at each position (board order). Each is taken in a copy of the draft, which then
    finishes with :class:`BoardPolicy` for me and ``n_plan_draws`` fresh room draws for everyone
    else; the candidate with the best mean final lineup wins. Later picks defer to the board.
    Planning draws come from this policy's own ``rng``, so it never sees the room it is scored in.
    """

    name = "lookahead"

    def __init__(self, *, n_plan_draws: int = 10, per_pos: int = 3, n_plan_picks=None, rng=None):
        self.n_plan_draws, self.per_pos, self.n_plan_picks = n_plan_draws, per_pos, n_plan_picks
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.base = BoardPolicy()

    def candidates(self, sim: DraftSim, state: DraftState, team: int) -> list[int]:
        ok = sim.allowed(state, team) & sim.pool.scored
        out, seen = [], dict.fromkeys(range(4), 0)
        for i in sim.pool.board_order:
            p = int(sim.pool.position[i])
            if ok[i] and seen[p] < self.per_pos:
                out.append(int(i))
                seen[p] += 1
        return out

    def pick(self, sim: DraftSim, state: DraftState, team: int, k: int) -> int:
        n_plan = self.n_plan_picks
        if n_plan is None:
            n_plan = sum(sim.league.starters.values()) + 1
        my_pick_number = int((sim.order[:k] == team).sum())
        if my_pick_number >= n_plan:
            return self.base.pick(sim, state, team, k)
        cands = self.candidates(sim, state, team)
        if len(cands) <= 1:
            return cands[0] if cands else self.base.pick(sim, state, team, k)
        reads = [sim.room(self.rng) for _ in range(self.n_plan_draws)]
        best, best_v = cands[0], -np.inf
        for c in cands:
            v = 0.0
            for read in reads:
                fork = state.copy()
                sim.take(fork, team, c)
                sim.run(fork, read, me=team, policy=self.base, start=k + 1)
                v += sim.my_value(fork, team)
            if v > best_v:
                best, best_v = c, v
        return best


# -- calibration -----------------------------------------------------------------------------------
def simulated_pick_spread(sim: DraftSim, *, n_drafts: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of each player's overall pick over room-only drafts.

    Players still undrafted when the draft ends count as going at the pick after the last one,
    which is how a mock-draft site's ADP treats them too (bounded, not missing).
    """
    n, last = len(sim.pool), len(sim.order)
    picks = np.full((n_drafts, n), float(last + 1))
    for d in range(n_drafts):
        read = sim.room(rng)
        state = sim.new_state()
        for k in range(last):
            i = sim.room_pick(state, int(sim.order[k]), read)
            sim.take(state, int(sim.order[k]), i)
            picks[d, i] = k + 1
    return picks.mean(axis=0), picks.std(axis=0, ddof=1)


def calibrate_noise(sim: DraftSim, observed_sd, *, grid=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                    n_drafts: int = 60, max_adp=None, seed: int = 0):
    """Pick the ``noise_scale`` whose simulated pick spread best matches ``observed_sd``.

    Compared in log space over players with a positive observed spread whose ADP lies inside the
    draft (``max_adp`` defaults to 80% of the pick count — the tail is truncated by the end of the
    draft and would bias the fit low). Returns ``(best_scale, {scale: log_rmse})``.
    """
    observed_sd = np.asarray(observed_sd, dtype=float)
    max_adp = max_adp or 0.8 * len(sim.order)
    adp = sim.pool.adp
    mask = np.isfinite(adp) & (adp <= max_adp) & (observed_sd > 0)
    scores = {}
    original = sim.noise_scale
    try:
        for scale in grid:
            sim.noise_scale = float(scale)
            _, sd = simulated_pick_spread(sim, n_drafts=n_drafts, rng=np.random.default_rng(seed))
            ok = mask & (sd > 0)
            scores[float(scale)] = float(np.sqrt(np.mean((np.log(sd[ok]) - np.log(observed_sd[ok])) ** 2)))
    finally:
        sim.noise_scale = original
    best = min(scores, key=scores.get)
    return best, scores
