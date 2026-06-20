"""csp_core.py — the GENERAL, PREDICATE-DRIVEN factor-graph CSP search core.

This is Layer 1 + the shared Layer-2 skeleton of the spec
(docs/general_factor_graph_search.md §2, §3, §6.1). It contains ZERO domain
identifiers: no `edges`, no `adj`, no `color`, no `k`-as-color-count, no
`not_equal`, no `cage`, no `gate`, no `dsatur`, no `ac3`, no `sudoku`. Every
operation is derived from a per-FACTOR-TYPE three-valued predicate looked up in a
registry (mycelium/csp_registry.py); the only domain content lives in
mycelium/csp_domains.py.

What is general here, all derived from the predicate:
  * verify_complete       — the sole `solved` arbiter (every factor == SAT)
  * is_consistent_partial — the soundness gate (no factor == VIOLATED)
  * gac_propagate         — generic GAC: prune a value with no supporting tuple
                            over the other scope members' domains, commit ONLY
                            forced singletons, with the arity_cap cost guard
                            (mirrors build_kenken_data.propagate():217's >20000).
  * mrv_varorder          — fewest-legal-values, tie-break by factor degree, prior,
                            index. (DSATUR falls out for the not-equal registry.)
  * lcv_valorder          — least-constraining value over co-factor members.
  * SystematicDFS / backtrack_search / solve_symbolic — the complete, sound,
                            budgeted DFS skeleton with Fix-1 (can_certify_unsat +
                            fallback) and Fix-2-compatible counting.

PROVEN-BY-CONSTRUCTION (pinned by parity selftests, NOT taken on faith):
  * GAC on a 2-ary not-equal predicate == AC-3 arc-consistency.
  * MRV on a not-equal registry == DSATUR variable ordering (same order/decision
    counts on fixtures — §5 L-TIE: order matches, not the byte-identical tie-stream).
The core never knows it is doing AC-3 or DSATUR; that is what generic GAC/MRV
COMPUTE on the not-equal registry.

CONTRACTS (load-bearing):
  * The predicate is THREE-VALUED (Consistency: VIOLATED/UNVIOLATED/SAT) because
    GAC and the partial-soundness gate both need "can this PARTIAL tuple still be
    completed?" which a boolean cannot express for n-ary/arithmetic factors.
  * The predicate must be hole-MONOTONE (§5 L-MONO): assigning a hole never turns
    VIOLATED into not-VIOLATED, so the partial check is a sound prune. A randomized
    monotonicity selftest is provided (assert_hole_monotone) for registration-time.
  * The neural signal enters ONLY as the `prior` argument to the ORDERERS. There is
    deliberately NO prior-biased propagate: a commit must be logically FORCED
    (sole-survivor singleton), never a 0.9-confident guess (Bryce's Jun-19 fix).

GPU-FREE: pure python + numpy-optional. ast.parse clean, CPU import clean.
"""

from __future__ import annotations

import ast
import enum
import os
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

UNASSIGNED = -1


# ===========================================================================
# THE THREE-VALUED CONSISTENCY VERDICT (the predicate's codomain)
# ===========================================================================

class Consistency(enum.IntEnum):
    """A factor's verdict on a (possibly partial) value-tuple over its scope.

    VIOLATED   — no completion of the holes can satisfy this factor (dead).
    UNVIOLATED — partial tuple, still extendable (or fully-assigned-but-no-verdict
                 by a partial-only predicate, though complete tuples should resolve
                 to SAT/VIOLATED).
    SAT        — fully assigned AND satisfied.
    """
    VIOLATED = 0
    UNVIOLATED = 1
    SAT = 2


# ===========================================================================
# THE DATA MODEL — variables + typed factors + membership incidence
# ===========================================================================

@dataclass(frozen=True)
class FactorType:
    """The single domain seam: one entry per relation class.

    predicate(ftype, params, member_values) -> Consistency is the ONLY domain code.
    `member_values` are the scope members' values in scope order; UNASSIGNED marks a
    hole. `params` are the factor's per-instance params (None for a symmetric binary
    relation; an (op, target) pair for an arithmetic factor; a type id for a gate).

    specialized_propagator is the OPTIONAL escape hatch for large-arity factors that
    blow past the arity_cap (§2.4 / L-ALLDIFF): a pure-speed swap behind an
    equivalence test, NEVER changing semantics. arity_hint feeds the cost guard.
    """
    name: str
    predicate: Callable[[int, Any, tuple], Consistency]
    specialized_propagator: Optional[Callable[["CSPState", "Factor"], "CSPState"]] = None
    arity_hint: Optional[int] = None


@dataclass(frozen=True)
class Factor:
    """One factor instance.

    ftype  : selects the FactorType (and thus the predicate) in the registry.
    scope  : ORDERED tuple of variable ids the factor touches (an ORDERED clique;
             see §5 L-ASYM — order is a registry-understood convention, e.g.
             element 0 = output for an asymmetric gate; symmetric relations ignore it).
    params : per-instance params (None | (op_id, target_int) | type_id) — NO domain
             type strings baked into the field semantics (the predicate interprets it).
    """
    ftype: int
    scope: tuple
    params: Any = None


@dataclass(frozen=True)
class Problem:
    """An immutable factor-graph CSP instance.

    n_vars      : number of variables.
    domains0    : per-variable INITIAL legal-value sets (varying domains are free:
                  givens are singletons, full alphabets are the degenerate case).
    factors     : the typed factor instances.
    var_factors : variable -> list of factor indices touching it (the GENERAL
                  constraint-graph incidence that replaces a coloring-only adjacency).
    registry    : ftype -> FactorType (the ONLY domain seam).
    """
    n_vars: int
    domains0: list
    factors: list
    var_factors: list
    registry: dict


@dataclass
class CSPState:
    """The mutable-by-copy search state (domain-agnostic).

    values     : len n_vars; UNASSIGNED = unassigned. (Replaces coloring's `colors`.)
    domains    : per-variable remaining legal VALUE sets (sets of VALUES drawn from
                 domains0, NOT 0..k-1).
    n_assigned : count of assigned variables.
    meta       : free-form plug-in scratch (beliefs/entropy). The skeleton never
                 reads it for decisions — it is for ordering priors only.
    """
    problem: Problem
    values: list
    domains: list
    n_assigned: int = 0
    meta: dict = field(default_factory=dict)

    def copy(self) -> "CSPState":
        """Branch-copy: copy values/domains/meta; SHARE the immutable Problem."""
        return CSPState(
            problem=self.problem,
            values=list(self.values),
            domains=[set(d) for d in self.domains],
            n_assigned=self.n_assigned,
            meta=dict(self.meta),
        )

    def assigned(self) -> dict:
        """{var: value} of currently-assigned variables."""
        return {v: c for v, c in enumerate(self.values) if c != UNASSIGNED}

    def unassigned_vars(self) -> list:
        return [v for v in range(self.problem.n_vars) if self.values[v] == UNASSIGNED]


def make_initial_state(problem: Problem) -> CSPState:
    """Fresh root: nothing assigned, every variable's domain = its domains0 set.

    A variable whose domains0 is a singleton is treated as a pre-fixed given:
    its value is committed and counted as assigned (givens / pre-set cells flow
    through with no skeleton change)."""
    values = [UNASSIGNED] * problem.n_vars
    domains = [set(d) for d in problem.domains0]
    n_assigned = 0
    for v in range(problem.n_vars):
        if len(domains[v]) == 1:
            (a,) = tuple(domains[v])
            values[v] = a
            n_assigned += 1
    return CSPState(problem=problem, values=values, domains=domains,
                    n_assigned=n_assigned, meta={})


def assign_var(state: CSPState, v: int, a: int) -> CSPState:
    """Return a NEW state with variable v = value a, and a pruned from the domains of
    v's still-unassigned CO-FACTOR members for which the predicate makes a's presence
    impossible. Generic forward-check: for each factor touching v, after fixing v=a,
    prune from each other member u any value b that cannot extend (with u=b, v=a, the
    rest holes) without VIOLATED. Does NOT validate global properness — the caller
    checks via is_consistent_partial / empty-domain.

    For a 2-ary not-equal factor this reduces EXACTLY to unit forward-check (remove a
    from the neighbour's domain): the only b that VIOLATES {u=b, v=a} is b==a, so a is
    the sole value pruned from u. (AC-3-is-GAC, the binary case.)
    """
    s = state.copy()
    s.values[v] = a
    s.domains[v] = {a}
    s.n_assigned += 1
    for fi in s.problem.var_factors[v]:
        f = s.problem.factors[fi]
        pred = s.problem.registry[f.ftype].predicate
        # Build the current value-tuple over this factor's scope (holes = UNASSIGNED).
        base = [s.values[m] for m in f.scope]
        for i, u in enumerate(f.scope):
            if u == v or s.values[u] != UNASSIGNED:
                continue
            survivors = set()
            for b in s.domains[u]:
                trial = list(base)
                trial[i] = b
                if pred(f.ftype, f.params, tuple(trial)) != Consistency.VIOLATED:
                    survivors.add(b)
            if survivors != s.domains[u]:
                s.domains[u] = survivors
    return s


def has_empty_domain(state: CSPState) -> bool:
    """Forward-check dead-end test: any UNASSIGNED variable with no legal value left."""
    for v in range(state.problem.n_vars):
        if state.values[v] == UNASSIGNED and len(state.domains[v]) == 0:
            return True
    return False


# ===========================================================================
# THE FOUR DERIVED OPERATIONS (all from the predicate — no domain code)
# ===========================================================================

def verify_complete(state: CSPState) -> bool:
    """SUCCESS arbiter: every variable assigned AND every factor's predicate == SAT
    on the complete tuple. The SOLE `solved` arbiter; frame-invariant (any valid
    assignment passes). Never a propagator, never the deducer."""
    if state.n_assigned != state.problem.n_vars:
        return False
    for v in range(state.problem.n_vars):
        if state.values[v] == UNASSIGNED:
            return False
    reg = state.problem.registry
    for f in state.problem.factors:
        vals = tuple(state.values[m] for m in f.scope)
        if reg[f.ftype].predicate(f.ftype, f.params, vals) != Consistency.SAT:
            return False
    return True


def is_consistent_partial(state: CSPState) -> bool:
    """SOUNDNESS gate (pre-extend + post-propagate): no factor predicate == VIOLATED
    on its (partial) tuple. Sound by the hole-monotonicity contract (§5 L-MONO)."""
    reg = state.problem.registry
    for f in state.problem.factors:
        vals = tuple(state.values[m] for m in f.scope)
        if reg[f.ftype].predicate(f.ftype, f.params, vals) == Consistency.VIOLATED:
            return False
    return True


def _commit_forced_singletons(state: CSPState) -> None:
    """Bryce's Jun-19 fix made structural: commit a value into the assignment ONLY
    when it is the SOLE survivor of a variable's domain — a logically FORCED move,
    never a confidence guess. Keeps values/n_assigned consistent with the tightened
    domains so the verifier + var-orderers see the forced commits. Mutates in place."""
    for v in range(state.problem.n_vars):
        if state.values[v] == UNASSIGNED and len(state.domains[v]) == 1:
            (a,) = tuple(state.domains[v])
            state.values[v] = a
            state.n_assigned += 1


def _scope_domain_product(state: CSPState, scope: tuple) -> int:
    p = 1
    for u in scope:
        p *= len(state.domains[u])
        if p > (1 << 62):
            return p
    return p


def _product(domain_lists):
    """itertools.product over a list of sorted value-lists (generic GAC enumerator)."""
    import itertools
    return itertools.product(*domain_lists)


def gac_propagate(state: CSPState, arity_cap: int = 20000) -> CSPState:
    """Generic Generalized-Arc-Consistency, lifted from build_kenken_data.propagate().

    For each factor in the queue: if a scope member's value `a` has NO supporting
    tuple over the OTHER members' current domains that keeps the factor non-VIOLATED,
    prune `a`. Requeue the affected members' factors. After the fixpoint, commit ONLY
    forced singletons (sole survivors). Returns a NEW state; an emptied domain is left
    so the caller's forward-check prunes the branch.

    THE COST GUARD (§2.4 / L-ARITY, mirrors propagate():217's `> 20000: continue`):
    a factor whose ∏|D(member)| exceeds arity_cap is SKIPPED by generic GAC (still
    forward-checked + verified, never unsound) and is the registration point for an
    OPTIONAL specialized_propagator. If a FactorType supplies one, it is invoked
    instead of the enumerator (a pure-speed swap behind an equivalence test).

    DSATUR / AC-3 are what this COMPUTES on the not-equal registry — there is no
    domain branch here.
    """
    s = state.copy()
    reg = s.problem.registry
    nf = len(s.problem.factors)
    in_queue = [True] * nf
    q = deque(range(nf))
    while q:
        fi = q.popleft()
        in_queue[fi] = False
        f = s.problem.factors[fi]
        ft = reg[f.ftype]
        # Large-arity escape hatch: prefer a specialized propagator if registered,
        # else skip generic enumeration (the cost guard) — never unsound.
        if _scope_domain_product(s, f.scope) > arity_cap:
            if ft.specialized_propagator is not None:
                before = [set(s.domains[u]) for u in f.scope]
                s = ft.specialized_propagator(s, f)
                for i, u in enumerate(f.scope):
                    if s.domains[u] != before[i]:
                        if not s.domains[u]:
                            return s
                        for g in s.problem.var_factors[u]:
                            if not in_queue[g]:
                                in_queue[g] = True
                                q.append(g)
            continue
        pred = ft.predicate
        support = [set() for _ in f.scope]
        for combo in _product([sorted(s.domains[u]) for u in f.scope]):
            if pred(f.ftype, f.params, combo) != Consistency.VIOLATED:
                for i, val in enumerate(combo):
                    support[i].add(val)
        for i, u in enumerate(f.scope):
            removed = s.domains[u] - support[i]
            if removed:
                s.domains[u] -= removed
                if not s.domains[u]:
                    return s                      # empty domain -> caller prunes
                for g in s.problem.var_factors[u]:
                    if not in_queue[g]:
                        in_queue[g] = True
                        q.append(g)
    _commit_forced_singletons(s)
    return s


def noop_propagate(state: CSPState) -> CSPState:
    """The 'no propagation' control: return the state unchanged (only the implicit
    forward-check from assign_var is in effect)."""
    return state


def mrv_varorder(state: CSPState, prior: Optional[list] = None) -> int:
    """Minimum-Remaining-Values: the unassigned variable with the fewest legal values,
    tie-broken by factor degree (len var_factors), then a higher prior score, then a
    lower index. Returns UNASSIGNED if all assigned.

    DSATUR-is-MRV (proven): for not-equal factors, after forward-check
    |D(v)| = alphabet - #distinct-values-among-assigned-co-members, so "fewest legal
    values" <=> "highest saturation"; the degree tie-break <=> DSATUR's. No domain
    branch — DSATUR is what generic MRV computes on the not-equal registry.

    `prior` (optional, len n_vars float) is the neural ORDERING signal — a tie-break
    only (finding c: structural ordering dominates the learned signal). Never gates
    soundness, never commits.
    """
    best, best_key = UNASSIGNED, None
    for v in range(state.problem.n_vars):
        if state.values[v] != UNASSIGNED:
            continue
        key = (
            -len(state.domains[v]),
            len(state.problem.var_factors[v]),
            (prior[v] if prior is not None else 0.0),
            -v,
        )
        if best_key is None or key > best_key:
            best_key, best = key, v
    return best


def lcv_valorder(state: CSPState, v: int, prior: Optional[list] = None) -> list:
    """Least-Constraining-Value: order v's legal values so the one removing the FEWEST
    options from co-factor members' domains comes first. Only values in D(v) are
    returned (forward-check + GAC already pruned).

    `prior` (optional, len = alphabet, float) is the neural value-ordering signal — a
    secondary tie-break (value-ordering is 2nd-order at small domains; finding c).
    """
    legal = sorted(state.domains[v])

    def _removed(a: int) -> int:
        s2 = assign_var(state, v, a)
        total = 0
        seen = set()
        for fi in state.problem.var_factors[v]:
            for u in state.problem.factors[fi].scope:
                if u == v or u in seen or state.values[u] != UNASSIGNED:
                    continue
                seen.add(u)
                total += len(state.domains[u]) - len(s2.domains[u])
        return total

    return sorted(
        legal,
        key=lambda a: (_removed(a), -(prior[a] if prior is not None else 0.0), a),
    )


def random_varorder(seed: int) -> Callable:
    """A seeded random variable-orderer (the 'no good heuristic' control)."""
    rng = random.Random(seed)

    def _pick(state: CSPState) -> int:
        un = state.unassigned_vars()
        if not un:
            return UNASSIGNED
        return rng.choice(un)

    return _pick


def random_valorder(seed: int) -> Callable:
    """A seeded random value-orderer (the arbitrary-order control)."""
    rng = random.Random(seed)

    def _order(state: CSPState, v: int) -> list:
        legal = sorted(state.domains[v])
        rng.shuffle(legal)
        return legal

    return _order


# ===========================================================================
# THE PLUGGABLE DFS SKELETON (shared by ALL strategies — fairness lives here)
# ===========================================================================

def _count_assigned(values: list) -> int:
    return sum(1 for c in values if c != UNASSIGNED)


def backtrack_search(
    problem: Problem,
    propagate_fn: Callable[[CSPState], CSPState],
    varorder_fn: Callable[[CSPState], int],
    valorder_fn: Callable[[CSPState, int], list],
    budget: int = 100,
    seed: int = 0,
    can_certify_unsat: bool = True,
    fallback_fn: Optional[Callable[[], dict]] = None,
) -> dict:
    """Depth-first branch-and-propagate over a general factor-graph Problem.

    Identical search logic for every strategy; ONLY the three plug-ins differ
    (fair-by-construction). Soundness/completeness guarantees live HERE, derived from
    the predicate via verify_complete / is_consistent_partial / gac_propagate.

    Args:
      problem      : the factor-graph CSP instance.
      propagate_fn(state) -> state         : tighten domains / commit forced singletons.
      varorder_fn(state)  -> int           : choose an UNASSIGNED var (UNASSIGNED if none).
      valorder_fn(state,v)-> list[int]     : ordered candidate values for v (in D(v)).
      budget : max DECISION-NODES expanded (one varorder pick + its value loop). On
               exhaustion -> status='budget', best-effort return.
      seed   : reserved for stochastic plug-in coordination (random plug-ins carry
               their own RNG; here for signature stability).
      can_certify_unsat : TRUE for SOUND propagators (GAC, no-op): a tree exhausted
               with no trusted commits is a real unsat certificate. FALSE for an
               advisory propagator that AUTO-COMMITS confident guesses: a confidently-
               WRONG commit can manufacture a conflict at/near the root and dead-end a
               SOLVABLE instance with nothing to backtrack — a FALSE unsat (the Fix-1
               fairness bug). When FALSE, any would-be 'unsat' (root-dead OR exhausted
               tree) is NEVER certified; instead fallback_fn runs and its
               decisions/backtracks are MERGED with the wasted ones spent here.
      fallback_fn : () -> dict, a complete-search fallback for a dead-end when this
               config may not certify unsat. Its result REPLACES the would-be unsat;
               wasted decisions/backtracks are added on top. If None, the would-be
               unsat downgrades to status='budget' (never a false certificate).

    Returns dict (the SearchResult shape):
      status        : 'solved' | 'budget' | 'unsat'
      assignment    : list[int] length n_vars (-1 = unassigned) — SOLUTION if solved,
                      else the best-effort most-complete consistent partial.
      decisions     : int  decision-nodes expanded
      backtracks    : int  failed-branch unwinds
      best_partial  : list[int]  the most-complete consistent partial encountered
      n_assigned_best : int  how many vars the best_partial assigns
      (+ fellback / fallback_reason if a config dead-ended and re-ran the fallback)

    SOUND: 'solved' is returned ONLY when verify_complete passes the exact predicate
    verifier. 'unsat' is certified ONLY by exhaustive enumeration with a propagator
    allowed to certify (can_certify_unsat=True) or by GAC proving an empty domain;
    advisory commits can only ACCELERATE or fall back, never certify.
    """
    root = make_initial_state(problem)
    root = propagate_fn(root)

    counters = {"decisions": 0, "backtracks": 0}
    best = {"values": list(root.values), "n_assigned": _count_assigned(root.values)}

    def _track_best(state: CSPState):
        na = state.n_assigned
        if na > best["n_assigned"] and is_consistent_partial(state):
            best["n_assigned"] = na
            best["values"] = list(state.values)

    def _resolve_deadend(reason: str) -> dict:
        """A tree that dead-ended with NO solution (root conflict OR exhausted, budget
        NOT hit). FAIR + SOUND: a sound propagator -> real unsat certificate; an
        advisory one -> re-run the complete fallback (never a false unsat), merging the
        wasted work; no fallback -> downgrade to 'budget'."""
        if can_certify_unsat:
            return {
                "status": "unsat",
                "assignment": best["values"],
                "decisions": counters["decisions"],
                "backtracks": counters["backtracks"],
                "best_partial": best["values"],
                "n_assigned_best": best["n_assigned"],
            }
        if fallback_fn is not None:
            fb = dict(fallback_fn())
            fb["decisions"] = fb.get("decisions", 0) + counters["decisions"]
            fb["backtracks"] = fb.get("backtracks", 0) + counters["backtracks"]
            fb["fellback"] = True
            fb["fallback_reason"] = reason
            return fb
        return {
            "status": "budget",
            "assignment": best["values"],
            "decisions": counters["decisions"],
            "backtracks": counters["backtracks"],
            "best_partial": best["values"],
            "n_assigned_best": best["n_assigned"],
        }

    # Root sanity: an immediate empty domain or VIOLATED partial => dead-end.
    if has_empty_domain(root) or not is_consistent_partial(root):
        return _resolve_deadend("root_conflict")

    _track_best(root)
    solution_box = {"values": None}

    def _dfs(state: CSPState) -> bool:
        if verify_complete(state):
            solution_box["values"] = list(state.values)
            return True
        if counters["decisions"] >= budget:
            return False
        v = varorder_fn(state)
        if v == UNASSIGNED or v < 0:
            return False
        counters["decisions"] += 1
        candidates = valorder_fn(state, v)
        for a in candidates:
            if a not in state.domains[v]:
                continue
            child = assign_var(state, v, a)
            if not is_consistent_partial(child):
                counters["backtracks"] += 1
                continue
            child = propagate_fn(child)
            if has_empty_domain(child):
                counters["backtracks"] += 1
                continue
            if not is_consistent_partial(child):
                counters["backtracks"] += 1
                continue
            _track_best(child)
            if _dfs(child):
                return True
            counters["backtracks"] += 1
            if counters["decisions"] >= budget:
                return False
        return False

    solved = _dfs(root)

    if solved:
        return {
            "status": "solved",
            "assignment": solution_box["values"],
            "decisions": counters["decisions"],
            "backtracks": counters["backtracks"],
            "best_partial": solution_box["values"],
            "n_assigned_best": problem.n_vars,
        }
    if counters["decisions"] >= budget:
        return {
            "status": "budget",
            "assignment": best["values"],
            "decisions": counters["decisions"],
            "backtracks": counters["backtracks"],
            "best_partial": best["values"],
            "n_assigned_best": best["n_assigned"],
        }
    return _resolve_deadend("tree_exhausted")


def solve_symbolic(problem: Problem, budget: int = 100000, seed: int = 0) -> dict:
    """The complete symbolic solver = the B3 CEILING / no-neural control. Plugs
    {gac_propagate, mrv_varorder, lcv_valorder} into backtrack_search.

    With a large budget this is complete: status='solved' returns a valid assignment;
    status='unsat' is a CERTIFICATE the instance is unsatisfiable (whole tree exhausted
    under budget); status='budget' means the budget cut search short."""
    return backtrack_search(
        problem,
        propagate_fn=gac_propagate,
        varorder_fn=mrv_varorder,
        valorder_fn=lcv_valorder,
        budget=budget,
        seed=seed,
        can_certify_unsat=True,
    )


# ===========================================================================
# REGISTRATION-TIME SAFETY: the hole-monotonicity selftest (§5 L-MONO)
# ===========================================================================

def assert_hole_monotone(ftype: int, params: Any, predicate, alphabet,
                         arity: int, samples: int = 200, seed: int = 0) -> bool:
    """Randomized check of the L-MONO contract the partial-soundness gate relies on:
    filling a hole never moves a partial tuple's verdict from VIOLATED to
    {UNVIOLATED, SAT}. (Equivalently: a VIOLATED partial stays VIOLATED as holes are
    filled — so is_consistent_partial pruning a VIOLATED partial is sound, because no
    completion can recover it.) Samples random partial tuples; for each that is already
    VIOLATED, fills one random hole and asserts it is still VIOLATED. Returns True if
    no violation is found in `samples` draws (cheap, run once at registration)."""
    rng = random.Random(seed)
    alpha = list(alphabet)
    if not alpha:
        return True
    for _ in range(samples):
        tup = [rng.choice(alpha) if rng.random() < 0.7 else UNASSIGNED
               for _ in range(arity)]
        verdict = predicate(ftype, params, tuple(tup))
        if verdict != Consistency.VIOLATED:
            continue
        holes = [i for i, x in enumerate(tup) if x == UNASSIGNED]
        if not holes:
            continue
        i = rng.choice(holes)
        filled = list(tup)
        filled[i] = rng.choice(alpha)
        if predicate(ftype, params, tuple(filled)) != Consistency.VIOLATED:
            return False
    return True


# ===========================================================================
# AST parse gate
# ===========================================================================

def _ast_parse_ok() -> bool:
    with open(os.path.abspath(__file__)) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


if __name__ == "__main__":
    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] ok={parse_ok}", flush=True)
    sys.exit(0 if parse_ok else 1)
