# General factor-graph search tier — design spec (two-layer, predicate-driven)

**Status:** SPEC — not built. Refactor target of `mycelium/csp_search.py`
(coloring-only today) + `scripts/search_coloring.py` (the neural driver). Author:
Bryce + Claude.

This tier WRAPS Tier 3 (the validated breathing deducer): **the deducer proposes the
order, complete systematic search disposes.** It is general over the factor-graph
abstraction the engine already speaks — variable nodes + typed factor nodes +
membership — and is explicitly *not a one-trick pony*. KenKen is the first generality
test: the same tier, plus a KenKen cage predicate, with **zero coloring code**.

---

## 0. The goal + the ONE-TRICK-PONY GUARANTEE

**Goal.** A reusable, sound, complete search tier that solves *any* verifiable CSP the
engine can represent, and into which new SEARCH STRATEGIES (MCTS, local search) and new
DOMAINS slot without rewriting the search.

**THE GUARANTEE (the thesis of this spec).** The *only* domain-specific code anywhere in
the tier is **a per-FACTOR-TYPE predicate** — one function per relation class:

```python
# member_values are the scope members' values in scope order; UNASSIGNED = -1 marks holes.
# params are the factor's per-instance params (None for not-equal; (op, target) for a
# cage; a gate-type id for a logic gate).
predicate(ftype, params, member_values) -> Consistency   # SAT | UNVIOLATED | VIOLATED
```

plus, per domain, a thin **bridge** that reads scopes/types/params out of the engine's
tensors into a `Problem`. **Everything else is general** and contains zero domain
identifiers (no `edges`, no `adj`, no `colors`, no `k`-as-color-count, no `not-equal`,
no `cage_op`):

| derived operation | how it falls out of the predicate (no domain code) |
|---|---|
| **verifier** (the sole `solved` arbiter) | every factor's predicate `== SAT` on the complete assignment |
| **partial soundness gate** | no factor predicate `== VIOLATED` on its (partial) tuple |
| **GAC propagation** | prune value `a` from `D(x)` if no assignment of `x`'s other scope members (from their domains) leaves the factor non-`VIOLATED` with `x=a`; commit only FORCED singletons |
| **MRV var-ordering** | unassigned var with fewest legal values, tie-break by factor-degree then index |
| **LCV val-ordering** | value that removes fewest options from co-factor variables' domains |

**Why this is exactly enough, and not more.** The predicate is the *same knowledge the
data generators and the existing verifier already encode* — there is no new domain
surface, only a refactor that exposes it as a registry. The grounding is already in
the tree, twice:

- `scripts/build_kenken_data.py:cage_ok(typ, tgt, asg) -> bool` (`:175`) IS the KenKen
  complete-tuple predicate; `mycelium/circuit_data.py:_eval_gate(gtype, operands)`
  (`:162`) IS the circuit one; `u != v` IS coloring's.
- `scripts/build_kenken_data.py:propagate()` (`:190`) ALREADY does generic GAC:
  `itertools.product` over member domains filtered by `cage_ok`, building a per-position
  *support set* and intersecting domains against it — **including a
  `_prod(domain_sizes) > 20000: continue` arity/cost guard** (`:217`), the exact caveat
  this spec must address, already discovered.
- `mycelium/csp_search.py` is *already* a strategy-shaped skeleton: `backtrack_search`
  consumes black-box `(propagate_fn, varorder_fn, valorder_fn)` and three coloring
  specializations (`ac3_propagate` = GAC-on-not-equal, `dsatur_varorder` = MRV-on-
  coloring, `lcv_valorder` = LCV-on-coloring).

The refactor *lifts* these three coloring-specific functions and the one-per-domain
predicate into a single predicate-driven core. **DSATUR-is-MRV and AC-3-is-GAC are not
claims to take on faith — they are proven by construction (§2) and pinned by parity
selftests (§6).** If any piece of the tier needs coloring-specific (or any single-
domain) code beyond the predicate + bridge, that is a DESIGN BUG to surface and fix.

---

## 1. Motivation — why search must be GENERAL, SYMBOLIC-SOUND, with NEURAL ORDERING

Three findings (Jun-19) fix the architecture of this tier.

**(a) The Path-B honest negative.** Neural-propagation-inside-backtracking on hard
3-coloring LOST to symbolic search, for free: AC-3 ceiling 0.95, no-prop 0.85; neural
probabilistic propagation was NET-NEGATIVE. The viability gate
(`scripts/analyze_search_guidance_gate.py`) had already shown why: **G1 FAIL** — the
learned policy is *confidently wrong* at the vertices it gets wrong (below-chance mass
on gold), so it is unusable as a value/prior; **G2 PASS** (AUC 0.69) — per-vertex
entropy localizes *where* the deducer is unsure; **G4 PASS + monotonic** — clamping a
few vertices and re-deducing improves the rest.

**(b) The "commit only when forced" fix (Bryce's insight).** A sound propagation commit
must be a LOGICALLY FORCED move (100%), like AC-3 / unit-propagation. Committing the
deducer's 0.9-confident GUESSES conflates guessing with deducing and *causes*
backtracks (the Path-B net-negative). Therefore the neural signal's role is **ORDERING**
(which branch first), **NOT committing**. Only symbolic GAC may commit, and it only ever
commits the sole survivor of a domain.

**(c) DSATUR >> neural-entropy var-order.** Structural variable-ordering dominates the
learned signal; value-ordering is second-order at small domains (it matters more with
>3 choices).

The design these force:
1. **Symbolic systematic DFS (MRV/GAC/LCV) is the sound, complete default** and the
   reproducibility anchor — the permanent fallback (CLAUDE.md's additive/gated law).
2. **Neural is ADVISORY/REVERTIBLE: ordering only.** It biases variable/value ordering;
   it never commits, never certifies `unsat`. The refuted auto-commit path is demoted
   to a kept-but-not-default ablation arm (§3).
3. **The exact verifier is the sole arbiter of `solved`** — never a propagator, never
   the deducer.
4. **The tier must be general** so the *same* search machinery is re-tested across
   domains; coloring's honest negative is only meaningful if KenKen (weaker propagation,
   domain 7) can re-ask "does neural ordering earn its keep where symbolic propagation
   leaves headroom?" — which requires zero per-domain search code to be a fair test.

---

## 2. Layer 1 — the general factor-graph CSP interface

### 2.1 The data model (replaces `CSPState`'s coloring fields)

A factor graph is `(variables, per-variable domains, typed factors)`. `var_factors`
(variable → the factors touching it) is the GENERAL replacement for coloring's `adj` —
the FACTS' missing "for each variable, list of constraint indices, not neighbours."

```python
UNASSIGNED = -1

class Consistency(enum.IntEnum):
    VIOLATED  = 0   # no completion can satisfy this factor (dead)
    UNVIOLATED = 1  # partial tuple, still extendable
    SAT       = 2   # fully assigned AND satisfied

@dataclass(frozen=True)
class Factor:
    ftype:  int               # latent_type value -> selects the predicate
    scope:  tuple[int, ...]   # variable ids (a membership row's 1-indices); ORDERED (see §5 L-ASYM)
    params: Any               # None | (op_id, target_int) | gate_type_id  (NO domain types in the field)

@dataclass(frozen=True)
class Problem:
    n_vars:    int
    domains0:  list[set[int]]            # per-variable INITIAL domains (Sudoku-style varying domains free)
    factors:   list[Factor]
    var_factors: list[list[int]]         # var -> factor indices touching it (the general "constraint graph")
    registry:  dict[int, FactorType]     # ftype -> FactorType (the ONLY domain seam)

@dataclass
class CSPState:                          # domain-agnostic; replaces colors/adj/k
    problem:    Problem
    values:     list[int]                # len n_vars; -1 = unassigned (was `colors`)
    domains:    list[set[int]]           # per-var remaining legal VALUES (not 0..k-1)
    n_assigned: int
    meta:       dict                     # free-form plug-in scratch (beliefs/entropy) — unchanged
    def copy(self) -> "CSPState": ...    # copy values/domains; share immutable Problem
```

`domains[v]` is a set of VALUES drawn from `domains0` (read from the engine's
`value_domain_mask`), not `0..k-1` — so per-variable varying domains (Sudoku givens,
KenKen restrictions, SAT `{0,1}`) flow through with no skeleton change; coloring is the
degenerate "every real var starts at the full alphabet" case.

### 2.2 The predicate — the single domain seam

```python
@dataclass(frozen=True)
class FactorType:
    name:      str
    predicate: Callable[[int, Any, tuple], Consistency]   # the ONLY domain code
    # OPTIONAL escape hatch for large-arity factors (§2.4), behind an equivalence test:
    specialized_propagator: Optional[Callable[["CSPState", Factor], "CSPState"]] = None
    arity_hint: Optional[int] = None                       # for the GAC cost guard
```

The predicate is THREE-VALUED, not boolean, because GAC and the partial-soundness gate
both need "can this PARTIAL tuple still be completed?" which a boolean cannot express
for n-ary/arithmetic factors. **Contract (load-bearing, see §5 L-MONO):** the predicate
must be hole-MONOTONE — assigning a hole never turns `VIOLATED` into not-`VIOLATED` — so
the partial check is a sound prune.

Every domain's predicate REUSES the function the generator/verifier already has:

```python
def not_equal_pred(_ft, _p, vals):       # coloring: scope=(u,v), params=None
    a, b = vals
    if a == UNASSIGNED or b == UNASSIGNED: return Consistency.UNVIOLATED
    return Consistency.SAT if a != b else Consistency.VIOLATED

def all_diff_pred(_ft, _p, vals):        # KenKen row/col, Sudoku unit: n-ary, no params
    seen = [v for v in vals if v != UNASSIGNED]
    if len(seen) != len(set(seen)):       return Consistency.VIOLATED
    return Consistency.SAT if UNASSIGNED not in vals else Consistency.UNVIOLATED

def cage_pred(_ft, params, vals):        # KenKen cage: params=(op_id, target_INT)
    if any(v == UNASSIGNED for v in vals): return Consistency.UNVIOLATED   # generic partial = no-info
    op, target = params
    return Consistency.SAT if cage_ok(op, target, vals) else Consistency.VIOLATED  # REUSE build_kenken_data.cage_ok

def gate_pred(_ft, params, vals):        # circuit: scope=(g, *operands), params=gate_type
    if any(v == UNASSIGNED for v in vals): return Consistency.UNVIOLATED
    out, ins = vals[0], vals[1:]
    return Consistency.SAT if out == _eval_gate(params, ins) else Consistency.VIOLATED  # REUSE circuit_data._eval_gate
```

The default partial mode is "`UNVIOLATED` unless a sub-relation is already decidable"
(vacuously true). For correctness only the COMPLETE form is REQUIRED; generic GAC prunes
by enumerating the other members' domains and calling the complete predicate (the
`cage_ok` path in `propagate()`). A domain MAY supply a sharper partial check (e.g.
arithmetic interval bounds for big add/mul cages) — optional, not required (§5 L-PART).

### 2.3 The four operations — ALL derived from the predicate

```python
def verify_complete(state) -> bool:          # SUCCESS arbiter
    if state.n_assigned != state.problem.n_vars: return False
    for f in state.problem.factors:
        vals = tuple(state.values[m] for m in f.scope)
        if state.problem.registry[f.ftype].predicate(f.ftype, f.params, vals) != Consistency.SAT:
            return False
    return True

def is_consistent_partial(state) -> bool:    # SOUNDNESS gate (pre-extend + post-propagate)
    for f in state.problem.factors:
        vals = tuple(state.values[m] for m in f.scope)
        if state.problem.registry[f.ftype].predicate(f.ftype, f.params, vals) == Consistency.VIOLATED:
            return False
    return True

def gac_propagate(state, arity_cap=20000) -> CSPState:   # generic GAC (lifted from propagate())
    s = state.copy(); q = deque(range(len(s.problem.factors)))
    while q:
        f = s.problem.factors[q.popleft()]; pred = s.problem.registry[f.ftype].predicate
        if prod(len(s.domains[u]) for u in f.scope) > arity_cap:    # the cost guard (mirrors propagate():217)
            continue                                                # large-arity -> specialized propagator slot (§2.4)
        support = [set() for _ in f.scope]
        for combo in product(*(sorted(s.domains[u]) for u in f.scope)):
            if pred(f.ftype, f.params, combo) != Consistency.VIOLATED:
                for i, v in enumerate(combo): support[i].add(v)
        for i, u in enumerate(f.scope):
            removed = s.domains[u] - support[i]
            if removed:
                s.domains[u] -= removed
                if not s.domains[u]: return s                       # empty domain -> caller prunes
                for g in s.problem.var_factors[u]: q.append(g)
    _commit_forced_singletons(s)                                   # ONLY sole-survivor commits (100% forced)
    return s

def mrv_varorder(state, prior=None) -> int:  # DSATUR falls out for not-equal
    best, best_key = UNASSIGNED, None
    for v in range(state.problem.n_vars):
        if state.values[v] != UNASSIGNED: continue
        key = (-len(state.domains[v]), len(state.problem.var_factors[v]),
               (prior[v] if prior else 0.0), -v)
        if best_key is None or key > best_key: best_key, best = key, v
    return best

def lcv_valorder(state, v, prior=None) -> list[int]:
    def removed(a):
        s2 = assign_var(state, v, a)                               # tentative forward-check over v's factors
        return sum(len(state.domains[u]) - len(s2.domains[u])
                   for f in state.problem.var_factors[v]
                   for u in state.problem.factors[f].scope if u != v)
    return sorted(state.domains[v],
                  key=lambda a: (removed(a), -(prior[v][a] if prior else 0.0), a))
```

`_commit_forced_singletons` is **Bryce's Jun-19 fix made structural**: GAC commits a
value only when it is the SOLE survivor of a domain — a logically forced move, never a
0.9-confident guess. The neural signal is barred from this path; it enters only as the
`prior` argument to the ORDERERS (§3).

**DSATUR-is-MRV (proven).** After forward-check, `|D(v)| = k − #distinct-colors-among-
assigned-neighbours` for not-equal factors, so "fewest legal values" ⇔ "highest
saturation," and the degree tie-break ⇔ DSATUR's degree tie-break. **No coloring branch
exists** — DSATUR is what generic MRV *computes* on the not-equal registry. (Caveat: the
recovered ORDER matches; the byte-identical tie-stream is checked by parity test on
decision counts, not node-by-node identity — §5 L-TIE.)

**AC-3-is-GAC (proven).** GAC on `not_equal_pred` with `|scope|=2`: the support set of
`D(u)` excludes `a` exactly when `D(v) == {a}` — byte-equivalent to `ac3_propagate`'s
`_revise`. One propagator yields AC-3 (coloring), unit-propagation (SAT clauses),
naked/hidden singles (Sudoku units), and arithmetic pruning (KenKen cages) — zero
per-domain branches. Pinned by a parity selftest (GAC-not_equal vs `ac3_propagate` on
the existing C5/K4 fixtures).

### 2.4 The arity / cost caveat (addressed head-on)

Generic GAC's `product(*domains)` is cheap exactly where the launch domains live:
coloring `|scope|=2` → ≤ k²; gates 2–3 → ≤ 8; KenKen cages 1–4 cells → ≤ 7⁴ = 2401 —
all under the cap. The `arity_cap` (mirroring `propagate():217`'s `> 20000`) makes the
guard EXPLICIT: a factor whose domain product exceeds the cap is SKIPPED by generic GAC
(still forward-checked + verified, never unsound) and is the registration point for an
OPTIONAL **specialized propagator** keyed by factor type (`FactorType.specialized_
propagator`, behind an equivalence test vs the enumerator — a pure speed swap, identical
semantics). This is the ONLY place arity-specific code may live, and none of the launch
domains need it — **except** KenKen row/col all-different as a single n-ary factor
(§5 L-ALLDIFF, the first predictable specialized-propagator requirement).

### 2.5 The registry + the bridge (adding a domain = a tiny table)

```python
def register(registry, ftype, pred, **kw): registry[ftype] = FactorType(pred=pred, **kw)
# coloring:  register(LTYPE_EDGE, not_equal_pred, arity_hint=2)
# kenken:    register(ROW, all_diff_pred); register(COL, all_diff_pred); register(CAGE, cage_pred)
# circuit:   register(AND, gate_pred); register(OR, gate_pred); register(NOT, gate_pred)
```

`factorgraph_from_membership` generalizes the coloring-only `edges_from_membership`
(which assumes `|scope|==2`, `ltype==0`):

```python
def factorgraph_from_membership(membership_np, latent_type_np, value_domain_mask_np, b,
                                registry, factor_params, pad_type):
    factors, var_factors = [], [[] for _ in range(n_vars)]
    for l in range(L):
        ft = int(latent_type_np[b, l])
        if ft == pad_type: continue                            # GENERAL parameterized sentinel (was ltype != 0)
        scope = tuple(int(i) for i in np.where(membership_np[b, l] > 0)[0])
        if not scope: continue
        f = Factor(ft, scope, factor_params(b, l))             # params from the domain side-tensors (§5 L-PARAM)
        for v in scope: var_factors[v].append(len(factors)); factors.append(f)
    domains0 = [set(np.where(value_domain_mask_np[b, v] > 0)[0]) for v in range(n_vars)]
    return Problem(n_vars, domains0, factors, var_factors, registry)
```

`factor_params(b, l)` is a small per-domain callback that pulls per-instance params from
the engine's side-tensors (`{}` for edges; `(op_id, EXACT target)` for cages — NOT
`target_to_bucket`, which is an inlet-only lossy encoding; `gate_type` for gates). Per
domain: a few `register()` calls + this callback. Zero search code.

---

## 3. Layer 2 — the pluggable search-strategy contract

### 3.1 The `Strategy` contract

A strategy is any callable consuming ONLY Layer-1 surfaces (`Problem` + the four derived
ops). Today's `backtrack_search` is already strategy-shaped; the refactor names the whole
bundle a `Strategy` and makes the verifier/soundness/completeness guarantees part of the
*contract* rather than properties of one function.

```python
@dataclass(frozen=True)
class SearchResult:
    status: str                 # 'solved' | 'budget' | 'unsat'
    assignment: list[int]       # len n_vars, -1=unassigned; solution if solved else best-effort proper partial
    decisions: int              # decision-nodes (the cross-strategy work unit)
    backtracks: int
    best_partial: list[int]
    n_assigned_best: int
    forwards: int = 0           # neural deduce_fn calls (Fix-2 honest compute; 0 for pure-symbolic)
    extra: dict = field(default_factory=dict)   # fellback / fallback_reason / sims / ...

class Strategy(Protocol):
    def solve(self, problem, *, budget, seed=0, priors=None) -> SearchResult: ...
```

**Mandatory guarantees (enforced by the shared core, not re-implemented per strategy):**
- **SOUND** — `status='solved'` ONLY when `verify_complete` passes the exact predicate
  verifier. No propagator/policy/simulation may certify success.
- **DOMAIN-RESPECTING** — never assigns a value outside `domains[var]`; advisory
  orderings are filtered to the live domain (the domain is the enforcer).
- **UNSAT-HONEST** — may emit `unsat` ONLY if complete AND it made no unsound commits
  (`can_certify_unsat=True`). A strategy that makes advisory commits runs with
  `can_certify_unsat=False` + a complete `fallback`, so a wrong commit degrades to
  complete search, never a false certificate (Fix-1, generalized).
- **HONEST COST** — always reports `decisions` AND `forwards`; never hides neural cost
  behind a low decision count (Fix-2).

The shared core is today's `backtrack_search` body lifted to Layer-1 calls
(`verify_complete`/`is_consistent_partial`/`gac_propagate`/`assign_var` instead of the
coloring functions). Fix-1 (`can_certify_unsat` + `fallback_fn`) and Fix-2 (the
`ForwardCounter`) are ALREADY domain-agnostic and carry over verbatim. The complete
exact ceiling is:

```python
def solve_symbolic(problem, budget=100000, seed=0):     # generalizes solve_symbolic, the B3 ceiling
    return SystematicDFS(gac_propagate, mrv_varorder, lcv_valorder,
                         can_certify_unsat=True).solve(problem, budget=budget, seed=seed)
```

A new strategy (MCTS, local search, CDCL-for-SAT) is a new class consuming the SAME
Layer-1 ops — zero domain code, because all domain knowledge is the predicate registry.
This is the literal mechanism for "add other search algorithms generally."

### 3.2 The clean seam — marginals as ORDERING PRIORS, never commits

This encodes finding (b): **a commit must be logically forced; the neural signal's role
is ordering.** Priors NEVER reach `propagate`/`assign` and never gate soundness — they
only reorder the candidate lists the verifier+GAC will prune anyway.

```python
@dataclass(frozen=True)
class OrderingPriors:
    marginals: Callable[[CSPState], np.ndarray]   # state -> (n_vars, max_dom) softmax; lazy, cached in state.meta
    # derived reads: entropy(state) (G2: WHERE the deducer is unsure), top_value, prob

def make_ordering_priors(deduce_fn, problem) -> OrderingPriors: ...
    # GENERIC boundary adapter over the engine's per-variable value-codebook marginals.
    # deduce_fn(input_cells (S,), value_domain_mask (S,N)) -> probs (S,N) IS already the
    # engine's interface: factor_breathing_forward's value_logits_history[-1].softmax.
    # Domain-agnostic — the only boundary detail is the +1/-1 value-encoding offset.
```

Two prior-biased orderers, built by generic combinators (no domain code):

```python
def neural_varorder(state):   # MRV primary; deducer-ENTROPY tie-break among the MRV-optimal set (G2)
    return mrv_varorder(state, prior=priors.entropy(state))   # demotes pure-entropy to a tie-break (finding c)
def neural_valorder(state, v):  # LCV order biased by deducer marginal; advisory, domain-filtered
    return lcv_valorder(state, v, prior=priors.marginals_for(v))   # prob-then-LCV (value-order 2nd-order; finding c)
```

These WRAP, never replace, the structural orderers — strictly additive: they reduce to
pure MRV/LCV when the deducer is absent or uninformative. **There is deliberately NO
`prior_biased_propagate`** — no sanctioned path from a soft marginal to a domain prune or
commit. The legacy `neural_propagate` auto-commit path is retained ONLY behind the
`can_certify_unsat=False` + fallback harness, as the *refuted* ablation arm B2 (kept so
the negative result stays reproducible, NOT recommended, NOT default).

---

## 4. MCTS — where it does, and does NOT, belong

MCTS is **NOT** offered for the verifiable-CSP regime, and the tier must not pretend
otherwise (consistent with the prior AlphaZero/PUCT rejection):

- Systematic DFS is **complete + exact** here and empirically BEAT neural propagation
  (AC-3 ceiling 0.95 vs net-negative neural prop). MCTS *sampling* discards the exact
  verifier + GAC pruning that make the verifiable regime cheap.
- PUCT needs a trustworthy policy/value prior; the gate showed **G1 FAIL** — we lack the
  prior MCTS requires.

MCTS's correct home is the **non-verifiable frontier**, where it slots in as *just
another `Strategy`* over the same Layer-1 interface: optimization with soft/partial
objectives (no boolean `verify_complete`, instead a `score(state) -> float`),
astronomical branching with no cheap completeness, adversarial/sequential structure —
i.e. the NL-specified / learned-constraint problems (the Phase-1-parser future). Layer 1
exposes an OPTIONAL `objective(state) -> Optional[float]`; a strategy consuming
`objective` is a "soft-objective" strategy, one consuming only `verify_complete` is an
"exact" strategy. Same `Strategy.solve` signature, same `var_factors`/predicate surface,
zero domain code. A strategy that ignores `verify_complete` on a verifiable problem is a
CONFIGURATION ERROR, not a silent slowdown (§5 L-SOFT).

This is co-solution, not replacement, of Tier 3: K breaths remain factor-graph iterative
inference; search WRAPS the deducer and respects the gold-free convergence instrument
(the marginals are an online prior, never a peek at the answer).

---

## 5. Generality evidence + the honest LEAK TABLE

### 5.1 Three instantiations (the predicate is the entire domain surface)

```python
SAT     = {CLAUSE: FactorType("clause", clause_pred)}                 # arity = clause width
SUDOKU  = {ALLDIFF: FactorType("alldiff", all_diff_pred, arity_hint=9)}  # 27 instances; needs all-diff fast path
NQUEENS = {COL:  FactorType("col",  lambda f,p,v: -1 in v or v[0]!=v[1], arity_hint=2),
           DIAG: FactorType("diag", lambda f,p,v: -1 in v or abs(v[0]-v[1])!=abs(p[0]-p[1]), arity_hint=2)}
```

- **Boolean SAT (CNF).** Vars bool, `domains0[v]={0,1}`, one type `clause`
  (`params`=literals over scope). GAC on a clause = unit propagation; `_commit_forced_
  singletons` does exactly the unit-clause force. MRV ≈ most-constrained-var, LCV ≈ phase
  preference. CORRECT + complete with zero new search code — but a *baseline* SAT solver,
  not CDCL (which is a Layer-2 strategy swap, §5.2 L-CDCL).
- **Sudoku-9 (the large-domain test).** 81 vars, `domains0[cell]={0..8}` (givens =
  singleton), 27 all-different factors (9 rows/cols/boxes). GAC = naked/hidden-single
  elimination; MRV picks fewest-candidates; **LCV is genuinely load-bearing at k=9** —
  the cleanest proof the value-ordering generality survives at large domain. (Requires
  the all-diff fast path, L-ALLDIFF.)
- **N-Queens (the structurally-distant test).** One var per row, `domains0[r]={0..N-1}`
  (the queen's column); two binary types `col_diff` + `diag_diff(params=(r1,r2))`. GAC =
  forward-checking on columns/diagonals; MRV = most-constrained row; LCV = least-
  eliminating column. Deliberately NOT a coloring clone — and it exposes the O(N²)
  binary-factor blow-up honestly (L-DENSE). (Job-shop scheduling was REJECTED as an
  in-scope example precisely because it is optimization, not satisfaction → routed to
  the MCTS/optimization frontier, L-OPT.)

### 5.2 The leak table (the tier's boundary — not hidden)

| leak | verdict | resolution |
|---|---|---|
| **L-ALLDIFF**: KenKen row/col as ONE n-ary all-different is `7^7 ≈ 823k` > cap → generic GAC SKIPS it (forward-check only). | **DOCUMENTED EXTENSION POINT** | The predictable first specialized propagator: a Hall-interval / value-occurrence all-different via `FactorType.specialized_propagator`. KenKen is the first generality test AND the first place this is needed. Until shipped, KenKen rows/cols rely on forward-check, weaker than AC-3-strength (report this). |
| **L-ARITY**: generic GAC enumerates `∏|D(other members)|` per revise — exponential in arity; safe for launch domains, blows up for wide cages / 9-cell sum cages. | **ABSORB w/ GUARD** | The `arity_cap` (mirrors `propagate():217`) makes the guard explicit; over-cap factors get an opt-in specialized propagator. The generic path is the CORRECTNESS reference; specialized ones are pure-speed swaps behind an equivalence test. |
| **L-ASYM**: asymmetric n-ary factors (gate `out == f(ins)`; KenKen sub/div order-dependent) need to know WHICH scope member is the output / operand order, but membership encodes scope as an UNORDERED clique. | **DOCUMENTED — bridge boundary** | `Factor.scope` is an ORDERED tuple with a registry-understood convention (element 0 = output for gates). The bridge cannot recover order from a symmetric clique alone → the domain's `factor_params`/bridge supplies ordered scope from the data generator's side-tensors, NOT from membership. A real generality gap at the Layer-1/bridge boundary, resolved there, never in Layer 2. |
| **L-MONO**: the hole-monotonicity contract (assigning a hole never flips `VIOLATED`→not) is REQUIRED for sound partial-pruning but not enforced; a non-monotone predicate (e.g. XOR-parity) silently breaks the gate. | **ABSORB w/ selftest** | Document loudly; add a per-registered-type randomized monotonicity selftest at registration (sample partial→full extensions, cheap, run once). |
| **L-PARAM**: `edges_from_membership` assumes 2-ones rows + no params; KenKen op/target live in `cage_op`/`cage_target`, not membership. | **ABSORB** | The bridge takes a `factor_params(b, l)` callback to pull per-instance params from the domain's side-tensors. Without it only param-free types bridge; arithmetic factors need the plumbing (cleanly, one callback). |
| **L-TIE**: DSATUR-as-MRV recovers the same ORDER, not the byte-identical tie-stream of `dsatur_varorder` (sat-then-degree-then-index vs remaining-then-factors-then-index). | **ABSORB, note** | They coincide structurally on not-equal; the regression test compares DECISION/SOLVE counts on a fixed fixture set, not node-by-node identity. Risk is only to the "byte-identical refactor" phrasing, not correctness. |
| **L-DENSE**: N-Queens / coloring materialize O(n²) binary factors; `var_factors` + the GAC queue scale with factor count. | **DOCUMENTED EXTENSION POINT** | Allow an "implicit/parametric" FactorType (one instance + a scope-generator covering an all-pairs relation) so the propagator iterates pairs without materializing n² objects. Until then correct but slow on dense domains. |
| **L-CDCL**: SAT at scale wants CDCL (clause learning, VSIDS, watched literals), not DFS+GAC. | **OUT-OF-SCOPE → Layer-2 strategy** | "SAT falls out" means CORRECT + complete, NOT a good SAT solver. CDCL is a new Layer-2 strategy (zero domain code), which the contract explicitly supports. |
| **L-OPT**: optimization vs satisfaction (job-shop makespan, MAX-SAT, weighted-CSP, min-coloring) — no boolean verifier, soft objective, completeness meaningless. | **OUT-OF-SCOPE → MCTS/optimization frontier** | The verifier-as-arbiter + complete-DFS guarantees DO NOT apply. Route to a Layer-2 soft-objective strategy (§4). The CSP tier REFUSES these (or solves only the feasibility sub-problem). |
| **L-CONT**: continuous / unbounded-integer variables (temporal `x−y ≤ d` over reals, linear arithmetic). | **OUT-OF-SCOPE / partial extension** | `domains0` as finite `set[int]` is baked everywhere. Bounded-integer temporal CSPs can be discretized (extension point); true continuous domains need interval propagation / SMT — a different tier. |
| **L-GLOBAL**: truly global constraints not expressible as a per-factor tuple predicate (e.g. "≤ k colors total", "sum over ALL vars = C"). | **OUT-OF-SCOPE → route** | A tuple predicate is scoped to one factor; a global constraint has scope = all vars, collapsing GAC to brute force. Decompose into small factors if possible, else specialized propagator or a different strategy. |

---

## 6. Integration + phased build/test plan

### 6.1 Module split

```
mycelium/csp_search.py  ->  SPLIT (coloring becomes a registered domain, not a deletion):
  mycelium/csp_core.py       # GENERAL: Factor, Problem, CSPState, the shared core
                             #   (backtrack_search/SystematicDFS), verify_complete,
                             #   is_consistent_partial, gac_propagate, mrv/lcv, assign_var,
                             #   solve_symbolic(problem). Fix-1/Fix-2 intact, domain-free.
  mycelium/csp_registry.py   # FactorType + the predicates: not_equal, all_diff, cage, gate
                             #   (cage REUSES build_kenken_data.cage_ok; gate REUSES
                             #    circuit_data._eval_gate — single source of constraint truth).
  mycelium/csp_domains.py    # BRIDGES: problem_from_coloring/_kenken/_circuit, each building
                             #   a Problem from (membership, latent_type, params) tensors.
                             #   edges_from_membership becomes the not-equal case of the
                             #   generic factorgraph_from_membership.
```

- Coloring functions become registry/bridge entries: `dsatur_varorder`→ the coloring
  instantiation of `mrv_varorder`; `ac3_propagate`→ the not-equal predicate consumed by
  generic `gac_propagate`; `lcv_valorder`→ generic LCV; `is_proper_partial`/
  `is_complete_proper`→ thin coloring wrappers over the generic ops (kept so the coloring
  selftest is unchanged).
- `scripts/search_coloring.py`: the GPU `deduce_fn` / `_make_engine_deduce_fn` /
  `ForwardCounter` / ablation grid / metrics move to a general
  `scripts/search_eval.py --domain {coloring,kenken,circuit}`; `search_coloring.py`
  becomes a thin shim invoking `--domain coloring` so existing run commands + the
  published coloring numbers are reproducible.

### 6.2 Phased plan + gating (each phase gates the next)

- **Phase 0 — refactor under green tests (no behavior change).** Split the module;
  express coloring as `not_equal` + `problem_from_coloring`. **GATE:** the existing
  coloring CPU selftest passes UNCHANGED; `solve_symbolic` decisions/backtracks
  byte-identical on the C5/K4/3-prism fixtures; the `search_coloring.py` shim reproduces
  the published GPU coloring ablation numbers.
- **Phase 1 — generic Layer-1 equivalence.** Prove `gac_propagate(not_equal)` reproduces
  `ac3_propagate` (decisions/backtracks identical on the coloring fixtures), MRV-pick ==
  DSATUR-pick on a forward-checked coloring state (incl. ties), and enumerator ==
  specialized propagator where one exists. **GATE:** byte-identical coloring B3; the CPU
  mock end-to-end still solves the 3-prism.
- **Phase 2 — KenKen port (the FIRST GENERALITY TEST).** Add `all_diff` + `cage` registry
  entries (cage REUSES `cage_ok`, EXACT integer target via the bridge, not the bucket) +
  `problem_from_kenken`. Ship the all-different specialized propagator (L-ALLDIFF).
  **GATE:** ZERO coloring identifiers in the KenKen path; symbolic B3 solves ≥ the
  leak-free corpus's known-solvable set within budget (cross-check vs `build_kenken_
  data.propagate` as ground truth).
- **Phase 3 — neural ordering on KenKen.** Wire the KenKen deducer ckpt as ordering
  priors (MRV-tie entropy + LCV bias) via the SAME `_make_engine_deduce_fn` (already
  domain-agnostic; re-run `VERIFY_PARITY` on KenKen's 3-type membership before trusting
  it). Run the ablation grid (B0 deducer-argmax-no-search; B1 search-no-prop; B2/B2b
  neural ordering + GAC; B3 symbolic ceiling). **READS:** (a) does B3 leave MORE residual
  tree on KenKen than coloring (weaker arithmetic propagation)? — this is the regime where
  ordering priors can earn their keep; (b) does neural ordering lower `decisions_to_solve`
  vs B1/B3 *at honest `forwards_per_solve`*; (c) value-ordering at domain=7 vs coloring's
  3. **GATE for "neural ordering helps generally":** lower `decisions_to_solve` than B3 on
  the search-hard band with forward-cost shown alongside, AND no regression of the coloring
  honest-negative (ordering-only is safe by construction; verify it neither helps nor hurts
  coloring beyond noise). Precondition: a KenKen analog of the G2 entropy-localization read
  (does per-var entropy localize uncertainty on KenKen?) — without it Phase 3 isn't worth
  running.
- **Phase 4 (deferred, gated on a DAG testbed) — circuit port + MCTS slot.** Circuit is a
  3rd registry entry (`gate_pred`) + bridge (L-ASYM resolved at the bridge: ordered scope,
  element 0 = output); it exercises the hierarchical/DAG geometry the radial-depth prize
  needs. MCTS enters ONLY here and ONLY as a Layer-2 strategy for the non-verifiable
  frontier (§4) — never for verifiable CSPs.

### 6.3 Substrate / perf

- **deduce_fn**: REUSE the proven JIT-once/replay + assign-in-place buffer pattern from
  `_make_engine_deduce_fn` verbatim — it is already domain-agnostic (it replays
  `factor_breathing_forward` with live membership/latent_type/cell_valid buffers; the
  mask builder is pure tensor ops, so the per-instance mask-replay correctness argument +
  the `VERIFY_PARITY` gate apply to KenKen's 3 types too). No new JIT graph paths; no
  `float32` literal in the step; softmax in numpy on materialized logits. A silent
  stale-mask bake-in would corrupt ordering priors invisibly → re-run `VERIFY_PARITY` on
  KenKen before trusting it.
- **Generic GAC cost**: bounded by `∏|D(other members)|` per arc (§2.4); cheap for all
  launch domains. Keep the `assign_var` shallow-copy pattern (copy values/domains, share
  immutable `Problem`); the fixpoint round cap carries over; order GAC arcs
  smallest-domain-first (generic perf win, not domain code).
- **Budget unit**: decision-nodes is the cross-strategy comparable work unit; pair it
  with `forwards_per_solve` (Fix-2) so "neural collapses the tree" is only ever claimed
  with BOTH numbers shown.

---

## 7. Three-tier consistency

### 7.0 The two channels — where the predicate registry sits (it is NOT the Poincaré ball)

A factor graph decomposes into two **orthogonal** channels, and conflating them is an
architectural error we have already paid for (v100's "C2 death"):

- **TOPOLOGY** — *who connects to whom* (membership: which variables share a factor).
- **SEMANTICS** — *what relation must hold* (the predicate: which value-tuples a factor
  allows).

These are independent: graph coloring and a hypothetical "adjacent-vertices-must-be-EQUAL"
problem have the **same** topology (identical edges/membership) but **opposite** predicates.
Topology alone cannot tell them apart; the predicate is what distinguishes them.

Every consumer needs **both** channels, which gives a 2×2 — and the predicate registry and
the Poincaré ball sit on **opposite diagonals**:

| | TOPOLOGY (wiring) | SEMANTICS (relation) |
|---|---|---|
| **neural (Tier 3 deducer)** | per-head attention **masks** ← the **Poincaré ball** (Tier 2) generates these | the **verification inlet** (op-type + target features) |
| **symbolic (this search tier)** | `var_factors` incidence + the GAC queue | the **predicate registry** (§2.2) |

So the predicate registry is **not** the Poincaré ball. The ball lives top-left — it
*generates the wiring* (continuous coords → masks), carrying zero "what relation" content.
The registry lives bottom-right. Its true sibling is the **verification inlet**: both encode
the *same* constraint semantics, one fed to the neural deducer, one to the symbolic search —
a single source of constraint truth per domain (the data generator's `cage_ok` / `_eval_gate`
/ `u≠v`), consumed three ways.

**Why they must stay separate (hard-won).** v100 tried folding constraint semantics (op-type)
into the attention/mask channel and it FAILED — hence CLAUDE.md's "arithmetic as VERIFICATION,
never an op-type mask channel." The geometry encodes *where* things sit (abstraction level,
relation grouping → masks); the predicate encodes *what must hold* (→ verification /
propagation). Folding the predicate into the geometry is the refuted move; this spec keeps
the semantics channel (the registry) strictly out of the topology channel (membership/masks).

**Frontier nuance.** When constraints are *learned* from NL rather than given as clean
predicates (the eventual Phase-1 parser world), the semantics channel becomes a learned
embedding too — but that is a constraint-semantics representation, still **distinct** from
the topology (Poincaré) ball.

### 7.1 Tier discipline

- **This tier WRAPS Tier 3, it does not replace it.** Tier 3 (the validated breathing
  deducer) stays the deducer; the search tier is the "deducer proposes, complete search
  disposes" layer. K breaths remain factor-graph iterative inference, NOT tree search —
  MCTS is absent from the verifiable path (§4), consistent with CLAUDE.md.
- **The two-phase split holds.** Phase 1 = structure-finder → factor graph (for CSPs a
  free deterministic spec-reader). Phase 2 = the mask-flexible executor — now extended
  with this search tier, which is itself general over any factor graph the executor
  consumes.
- **Additive + gated discipline (CLAUDE.md §5).** Symbolic systematic search
  (`solve_symbolic`) is the sound, complete DEFAULT and the permanent fallback at every
  phase; neural ordering is the additive, separately-toggled layer that ships only if it
  beats the symbolic ceiling on the search-hard band at honest forward cost. The Phase-0
  reproducibility gate is the byte-identical-fallback analog of the hyperbolic spec's
  "frozen-off = byte-identical."
- **Gold-free preserved.** Every Layer-1 function reads only `(membership, types,
  domains, params)`; the deducer's marginals are an online ordering prior, never a peek at
  gold; `reference_solution` stays metrics-only. The exact predicate verifier (not gold
  match) is the sole `solved` arbiter — frame-invariant (any valid solution passes).
- **Consistent with the validated findings.** Tiers 1–2 remain SPEC-STAGE; this spec
  claims nothing about the Poincaré embedding or hyperbolic generator. The honest-negative
  protocol (symbolic ceiling vs neural arm, `rho_no_ceiling`-style controls, pre-registered
  bars) is the same discipline that produced the Jun-19 coloring result, now domain-agnostic.

---

**Status restated:** SPEC — not built. The grounding (coloring skeleton, `cage_ok`,
`propagate()`'s GAC + arity guard, `_eval_gate`, the engine's per-variable marginals)
is all in-tree and verified; this document specifies the refactor that lifts it into one
predicate-driven, factor-graph-general search tier, with KenKen as the first
zero-coloring-code generality test.
