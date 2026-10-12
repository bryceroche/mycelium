"""solver_pool.py — THE SOLVER PIPELINE's pool + memo for FORM 3's release test
(2026-10-11, perf lever A; ledger "ARCHITECTURE v2 REGISTERED ... THE PERF PLAN").

Of the three places a per-row symbolic solve can run inside a consult,
two are ALREADY pooled:
  - facts_pool.py (ALG_ALT2/ALT3's alt2_fact_buf -> alternator_bridge.ping, a GAC
    propagation only — spawn context, auto-sized by ALG_FACTS_WORKERS, default ON)
  - alternator_bridge.core_rows / core_rows_budget (THE WHEEL's refuse_and_core —
    spawn context, default ON where the wheel is used)
rack_release_rows' per-row rack_release_row call (phase1_algebra_head.py, FORM 3,
ALG_RACK_RELEASE) is the one left sequential, in-process, SIGALRM-walled alone —
the exact call whose unbounded decision cost (NOT its wall-clock, which was
already fixed 2026-10-08) hung RK3X_241 for 13 h. This module extends the SAME
map-reduce pattern to it, behind three env doors that all default to today's
values (phase1_algebra_head.py only calls into this module when at least one of
them asks for something other than today's behavior):

  ALG_SOLVER_POOL=N     fork N workers (csp_core/csp_domains are pure python —
                         no tinygrad/torch import — so fork is safe and cheap:
                         no DEV=CPU re-init needed, nothing here ever touches a
                         GPU handle). N<=1 runs in-process, unchanged order.
  ALG_SOLVER_MEMO=1     memoize the solve by (canonical WL digest of the dry
                         subgraph's factors, m, budget) — an in-process dict,
                         size-capped, with an optional on-disk pickle
                         (.cache/solver_memo_<tag>.pkl) for a cross-run warm
                         start (read at pool/process init only; see save_memo).
  ALG_CONSULT_BUDGET=K  the decision budget used by this solve at TRAINING
                         (default 5000, today's rack_release_row default) — the
                         read-time certificate budget is a different call site
                         entirely and is untouched by this module.

Correctness contract: solve_release_rows(tasks, ...) returns exactly the same
list of status strings ("solved" | "unsat" | "budget"), in task order, that
calling rack_release_row(*task) sequentially with the same budget would —
whether pooled, memoized, both, or neither. _solve_one is the ONE place that
contract is implemented; it reuses phase1_algebra_head._rack_build_problem so
the pooled path can never build a different csp_core.Problem than the
sequential path does from the same inputs.
"""
import os
import pickle
import time

_MEMO_CAP = int(os.environ.get("ALG_SOLVER_MEMO_CAP", "200000"))
_memo_stats = [0, 0]   # [hits, misses] — the sequential/main-process path only (see solve_release_rows)
_calls = [0]
_rows_seen = [0]
_rows_budget = [0]   # rows answered "budget" (exhaustion OR a wall hit — see solve_release_rows' print)
_PRINT_EVERY = 500

_POOL = None
_POOL_N = None


def _head():
    """Lazy import of the head module — scripts/ is on sys.path by the time
    this is ever called (phase1_algebra_head.py is always the caller's own
    module, or a forked copy of it)."""
    import sys
    if "scripts" not in sys.path:
        sys.path.insert(0, "scripts")
    import phase1_algebra_head as H
    return H


def _wl_digest_no_query(factors, nv):
    """THE SAME WL CANONICALIZATION hash_audit_iso.canon() uses (fdesc + 6
    rounds of color refinement + a sorted-signature hash), ADAPTED (not
    reused unmodified — tried first, see the ledger note below) to drop the
    query-variable distinguishing step: canon() marks one var "Q" and ends by
    hashing in col[q], and EVERY existing caller (canonicalizer_bar.py,
    meta_read.py, wild_signal_pricing.py, paraphrase_collapse_read.py) passes
    a real query var — canonical_digest(factors, None, ...) raises KeyError
    (col has no key None) because the door was never built for a query-free
    identity question. A release test asks "is this subgraph consistent",
    never "what is var Q" — there IS no query here — and faking one (e.g.
    picking some used var as "the" query) would bias the coloring toward an
    arbitrary variable and risk the exact failure a memo must never have: two
    DIFFERENT subgraphs colliding on the same key. Dropping the query step
    entirely keeps the WL refinement (and so the cross-row identity it buys —
    two rows whose dry subgraphs are isomorphic up to variable relabeling
    memo-hit each other, not just the identical row recurring) while staying
    sound: no var is privileged, so no var's identity can be lost or faked.
    nv only has to cover every variable index some factor's scope touches —
    unused slots never enter any factor's member list, so they never reach
    `sig` and cannot affect the digest (fixture: solverpipe_correctness_test
    does not probe this directly here, but canon()'s own structure makes it
    true by inspection — sig is built from `facs` alone)."""
    import hashlib
    from collections import defaultdict
    from hash_audit_iso import fdesc
    facs = [fdesc(f) for f in factors]
    col = {v: "." for v in range(nv)}
    for _ in range(6):
        fcols = []
        for kind, mem in facs:
            aa = tuple(sorted(col[m] for r, m in mem if r == "a"))
            rest = tuple((r, col[m]) for r, m in mem if r != "a")
            fcols.append((kind, aa, rest))
        inc = defaultdict(list)
        for (kind, mem), fc in zip(facs, fcols):
            for r, m in mem:
                inc[m].append((fc, r))
        col = {v: (col[v], tuple(sorted(map(repr, inc[v])))) for v in range(nv)}
        ranks = {c: i for i, c in enumerate(sorted(set(map(repr, col.values()))))}
        col = {v: ranks[repr(c)] for v, c in col.items()}
    sig = sorted(repr((kind, tuple(sorted(col[m] for r, m in mem if r == "a")),
                       tuple((r, col[m]) for r, m in mem if r != "a")))
                 for kind, mem in facs)
    return hashlib.sha256("|".join(sig).encode()).hexdigest()


def _digest_key(gv, rel_facs, nv, m, budget):
    """THE MEMO KEY (honestly enumerated): the dry subgraph's query-free WL
    digest (_wl_digest_no_query, above) over its given VALUES and relation
    structure (fdesc bakes a given's value into its descriptor — "giv" carries
    int(f["value"]) — so two subgraphs with the same shape but different
    given numbers never collide) + the domain bound m (a value can be
    satisfiable under one m and not another) + the decision budget (budget
    changes 'budget' vs a proven status, so two calls differing only in
    budget are NOT the same question — a memo that ignored this could hand
    back a stale 'budget' after a later call affords more search, or vice
    versa)."""
    factors = ([{"ftype": "given", "var": int(v), "value": int(val)} for v, val in gv.items()]
               + [{"ftype": "rel", "op": f["op"], "args": list(f["args"]), "result": f["result"]}
                  for f in rel_facs])
    used = {int(v) for v in gv} | {int(a) for f in rel_facs for a in list(f["args"]) + [f["result"]]}
    nv_cover = max([int(nv)] + [v + 1 for v in used])
    digest = _wl_digest_no_query(factors, nv_cover)
    return (digest, int(m), int(budget))


def _solve_one(task, budget, memo=None, stats=None):
    """task = (by_slot, dry_slots, m, n_vars). Returns the status string
    rack_release_row(*task, budget=budget) would — same wall, same
    contradiction-shortcut, same budget semantics — with an optional memo
    lookup/store around the ONE expensive branch (an actual solve_symbolic
    call; the 'unsat'-by-conflict and 'solved'-by-no-relations branches are
    already free and are never memoized, matching rack_release_row's own
    cheapest-first order). stats, if given, is a 2-list [hits, misses]
    bumped explicitly (never inferred from memo size, which a digest failure
    would make ambiguous)."""
    H = _head()
    by_slot, dry_slots, m, n_vars = task
    kind, prob, gv, rel_facs = H._rack_build_problem(by_slot, dry_slots, m, n_vars)
    if kind != "prob":
        return kind
    key = None
    if memo is not None:
        try:
            key = _digest_key(gv, rel_facs, n_vars, m, budget)
        except Exception:
            key = None          # a digest failure never blocks the solve — it just skips the memo
        if key is not None and key in memo:
            if stats is not None:
                stats[0] += 1
            return memo[key]
        if stats is not None:
            stats[1] += 1
    from mycelium.csp_core import solve_symbolic
    wall = float(os.environ.get("ALG_FACTS_ROW_WALL", "5"))
    if wall <= 0:
        status = solve_symbolic(prob, budget=budget, seed=0)["status"]
    else:
        import signal as _sg
        armed = [True]
        def _al(signum, frame):
            if armed[0]:
                raise H._FactsTimeout()
        _old = _sg.signal(_sg.SIGALRM, _al)
        try:
            _sg.setitimer(_sg.ITIMER_REAL, wall)
            try:
                status = solve_symbolic(prob, budget=budget, seed=0)["status"]
                armed[0] = False
            except H._FactsTimeout:
                armed[0] = False
                H._RACK_WALL_N[0] += 1
                status = "budget"       # a wall hit is NEVER a contradiction certificate (the law)
        finally:
            armed[0] = False
            _sg.setitimer(_sg.ITIMER_REAL, 0)
            _sg.signal(_sg.SIGALRM, _old)
    if key is not None:
        if len(memo) >= _MEMO_CAP:
            memo.pop(next(iter(memo)))   # a cheap FIFO-ish evict, not LRU — the cap is a safety
                                          # valve against unbounded growth, not a perf claim
        memo[key] = status
    return status


# ---------------------------------------------------------------------------
# THE IN-PROCESS MEMO (sequential and pool-worker-local)
# ---------------------------------------------------------------------------
_SEQ_MEMO = {}


def load_memo(tag):
    path = f".cache/solver_memo_{tag}.pkl"
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_memo(tag, memo):
    path = f".cache/solver_memo_{tag}.pkl"
    with open(path, "wb") as f:
        pickle.dump(dict(list(memo.items())[:_MEMO_CAP]), f)


# ---------------------------------------------------------------------------
# THE POOL (fork; hang-proof map — the SAME recipe as facts_pool.py's
# _map_hangproof / alternator_bridge.core_rows, ported to a fork context
# since this solver never imports tinygrad/torch)
# ---------------------------------------------------------------------------
def _worker_init(budget, memo_tag):
    global _WORKER_BUDGET, _WORKER_MEMO
    _WORKER_BUDGET = budget
    _WORKER_MEMO = load_memo(memo_tag) if memo_tag else {}


_WORKER_BUDGET = 5000
_WORKER_MEMO = None


_WORKER_STATS = [0, 0]


def _worker_tagged(args):
    i, task = args
    status = _solve_one(task, _WORKER_BUDGET, _WORKER_MEMO, _WORKER_STATS)
    return i, status


def _pool_map(tasks, workers, budget, memo_tag, wall=None):
    global _POOL, _POOL_N
    import multiprocessing as mp
    wall = wall or float(os.environ.get("ALG_FACTS_ROW_WALL", "5")) * 4 + 20.0
    if _POOL is None or _POOL_N != workers:
        if _POOL is not None:
            try:
                _POOL.terminate()
            except Exception:
                pass
        ctx = mp.get_context("fork")
        _POOL = ctx.Pool(workers, initializer=_worker_init, initargs=(budget, memo_tag))
        _POOL_N = workers
    out = [None] * len(tasks)
    todo = list(range(len(tasks)))
    for attempt in range(2):
        if not todo:
            break
        it = _POOL.imap_unordered(_worker_tagged, [(i, tasks[i]) for i in todo], chunksize=1)
        try:
            for _ in range(len(todo)):
                i, status = it.next(timeout=wall)
                out[i] = status
        except mp.TimeoutError:
            pass
        todo = [i for i in todo if out[i] is None]
        if todo:
            print(f"[solver-pool] {len(todo)} row(s) never returned within {wall:.0f} s "
                  f"(attempt {attempt + 1}) — the pool is rebuilt", flush=True)
            try:
                _POOL.terminate()
            except Exception:
                pass
            _POOL = None
            if attempt == 0:
                ctx = mp.get_context("fork")
                _POOL = ctx.Pool(workers, initializer=_worker_init, initargs=(budget, memo_tag))
                _POOL_N = workers
    for i in todo:
        # THE HANG-PROOF MAP's fallback (facts_pool.py's precedent, 2026-09-17): a row that never
        # returns from a pool, twice, is answered IN-PROCESS rather than lost — _solve_one carries
        # its OWN SIGALRM wall (identical mechanism, main thread, safe to retry here) so this call
        # is bounded exactly like the sequential path's ever was; a wall hit still returns "budget",
        # never "unsat" (the law), so even a row that times out again is answered conservatively.
        print(f"[solver-pool] row {i} computed in-process (pool gave up twice)", flush=True)
        try:
            out[i] = _solve_one(tasks[i], budget, None, None)
        except Exception as e:
            print(f"[solver-pool] row {i} in-process fallback raised {type(e).__name__} — answered 'budget'", flush=True)
            out[i] = "budget"
    return out


def solve_release_rows(tasks, workers=0, budget=5000, use_memo=False, memo_tag="chain"):
    """tasks: list of (by_slot, dry_slots, m, n_vars). Returns a list of status
    strings in the SAME order — the contract rack_release_rows' reduce step
    relies on. workers<=1 or len(tasks)<=1: sequential, in-process (no fork).
    use_memo: the sequential path shares ONE process-wide dict (_SEQ_MEMO,
    grown across calls — load_memo/save_memo let a caller warm/bank it from
    disk, not wired into the training loop itself, per the brief: build, don't
    fire); the pooled path gives each worker its OWN dict, seeded once from
    disk at worker init (load_memo(memo_tag)) and never written back
    automatically (cross-process write-back needs a Manager or a merge step;
    left as a documented gap — see the report, not silently assumed away)."""
    _calls[0] += 1
    if not tasks:
        out = []
    else:
        memo = _SEQ_MEMO if use_memo else None
        if workers <= 1 or len(tasks) <= 1:
            out = [_solve_one(t, budget, memo, _memo_stats if memo is not None else None)
                   for t in tasks]
        else:
            out = _pool_map(tasks, workers, budget, memo_tag if use_memo else None)
    _rows_seen[0] += len(out)
    _rows_budget[0] += sum(1 for s in out if s == "budget")
    if _calls[0] % _PRINT_EVERY == 0:
        # ALG_CONSULT_BUDGET's ask: how often the training budget is exhausted (a reject) vs the
        # default — "budget" covers BOTH a genuine decision-node exhaustion and a SIGALRM wall hit
        # (the law: a wall is never a contradiction certificate, so it is folded into the same
        # conservative status); this print does not distinguish the two causes.
        print(f"[solver-pool] calls={_calls[0]} rows={_rows_seen[0]} budget-exhausted="
              f"{_rows_budget[0]} ({100.0 * _rows_budget[0] / max(_rows_seen[0], 1):.1f}%) "
              f"memo hits={_memo_stats[0]} misses={_memo_stats[1]} (main-process path only; "
              f"pool workers keep their own, unreported)", flush=True)
    return out
