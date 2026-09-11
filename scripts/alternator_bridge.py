"""alternator_bridge.py — ALTERNATOR V2's symbolic half (2026-09-01,
word given: per-breath ping-pong between the neural parse and symbolic
propagation). The bridge consumes a PARTIAL committed parse (the
confident subset of slots at breath k), calls the June core's
propagation layer ONLY (gac_propagate + forced singletons — no search,
no branching; the meter-divergence law: this check calls its organ),
and returns (facts, domain_mass) for injection into breath k+1.

Contract (the dual-terminal law, rotational_bus.md §6): everything
returned here is DETACHED FACT — it enters the forward pass as
conditioning (value channel + mask channel), never as supervision;
dL/dp through this module is identically zero by construction.

The commit adapter (o-dict -> committed factors) lives with the
forward patch, not here: this module is head-agnostic on purpose.
"""
import time

from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import make_initial_state, gac_propagate


def ping(n_vars, factors, m, max_rounds=8, arity_cap=20000):
    """One symbolic breath. factors: the COMMITTED subset (given/rel
    dicts, grammar form). Returns (facts, mass, rounds):
      facts: {var: value} for every variable forced to a singleton
      mass:  per-var domain size after propagation (the matryoshka
             radius — feeds the settling meter, diagnostic register
             ONLY, never a loss)
      rounds: propagation sweeps used
    Contradiction (emptied domain) returns ({}, None, r) — the neural
    side committed an impossible partial parse; injection layer treats
    this as 'no facts' (silence, never a gradient)."""
    gv = {f["var"]: f["value"] for f in factors if f["ftype"] == "given"}
    prob = problem_from_algebra3(n_vars, factors, gv, m)
    st = make_initial_state(prob)
    for r in range(max_rounds):
        before = [len(d) for d in st.domains]
        st = gac_propagate(st, arity_cap=arity_cap)
        after = [len(d) for d in st.domains]
        if any(a == 0 for a in after):
            return {}, None, r + 1
        if after == before:
            break
    facts = {v: next(iter(d)) for v, d in enumerate(st.domains)
             if len(d) == 1}
    mass = [len(d) for d in st.domains]
    return facts, mass, r + 1


def ladder_replay(row, prefix_fracs=(0.25, 0.5, 0.75, 1.0)):
    """Diagnostic: replay a wild row as the ping-pong would see it —
    commit growing prefixes of the factor list (a proxy for the parse
    firming up across breaths) and report how many facts the symbolic
    jaw hands back at each stage. The alternator's value hypothesis is
    exactly this curve: facts arrive EARLY on ladder-shaped problems."""
    fs = row["factors"]
    out = []
    for frac in prefix_fracs:
        k = max(1, round(len(fs) * frac))
        facts, mass, r = ping(row["n_vars"], fs[:k], row["m"])
        out.append((k, len(facts), r,
                    None if mass is None else sum(mass)))
    return out


if __name__ == "__main__":
    import json
    rows = [json.loads(l) for l in open(".cache/gsm8k_wild_drafts.jsonl")]
    import random
    rng = random.Random(11)
    samp = rng.sample(rows, 200)
    t0 = time.time()
    calls = 0
    early = 0          # rows where >=1 non-given fact exists at half-parse
    full_ok = 0        # rows fully forced at full commit (density check)
    for row in samp:
        rep = ladder_replay(row)
        calls += len(rep)
        giv = sum(1 for f in row["factors"] if f["ftype"] == "given")
        if rep[1][1] > giv:
            early += 1
        if rep[-1][1] == row["n_vars"]:
            full_ok += 1
    dt = time.time() - t0
    print(f"[bridge] {len(samp)} rows x 4 prefix pings = {calls} calls "
          f"in {dt:.2f}s ({dt / calls * 1000:.2f} ms/call)")
    print(f"[bridge] facts-before-full-parse (the ping-pong's fuel): "
          f"{early}/{len(samp)} rows hand back derived facts at "
          f"half-commit")
    print(f"[bridge] fully-forced at full commit: {full_ok}/{len(samp)} "
          f"(ladder density check — extractor promises 100%)")


def refuse_and_core(n_vars, factors, m, budget=20000, seed=0):
    """THE STEERING WHEEL's driver (2026-09-10, the word): solve the
    committed parse COMPLETELY (the June core's search, budgeted); on a
    certified refusal, the deletion-based MINIMAL UNSATISFIABLE CORE over
    the parse's OWN items — factor dicts, givens included (a wrong given
    is a wrong slot too). Returns
      {'status': 'sat' | 'unsat' | 'budget' | 'unbuildable',
       'core': [indices into `factors`], 'checks': n}
    'unsat' + core = the proof the wheel steers by: the smallest set of
    slots that cannot coexist. Everything here is detached fact (the
    dual-terminal contract): it conditions, it never supervises."""
    from mycelium.csp_core import solve_symbolic, deletion_core

    def _build(fs):
        gv = {f["var"]: f["value"] for f in fs if f["ftype"] == "given"}
        return problem_from_algebra3(n_vars, fs, gv, m)

    def _is_unsat(fs):
        try:
            return solve_symbolic(_build(fs), budget=budget,
                                  seed=seed)["status"] == "unsat"
        except (ValueError, KeyError, IndexError):
            return False            # an unbuildable subset is never a proof
    try:
        r = solve_symbolic(_build(list(factors)), budget=budget, seed=seed)
    except (ValueError, KeyError, IndexError) as e:
        return {"status": "unbuildable", "core": [], "checks": 1,
                "why": type(e).__name__}
    if r["status"] != "unsat":
        return {"status": r["status"], "core": [], "checks": 1}
    idx = list(range(len(factors)))
    core, checks = deletion_core(
        idx, lambda keep: _is_unsat([factors[i] for i in keep]))
    return {"status": "unsat", "core": core, "checks": checks + 1}


# ---------------------------------------------------------------------------
# THE WHEEL'S POOL (2026-09-11): the solver's cores across CPU cores. The
# pass is solver-bound; a spawn-context pool keeps the children clear of
# the parent's GPU handle (the AM single-process law).
# ---------------------------------------------------------------------------
_POOL = None


def _core_worker(args):
    n_vars, parse, m = args
    r = refuse_and_core(n_vars, parse, m)
    return r["status"], r["core"]


def core_rows(rows, workers=None):
    """rows: list of (n_vars, parse, m). Returns [(status, core), ...] in
    order. workers=None -> os.cpu_count()-2; workers<=1 -> in-process."""
    import os as _os
    global _POOL
    if workers is None:
        workers = max(1, (_os.cpu_count() or 2) - 2)
    if workers <= 1 or len(rows) <= 1:
        return [_core_worker(a) for a in rows]
    if _POOL is None:
        import multiprocessing as _mp
        _POOL = _mp.get_context("spawn").Pool(workers)
    return _POOL.map(_core_worker, rows, chunksize=1)
