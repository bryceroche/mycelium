"""THE PROOF IS THE PROPAGATION (2026-09-12): patch alternator_bridge.refuse_and_core
so every unsat test tries GAC propagation to a fixpoint FIRST (an emptied domain is a
sound certificate) and falls back to the budgeted search only when propagation does
not refute. Measured on the 960-row wheel fixture: 50/51 unsat rows refuted by
propagation alone; the deletion core 397.7 s -> 71.6 s with IDENTICAL cores 51/51.
Idempotent. Run only when no chain re-imports the bridge."""
import re, sys
p = "scripts/alternator_bridge.py"; s = open(p).read()
if "_prop_refutes" in s:
    print("[apply] already applied"); sys.exit(0)
old = '''    def _is_unsat(fs):
        try:
            return solve_symbolic(_build(fs), budget=budget,
'''
new = '''    def _prop_refutes(prob, rounds=8):
        # THE PROOF IS THE PROPAGATION (2026-09-12): GAC to a fixpoint; an
        # emptied domain is a sound unsat certificate (the wheel fixture:
        # 50/51 refusals fall here; the search's own unsat proofs were the
        # 80 s rows). Sat is never claimed here — that stays the search's.
        st = make_initial_state(prob)
        for _ in range(rounds):
            before = [len(d) for d in st.domains]
            st = gac_propagate(st)
            after = [len(d) for d in st.domains]
            if any(a == 0 for a in after):
                return True
            if after == before:
                return False
        return False

    def _is_unsat(fs):
        try:
            pb = _build(fs)
        except (ValueError, KeyError, IndexError):
            return False            # an unbuildable subset is never a proof
        if _prop_refutes(pb):
            return True
        try:
            return solve_symbolic(pb, budget=budget,
'''
assert s.count(old) == 1, s.count(old); s = s.replace(old, new)
old2 = '''    try:
        r = solve_symbolic(_build(list(factors)), budget=budget, seed=seed)
    except (ValueError, KeyError, IndexError) as e:
        return {"status": "unbuildable", "core": [], "checks": 1,
                "why": type(e).__name__}
    if r["status"] != "unsat":
        return {"status": r["status"], "core": [], "checks": 1}
'''
new2 = '''    try:
        pb0 = _build(list(factors))
    except (ValueError, KeyError, IndexError) as e:
        return {"status": "unbuildable", "core": [], "checks": 1,
                "why": type(e).__name__}
    if not _prop_refutes(pb0):
        r = solve_symbolic(pb0, budget=budget, seed=seed)
        if r["status"] != "unsat":
            return {"status": r["status"], "core": [], "checks": 1}
'''
assert s.count(old2) == 1, s.count(old2); s = s.replace(old2, new2)
open(p, "w").write(s); print("[apply] prop-first refusal + core applied to", p)
