"""gsm_order_audit.py -- THE ORDER AUDIT (ledger 2026-11-10 pin): among the problems SEL picked an ANSWER-RIGHT candidate, how often is
that candidate's program not slot-aligned with gold (some step_all miss)? And among those, how many are the gold steps re-ordered
(the same multiset of (op, operand-kind) steps) vs a different route? Zero GPU on .cache/gsm_select_{ARM}{.npz,_progs.json}."""
import json, os, sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_bench_data as GD

ARM = os.environ.get("SR_ARM", "MACRO_s0"); K = 12
its = json.load(open(GD.OUT))["TEST"]; d = np.load(f".cache/gsm_select_{ARM}.npz"); P = json.load(open(f".cache/gsm_select_{ARM}_progs.json"))
m = [min(len(it["prog"]), K) for it in its]; off = np.concatenate([[0], np.cumsum(m)])
S, A, pick = d["S"], d["A"], d["pick_SEL"]


def sig(prog):     # a step's shape without its slot references: op + sorted operand kinds/values-tokens
    return Counter((o, tuple(sorted((a[0], a[1] if a[0] != "s" else -1) for a in (A_, B_)))) for o, A_, B_ in prog)


n = al = reo = 0; rows = []
for j, it in enumerate(its):
    c = pick[j]
    if not A[c, j]: continue
    n += 1; ok = S[c, off[j]:off[j + 1]].all()
    if ok: al += 1; continue
    pr = [(o, tuple(a), tuple(b)) for o, a, b in P[c][j]]; gp = [(o, (a["t"] and ("t", a["t"][0])) or (a["s"] and ("s", a["s"][0])) or ("c", a["c"][0]),
                                                            (b["t"] and ("t", b["t"][0])) or (b["s"] and ("s", b["s"][0])) or ("c", b["c"][0])) for o, a, b in it["prog"]]
    reo += sig(pr) == sig(gp)
gr = A[0].sum()
out = (f"{ARM}: SEL answer-right {n} (greedy {gr}) | slot-aligned (every step right) {al} | misaligned {n - al}: same step multiset "
       f"(re-ordered) {reo}, other route {n - al - reo} | share misaligned {(n - al) / max(n, 1):.3f}")
print(out); open(f".cache/gsm_order_audit_{ARM}.txt", "w").write(out + "\n")
