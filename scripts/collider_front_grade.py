"""collider_front_grade.py — THE COLLIDER, RUNG 1 grader (CPU). Readings
A (the champion open), B, C (rivals). Per present gold slot: wrongA;
disagree(A,X) = signature differs. COVERAGE = P(disagree | wrongA);
LIFT = P(wrongA | disagree) / P(wrongA). Bars (ledger 2026-09-09):
coverage >= 0.40 AND lift >= 3.0 on at least one rival pair."""
import sys, numpy as np
A = np.load(sys.argv[1]); rivals = [(n, np.load(f)) for n, f in (a.split("=") for a in sys.argv[2:])]
pres = A["present"].astype(bool); wrongA = (A["correct"] == 0) & pres
base = wrongA.sum() / pres.sum()
print(f"[collider-grade] A: slots={pres.sum()} wrong={wrongA.sum()} base wrong rate={base:.4f}  A fac-exact={1-base:.4f}")
passed = False
for name, R in rivals:
    dis = (A["sig"] != R["sig"]).any(-1) & pres
    cov = (dis & wrongA).sum() / max(wrongA.sum(), 1)
    prec = (dis & wrongA).sum() / max(dis.sum(), 1)
    lift = prec / base
    rows_dis = dis.any(-1).mean()
    ok = cov >= 0.40 and lift >= 3.0
    passed |= ok
    print(f"[collider-grade] A vs {name}: R fac-exact={R['correct'][pres].mean():.4f} disagree slots={dis.sum()} ({dis.sum()/pres.sum():.3f} of slots; {rows_dis:.3f} of rows) "
          f"COVERAGE={cov:.3f} (bar>=0.40) P(wrong|dis)={prec:.3f} LIFT={lift:.2f}x (bar>=3.0) -> {'PASS' if ok else 'FAIL'}")
print(f"[collider-grade] RUNG 1 {'PASS' if passed else 'KILL — the front is too thin to steer by'}")
