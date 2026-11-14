"""(ledger 11-13 16:15) the scorer is the tinygrad forward whenever UNIT or DAMP (the torch net has neither); a UNIT run keeps the
torch-scored per-item DEEP read as deep_per_torch (the determinism witness against runs banked before this fix)"""
import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
rep('''SC = (lambda its: tg_score(P, its, UNIT)) if DAMP else''', '''SC = (lambda its: tg_score(P, its, UNIT)) if DAMP or UNIT else''')
rep('''        agree = None if DAMP else float((tg_score(P, deep, UNIT) == S["deep"]).mean())   # the tinygrad scorer's witness on the control path
''', '''        agree = None if DAMP or UNIT else float((tg_score(P, deep, UNIT) == S["deep"]).mean())   # the tinygrad scorer's witness on the control path
        dpt = [int(x) for x in (to_torch(P, tn), tn.eval(), RB.score(tn, deep))[2]] if UNIT else None   # torch-scored: the pre-fix witness only
''')
rep('''tg_score_agree=agree), open''', '''tg_score_agree=agree, deep_per_torch=dpt), open''')
open(p, "w").write(s); print("unit-score patch applied")
