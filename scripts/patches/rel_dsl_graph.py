"""(ledger 11-14) GRAPH-ROUTED HEADS: RD_GRAPH=h masks the first h heads to each token's factor-graph neighbourhood (var: itself + its
incident factors; factor: itself + its incident vars); the scorer is the tinygrad forward for any form the torch net lacks"""
import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]; s = s.replace(a, b)
rep('''SCHED = int(os.environ.get("RD_SCHED", "0"))''', '''GRAPH = int(os.environ.get("RD_GRAPH", "0"))   # (GRAPH, ledger 11-14) heads 0..GRAPH-1 routed by the factor graph
SCHED = int(os.environ.get("RD_SCHED", "0"))''')
rep('''{'D' if DAMP else ''}_depth2c''', '''{'D' if DAMP else ''}{f'G{GRAPH}' if GRAPH else ''}_depth2c''')
rep('''assert not SCHED or UNIT, "SCHED is a form of the unitary breath"''', '''assert not SCHED or UNIT, "SCHED is a form of the unitary breath"
assert 0 <= GRAPH <= NH, "RD_GRAPH = the number of routed heads"
TORCH_OK = not (UNIT or DAMP or GRAPH)          # the torch net has none of these: every other form is scored on the tinygrad forward''')
rep('''    x = Tensor.cat(V, Fz, dim=1); valid = pad.logical_not().cast(dtypes.float)
''', '''    x = Tensor.cat(V, Fz, dim=1); valid = pad.logical_not().cast(dtypes.float)
    if GRAPH:                                                                     # THE ROUTED HEADS: attend only along factor-graph edges
        ig = ((A + Rz + Mz) > 0).cast(dtypes.float).transpose(1, 2)              # B x V x F
        nbr = Tensor.cat(Tensor.cat(Tensor.eye(nv).reshape(1, nv, nv).expand(B, nv, nv), ig, dim=2),
                         Tensor.cat(ig.transpose(1, 2), Tensor.eye(nf).reshape(1, nf, nf).expand(B, nf, nf), dim=2), dim=1)   # B x L x L
        gm = (nbr > 0).where(0.0, -1e4).reshape(B, 1, L, L)
        mask = mask + Tensor.cat(*([gm] * GRAPH + [gm * 0] * (NH - GRAPH)), dim=1)
''')
rep('''SC = (lambda its: tg_score(P, its, UNIT)) if DAMP or UNIT else''', '''SC = (lambda its: tg_score(P, its, UNIT)) if not TORCH_OK else''')
rep('''        agree = None if DAMP or UNIT else float''', '''        agree = None if not TORCH_OK else float''')
open(p, "w").write(s); print("graph patch applied")
