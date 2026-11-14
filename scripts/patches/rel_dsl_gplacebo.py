"""(ledger 11-14) GRAPH PLACEBO: RD_GPLACEBO=1 (with RD_GRAPH=h) routes the h heads along a RELABELLED factor graph -- each item's real
factors are shuffled by a uniform random permutation (fresh per read, its own rng: the item stream is untouched), so the routing graph
keeps the exact degree sequence and shape of the true one but points each var at the wrong factors. Rides as an 11th batch array (F x F
permutation, identity on pads); the factor tokens' content (A, Rz, Mz) is untouched -- only WHERE the routed heads may look moves."""
import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]; s = s.replace(a, b)
rep('''GRAPH = int(os.environ.get("RD_GRAPH", "0"))''', '''GPLACEBO = int(os.environ.get("RD_GPLACEBO", "0"))   # (GRAPH PLACEBO, ledger 11-14) route along a relabelled graph
GRAPH = int(os.environ.get("RD_GRAPH", "0"))''')
rep('''{f'G{GRAPH}' if GRAPH else ''}_depth2c''', '''{f'G{GRAPH}' if GRAPH else ''}{'P' if GPLACEBO else ''}_depth2c''')
rep('''assert 0 <= GRAPH <= NH, "RD_GRAPH = the number of routed heads"''', '''assert 0 <= GRAPH <= NH, "RD_GRAPH = the number of routed heads"
assert not GPLACEBO or GRAPH, "GPLACEBO is a form of the routed heads"
PRNG = np.random.default_rng(1000 + RB.SEED)       # the placebo's own stream (the item / symbol rng is untouched)''')
rep('''    vf, vm, fo, fm, A, Rz, Mz, pm, tg, dv = bt
    B, nv, nf = vf.shape[0], vf.shape[1], fo.shape[1]; L = nv + nf; pad''', '''    vf, vm, fo, fm, A, Rz, Mz, pm, tg, dv = bt[:10]
    B, nv, nf = vf.shape[0], vf.shape[1], fo.shape[1]; L = nv + nf; pad''')
rep('''        ig = ((A + Rz + Mz) > 0).cast(dtypes.float).transpose(1, 2)              # B x V x F
''', '''        ig = ((A + Rz + Mz) > 0).cast(dtypes.float).transpose(1, 2)              # B x V x F
        if GPLACEBO: ig = ig @ bt[10]                                            # the placebo: var -> the wrong factors (same degrees)
''')
rep('''    return out, np.pad(ids, [(0, 0), (0, NV - nv)])
''', '''    if GPLACEBO:                                                                 # per item: a random permutation of its real factors
        Pm = np.zeros((out[2].shape[0], NF, NF), np.float32)
        for i, f in enumerate(out[3]):
            r = np.nonzero(f)[0]; q = np.arange(NF); q[r] = PRNG.permutation(r); Pm[i, np.arange(NF), q] = 1.0
        out.append(Pm)
    return out, np.pad(ids, [(0, 0), (0, NV - nv)])
''')
rep('''        bt, ids, ss, bc2s = list(a[:10]), a[10], a[11], a[12]''', '''        bt, ids, ss, bc2s = list(a[:-3]), a[-3], a[-2], a[-1]''')
open(p, "w").write(s); print("graph placebo patch applied")
