"""(ledger 11-15) SOFT-BIAS + WITNESS-LOCK. RD_GSOFT=b: the routed heads get +b on proposed edges instead of the hard -1e4 mask (soft
evidence only biases; b fixed, never learned). RD_GWIT=r: two INDEPENDENT r%-rewired copies of the graph (the GNOISE rewire, drawn twice);
the routed heads' hard mask = their intersection (lock only what both witnesses propose)."""
import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]; s = s.replace(a, b)
rep('''GCERT = int(os.environ.get("RD_GCERT", "0"))''',
    '''GSOFT = float(os.environ.get("RD_GSOFT", "0"))   # (SOFT-BIAS, ledger 11-15) routed heads: +b on proposed edges, no hard mask
GWIT = int(os.environ.get("RD_GWIT", "0"))       # (WITNESS-LOCK, ledger 11-15) hard mask = the intersection of two independent r%-rewired copies
GCERT = int(os.environ.get("RD_GCERT", "0"))''')
rep("{f'C{GCERT}' if GCERT else ''}", "{f'C{GCERT}' if GCERT else ''}{f'W{GWIT}' if GWIT else ''}{f'B{GSOFT:g}' if GSOFT else ''}")
rep('''assert 0 <= GCERT < 100,''', '''assert not (GSOFT or GWIT) or (GRAPH and not GKNN), "GSOFT / GWIT are forms of the routed heads"
assert 0 <= GCERT < 100,''')
rep('''+ (GCERT > 0) <= 1,''', '''+ (GCERT > 0) + (GWIT > 0) <= 1,''')
rep('''        if GNOISE or GCERT: ig = bt[10]''', '''        if GNOISE or GCERT or GWIT: ig = bt[10]''')
rep('''        gm = (nbr > 0).where(0.0, -1e4).reshape(B, 1, L, L)''', '''        gm = (nbr > 0).where(GSOFT if GSOFT else 0.0, 0.0 if GSOFT else -1e4).reshape(B, 1, L, L)''')
rep('''    if GNOISE:                                                                   # per var: r% of its edges rewired to a factor it does not touch
        ig = ((out[4] + out[5] + out[6]) > 0).transpose(0, 2, 1).astype(np.float32)''',
    '''    def rewire(r):                                                               # per var: r% of its edges rewired to a factor it does not touch
        ig = ((out[4] + out[5] + out[6]) > 0).transpose(0, 2, 1).astype(np.float32)''')
rep('''                    if len(free) and PRNG.random() < GNOISE / 100:
                        g = PRNG.choice(free); ig[i, v, f] = 0.0; ig[i, v, g] = 1.0; free = np.setdiff1d(free, [g])
        out.append(ig)''',
    '''                    if len(free) and PRNG.random() < r / 100:
                        g = PRNG.choice(free); ig[i, v, f] = 0.0; ig[i, v, g] = 1.0; free = np.setdiff1d(free, [g])
        return ig
    if GNOISE: out.append(rewire(GNOISE))
    if GWIT: out.append(rewire(GWIT) * rewire(GWIT))                              # lock only what both independent witnesses propose''')
open(p, "w").write(s); print("soft + witness applied")
