"""(ledger 11-14) THE ROUTING QUEUE: two new forms of the routed heads (RD_GRAPH=h), toward a DISCOVERED routing graph.
- RD_GNOISE=r (percent, tag N{r}): the routing graph is the true one with r% of each var's edges rewired to a uniformly random real
  factor it does not touch (var degrees kept; fresh per read, the placebo's own rng) -- how much of the lift survives an imperfect graph.
- RD_GKNN=k (tag K{k}): the routed heads attend to themselves + their k nearest tokens by a LEARNED similarity of the breath's own
  normalized state (knn_w: D x 32, per breath, weight-tied); the hard top-k is chosen on the detached similarity, and the similarity
  also enters the routed heads' scores as a bias (the projection's gradient road). No factor-graph input to the mask at all: the
  structure is discovered from state (KNN message passing)."""
import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]; s = s.replace(a, b)
rep('''GPLACEBO = int(os.environ.get("RD_GPLACEBO", "0"))''', '''GNOISE = int(os.environ.get("RD_GNOISE", "0"))     # (ROUTING QUEUE, ledger 11-14) percent of each var's routing edges rewired
GKNN = int(os.environ.get("RD_GKNN", "0"))         # (ROUTING QUEUE) routed heads by learned-state KNN, k neighbours (no graph in the mask)
KD = 32
GPLACEBO = int(os.environ.get("RD_GPLACEBO", "0"))''')
rep('''{'P' if GPLACEBO else ''}_depth2c''', '''{'P' if GPLACEBO else ''}{f'N{GNOISE}' if GNOISE else ''}{f'K{GKNN}' if GKNN else ''}_depth2c''')
rep('''assert not GPLACEBO or GRAPH, "GPLACEBO is a form of the routed heads"''', '''assert not GPLACEBO or GRAPH, "GPLACEBO is a form of the routed heads"
assert not (GNOISE or GKNN) or GRAPH, "GNOISE / GKNN are forms of the routed heads"
assert (GPLACEBO > 0) + (GNOISE > 0) + (GKNN > 0) <= 1, "one routing form at a time"''')
# the noisy graph rides as the 11th batch array (B x V x F routing incidence)
rep('''        out.append(Pm)
    return out, np.pad(ids, [(0, 0), (0, NV - nv)])
''', '''        out.append(Pm)
    if GNOISE:                                                                   # per var: r% of its edges rewired to a factor it does not touch
        ig = ((out[4] + out[5] + out[6]) > 0).transpose(0, 2, 1).astype(np.float32)
        for i in range(ig.shape[0]):
            rf = np.nonzero(out[3][i])[0]
            for v in np.nonzero(out[1][i])[0]:
                nb = np.nonzero(ig[i, v])[0]; free = np.setdiff1d(rf, nb)
                for f in nb:
                    if len(free) and PRNG.random() < GNOISE / 100:
                        g = PRNG.choice(free); ig[i, v, f] = 0.0; ig[i, v, g] = 1.0; free = np.setdiff1d(free, [g])
        out.append(ig)
    return out, np.pad(ids, [(0, 0), (0, NV - nv)])
''')
rep('''        if GPLACEBO: ig = ig @ bt[10]                                            # the placebo: var -> the wrong factors (same degrees)
''', '''        if GPLACEBO: ig = ig @ bt[10]                                            # the placebo: var -> the wrong factors (same degrees)
        if GNOISE: ig = bt[10]                                                   # the noisy graph (r% of each var's edges rewired)
''')
rep('''        gm = (nbr > 0).where(0.0, -1e4).reshape(B, 1, L, L)
        mask = mask + Tensor.cat(*([gm] * GRAPH + [gm * 0] * (NH - GRAPH)), dim=1)
''', '''        gm = (nbr > 0).where(0.0, -1e4).reshape(B, 1, L, L)
        if not GKNN: mask = mask + Tensor.cat(*([gm] * GRAPH + [gm * 0] * (NH - GRAPH)), dim=1)
    if GKNN:                                                                      # the routed heads' rows are rebuilt each breath from state
        eyeL = Tensor.eye(L).reshape(1, L, L); padk = pad.reshape(B, 1, L); knl = []
''')
rep('''        x = write(x, mha(P, lnorm(P, x, "blocks.0.ln1"), mask), unit, 2)
''', '''        if GKNN:                                                                  # KNN MESSAGE PASSING: neighbours by learned state similarity
            y = lnorm(P, x, "blocks.0.ln1"); e = y @ P["knn_w"]; sim = (e @ e.transpose(1, 2)) * (1 / math.sqrt(KD))
            sd = (padk | (eyeL > 0)).where(-1e9, sim.detach())                   # self and pads never compete for the k slots
            rk = (sd.unsqueeze(2) > sd.unsqueeze(3)).cast(dtypes.float).sum(-1)  # B x L x L: how many keys beat key j for query i
            sel = ((rk < GKNN) & (sd > -1e8)) | (eyeL > 0)
            km = (sel.where(0.0, -1e4) + sim).reshape(B, 1, L, L)
            if ex: knl.append(sel)
            x = write(x, mha(P, y, mask + Tensor.cat(*([km] * GRAPH + [km * 0] * (NH - GRAPH)), dim=1)), unit, 2)
        else:
            x = write(x, mha(P, lnorm(P, x, "blocks.0.ln1"), mask), unit, 2)
''')
rep('''    if ex: r.update(known=kn, kt=ktl, nrm=nrm, valid=valid, xs=xs, sig=sgl)''', '''    if ex: r.update(known=kn, kt=ktl, nrm=nrm, valid=valid, xs=xs, sig=sgl, knn=knl if GKNN else [])''')
rep('''def add_dial(P):''', '''def add_knn(P):
    P["knn_w"] = Tensor(np.random.default_rng(4242 + RB.SEED).standard_normal((D, KD)).astype(np.float32) / math.sqrt(D)).contiguous().realize()
    return P


def add_dial(P):''')
rep('''        if DIAL: P = add_dial(P); npar += D + 5
''', '''        if DIAL: P = add_dial(P); npar += D + 5
        if GKNN: P = add_knn(P); npar += D * KD
''')
open(p, "w").write(s); print("routing queue patch applied")
