import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a)); s = s.replace(a, b)
rep('''ARM = os.environ.get("RD_ARM", "NOMLP"); UNIT = int(os.environ.get("RD_UNIT", "0")); COV = float(os.environ["RD_COV"])''',
    '''ARM = os.environ.get("RD_ARM", "NOMLP"); UNIT = int(os.environ.get("RD_UNIT", "0")); COV = float(os.environ["RD_COV"])
DAMP = int(os.environ.get("RD_DAMP", "0"))   # (DAMP, ledger 11-13) a learned pole per complex plane on the carried state, each breath''')
rep('''TAG = os.environ.get("RD_TAG", f"TG{ARM}{'U' if UNIT else ''}_depth2c{COV:g}_s{RB.SEED}")''',
    '''TAG = os.environ.get("RD_TAG", f"TG{ARM}{'U' if UNIT else ''}{'D' if DAMP else ''}_depth2c{COV:g}_s{RB.SEED}")
assert not (DAMP and UNIT), "DAMP and UNIT are separate forms"''')
rep('''def write(x, d, unit):''', '''def damp(P, x):
    """(DAMP) the carried state times a learned pole per complex plane (adjacent dims paired): p_k = sigmoid(a_k) e^{i phi_k}"""
    r = P["damp_a"].sigmoid(); c, sn = r * P["damp_phi"].cos(), r * P["damp_phi"].sin(); re, im = x[..., 0::2], x[..., 1::2]
    return Tensor.stack(re * c - im * sn, re * sn + im * c, dim=-1).reshape(*x.shape)


def add_damp(P):
    P["damp_a"] = Tensor.full((D // 2,), 3.0).contiguous().realize(); P["damp_phi"] = Tensor.zeros(D // 2).contiguous().realize()
    return P


def write(x, d, unit):''')
rep('''    out, kn, ktl, nrm = [], [], [], [(x * x).sum(-1).sqrt()] if ex else []
    for b in range(NB):
        x = write(x, lin(P, P["bst"][b], "bproj").reshape(1, 1, D), unit)''',
    '''    out, kn, ktl, nrm = [], [], [], [(x * x).sum(-1).sqrt()] if ex else []; xs = [x] if ex else []
    for b in range(NB):
        if DAMP: x = damp(P, x)
        x = write(x, lin(P, P["bst"][b], "bproj").reshape(1, 1, D), unit)''')
rep('''        if ex: nrm.append((x * x).sum(-1).sqrt())''', '''        if ex: nrm.append((x * x).sum(-1).sqrt()); xs.append(x)''')
rep('''    if ex: r.update(known=kn, kt=ktl, nrm=nrm, valid=valid)''', '''    if ex: r.update(known=kn, kt=ktl, nrm=nrm, valid=valid, xs=xs)''')
rep('''def torch_flags(''', '''def pole_census(P, items, unit):
    """(POLE, ledger 11-13) per-breath states of the real tokens (256 DEEP items, the census batch), centered per breath, pooled least
    squares x_{b+1} ~ x_b A over breaths 1..NB-1 -> eigenvalues of A: a local linearization around the trajectory, R^2 reported"""
    rng = np.random.default_rng(777); its = [items[i % len(items)] for i in range(256)]
    arrs, ids = make_batch(its, rng); bt, idt = tens(arrs, ids); r = forward(P, bt, idt, unit, ex=True); val = r["valid"].numpy() > 0
    X = [x.numpy()[val].astype(np.float64) for x in r["xs"]]; X = [x - x.mean(0) for x in X]
    Xa, Xb = np.concatenate(X[1:-1]), np.concatenate(X[2:]); A, *_ = np.linalg.lstsq(Xa, Xb, rcond=None)
    r2 = 1 - ((Xb - Xa @ A) ** 2).sum() / (Xb ** 2).sum(); lam = np.abs(np.linalg.eigvals(A))
    out = dict(r2=float(r2), q=[float(x) for x in np.quantile(lam, [0.25, 0.5, 0.75])], inside=float((lam < 0.99).mean()),
               on=float(((lam >= 0.99) & (lam <= 1.01)).mean()), outside=float((lam > 1.01).mean()), top=[float(x) for x in np.sort(lam)[::-1][:5]])
    if DAMP:
        rad = 1 / (1 + np.exp(-P["damp_a"].numpy().astype(np.float64))); ph = np.degrees(np.abs(P["damp_phi"].numpy()))
        out["learned"] = dict(radius_q=[float(x) for x in np.quantile(rad, [0.25, 0.5, 0.75])], share_lt_0p9=float((rad < 0.9).mean()),
                              angle_deg_q=[float(x) for x in np.quantile(ph, [0.25, 0.5, 0.75])])
    return out


def tg_score(P, items, unit, seed=0):
    """RB.score on the tinygrad forward (same batches: RB.batch with the same rng; the last breath's argmax plan, the executor grades);
    needed for DAMP (the torch net has no poles); checked equal to RB.score on the control path"""
    rng = np.random.default_rng(10_000 + seed); ok = []
    for s0 in range(0, len(items), 256):
        its = items[s0:s0 + 256]; arrs, ids = make_batch(its, rng); bt, idt = tens(arrs, ids); lg = forward(P, bt, idt, unit)["lg"].numpy()[:, -1]
        for k, t in enumerate(its): ok.append(RB.RC.execute(t["facs"], {v: int(lg[k, v].argmax()) for v in t["der"]}, t["query"]) == t["gold"])
    return np.asarray(ok)


def torch_flags(''')
rep('''        torch.manual_seed(SEED); rng = np.random.default_rng(SEED); tn = mk(); P = from_torch(tn.state_dict()); npar = sum(p.numel() for p in tn.parameters())''',
    '''        torch.manual_seed(SEED); rng = np.random.default_rng(SEED); tn = mk(); P = from_torch(tn.state_dict()); npar = sum(p.numel() for p in tn.parameters())
        if DAMP: P = add_damp(P); npar += D
        SC = (lambda its: tg_score(P, its, UNIT)) if DAMP else (lambda its: (to_torch(P, tn), tn.eval(), RB.score(tn, its), tn.train())[2])''')
rep('''                to_torch(P, tn); tn.eval(); s_ = RB.score(tn, shallow).mean(); d_ = RB.score(tn, deep).mean(); tn.train()''',
    '''                s_ = SC(shallow).mean(); d_ = SC(deep).mean()''')
rep('''        to_torch(P, tn); tn.eval(); S = {k: RB.score(tn, v) for k, v in (("shallow", shallow), ("mid", mid), ("deep", deep))}''',
    '''        S = {k: SC(v) for k, v in (("shallow", shallow), ("mid", mid), ("deep", deep))}
        agree = None if DAMP else float((tg_score(P, deep, UNIT) == S["deep"]).mean())   # the tinygrad scorer's witness on the control path
        pc = pole_census(P, deep, UNIT)''')
rep('''engine="tinygrad", unit=UNIT, norm_census=nc, s_per_step=sps)''', '''engine="tinygrad", unit=UNIT, damp=DAMP, norm_census=nc, s_per_step=sps, pole=pc, tg_score_agree=agree)''')
rep('''+ " | norms(all) " + " ".join(f"{x:.2f}" for x in nc["all"]), flush=True)''',
    '''+ " | norms(all) " + " ".join(f"{x:.2f}" for x in nc["all"]), flush=True)
        print(f"[tg] {TAG} POLE R2 {pc['r2']:.3f} | |lambda| q25/50/75 " + "/".join(f"{x:.3f}" for x in pc["q"]) + f" | inside {pc['inside']:.3f} on {pc['on']:.3f} "
              f"outside {pc['outside']:.3f} | top " + " ".join(f"{x:.3f}" for x in pc["top"]) + (f" | learned {pc['learned']}" if DAMP else f" | tg-score agree {agree}"), flush=True)''')
open(p, "w").write(s)
