"""BT8 THE FACTORED SILHOUETTE (ledger 2026-11-03, word given: "the SQL cross join is our nemesis ... complex phase binding and
unbinding with the conjugate and Tree database"): gsm_breath_reader.py gains BT_FACT=1.

Today the silhouette is classified against 37 flat atoms = op x kind(a) x kind(b) + stop: the CROSS JOIN, each combination its own
centroid that must be seen to be learned. Under BT_FACT=1 the step type is BOUND from three factors on the silhouette's complex planes:
  z ~ e^{i psi_op} C_op[op] + e^{i psi_a} C_a[kind a] + e^{i psi_b} C_b[kind b]        (psi_r fixed random role phases: addresses)
  UNBIND   u_r = z * conj(e^{i psi_r})   (the conjugate mirror, never 1/z)
  SNAP     each u_r classified against its OWN small Welford codebook (5 ops incl. stop, 3 kinds, 3 kinds = 11 atoms, not 37)
  TYPE     log p(t) = log p(op) + log p(a) + log p(b)   (stop = log p(op = stop)); normalized by construction
  REBIND   the snapped atom re-entering the breath = sum_r e^{i psi_r} C_r[factor_r(t)]  (the notebook's committed atoms, the soft
           notebook, the pointer expansion all read the rebound composite)
The zoom's coarse level IS the op factor (log p(type | op) = log p(a) + log p(b)). The Welford factor atoms follow the unbound gold
silhouettes (no gradient, count capped), as the 37 atoms did. Applied by unit lfbt8 after lf8m launched its BT4M8 runs and lfbeam2read
finished (both chains re-read this file). Asserts each anchor once.
"""
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[bt8] {f}: {len(pairs)} edit(s)")


edit("gsm_breath_reader.py", [
    ('T6OFF = os.environ.get("BT_T6OFF") == "1"',
     'FACT = os.environ.get("BT_FACT") == "1"              # (ledger 11-03 BT8) the factored silhouette: op / kind a / kind b phase-bound, conjugate unbind\n'
     'T6OFF = os.environ.get("BT_T6OFF") == "1"'),
    ('PARENT = np.array([GD.OPS.index(o) for o, _, _ in TYPES] + [NOP - 1])\n',
     'PARENT = np.array([GD.OPS.index(o) for o, _, _ in TYPES] + [NOP - 1])\n'
     'FKA = np.array([KINDS.index(a) for _, a, _ in TYPES] + [0]); FKB = np.array([KINDS.index(b) for _, _, b in TYPES] + [0])   # (BT8) stop: no kinds\n'),
    ('            s.phi = nn.Parameter(torch.rand(NT, 2, NPA - NTK) * 2 * math.pi)                   # type-role phases on the bus planes\n',
     '            s.phi = nn.Parameter(torch.rand(NT, 2, NPA - NTK) * 2 * math.pi)                   # type-role phases on the bus planes\n'
     '            if FACT:   # (BT8) three factor codebooks (Welford) + fixed random role phases on the silhouette\'s DB/2 planes\n'
     '                g_ = torch.Generator().manual_seed(8)\n'
     '                s.register_buffer("psi", torch.rand(3, DB // 2, generator=g_) * 2 * math.pi)\n'
     '                s.register_buffer("Fop", torch.randn(NOP, DB) * 0.1); s.register_buffer("Fopn", torch.zeros(NOP))\n'
     '                s.register_buffer("Fa", torch.randn(3, DB) * 0.1); s.register_buffer("Fan", torch.zeros(3))\n'
     '                s.register_buffer("Fb", torch.randn(3, DB) * 0.1); s.register_buffer("Fbn", torch.zeros(3))\n'
     '                s.tauf = nn.Parameter(torch.zeros(3))\n'),
    ('        def breath(s, t, xq, nb, cond, cyc=None, sp=None):\n',
     '        def unbind(s, z):\n'
     '            """(BT8) the conjugate mirror: u_r = z * conj(e^{i psi_r}) for r = op, a, b -> (3, ..., DB)"""\n'
     '            return torch.stack([rot(z, -s.psi[r]) for r in range(3)])\n\n'
     '        def atoms(s):\n'
     '            """(BT8) the rebound composite per type: sum_r e^{i psi_r} C_r[factor_r(t)] (stop: the op factor alone) -> (NT, DB)"""\n'
     '            a = rot(s.Fop[par], s.psi[0]) + (rot(s.Fa[fka], s.psi[1]) + rot(s.Fb[fkb], s.psi[2])) * isstep[:, None]\n'
     '            return a\n\n'
     '        def breath(s, t, xq, nb, cond, cyc=None, sp=None):\n'),
    ('            typ, r, cf, lev, cons = nb\n            com = typ >= 0\n            re = s.reent(r[..., None] * s.C[typ.clamp(min=0)])',
     '            typ, r, cf, lev, cons = nb\n            com = typ >= 0\n            CA = s.atoms() if FACT else s.C                                                   # (BT8) the rebound atoms\n'
     '            re = s.reent(r[..., None] * CA[typ.clamp(min=0)])'),
    ('            unc = s.maskv if sp is None else s.maskv + s.sre(sp @ s.C)',
     '            unc = s.maskv if sp is None else s.maskv + s.sre(sp @ CA)'),
    ('            logit = -((z[:, :, None] - s.C[None, None]) ** 2).sum(-1) / math.sqrt(DB) * s.tau.exp()\n            if ZOOM:',
     '            logit = -((z[:, :, None] - s.C[None, None]) ** 2).sum(-1) / math.sqrt(DB) * s.tau.exp()\n'
     '            if FACT:   # (BT8) unbind each factor, snap it against its own codebook, compose: log p(op) + log p(a) + log p(b)\n'
     '                u = s.unbind(z)\n'
     '                lf = [(-((u[r][:, :, None] - F_[None, None]) ** 2).sum(-1) / math.sqrt(DB) * s.tauf[r].exp()).log_softmax(-1)\n'
     '                      for r, F_ in enumerate((s.Fop, s.Fa, s.Fb))]\n'
     '                lfine = (lf[1][..., fka] + lf[2][..., fkb]) * isstep\n'
     '                logit = lf[0][..., par] + lfine\n'
     '            elif ZOOM:'),
    ('                u = s.up(z) + s.upc(rr * s.C[c])                                                   # expand: silhouette + snapped atom x r',
     '                u = s.up(z) + s.upc(rr * CA[c])                                                    # expand: silhouette + snapped atom x r'),
    ('    par = torch.as_tensor(PARENT); grp = torch.zeros(NOP, NT, dtype=torch.bool); grp[par, torch.arange(NT)] = True\n',
     '    par = torch.as_tensor(PARENT); grp = torch.zeros(NOP, NT, dtype=torch.bool); grp[par, torch.arange(NT)] = True\n'
     '    fka, fkb = torch.as_tensor(FKA), torch.as_tensor(FKB); isstep = (torch.arange(NT) != STOP).float()   # (BT8) the factor indices per type\n'),
    ('        if ZOOM:\n            to = torch.as_tensor(PARENT)[tt]',
     '        if FACT:   # (BT8) the factor atoms follow the UNBOUND gold silhouettes (stop slots feed the op codebook only)\n'
     '            u = net.unbind(zz); st_ = tt != STOP\n'
     '            for F_, Fn, ur, idx_, sel in ((net.Fop, net.Fopn, u[0], torch.as_tensor(PARENT)[tt], torch.ones_like(st_)),\n'
     '                                          (net.Fa, net.Fan, u[1], torch.as_tensor(FKA)[tt], st_), (net.Fb, net.Fbn, u[2], torch.as_tensor(FKB)[tt], st_)):\n'
     '                ii, uu = idx_[sel], ur[sel]; cnt = torch.bincount(ii, minlength=len(Fn)).float()\n'
     '                sm = torch.zeros(len(Fn), DB).index_add_(0, ii, uu); nn_ = Fn + cnt; upd = cnt > 0\n'
     '                F_[upd] += (sm[upd] - cnt[upd, None] * F_[upd]) / nn_[upd, None]; Fn.copy_(nn_.clamp(max=NCAP))\n'
     '        if ZOOM:\n            to = torch.as_tensor(PARENT)[tt]'),
])
