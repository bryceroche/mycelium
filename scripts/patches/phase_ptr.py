"""PHASE-BOUND POINTERS (ledger 2026-11-06 06:38, Bryce: "word given for phase bound pointers"): BT_PHPTR=1 in both engines.

  A native phase embedding on the token side: theta_t = Linear(D -> PR=32, no bias)(the BT's token state xk_q); the key e^{i theta_t} on PR
  planes of its own. Three context phasors per slot: qn (the question noun's e^{i theta}; the CHART QUERY's qnoun heuristic), prev (the mean
  e^{i theta} of the token operands of the nearest committed earlier step; the notebook writes them back as a 6th array), sib (role b only:
  sum_j softmax(pa)_j e^{i theta_j}, role a's soft choice). rel_m(j) = mean_p Re(ctx_m,p e^{i psi_m,r,p} conj(e^{i theta_j,p})); the token
  pointer logits += sum_m g_m,r rel_m(j) (gains init at the oracle's signs: qn_a +1.6, prev_b +2.8, sib_b -3.0, the rest 0; psi init 0).
  New modules are created only under PHPTR and LAST in Net.__init__, so BT_PHPTR=0 leaves every init draw, tensor and rng draw as before
  (the 5-array notebook, nb_tensors and nb_t return the same tensors). Not with SEAL / SOFT. The tinygrad scorer is not ported (train mode
  scores with BR.score); parity mode skips its score block under PHPTR.
Asserts each anchor once. Usage: phase_ptr.py ROOT  (dry runs pass a copy)
"""
import os, sys

R = os.path.join(sys.argv[1], "scripts")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[phptr] {f}: {len(pairs)} edit(s)")


edit("gsm_breath_reader.py", [
    ('DW, DNH = 64, 4                                       # (SEAL) the DTO head: width, heads\n',
     'DW, DNH = 64, 4                                       # (SEAL) the DTO head: width, heads\n'
     'PHPTR = os.environ.get("BT_PHPTR") == "1"            # (ledger 11-06 PP) PHASE-BOUND POINTERS: token phases, relations by the conjugate\n'
     'PR = 32                                               # (PHPTR) the token phase planes\n'
     'assert not PHPTR or not (SEAL or SOFT), "PHPTR: not with SEAL / SOFT"\n'),
    ('def step_type(op, a, b):\n',
     'def qnoun(it):\n'
     '    """(PHPTR) the question noun: the content word after "how many / much" in the last sentence, else the token before the last "?"\n'
     '    (the CHART QUERY oracle\'s heuristic, verbatim in effect)"""\n'
     '    toks, sid = it["toks"], it["sid"]; q0 = sid.index(max(sid)); q = toks[q0:]\n'
     '    for i in range(len(q) - 1):\n'
     '        if q[i] == "how" and q[i + 1] in ("many", "much"):\n'
     '            j, _ = GU.content(q, i + 2)\n'
     '            if j is not None: return q0 + j\n'
     '    last = max(i for i, t in enumerate(toks) if t == "?") if "?" in toks else len(toks) - 1\n'
     '    return max(0, last - 1)\n\n\n'
     'def step_type(op, a, b):\n'),
    ('    return typ, r, cf, lev, cons\n',
     '    if PHPTR:   # (ledger 11-06 PP) per slot: the token operands of the nearest committed earlier step (the prev context)\n'
     '        pm = np.zeros((B, K, L), bool)\n'
     '        for b, e in enumerate(E):\n'
     '            for k in range(K):\n'
     '                js = [j for j in range(k) if committed[b, j] and progs[b][j][0] != "."]\n'
     '                if not js: continue\n'
     '                for kind, i in progs[b][js[-1]][1:]:\n'
     '                    if kind == "t" and i < e["n"]: pm[b, k, i] = True\n'
     '        return typ, r, cf, lev, cons, pm\n'
     '    return typ, r, cf, lev, cons\n'),
    ('    typ, r, cf, lev, cons = arrs\n    return (torch.as_tensor(typ), torch.as_tensor(r), torch.as_tensor(cf), torch.as_tensor(lev), torch.as_tensor(cons))\n',
     '    return tuple(torch.as_tensor(a) for a in arrs)                                     # (PHPTR: a 6th array, the prev mask)\n'),
    ('            typ, r, cf, lev, cons = nb\n',
     '            typ, r, cf, lev, cons = nb[:5]; prevm = nb[5] if PHPTR else None\n'),
    ('                s.dto_att = nn.ModuleList([DAttn() for _ in range(2)]); s.dto_ln = nn.LayerNorm(DW); s.dto_out = nn.Linear(DW, len(GD.OPS))\n',
     '                s.dto_att = nn.ModuleList([DAttn() for _ in range(2)]); s.dto_ln = nn.LayerNorm(DW); s.dto_out = nn.Linear(DW, len(GD.OPS))\n'
     '            if PHPTR:   # (ledger 11-06 PP) the token phase map, the relation phases psi (m = qn / prev / sib, role), the gains\n'
     '                s.pth = nn.Linear(D, PR, bias=False); s.ppsi = nn.Parameter(torch.zeros(3, 2, PR))\n'
     '                s.pg = nn.Parameter(torch.tensor([[1.6, 0.0], [0.0, 2.8], [0.0, -3.0]]))\n'),
    ('                outs = []\n',
     '                if PHPTR:   # (ledger 11-06 PP) the native phase embedding: e^{i theta_t} on PR planes of its own\n'
     '                    thx = s.pth(xk_q); ez = torch.stack([thx.cos(), thx.sin()], -1)                  # (B, Lq, PR, 2)\n'
     '                    cq = ez[torch.arange(B), t["qn"]][:, None].expand(-1, K, -1, -1)              # the question noun\n'
     '                    pm_ = prevm[:, :, :Lq].float()\n'
     '                    cp = torch.einsum("bkl,blpc->bkpc", pm_, ez) / pm_.sum(-1).clamp(min=1)[..., None, None]   # the prev step\'s tokens\n'
     '\n'
     '                    def rel(ctx, m, role):   # bind the context by e^{i psi}, unbind against each token by the conjugate: mean_p Re(.)\n'
     '                        a_ = s.ppsi[m, role]; c_, s_ = a_.cos(), a_.sin(); re_, im_ = ctx[..., 0], ctx[..., 1]\n'
     '                        cr = torch.stack([re_ * c_ - im_ * s_, re_ * s_ + im_ * c_], -1)\n'
     '                        return torch.einsum("bkpc,blpc->bkl", cr, ez) / PR\n'
     '                outs = []\n'),
    ('                    lt = torch.einsum("bkd,bld->bkl", q, kt) + s.pc * cnt[:, None, :Lq] * math.sqrt(D)\n',
     '                    lt = torch.einsum("bkd,bld->bkl", q, kt) + s.pc * cnt[:, None, :Lq] * math.sqrt(D)\n'
     '                    if PHPTR:   # the relations (sib: role a\'s soft choice, the continuous mirror)\n'
     '                        ctxs = [cq, cp] + ([torch.einsum("bkl,blpc->bkpc", outs[0].softmax(-1)[..., :Lq], ez)] if role == 1 else [])\n'
     '                        lt = lt + sum(s.pg[m, role] * rel(cx, m, role) for m, cx in enumerate(ctxs)) * math.sqrt(D)\n'),
    ('    return t, E\n',
     '    if PHPTR: t["qn"] = torch.as_tensor(np.array([qnoun(it) for it in its], np.int64))   # (PHPTR) the question noun\n'
     '    return t, E\n'),
])

edit("gsm_breath_tg.py", [
    ('SEAL = BR.SEAL                                                                               # (ledger 11-05) the sealed op head\n',
     'SEAL = BR.SEAL                                                                               # (ledger 11-05) the sealed op head\n'
     'PHPTR, PR = BR.PHPTR, BR.PR                                                                  # (ledger 11-06) phase-bound pointers\n'),
    ('    typ, r, cf, lev, cons = nb\n',
     '    typ, r, cf, lev, cons = nb[:5]; prevm = nb[5] if PHPTR else None\n'),
    ('lev=lev, ls=ls)\n', 'lev=lev, ls=ls, prevm=prevm)\n'),
    ('    outs = []\n',
     '    if PHPTR:   # (ledger 11-06 PP) the native phase embedding (torch ptrs, mirrored; contexts by matmul over the flattened planes)\n'
     '        thx = lin(P, xk_q, "pth", bias=False); ezf = Tensor.stack(thx.cos(), thx.sin(), dim=-1).reshape(B, Lq, PR * 2)\n'
     '        cq = (t["qn"].one_hot(Lq).cast(dtypes.float).reshape(B, 1, Lq) @ ezf).expand(B, K, PR * 2)\n'
     '        pm_ = o["prevm"][:, :, :Lq].cast(dtypes.float); cp = (pm_ @ ezf) / pm_.sum(-1).maximum(1).unsqueeze(-1)\n'
     '\n'
     '        def rel(ctx, m, role):\n'
     '            a_ = P["ppsi"][m, role]; c_, s_ = a_.cos(), a_.sin(); cc = ctx.reshape(B, K, PR, 2); re_, im_ = cc[..., 0], cc[..., 1]\n'
     '            cr = Tensor.stack(re_ * c_ - im_ * s_, re_ * s_ + im_ * c_, dim=-1).reshape(B, K, PR * 2)\n'
     '            return (cr @ ezf.transpose(-1, -2)) * (1 / PR)\n'
     '    outs = []\n'),
    ('        lt = q @ kt.transpose(-1, -2) + P["pc"] * cnt[:, None, :Lq] * math.sqrt(D)\n',
     '        lt = q @ kt.transpose(-1, -2) + P["pc"] * cnt[:, None, :Lq] * math.sqrt(D)\n'
     '        if PHPTR:\n'
     '            ctxs = [cq, cp] + ([outs[0].softmax(-1)[..., :Lq] @ ezf] if role == 1 else [])\n'
     '            lt = lt + sum(P["pg"][m, role] * rel(cx, m, role) for m, cx in enumerate(ctxs)) * math.sqrt(D)\n'),
    ('    for k in FIELDS:\n', '    for k in FIELDS + ("qn",):                                                            # (PHPTR) + the question noun\n'),
    ('def nb_t(arrs):\n    typ, r, cf, lev, cons = arrs\n    return Tensor(typ.astype(np.int32)), Tensor(r.astype(np.float32)), Tensor(cf.astype(np.float32)), Tensor(lev.astype(np.int32)), Tensor(cons)\n',
     'def nb_t(arrs):   # (PHPTR: a 6th array, the prev mask; the first five as before)\n'
     '    return tuple(Tensor(a.astype(np.int32)) if a.dtype in (np.int64, np.int32) else Tensor(a.astype(np.float32)) if a.dtype != bool\n'
     '                 else Tensor(a) for a in arrs)\n'),
    ('cyc, sp, dto=None):\n', 'cyc, sp, *ex):\n'),
    ('        if SEAL: t["dto"] = dto\n',
     '        ex = list(ex); prevm = None\n'
     '        if SEAL: t["dto"] = ex.pop(0)\n'
     '        if PHPTR: t["qn"] = ex.pop(0); prevm = ex.pop(0)                                       # (PHPTR) the question noun, the prev mask\n'),
    ('breath(P, t, xq, (typ, r, cf, lev, cons), cyc,', 'breath(P, t, xq, (typ, r, cf, lev, cons) + ((prevm,) if PHPTR else ()), cyc,'),
    ('        key = (Lb, soft)\n',
     '        px = [tt["qn"], nbt[5]] if PHPTR else []                                              # (PHPTR) after the DTO, in _step\'s order\n'
     '        key = (Lb, soft)\n'),
    ('*nbt, gtt,', '*nbt[:5], gtt,'),
    ('cyct, sp, *dx).numpy())', 'cyct, sp, *dx, *px).numpy())'),
    ('    assert not SEAL, "tg score: not ported under SEAL (train mode scores with BR.score)"\n',
     '    assert not SEAL, "tg score: not ported under SEAL (train mode scores with BR.score)"\n'
     '    assert not PHPTR, "tg score: not ported under PHPTR (train mode scores with BR.score)"\n'),
    ('for split in (() if SEAL else ("ORIG", "TEST")):', 'for split in (() if SEAL or PHPTR else ("ORIG", "TEST")):'),
])
