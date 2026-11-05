"""THE SEALED OP HEAD (ledger 2026-11-05, Bryce: "word given for sealed op head"): BT_SEAL=1 in both engines.

  p(type) = p_text(struct) x p_DTO(op). struct = the operand kinds (9) + stop: the text silhouette's 38 atom logits marginalized over the op
  (log-sum-exp per struct), so the text's op preference cannot reach the decision. The op comes from THE DTO HEAD (scripts/gsm_dto.py:
  10 field tokens of the chosen operands -> 2 plain self-attention blocks, width 64, 4 heads, no MLP -> 4 op log-probs); it never sees a
  token state. Pointers are op-blind: p(struct) as the radius, the struct-mean atom, the canonical type's role phases. Training: the gold
  operands' DTO (a slot's value / unit known iff committed); inference (BR.score): struct -> pointers -> the chosen operands' DTO -> op.
  SEAL excludes ZOOM / FACT / SOFT / GATE (zoom IS the text's p(op) factor). New modules are created only under SEAL, so BT_SEAL=0
  leaves every init draw, tensor and rng draw as before. Beside metric under SEAL: op_gold = the head's op on the GOLD operands.
  The tinygrad scorer is not ported (train mode scores with BR.score); parity mode skips its score block under SEAL.
Asserts each anchor once. Usage: seal_op.py ROOT  (dry runs pass a copy)
"""
import os, sys

R = os.path.join(sys.argv[1], "scripts")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[seal] {f}: {len(pairs)} edit(s)")


edit("gsm_breath_reader.py", [
    ('import gsm_units as GU\n', 'import gsm_units as GU\nimport gsm_dto as DTO\n'),
    ('T6OFF = os.environ.get("BT_T6OFF") == "1"',
     'SEAL = os.environ.get("BT_SEAL") == "1"              # (ledger 11-05) THE SEALED OP HEAD: p(type) = p_text(struct) x p_DTO(op)\n'
     'assert not SEAL or not (ZOOM or FACT or SOFT or GATE), "SEAL replaces zoom\'s p(op); not with FACT / SOFT / GATE"\n'
     'DW, DNH = 64, 4                                       # (SEAL) the DTO head: width, heads\n'
     'T6OFF = os.environ.get("BT_T6OFF") == "1"'),
    ('SPLITS = TR.SPLITS\n',
     'NS = 3 * len(KINDS) + 1                               # (SEAL) the structs: operand kinds (9) + stop\n'
     'SOF = np.array([3 * KINDS.index(a) + KINDS.index(b) for _, a, b in TYPES] + [NS - 1])        # type -> struct\n'
     'OPI = np.array([GD.OPS.index(o) for o, _, _ in TYPES] + [0])                                  # type -> op (stop: masked by isstep)\n'
     'CANON = np.array([TYPES.index((GD.OPS[0], a, b)) for a in KINDS for b in KINDS] + [STOP])    # struct -> its canonical type\n'
     'assert all(j == 9 * OPI[j] + SOF[j] for j in range(STOP)), "TYPES layout: type = 9 op + struct"\n'
     'SPLITS = TR.SPLITS\n'),
    ('    kmask_op = torch.stack([kmask[grp[g]].any(0) for g in range(NOP)])                    # the op-level mask: the union of its types\' kinds\n',
     '    kmask_op = torch.stack([kmask[grp[g]].any(0) for g in range(NOP)])                    # the op-level mask: the union of its types\' kinds\n'
     '    sof, opi, canon = torch.as_tensor(SOF), torch.as_tensor(OPI), torch.as_tensor(CANON)  # (SEAL) type -> struct / op; struct -> canonical\n'
     '    sgrp = torch.zeros(NS, NT, dtype=torch.bool); sgrp[sof, torch.arange(NT)] = True\n'
     '    smean = sgrp.float() / sgrp.float().sum(1, keepdim=True)                              # (SEAL) the struct-mean atom weights\n'),
    ('    class Net(nn.Module):\n',
     '    class DAttn(nn.Module):   # (SEAL) the DTO head\'s plain self-attention block: no rotation, no MLP\n'
     '        def __init__(s):\n'
     '            super().__init__(); s.ln = nn.LayerNorm(DW); s.q = nn.Linear(DW, DW); s.kv = nn.Linear(DW, 2 * DW); s.o = nn.Linear(DW, DW)\n\n'
     '        def forward(s, x):\n'
     '            B_, Lx, _ = x.shape; y = s.ln(x); dh = DW // DNH\n'
     '            q = s.q(y).reshape(B_, Lx, DNH, dh).transpose(1, 2)\n'
     '            k, v = (t_.reshape(B_, Lx, DNH, dh).transpose(1, 2) for t_ in s.kv(y).chunk(2, -1))\n'
     '            sc = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(dh))\n'
     '            return x + s.o((sc.softmax(-1) @ v).transpose(1, 2).reshape(B_, Lx, DW))\n\n'
     '    class Net(nn.Module):\n'),
    ('            s.kc = nn.Parameter(torch.randn(NC, D) * 0.02); s.pc = nn.Parameter(torch.zeros(1))\n',
     '            s.kc = nn.Parameter(torch.randn(NC, D) * 0.02); s.pc = nn.Parameter(torch.zeros(1))\n'
     '            if SEAL:   # (ledger 11-05) THE DTO HEAD: field tokens -> 2 plain attention blocks -> 4 op log-probs; never a token state\n'
     '                s.dto_e = nn.Embedding(DTO.NV, DW); s.dto_f = nn.Parameter(torch.randn(DTO.NF, DW) * 0.02)\n'
     '                s.dto_att = nn.ModuleList([DAttn() for _ in range(2)]); s.dto_ln = nn.LayerNorm(DW); s.dto_out = nn.Linear(DW, len(GD.OPS))\n\n'
     '        def ophead(s, ids):\n'
     '            """(SEAL) (B, K, NF) DTO field ids -> (B, K, 4) log p(op)"""\n'
     '            B_, K_, _ = ids.shape; x = s.dto_e(ids.reshape(B_ * K_, -1)) + s.dto_f\n'
     '            for blk in s.dto_att: x = blk(x)\n'
     '            return s.dto_out(s.dto_ln(x.mean(1))).log_softmax(-1).reshape(B_, K_, -1)\n'),
    ('            p = logit.softmax(-1)\n            def ptrs(c, zoom, free=False):',
     '            if SEAL:   # (ledger 11-05) the text keeps the struct (op marginalized out); the DTO head supplies the op (training: t["dto"])\n'
     '                lt_ = logit.log_softmax(-1)\n'
     '                lstr = torch.stack([lt_.masked_fill(~sgrp[g], -1e9).logsumexp(-1) for g in range(NS)], -1)   # (B, K, NS) log p(struct)\n'
     '                logit = lstr[..., sof]\n'
     '                if "dto" in t: logit = logit + s.ophead(t["dto"])[..., opi] * isstep\n'
     '            p = logit.softmax(-1)\n            def ptrs(c, zoom, free=False):'),
    ('                rr = p.gather(-1, c[..., None])                                                    # the radius: type confidence\n'
     '                u = s.up(z) + s.upc(rr * CA[c])                                                    # expand: silhouette + snapped atom x r\n',
     '                if SEAL:   # (ledger 11-05) op-blind: p(struct) as the radius, the struct-mean atom, the canonical type\'s phases\n'
     '                    cs = sof[c]; rr = lstr.exp().gather(-1, cs[..., None]); u = s.up(z) + s.upc(rr * (smean @ CA)[cs]); c = canon[cs]\n'
     '                else:\n'
     '                    rr = p.gather(-1, c[..., None])                                                    # the radius: type confidence\n'
     '                    u = s.up(z) + s.upc(rr * CA[c])                                                    # expand: silhouette + snapped atom x r\n'),
    ('    nb = nb_tensors(notebook(E, L, progs, ~masked, gt, np.ones((len(E), K), np.float32)))\n',
     '    if SEAL: t["dto"] = torch.as_tensor(DTO.dto_ids(its, progs, ~masked))                  # (SEAL) the gold operands\' DTO; slots known iff committed\n'
     '    nb = nb_tensors(notebook(E, L, progs, ~masked, gt, np.ones((len(E), K), np.float32)))\n'),
    ('def score(net, its, states, tree, seed=0):\n',
     'def seal_decide(net, its, progs, com, logit, pa, pb, dec):\n'
     '    """(SEAL) the text\'s struct, then the op from the DTO of the operands the pointers chose -> type ids, confidence p(struct) p(op)"""\n'
     '    import torch\n'
     '    B_ = logit.shape[0]; st_ = SOF[logit.argmax(-1).numpy()]; lps = logit.max(-1).values.numpy()\n'
     '    ia, ib = pa.argmax(-1).numpy(), pb.argmax(-1).numpy()\n'
     '    cand = [[(dec(ia[b, k]), dec(ib[b, k])) for k in range(K)] for b in range(B_)]\n'
     '    lop = net.ophead(torch.as_tensor(DTO.dto_ids(its, progs, com, cand))).numpy(); step = st_ != NS - 1\n'
     '    return np.where(step, 9 * lop.argmax(-1) + st_, STOP), np.exp(lps + np.where(step, lop.max(-1), 0)).astype(np.float32)\n\n\n'
     'def score(net, its, states, tree, seed=0):\n'),
    ('    res = dict(ans=[], prog=[], step_op=[], step_ptr=[], step_all=[], first_all=[], type_ok=[], num_for_slot=[], commit_cycle=[])\n',
     '    res = dict(ans=[], prog=[], step_op=[], step_ptr=[], step_all=[], first_all=[], type_ok=[], num_for_slot=[], commit_cycle=[])\n'
     '    if SEAL: res["op_gold"] = []                                                             # (SEAL) the head\'s op on the GOLD operands\n'),
    ('                pt = p.gather(-1, torch.as_tensor(ty)[..., None])[..., 0].numpy() if ORACLE else p.max(-1).values.numpy()\n',
     '                pt = p.gather(-1, torch.as_tensor(ty)[..., None])[..., 0].numpy() if ORACLE else p.max(-1).values.numpy()\n'
     '                if SEAL: ty, pt = seal_decide(net, part, progs, com, logit, pa, pb, dec)\n'),
    ('            gprogs, gt = gold(E)\n            for b, (it, e) in enumerate(zip(part, E)):\n',
     '            gprogs, gt = gold(E)\n'
     '            if SEAL: og = net.ophead(torch.as_tensor(DTO.dto_ids(part, gprogs, np.ones((B, K), bool)))).argmax(-1).numpy()\n'
     '            for b, (it, e) in enumerate(zip(part, E)):\n'),
    ('                    res["type_ok"].append(typ[b, k] == gt[b, k]); res["commit_cycle"].append(int(cyc[b, k]))\n',
     '                    res["type_ok"].append(typ[b, k] == gt[b, k]); res["commit_cycle"].append(int(cyc[b, k]))\n'
     '                    if SEAL: res["op_gold"].append(GD.OPS[og[b, k]] == gprogs[b][k][0])\n'),
])

edit("gsm_breath_tg.py", [
    ('W, WH, WT, WL = Tensor(_W).realize(), Tensor(_WH).realize(), Tensor(_WT).realize(), Tensor(_WL).realize()\n',
     'SEAL = BR.SEAL                                                                               # (ledger 11-05) the sealed op head\n'
     '_SOFM = np.zeros((BR.NS, NT), np.float32); _SOFM[BR.SOF, np.arange(NT)] = 1                # type -> struct (one-hot gathers)\n'
     '_OPM = np.zeros((len(BR.GD.OPS), NT), np.float32); _OPM[BR.OPI, np.arange(NT)] = 1; _OPM[:, STOP] = 0\n'
     '_CANM = np.zeros((BR.NS, NT), np.float32); _CANM[np.arange(BR.NS), BR.CANON] = 1             # struct -> its canonical type\n'
     'SOFM, SGRP, OPM, CANM = Tensor(_SOFM).realize(), Tensor(_SOFM > 0).realize(), Tensor(_OPM).realize(), Tensor(_CANM).realize()\n'
     'SMEAN = Tensor(_SOFM / _SOFM.sum(1, keepdims=True)).realize()\n'
     'W, WH, WT, WL = Tensor(_W).realize(), Tensor(_WH).realize(), Tensor(_WT).realize(), Tensor(_WL).realize()\n'),
    ('def read(P, t):\n',
     'def ophead(P, ids):\n'
     '    """(SEAL) the DTO head (torch Net.ophead): (B, K, NF) field ids -> (B, K, 4) log p(op); one-hot gathers, plain attention"""\n'
     '    B_, K_ = ids.shape[0], ids.shape[1]; DW, DNH = BR.DW, BR.DNH; dh = DW // DNH\n'
     '    x = ids.reshape(B_ * K_, -1).one_hot(BR.DTO.NV).cast(dtypes.float) @ P["dto_e.weight"] + P["dto_f"]; Lx = x.shape[1]\n'
     '    for i in range(2):\n'
     '        n = f"dto_att.{i}"; y = lnorm(P, x, n + ".ln"); kv = lin(P, y, n + ".kv")\n'
     '        q = lin(P, y, n + ".q").reshape(-1, Lx, DNH, dh).transpose(1, 2)\n'
     '        k = kv[..., :DW].reshape(-1, Lx, DNH, dh).transpose(1, 2); v = kv[..., DW:].reshape(-1, Lx, DNH, dh).transpose(1, 2)\n'
     '        sc = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(dh))\n'
     '        x = x + lin(P, (sc.softmax(-1) @ v).transpose(1, 2).reshape(-1, Lx, DW), n + ".o")\n'
     '    return lin(P, lnorm(P, x.mean(1), "dto_ln"), "dto_out").log_softmax(-1).reshape(B_, K_, -1)\n\n\n'
     'def read(P, t):\n'),
    ('    return dict(logit=logit, p=logit.softmax(-1), z=z, lfine=lfine, xk_q=xk_q, cons=cons, t=t, Lq=Lq, CA=CA, lev=lev)\n',
     '    ls = None\n'
     '    if SEAL:   # (ledger 11-05) the text keeps the struct (op marginalized out); the DTO head supplies the op (training: t["dto"])\n'
     '        lt_ = logit.log_softmax(-1)\n'
     '        ls = Tensor.stack(*[SGRP[g].where(lt_, -1e9).logsumexp(-1) for g in range(BR.NS)], dim=-1)\n'
     '        logit = ls @ SOFM\n'
     '        if "dto" in t: logit = logit + ophead(P, t["dto"]) @ OPM\n'
     '    return dict(logit=logit, p=logit.softmax(-1), z=z, lfine=lfine, xk_q=xk_q, cons=cons, t=t, Lq=Lq, CA=CA, lev=lev, ls=ls)\n'),
    ('    rr = (p * oc).sum(-1, keepdim=True)\n'
     '    u = lin(P, z, "up") + lin(P, rr * (oc @ o["CA"]), "upc"); hu = lnorm(P, u, "ln")\n',
     '    if SEAL:   # (ledger 11-05) op-blind: p(struct) as the radius, the struct-mean atom, the canonical type\'s phases\n'
     '        cs = oc @ SOFM.T; rr = (o["ls"].exp() * cs).sum(-1, keepdim=True)\n'
     '        u = lin(P, z, "up") + lin(P, rr * (cs @ (SMEAN @ o["CA"])), "upc"); oc = cs @ CANM\n'
     '    else:\n'
     '        rr = (p * oc).sum(-1, keepdim=True)\n'
     '        u = lin(P, z, "up") + lin(P, rr * (oc @ o["CA"]), "upc")\n'
     '    hu = lnorm(P, u, "ln")\n'),
    ('    def _step(s, x, theta, nid, isn, isnum, g, off, sid, pad, role, typ, r, cf, lev, cons, gt, mk, am, va, vb, cyc, sp):\n'
     '        P = s.P; t = dict(x=x, theta=theta, nid=nid, isn=isn, isnum=isnum, g=g, off=off, sid=sid, pad=pad, role=role)\n',
     '    def _step(s, x, theta, nid, isn, isnum, g, off, sid, pad, role, typ, r, cf, lev, cons, gt, mk, am, va, vb, cyc, sp, dto=None):\n'
     '        P = s.P; t = dict(x=x, theta=theta, nid=nid, isn=isn, isnum=isnum, g=g, off=off, sid=sid, pad=pad, role=role)\n'
     '        if SEAL: t["dto"] = dto\n'),
    ('        nb = BR.notebook(E, Lb + K, progs, ~masked, gt, np.ones((len(E), K), np.float32))\n',
     '        nb = BR.notebook(E, Lb + K, progs, ~masked, gt, np.ones((len(E), K), np.float32))\n'
     '        dx = [Tensor(BR.DTO.dto_ids(its, progs, ~masked).astype(np.int32))] if SEAL else []   # (SEAL) the gold operands\' DTO\n'),
    ('                                 Tensor((t["am"] & masked).astype(np.float32)), Tensor(t["va"]), Tensor(t["vb"]), cyct, sp).numpy())\n',
     '                                 Tensor((t["am"] & masked).astype(np.float32)), Tensor(t["va"]), Tensor(t["vb"]), cyct, sp, *dx).numpy())\n'),
    ('    """BR.score\'s schedule loop on the tinygrad net (eager) -> the same metric dict"""\n',
     '    """BR.score\'s schedule loop on the tinygrad net (eager) -> the same metric dict"""\n'
     '    assert not SEAL, "tg score: not ported under SEAL (train mode scores with BR.score)"\n'),
    ('        N = int(os.environ.get("TG_PARN", "128"))\n        for split in ("ORIG", "TEST"):\n',
     '        N = int(os.environ.get("TG_PARN", "128"))\n        for split in (() if SEAL else ("ORIG", "TEST")):   # (SEAL: the tg scorer is not ported)\n'),
])
