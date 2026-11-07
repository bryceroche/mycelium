"""THE CHART-SEALED POINTER (ledger 2026-11-06 17:07, pinned before the build): BT_CS=1 (with BT_ZQ=1: its charts t["ch"] and keys zck) in both engines.

  The token block of each pointer becomes a hierarchical softmax with an exact seal. F = q . kt / sqrt D over the open numerals:
  token logit = lse_T(F) + log p_r(chart | q) + F_t - lse_chart(t)(F), then + pc . cnt as before. lse_T keeps the token block's total mass (the
  token / slot / const competition is untouched); p_r = softmax(q . zck / sqrt D) over the charts present among the open numerals; F - lse_chart
  = log p(token | chart, q): q . kt cannot move mass between charts; only the zck road chooses the chart (the RESIDUAL-SEAL law).
  BT_CS unset leaves every array as before. Applied AFTER zoom_query (its anchors are the ZQ-patched files). Usage: chart_seal.py ROOT
"""
import os, sys

R = os.path.join(sys.argv[1], "scripts")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[cs] {f}: {len(pairs)} edit(s)")


edit("gsm_breath_reader.py", [
    ('CH = None                                             # (ZQ) the chart memmap, loaded on first use\n',
     'CH = None                                             # (ZQ) the chart memmap, loaded on first use\n'
     'CS = os.environ.get("BT_CS") == "1"                  # (ledger 11-06 CS) THE CHART-SEALED POINTER (rides on ZQ\'s charts + keys)\n'
     'assert not CS or (ZQ and not PHPTR), "CS: needs BT_ZQ=1, not with PHPTR"\n'
     '\n'
     '\n'
     'def chart_seal(F, q, zck, ch, m):\n'
     '    """(CS) F (B, K, Lq) natural token logits, m (B, K, Lq) open, ch (B, Lq) charts -> the sealed token logits (natural units)"""\n'
     '    import torch\n'
     '    F_ = torch.where(m, F, torch.full_like(F, -1e4))\n'
     '    oh = torch.nn.functional.one_hot(ch, NCH).float()                                          # (B, Lq, C)\n'
     '    inc = m[..., None] & (oh[:, None] > 0.5)                                                    # (B, K, Lq, C)\n'
     '    fm = torch.where(inc, F_[..., None], torch.full_like(inc, -1e4, dtype=F.dtype)).amax(2)    # (B, K, C) per-chart max\n'
     '    S = torch.where(inc, F_[..., None] - fm[:, :, None], torch.full_like(inc, -1e4, dtype=F.dtype)).exp().sum(2)\n'
     '    lsec = fm + S.clamp(min=1e-30).log()                                                        # lse within each chart\n'
     '    lchr = torch.where(S > 0.5, torch.einsum("bkd,cd->bkc", q, zck) / math.sqrt(D), torch.full_like(S, -1e4)).log_softmax(-1)\n'
     '    return F_.logsumexp(-1, keepdim=True) + torch.einsum("bkc,blc->bkl", lchr - lsec, oh) + F_\n'),
    ('                    if ZQ:   # (ledger 11-06 ZQ) the coarse level: the query searches the Tree\'s charts; + log p(the token\'s chart | q)\n',
     '                    if CS:   # (ledger 11-06 CS) the seal: chart by the zck road only, address within the chart by q . kt\n'
     '                        m_ = t["isnum"][:, None, :Lq] & km[:, :, role, 0:1]\n'
     '                        lt = (chart_seal(torch.einsum("bkd,bld->bkl", q, kt) / math.sqrt(D), q, s.zck, t["ch"][:, :Lq], m_)\n'
     '                              + s.pc * cnt[:, None, :Lq]) * math.sqrt(D)\n'
     '                    elif ZQ:   # (ledger 11-06 ZQ) the coarse level: the query searches the Tree\'s charts; + log p(the token\'s chart | q)\n'),
])

edit("gsm_breath_tg.py", [
    ('ZQ, NCH = BR.ZQ, BR.NCH                                                                      # (ledger 11-06) the zooming query\n',
     'ZQ, NCH = BR.ZQ, BR.NCH                                                                      # (ledger 11-06) the zooming query\n'
     'CS = BR.CS                                                                                   # (ledger 11-06) the chart-sealed pointer\n'
     '\n'
     '\n'
     'def chart_seal(F, q, zck, ch, m):\n'
     '    """(CS) as BR.chart_seal (natural units)"""\n'
     '    F_ = m.where(F, -1e4)\n'
     '    oh = ch.one_hot(NCH).cast(dtypes.float)                                                    # (B, Lq, C)\n'
     '    inc = m.unsqueeze(-1) & (oh.unsqueeze(1) > 0.5)                                            # (B, K, Lq, C)\n'
     '    fm = inc.where(F_.unsqueeze(-1), -1e4).max(2)                                              # (B, K, C)\n'
     '    S = inc.where(F_.unsqueeze(-1) - fm.unsqueeze(2), -1e4).exp().sum(2)\n'
     '    lsec = fm + S.maximum(1e-30).log()\n'
     '    lchr = (S > 0.5).where(q @ zck.T * (1 / math.sqrt(D)), -1e4).log_softmax(-1)\n'
     '    return F_.logsumexp(-1, keepdim=True) + (lchr - lsec) @ oh.transpose(-1, -2) + F_\n'),
    ('        if ZQ:   # (ledger 11-06 ZQ) the coarse level (torch ptrs, mirrored; the gather as a one-hot matmul)\n',
     '        if CS:   # (ledger 11-06 CS) the seal (torch ptrs, mirrored)\n'
     '            m_ = t["isnum"][:, None, :Lq] & km[:, :, role, 0:1]\n'
     '            lt = (chart_seal(q @ kt.transpose(-1, -2) * (1 / math.sqrt(D)), q, P["zck"], t["ch"][:, :Lq], m_)\n'
     '                  + P["pc"] * cnt[:, None, :Lq]) * math.sqrt(D)\n'
     '        elif ZQ:   # (ledger 11-06 ZQ) the coarse level (torch ptrs, mirrored; the gather as a one-hot matmul)\n'),
])
