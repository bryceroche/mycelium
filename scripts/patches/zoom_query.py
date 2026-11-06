"""THE ZOOMING QUERY (ledger 2026-11-06 11:59, Bryce: "word given for the whole queue"): BT_ZQ=1 in both engines.

  The pointer zooms: chart first, then address. The Tree's charts (.cache/gsmb_chart64_rb8.npy, scripts/gsm_chart_build.py: the ST64 node of
  every numeral, 64 = none) ride the batch as t["ch"]; 65 learned chart keys zck (created LAST, under ZQ only); per slot and role the bound
  query searches them: log p(chart | q) = log_softmax_c(q . zck_c / sqrt D), and every token's pointer logit += log p(its chart | q) + log 65
  (centred: a uniform search adds 0; the coarse level; q . kt stays the fine level). A mandatory road: no gain; no new loss. BT_ZQ=0 leaves every init draw, tensor and array as
  before. The tinygrad scorer is not ported (train mode scores with BR.score); parity mode skips its score block under ZQ.
Applied AFTER phase_ptr (its anchors are the PP-patched files). Asserts each anchor once. Usage: zoom_query.py ROOT  (dry runs pass a copy)
"""
import os, sys

R = os.path.join(sys.argv[1], "scripts")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[zq] {f}: {len(pairs)} edit(s)")


edit("gsm_breath_reader.py", [
    ('assert not PHPTR or not (SEAL or SOFT), "PHPTR: not with SEAL / SOFT"\n',
     'assert not PHPTR or not (SEAL or SOFT), "PHPTR: not with SEAL / SOFT"\n'
     'ZQ = os.environ.get("BT_ZQ") == "1"                  # (ledger 11-06 ZQ) THE ZOOMING QUERY: log p(the token\'s Tree chart | q) on the pointer\n'
     'NCH = 65                                              # (ZQ) 64 ST64 charts + none\n'
     'CH = None                                             # (ZQ) the chart memmap, loaded on first use\n'),
    ('                s.pg = nn.Parameter(torch.tensor([[1.6, 0.0], [0.0, 2.8], [0.0, -3.0]]))\n',
     '                s.pg = nn.Parameter(torch.tensor([[1.6, 0.0], [0.0, 2.8], [0.0, -3.0]]))\n'
     '            if ZQ:   # (ledger 11-06 ZQ) the chart keys (LAST)\n'
     '                s.zck = nn.Parameter(torch.randn(NCH, D) * 0.02)\n'),
    ('                        lt = lt + sum(s.pg[m, role] * rel(cx, m, role) for m, cx in enumerate(ctxs)) * math.sqrt(D)\n',
     '                        lt = lt + sum(s.pg[m, role] * rel(cx, m, role) for m, cx in enumerate(ctxs)) * math.sqrt(D)\n'
     '                    if ZQ:   # (ledger 11-06 ZQ) the coarse level: the query searches the Tree\'s charts; + log p(the token\'s chart | q)\n'
     '                        lch = (torch.einsum("bkd,cd->bkc", q, s.zck) / math.sqrt(D)).log_softmax(-1) + math.log(NCH)   # centred: uniform = 0\n'
     '                        lt = lt + lch.gather(-1, t["ch"][:, None, :Lq].expand(-1, K, -1)) * math.sqrt(D)\n'),
    ('    if PHPTR: t["qn"] = torch.as_tensor(np.array([qnoun(it) for it in its], np.int64))   # (PHPTR) the question noun\n',
     '    if PHPTR: t["qn"] = torch.as_tensor(np.array([qnoun(it) for it in its], np.int64))   # (PHPTR) the question noun\n'
     '    if ZQ:   # (ZQ) the Tree\'s ST64 chart per position (NCH - 1 = none)\n'
     '        global CH\n'
     '        if CH is None: CH = np.load(GD.V1.C("gsmb_chart64_rb8.npy"), mmap_mode="r")\n'
     '        ch = np.full(t["pad"].shape, NCH - 1, np.int64)\n'
     '        for b, it in enumerate(its): n = len(it["toks"]); ch[b, :n] = CH[it["st0"]:it["st0"] + n]\n'
     '        t["ch"] = torch.as_tensor(ch)\n'),
])

edit("gsm_breath_tg.py", [
    ('PHPTR, PR = BR.PHPTR, BR.PR                                                                  # (ledger 11-06) phase-bound pointers\n',
     'PHPTR, PR = BR.PHPTR, BR.PR                                                                  # (ledger 11-06) phase-bound pointers\n'
     'ZQ, NCH = BR.ZQ, BR.NCH                                                                      # (ledger 11-06) the zooming query\n'),
    ('            lt = lt + sum(P["pg"][m, role] * rel(cx, m, role) for m, cx in enumerate(ctxs)) * math.sqrt(D)\n',
     '            lt = lt + sum(P["pg"][m, role] * rel(cx, m, role) for m, cx in enumerate(ctxs)) * math.sqrt(D)\n'
     '        if ZQ:   # (ledger 11-06 ZQ) the coarse level (torch ptrs, mirrored; the gather as a one-hot matmul)\n'
     '            lch = (q @ P["zck"].T * (1 / math.sqrt(D))).log_softmax(-1) + math.log(NCH)   # centred: uniform = 0\n'
     '            lt = lt + (lch @ t["ch"][:, :Lq].one_hot(NCH).cast(dtypes.float).transpose(-1, -2)) * math.sqrt(D)\n'),
    ('    for k in ("x", "isn", "isnum", "sid", "off", "nid", "g", "pad", "role"):\n',
     '    for k in ("x", "isn", "isnum", "sid", "off", "nid", "g", "pad", "role", "ch"):          # (ZQ: ch pads are masked keys)\n'),
    ('    for k in FIELDS + ("qn",):                                                            # (PHPTR) + the question noun\n',
     '    for k in FIELDS + ("qn", "ch"):                                                       # (PHPTR) + the question noun, (ZQ) the charts\n'),
    ('        if PHPTR: t["qn"] = ex.pop(0); prevm = ex.pop(0)                                       # (PHPTR) the question noun, the prev mask\n',
     '        if PHPTR: t["qn"] = ex.pop(0); prevm = ex.pop(0)                                       # (PHPTR) the question noun, the prev mask\n'
     '        if ZQ: t["ch"] = ex.pop(0)                                                               # (ZQ) the charts\n'),
    ('        px = [tt["qn"], nbt[5]] if PHPTR else []                                              # (PHPTR) after the DTO, in _step\'s order\n',
     '        px = [tt["qn"], nbt[5]] if PHPTR else []                                              # (PHPTR) after the DTO, in _step\'s order\n'
     '        zx = [tt["ch"]] if ZQ else []                                                          # (ZQ) last\n'),
    ('cyct, sp, *dx, *px).numpy())', 'cyct, sp, *dx, *px, *zx).numpy())'),
    ('    assert not PHPTR, "tg score: not ported under PHPTR (train mode scores with BR.score)"\n',
     '    assert not PHPTR, "tg score: not ported under PHPTR (train mode scores with BR.score)"\n'
     '    assert not ZQ, "tg score: not ported under ZQ (train mode scores with BR.score)"\n'),
    ('for split in (() if SEAL or PHPTR else ("ORIG", "TEST")):', 'for split in (() if SEAL or PHPTR or ZQ else ("ORIG", "TEST")):'),
])
