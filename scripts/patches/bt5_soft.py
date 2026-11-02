"""bt5_soft.py -- THE SOFT NOTEBOOK (ledger 2026-11-01; the continuous mirror of the snap): adds BT_SOFT=1 to gsm_breath_reader.py.
Every UNCOMMITTED slot shows the next breath what the BT already believes -- the expected atom sum_t p(t) C_t (detached; never gold)
-- instead of the bare mask token, sharpened coarse-to-fine over the cycles: p = softmax(logit / tau_c), tau_c = 2 ** (1 - 2c/(NCYC-1))
(2.0 at cycle 0 -> 0.5 at the last). Training = diffusion self-conditioning: on half the batches a no-grad first pass makes p (its
logit is computed from the silhouette BEFORE the gold cond enters, so p carries no gold), then the trained pass reads it. Committed slots
are unchanged (the snapped atom x radius). sp = 0 reproduces the old breath exactly (the projection has no bias).
APPLY ONLY after unit lfbt4's last gsm_breath_reader launch. Refuses if already applied."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "scripts/gsm_breath_reader.py"; s = open(p).read()
assert "SOFT =" not in s, "already applied"
r = [('T6OFF = os.environ.get("BT_T6OFF") == "1"',
      'SOFT = os.environ.get("BT_SOFT") == "1"              # (ledger 11-01) the soft notebook: uncommitted slots show sum_t p(t) C_t, tau by cycle\n'
      'T6OFF = os.environ.get("BT_T6OFF") == "1"'),
     ('            s.lev = nn.Embedding(LEVMAX + 1, D); s.reent = nn.Linear(DB, D)\n',
      '            s.lev = nn.Embedding(LEVMAX + 1, D); s.reent = nn.Linear(DB, D)\n'
      '            s.sre = nn.Linear(DB, D, bias=False) if SOFT else None                             # the soft notebook\'s road (no bias: sp=0 is the old breath)\n'),
     ('        def breath(s, t, xq, nb, cond, cyc=None):\n',
      '        def breath(s, t, xq, nb, cond, cyc=None, sp=None):\n'),
     ('            h = s.slotk.weight[None] + torch.where(com[..., None], re, s.maskv) + s.cfp(cf) + s.lev(lev) + s.cons(cons[:, Lq:].long())\n',
      '            unc = s.maskv if sp is None else s.maskv + s.sre(sp @ s.C)                        # uncommitted: the mask + what the BT believes\n'
      '            h = s.slotk.weight[None] + torch.where(com[..., None], re, unc) + s.cfp(cf) + s.lev(lev) + s.cons(cons[:, Lq:].long())\n'),
     ('    logit, pa, pb, z, _ = net.breath(t, xq, nb, gtt, cyc)\n',
      '    sp = None\n'
      '    if SOFT and rng.random() < 0.5:                                                           # self-conditioning: the model\'s own detached belief\n'
      '        with torch.no_grad():\n'
      '            l0 = net.breath(t, xq, nb, gtt, cyc)[0]\n'
      '            sp = (l0 / TAU(cyc)[:, None, None]).softmax(-1)\n'
      '    logit, pa, pb, z, _ = net.breath(t, xq, nb, gtt, cyc, sp)\n'),
     ('            for c in range(NCYC):\n                nb = nb_tensors(notebook(E, L, progs, com, typ, rad))\n'
      '                logit, pa, pb, _, p = net.breath(t, xq, nb, torch.as_tensor(gt0) if ORACLE else None, torch.full((B,), c))\n',
      '            sp = None\n'
      '            for c in range(NCYC):\n                nb = nb_tensors(notebook(E, L, progs, com, typ, rad))\n'
      '                logit, pa, pb, _, p = net.breath(t, xq, nb, torch.as_tensor(gt0) if ORACLE else None, torch.full((B,), c), sp)\n'
      '                if SOFT and c + 1 < NCYC: sp = (logit / TAU(torch.full((B,), c + 1))[:, None, None]).softmax(-1)   # next cycle reads it\n'),
     ('def nb_tensors(arrs):\n',
      'def TAU(cyc):\n    """the coarse-to-fine schedule: tau 2.0 at cycle 0 -> 0.5 at the last (a (B,) float tensor)"""\n'
      '    return 2.0 ** (1.0 - 2.0 * cyc.float() / (NCYC - 1))\n\n\ndef nb_tensors(arrs):\n')]
for a, b in r:
    assert s.count(a) == 1, a[:70]; s = s.replace(a, b)
open(p, "w").write(s); print("BT5 soft notebook applied to", p)
