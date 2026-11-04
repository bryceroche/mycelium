"""BT9 THE ID SKIP (ledger 2026-11-03, word given: "yes we want the composite keys to bypass the compression"): gsm_breath_reader.py
gains BT_IDSKIP=1.

Today slot id (s.slotk) and tree level (s.lev) are ADDED to the slot state before the breath and squeezed through the 512 -> 64 waist;
only the T6 step tick bypasses it (a rotation on the pointer planes after expansion). Under BT_IDSKIP=1:
  - slot id and tree level leave the pre-bottleneck input (the waist carries the rule: content only)
  - the composite key rejoins AFTER the waist as phases, never content (a skip that cannot leak facts):
      slot id     = the T6 step tick (unchanged) on planes [0:NTK)
      tree level  = T6 gears x the executor's derivation depth on its own planes [NTK:NTK+NLV) of the slot pointers' q and k
      type role   = the bus planes [NTK+NLV:) (the snapped type, as before)
  - they meet in the slot-pointer attention: Re(q conj k) sums the aligned planes (Nazare: in phase they stack, out of phase cancel)
The T6 clock (not T9): the tags' spans are <= 8 and T6's slow gears (48, 96) already exceed them (the clock-range rule).
Applied by unit lfbt9 after lfw128 launched its runs (that chain re-reads this file). Asserts each anchor once.
"""
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[bt9] {f}: {len(pairs)} edit(s)")


edit("gsm_breath_reader.py", [
    ('T6OFF = os.environ.get("BT_T6OFF") == "1"',
     'IDSKIP = os.environ.get("BT_IDSKIP") == "1"          # (ledger 11-03 BT9) slot id + tree level bypass the waist as phase tags (the ID skip)\n'
     'T6OFF = os.environ.get("BT_T6OFF") == "1"'),
    ('    wt = torch.as_tensor(np.radians(CK.t6_plane_speeds(NTK)), dtype=torch.float32)        # pointer space: the step tick\'s planes\n',
     '    wt = torch.as_tensor(np.radians(CK.t6_plane_speeds(NTK)), dtype=torch.float32)        # pointer space: the step tick\'s planes\n'
     '    NLV = NTK // 2 if IDSKIP else 0                                                       # (BT9) the tree-level tag planes\n'
     '    wl = torch.as_tensor(np.radians(CK.t6_plane_speeds(max(NLV, 1))), dtype=torch.float32)\n'),
    ('            h = s.slotk.weight[None] + torch.where(com[..., None], re, unc) + s.cfp(cf) + s.lev(lev) + s.cons(cons[:, Lq:].long())',
     '            h = torch.where(com[..., None], re, unc) + s.cfp(cf) + s.cons(cons[:, Lq:].long())\n'
     '            if not IDSKIP: h = h + s.slotk.weight[None] + s.lev(lev)                         # (BT9: the IDs ride the skip instead)'),
    ('                def bind(q, role):                                                                 # bus planes: q * e^{i phi(type, role)}\n'
     '                    ph = s.phi[c, role]\n'
     '                    return torch.cat([q[..., :2 * NTK], rot(q[..., 2 * NTK:], ph)], -1)\n',
     '                def bind(q, role):                                                                 # bus planes: q * e^{i phi(type, role)}\n'
     '                    ph = s.phi[c, role][..., NLV:]                                                 # (BT9: the level planes are not bus)\n'
     '                    return torch.cat([q[..., :2 * (NTK + NLV)], rot(q[..., 2 * (NTK + NLV):], ph)], -1)\n\n'
     '                la = lev.float()[..., None] * wl                                                   # (BT9) the level tag per slot\n\n'
     '                def levrot(z_):                                                                    # (BT9) bind the level on its planes\n'
     '                    if not IDSKIP: return z_\n'
     '                    return torch.cat([z_[..., :2 * NTK], rot(z_[..., 2 * NTK:2 * (NTK + NLV)], la), z_[..., 2 * (NTK + NLV):]], -1)\n'),
    ('                kt = s.kt(xk_q); ksl = tickrot(s.ks(hu)); cnt = cons.float()',
     '                kt = s.kt(xk_q); ksl = levrot(tickrot(s.ks(hu))); cnt = cons.float()'),
    ('                    ls = torch.einsum("bkd,bjd->bkj", tickrot(q), ksl) + s.pc * cnt[:, None, Lq:] * math.sqrt(D)',
     '                    ls = torch.einsum("bkd,bjd->bkj", levrot(tickrot(q)), ksl) + s.pc * cnt[:, None, Lq:] * math.sqrt(D)'),
])
