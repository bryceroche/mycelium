"""BT8b THE CHAINED BINDING (ledger 2026-11-04, word given: "avoid the SQL cross join and use phase binding and conjugate unbinding
instead!"): gsm_breath_reader.py gains BT_FACT=2 (BT_FACT=1 = BT8, unchanged).

The BT8 diag: the factored silhouette decodes op / kind a / kind b as well as or better than the 37 atoms on TEST, but the JOINT falls:
independent decoding loses the conditioning between factors. BT8b keeps the 11 shared Welford atoms (5 ops incl. stop, 3 kinds, 3 kinds)
and no per-combination storage, and puts the conditioning into the KEYS (fixed random phases, an address, never a memorized cell):
  z ~ e^{i psi_op} F_op[op] + e^{i (psi_a + Theta_op)} F_a[ka] + e^{i (psi_b + Theta_op + Phi_ka)} F_b[kb]
  UNBIND kind a under each op's key, kind b under each (op, kind a)'s key (the conjugate mirror, never 1/z)
  log p(t) = log p(op) + log p(ka | op) + log p(kb | op, ka)       (each a softmax over the shared factor codebook)
A wrong op unbinds kind a under the wrong key and reads crosstalk, so the joint is coherent by construction (the chain of binding is the
zoom: op, then kind a in op's frame, then kind b in (op, ka)'s frame). The Welford atoms follow the gold silhouettes unbound under the
gold type's own keys. Asserts each anchor once.
"""
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[bt8b] {f}: {len(pairs)} edit(s)")


edit("gsm_breath_reader.py", [
    ('FACT = os.environ.get("BT_FACT") == "1"  ',
     'CHAIN = os.environ.get("BT_FACT") == "2"             # (ledger 11-04 BT8b) chained binding: kind keys bound to op (a) and op + kind a (b)\n'
     'FACT = os.environ.get("BT_FACT") in ("1", "2")  '),
    ('                s.tauf = nn.Parameter(torch.zeros(3))\n',
     '                s.tauf = nn.Parameter(torch.zeros(3))\n'
     '                if CHAIN:   # (BT8b) the fixed random key phases: op keys and kind-a keys (addresses, not cells)\n'
     '                    s.register_buffer("Theta", torch.rand(NOP, DB // 2, generator=g_) * 2 * math.pi)\n'
     '                    s.register_buffer("Phi", torch.rand(3, DB // 2, generator=g_) * 2 * math.pi)\n'),
    ('        def atoms(s):\n',
     '        def keys(s):\n'
     '            """(BT8b) per type: the kind-a and kind-b key phases -> (NT, DB/2) x2 (BT8: the bare role phases)"""\n'
     '            if not CHAIN: return s.psi[1].expand(NT, -1), s.psi[2].expand(NT, -1)\n'
     '            return s.psi[1] + s.Theta[par], s.psi[2] + s.Theta[par] + s.Phi[fka]\n\n'
     '        def unbind_gold(s, z, tt):\n'
     '            """(BT8b) the gold silhouettes unbound under the gold type\'s own keys -> (3, N, DB)"""\n'
     '            ka_, kb_ = s.keys()\n'
     '            return torch.stack([rot(z, -s.psi[0]), rot(z, -ka_[tt]), rot(z, -kb_[tt])])\n\n'
     '        def atoms(s):\n'),
    ('            a = rot(s.Fop[par], s.psi[0]) + (rot(s.Fa[fka], s.psi[1]) + rot(s.Fb[fkb], s.psi[2])) * isstep[:, None]\n',
     '            ka_, kb_ = s.keys()\n'
     '            a = rot(s.Fop[par], s.psi[0]) + (rot(s.Fa[fka], ka_) + rot(s.Fb[fkb], kb_)) * isstep[:, None]\n'),
    ('                lfine = (lf[1][..., fka] + lf[2][..., fkb]) * isstep\n',
     '                lfine = (lf[1][..., fka] + lf[2][..., fkb]) * isstep\n'
     '                if CHAIN:   # (BT8b) unbind each kind under every type\'s own key: p(ka | op), p(kb | op, ka)\n'
     '                    ka_, kb_ = s.keys(); zt = z[:, :, None]\n'
     '                    lca = [(-((rot(zt, -k_)[..., None, :] - F_) ** 2).sum(-1) / math.sqrt(DB) * s.tauf[r].exp()).log_softmax(-1)\n'
     '                           for r, k_, F_ in ((1, ka_, s.Fa), (2, kb_, s.Fb))]                    # (B, K, NT, 3) each\n'
     '                    lfine = (lca[0].gather(-1, fka.expand(*lca[0].shape[:2], -1)[..., None])[..., 0]\n'
     '                             + lca[1].gather(-1, fkb.expand(*lca[1].shape[:2], -1)[..., None])[..., 0]) * isstep\n'),
    ('            u = net.unbind(zz); st_ = tt != STOP\n',
     '            u = net.unbind(zz); st_ = tt != STOP\n'
     '            if CHAIN:   # (BT8b) the kinds unbound under the gold type\'s own keys\n'
     '                u = net.unbind_gold(zz, tt)\n'),
])
