"""SPH THE SMOOTH SPACE (ledger 2026-11-04, Bryce's brainstorm: "constrain the search space such that only the smaller representation
space with smooth, even, and flowing angles is allowed"; word given): gsm_breath_reader.py + gsm_breath_tg.py gain BT_SPH.

  BT_SPH=1  THE TORUS SILHOUETTE: every complex plane of the silhouette z (DB/2 planes) is put on the unit circle, z_p / |z_p|. Angles
            only: the waist lives on the torus T^(DB/2), smooth and corner-free. Radius is NOT lost: the Welford atoms average unit
            phasors, so an atom's radius per plane is its members' agreement (mean resultant length = consolidation), and the commit
            radius stays the type confidence p (the conjugate mirror rule).
  BT_SPH=2  + THE TURNING BREATH: each breath block's update is a ROTATION of the slot state's planes, h <- h * e^{i theta},
            theta = W_oa(attention output) (zero-init: the identity at birth), in place of the residual add h + W_o(...). Norm-preserving
            per plane: no breath can grow, shrink or blur the state. The reader blocks keep the residual.
Identical formulas in both engines (|z_p| = sqrt(re^2 + im^2 + 1e-12)). Applied by unit lfsph after lfq2 ends (lfgpuq / lfp9 / lfq2 /
lfag3 re-read both files). Asserts each anchor once. Usage: sph.py [dir]  (default scripts/; dry runs pass a COPY dir)
"""
import os, sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[sph] {f}: {len(pairs)} edit(s)")


edit("gsm_breath_reader.py", [
    ('T6OFF = os.environ.get("BT_T6OFF") == "1"',
     'SPH = os.environ.get("BT_SPH", "0")                 # (ledger 11-04 SPH) 1 = the torus silhouette; 2 = + the turning breath (rotation-only updates)\n'
     'T6OFF = os.environ.get("BT_T6OFF") == "1"'),
    ('    class Attn(nn.Module):\n        def __init__(s):\n            super().__init__(); s.lnq = nn.LayerNorm(D); s.lnk = nn.LayerNorm(D)\n',
     '    def torus(z):                                                                          # (SPH) every plane on the unit circle\n'
     '        zz = z.unflatten(-1, (-1, 2)); return (zz / (zz.pow(2).sum(-1, keepdim=True) + 1e-12).sqrt()).flatten(-2)\n\n'
     '    class Attn(nn.Module):\n        def __init__(s, turn=False):\n            super().__init__(); s.lnq = nn.LayerNorm(D); s.lnk = nn.LayerNorm(D)\n'
     '            s.turn = turn\n'
     '            if turn: s.oa = nn.Linear(D, D // 2); nn.init.zeros_(s.oa.weight); nn.init.zeros_(s.oa.bias)   # (SPH=2) the turn angles\n'),
    ('            return xq + s.o(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, Lq, D))',
     '            o_ = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, Lq, D)\n'
     '            return rot(xq, s.oa(o_)) if s.turn else xq + s.o(o_)                              # (SPH=2) a rotation, never an add'),
    ('            s.core = nn.ModuleList([Attn() for _ in range(NLB)])',
     '            s.core = nn.ModuleList([Attn(SPH == "2") for _ in range(NLB)])'),
    ('            z = s.down(h)                                                                      # THE SILHOUETTE',
     '            z = s.down(h)                                                                      # THE SILHOUETTE\n'
     '            if SPH != "0": z = torus(z)                                                        # (SPH) on the torus'),
])
edit("gsm_breath_tg.py", [
    ('def attn(P, n, xq, xk, aq, ak, mask):',
     'def torus(z):\n    """(SPH) every plane of z (..., 2P) on the unit circle (the torch formula)"""\n'
     '    sh = z.shape; zz = z.reshape(*sh[:-1], sh[-1] // 2, 2)\n'
     '    return (zz / ((zz * zz).sum(-1, keepdim=True) + 1e-12).sqrt()).reshape(sh)\n\n\n'
     'def attn(P, n, xq, xk, aq, ak, mask, turn=False):'),
    ('    return xq + lin(P, (sc.softmax(-1) @ v).transpose(1, 2).reshape(B, Lq, D), n + ".o")',
     '    o_ = (sc.softmax(-1) @ v).transpose(1, 2).reshape(B, Lq, D)\n'
     '    return rot(xq, lin(P, o_, n + ".oa")) if turn else xq + lin(P, o_, n + ".o")                 # (SPH=2) a rotation, never an add'),
    ('    for i in range(NLB): h = attn(P, f"core.{i}", h, Tensor.cat(xk_q, h, dim=1), sa, ka, mask)',
     '    for i in range(NLB): h = attn(P, f"core.{i}", h, Tensor.cat(xk_q, h, dim=1), sa, ka, mask, BR.SPH == "2")'),
    ('    z = lin(P, lnorm(P, h, "down.0"), "down.1")',
     '    z = lin(P, lnorm(P, h, "down.0"), "down.1")\n'
     '    if BR.SPH != "0": z = torus(z)                                                              # (SPH) on the torus'),
])
