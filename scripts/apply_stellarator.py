"""apply_stellarator.py — THE STELLARATOR, REFIRED (2026-09-10, the word:
"do we snip the residual connections? I'm inclined if you are" — yes).
Rewrites the August cell-3b door (ALG_STELLAR=1) for today's payload:
  * PER-SLOT junction: the notebook read `_rd` is (B, L, H) under
    NB_PERSLOT (the August form reshaped a blurred (B, H) read — it
    would crash on the per-slot lanes);
  * THE SNIP: w(kb) = cos^2(kb*pi / (2*(K_B-1))) — exactly 0 at the LAST
    loop breath (August's /(2*K_B) left 5% of the residual standing: a
    bypass under the residual-seal law); no cliff between;
  * THE CLOCK IS EXEMPT: under ALG_POLAR the blend applies to CONTENT
    dims only (g_c); the clock block (g_k) passes untouched — the sextet
    is a coordinate system, not a road, and the ink would overwrite it.
Unset = untouched (the block is under `if ALG_STELLAR`)."""
import sys
P = "scripts/phase1_algebra_head.py"
s = open(P).read()
old = ('        if ALG_STELLAR:                        # cell-3b: the twist in\n'
       '            _w = math.cos(kb * math.pi / (2 * K_B)) ** 2   # geometry —\n'
       '            cur = _w * cur + (1 - _w) * _rd.reshape(B, 1, -1)\n'
       '            q_extra = cur + p["breath_emb"][kb].reshape(1, 1, -1) + _rd.reshape(B, 1, -1)\n'
       '                                                  # no cliff, no gate\n')
new = ('        if ALG_STELLAR:                        # cell-3b: the twist in\n'
       '            # THE STELLARATOR, REFIRED (apply_stellarator.py, 2026-09-10):\n'
       '            # the helical handoff residual -> notebook junction, per-slot\n'
       '            # payload, SNIPPED to exactly 0 at the last loop breath, the\n'
       '            # clock block exempt (a coordinate system, not a road).\n'
       '            _w = math.cos(kb * math.pi / (2 * (K_B - 1))) ** 2   # 1 -> 0\n'
       '            _rdj = _rd if NB_PERSLOT else _rd.reshape(B, 1, -1)\n'
       '            if ALG_POLAR:\n'
       '                _, _, _sg_c, _sg_k, _ = _polar_sink()\n'
       '                _wv = _w * _sg_c.reshape(1, 1, -1) + _sg_k.reshape(1, 1, -1)\n'
       '            else:\n'
       '                _wv = _w\n'
       '            cur = _wv * cur + (1.0 - _wv) * _rdj\n'
       '            q_extra = cur + p["breath_emb"][kb].reshape(1, 1, -1) + _rdj\n'
       '                                                  # no cliff, no gate\n')
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(P, "w").write(s)
print("apply_stellarator: the door rewritten (per-slot, snipped, clock-exempt)")
