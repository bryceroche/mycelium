"""apply_stellarator2.py — STELLARATOR v2 (2026-09-10, the word): the TOTAL
cut. ALG_STELLAR=2 = the helical handoff with NO clock exemption — every
plane blends, the residual is extinguished at the last loop breath on
the whole state. The clock's bearing survives because the memory is
written in the canonical frame and re-phased at read (ALG_CLOCK_CANON);
without the canonical frame this is the READ-TIME meter of the
exemption's own cost (what content rode the exempt block across the cut).
ALG_STELLAR=1 keeps v1 (exempt) bit-identically."""
import sys
P = "scripts/phase1_algebra_head.py"
s = open(P).read()
if "ALG_STELLAR >= 2" in s:
    print("apply_stellarator2: target already carries the door — refusing"); sys.exit(2)
old = ('            if ALG_POLAR:\n'
       '                _, _, _sg_c, _sg_k, _ = _polar_sink()\n'
       '                _wv = _w * _sg_c.reshape(1, 1, -1) + _sg_k.reshape(1, 1, -1)\n'
       '            else:\n'
       '                _wv = _w\n')
new = ('            if ALG_POLAR and ALG_STELLAR < 2:\n'
       '                _, _, _sg_c, _sg_k, _ = _polar_sink()\n'
       '                _wv = _w * _sg_c.reshape(1, 1, -1) + _sg_k.reshape(1, 1, -1)\n'
       '            else:\n'
       '                _wv = _w            # v2 (ALG_STELLAR >= 2): the TOTAL cut, no exemption\n')
assert s.count(old) == 1, s.count(old)
open(P, "w").write(s.replace(old, new))
print("apply_stellarator2: the total-cut door added (ALG_STELLAR=2); v1 untouched")
