"""apply_clock_canon.py — THE CANONICAL CLOCK FRAME for the memories
(2026-09-10, the word: "fix the clock block"). Under ALG_POLAR the state's
clock planes turn one quantum per breath; the notebook ink and the garage
deposit are written from the ROTATED state (phase abs[kb-1]) through
dense maps (W_sil; W_bind1/gelu/W_bind2) that smear that phase across
every dim, and both are read at a later breath (state at abs[kb-2]) with
nothing undoing the phase: a stale bearing rides into the reading. The
fix (door ALG_CLOCK_CANON=1; unset = untouched): WRITE IN A CANONICAL
FRAME — de-rotate the clock planes to phase 0 before the ink and the
deposit are made — and RE-PHASE each read (the notebook read, the garage
injection) to the reader's own phase. Content planes are identity in the
wheel tables (cos 1, sin 0), so only the clock block moves. One
primitive, both memories."""
import sys
P = "scripts/phase1_algebra_head.py"
s = open(P).read()
if "ALG_CLOCK_CANON" in s:
    print("apply_clock_canon: target already carries the door — refusing"); sys.exit(2)
n0 = len(s)

def rep(old, new, count=1):
    global s
    c = s.count(old)
    assert c == count, (c, old[:90])
    s = s.replace(old, new)

# K0 — the dial + the primitive (beside the stellarator dial)
rep('ALG_STELLAR = int(os.environ.get("ALG_STELLAR", "0"))    # cell-3b: helical handoff\n',
    'ALG_STELLAR = int(os.environ.get("ALG_STELLAR", "0"))    # cell-3b: helical handoff\n'
    'ALG_CLOCK_CANON = int(os.environ.get("ALG_CLOCK_CANON", "0"))   # memories in a canonical clock frame\n'
    '\n'
    '\n'
    'def _clock_frame(x, k, sign, rot2):\n'
    '    """Rotate x\'s CLOCK planes by sign * (the wheel\'s absolute phase\n'
    '    after loop breath k); k < 1 = phase 0 = identity. Content planes\n'
    '    are identity in the tables. sign=-1 canonicalizes a write,\n'
    '    sign=+1 re-phases a read to the reader\'s frame."""\n'
    '    if k < 1:\n'
    '        return x\n'
    '    from tinygrad import Tensor as _Tf, dtypes as _df\n'
    '    _, _, _fac, _fas, _ = _polar_tables()\n'
    '    return rot2(x, _Tf(_fac[k - 1], dtype=_df.float),\n'
    '                _Tf(float(sign) * _fas[k - 1], dtype=_df.float))\n')

# K1 — the notebook read, re-phased to the reader (state at abs[kb-2] -> k = kb-1)
rep('            _rd = sum(_at[:, :, j:j + 1] * _nb[j] for j in range(len(_nb)))\n'
    '            if "notebook" in _SEVER:\n',
    '            _rd = sum(_at[:, :, j:j + 1] * _nb[j] for j in range(len(_nb)))\n'
    '            if ALG_CLOCK_CANON and ALG_POLAR:\n'
    '                _rd = _clock_frame(_rd, kb - 1, +1, _rot2)   # the reader\'s frame\n'
    '            if "notebook" in _SEVER:\n')

# K2 — the garage injection, re-phased to the reader
rep('        _inj4 = Tensor.cat(*_rds4, dim=-1) @ p["W_busr"]\n'
    '        _inj4o = _inj4 * 0.0 if "garage" in _SEVER else _inj4   # sever door\n',
    '        _inj4 = Tensor.cat(*_rds4, dim=-1) @ p["W_busr"]\n'
    '        if ALG_CLOCK_CANON and ALG_POLAR:\n'
    '            _inj4 = _clock_frame(_inj4, kb - 1, +1, _rot2)   # the reader\'s frame\n'
    '        _inj4o = _inj4 * 0.0 if "garage" in _SEVER else _inj4   # sever door\n')

# K3 — the ink write, canonical frame (the state is at abs[kb-1] here)
rep('        _nb.append((cur @ p["W_sil"]) if NB_PERSLOT\n'
    '                   else (_fed_core(cur).mean(1) @ p["W_sil"]))\n',
    '        _cur_w = (_clock_frame(cur, kb, -1, _rot2)\n'
    '                  if ALG_CLOCK_CANON and ALG_POLAR else cur)   # canonical frame\n'
    '        _nb.append((_cur_w @ p["W_sil"]) if NB_PERSLOT\n'
    '                   else (_fed_core(_cur_w).mean(1) @ p["W_sil"]))\n')

# K4 — the deposit, canonical frame
rep('        _wg4 = ((cur @ p["W_bind1"] + p["W_bind1_b"]).gelu()\n'
    '                @ p["W_bind2"])\n',
    '        _cur_g = (_clock_frame(cur, kb, -1, _rot2)\n'
    '                  if ALG_CLOCK_CANON and ALG_POLAR else cur)   # canonical frame\n'
    '        _wg4 = ((_cur_g @ p["W_bind1"] + p["W_bind1_b"]).gelu()\n'
    '                @ p["W_bind2"])\n')
open(P, "w").write(s)
print(f"apply_clock_canon: 1 dial + primitive, 4 anchors applied ({n0} -> {len(s)} bytes)")
