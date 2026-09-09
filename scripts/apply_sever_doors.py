"""apply_sever_doors.py — THE SEVERANCE LADDER (2026-09-09, the word: "make
sure the param ratios are well balanced"). Read-time doors that zero one
organ's injection so its REMOVAL COST can be read (the headroom
corollary: sever, read the drop, then decide). Env ALG_SEVER = comma list
of organ names; unset = the forward is untouched (bit-identical, the eq
gate proves it). Organs: notebook, garage, s3, s4, mixer, fedtwin, altv0,
ffn (loop breaths only), pforms. Every door keeps the params in the graph
(zero-mult, the None-grad lesson) — these are READ doors, never training.
Idempotent: refuses a target that already carries the doors."""
import re, sys
P = "scripts/phase1_algebra_head.py"
s = open(P).read()
if "_SEVER" in s:
    print("apply_sever_doors: target already carries the doors — refusing"); sys.exit(2)
n0 = len(s)

def rep(old, new, count=1):
    global s
    c = s.count(old)
    assert c == count, (c, old[:80])
    s = s.replace(old, new)

# A0 — the door registry (after PF_FORMS so it sits with the dials)
rep('PF_FORMS = int(os.environ.get("PF_FORMS", "3"))   # pointer/macro forms\n',
    'PF_FORMS = int(os.environ.get("PF_FORMS", "3"))   # pointer/macro forms\n'
    '# THE SEVERANCE LADDER (apply_sever_doors.py, 2026-09-09): read-time\n'
    '# organ severance for removal-cost reads. Unset = untouched forward.\n'
    '_SEVER_ORGANS = frozenset(("notebook", "garage", "s3", "s4", "mixer",\n'
    '                           "fedtwin", "altv0", "ffn", "pforms"))\n'
    '_SEVER = frozenset(x for x in os.environ.get("ALG_SEVER", "").split(",") if x)\n'
    'assert _SEVER <= _SEVER_ORGANS, \\\n'
    '    f"ALG_SEVER unknown organ(s) {sorted(_SEVER - _SEVER_ORGANS)}; " \\\n'
    '    f"known: {sorted(_SEVER_ORGANS)}"\n')

# A1 — FFN (loop breaths only: extra is None on the breath-0 read)
rep('        st = st + ((st @ p["ffn_w1"] + p["ffn_b1"]).gelu() @ p["ffn_w2"] + p["ffn_b2"])\n'
    '        return st, at.mean(1)\n',
    '        _ffn_sv = ((st @ p["ffn_w1"] + p["ffn_b1"]).gelu() @ p["ffn_w2"] + p["ffn_b2"])\n'
    '        if "ffn" in _SEVER and extra is not None:   # sever: loop breaths only\n'
    '            _ffn_sv = _ffn_sv * 0.0\n'
    '        st = st + _ffn_sv\n'
    '        return st, at.mean(1)\n')

# A2 — notebook (both lane forms)
rep('            _rd = sum(_at[:, :, j:j + 1] * _nb[j] for j in range(len(_nb)))\n'
    '            q_extra = q_extra + _rd           # (B, L, H) — no blur\n',
    '            _rd = sum(_at[:, :, j:j + 1] * _nb[j] for j in range(len(_nb)))\n'
    '            if "notebook" in _SEVER:\n'
    '                _rd = _rd * 0.0\n'
    '            q_extra = q_extra + _rd           # (B, L, H) — no blur\n')
rep('            _rd = sum(_at[:, j:j + 1] * _nb[j] for j in range(len(_nb)))\n'
    '            q_extra = q_extra + _rd.reshape(B, 1, -1)\n',
    '            _rd = sum(_at[:, j:j + 1] * _nb[j] for j in range(len(_nb)))\n'
    '            if "notebook" in _SEVER:\n'
    '                _rd = _rd * 0.0\n'
    '            q_extra = q_extra + _rd.reshape(B, 1, -1)\n')

# A3 — garage (the OPEN road only; the sealed road is a different meter)
rep('        _inj4 = Tensor.cat(*_rds4, dim=-1) @ p["W_busr"]\n',
    '        _inj4 = Tensor.cat(*_rds4, dim=-1) @ p["W_busr"]\n'
    '        _inj4o = _inj4 * 0.0 if "garage" in _SEVER else _inj4   # sever door\n')
rep('            _q_open = q_extra + _inj4 * p["bus_g"].reshape(1, 1, 1)\n',
    '            _q_open = q_extra + _inj4o * p["bus_g"].reshape(1, 1, 1)\n')
rep('        else:\n            q_extra = q_extra + _inj4 * p["bus_g"].reshape(1, 1, 1)\n',
    '        else:\n            q_extra = q_extra + _inj4o * p["bus_g"].reshape(1, 1, 1)\n')

# A4 — the sc2 slot mixer's own output
rep('    h_slot = (sc2.softmax(-1) @ bv) @ p["W_bo"] + p["W_bo_b"]\n',
    '    h_slot = (sc2.softmax(-1) @ bv) @ p["W_bo"] + p["W_bo_b"]\n'
    '    if "mixer" in _SEVER:\n'
    '        h_slot = h_slot * 0.0\n')

# A5 — the fed mixer twin
rep('        h_slot = h_slot + _mx_inj\n',
    '        if "fedtwin" in _SEVER:\n'
    '            _mx_inj = _mx_inj * 0.0\n'
    '        h_slot = h_slot + _mx_inj\n')

# A6 — the v0 alt bias (both sites: sc2 and station 4)
rep('    if _A5 is not None and "alt_g" in p:\n'
    '        # v0 soft bias rides alongside (facts wire attention)\n',
    '    if _A5 is not None and "alt_g" in p and "altv0" not in _SEVER:\n'
    '        # v0 soft bias rides alongside (facts wire attention)\n')
rep('        if _A5 is not None and "alt_g" in p:   # v0 bias, as station 2\n',
    '        if _A5 is not None and "alt_g" in p and "altv0" not in _SEVER:   # v0 bias, as station 2\n')

# A7 — ALT21 stations 3 and 4
rep('        h_slot = h_slot + _d21a + _d21b  # additive; zeros at birth\n',
    '        if "s3" in _SEVER:\n'
    '            _d21a = _d21a * 0.0\n'
    '        if "s4" in _SEVER:\n'
    '            _d21b = _d21b * 0.0\n'
    '        h_slot = h_slot + _d21a + _d21b  # additive; zeros at birth\n')

# A8 — the pointer forms (readout; the gains ARE the organ)
rep('    gk = "fed_pf_" + name + "_g"\n    if not (ALG_FED and gk in p):\n        return base\n',
    '    gk = "fed_pf_" + name + "_g"\n    if not (ALG_FED and gk in p) or "pforms" in _SEVER:\n        return base\n')

open(P, "w").write(s)
print(f"apply_sever_doors: 9 organs, 12 anchors applied ({n0} -> {len(s)} bytes)")
