"""apply_grad_tap.py — THE GRADIENT TAP (2026-09-10, the gut: "does the
farmer get tipped?"). A read-only probe: `_GTAP` (module global, None
everywhere except under scripts/grad_census.py) is a dict breath -> zero
Tensor (B, L, H) with requires_grad; breath_step adds it to the state
ENTERING that breath, forward adds "final" to the state the readout
reads, and the notebook read records its attention weights. After
loss.backward(), _GTAP[kb].grad IS dL/d(state entering breath kb): the
credit each breath receives. Zero-cost when None (the _IMP / _CENSUS
idiom). Never a training door."""
import sys
P = "scripts/phase1_algebra_head.py"
s = open(P).read()
if "_GTAP" in s:
    print("apply_grad_tap: target already carries the tap — refusing"); sys.exit(2)
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80]); s = s.replace(old, new)
rep('ALG_CLOCK_CANON = int(os.environ.get("ALG_CLOCK_CANON", "0"))   # memories in a canonical clock frame\n',
    'ALG_CLOCK_CANON = int(os.environ.get("ALG_CLOCK_CANON", "0"))   # memories in a canonical clock frame\n'
    '_GTAP = None      # THE GRADIENT TAP (apply_grad_tap.py): read-only probe leaves per breath\n')
rep('    cur = state["cur"]; breaths = state["breaths"]\n',
    '    cur = state["cur"]; breaths = state["breaths"]\n'
    '    if _GTAP is not None and kb in _GTAP:\n'
    '        cur = cur + _GTAP[kb]            # the probe leaf: dL/d(state entering breath kb)\n')
rep('            _at = _sc.softmax(-1)             # (B, L, k)\n',
    '            _at = _sc.softmax(-1)             # (B, L, k)\n'
    '            if _GTAP is not None:\n'
    '                _GTAP.setdefault("at", []).append((kb, _at.detach()))\n')
rep('    _s_final = breaths[-1]\n',
    '    _s_final = breaths[-1]\n'
    '    if _GTAP is not None and "final" in _GTAP:\n'
    '        _s_final = _s_final + _GTAP["final"]\n')
open(P, "w").write(s); print("apply_grad_tap: 4 anchors applied")
