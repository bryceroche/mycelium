"""apply_idle_breaths.py — THE IDLE-BREATH DOOR (2026-09-10, the middle
desert's headroom read): ALG_IDLE_BREATHS=3,4,5 makes those loop breaths
do nothing to the state — the breath gate closed (g = 0, no update) and
the stellarator blend skipped (w = 1, no junction) — so the state passes
through untouched (the clock still turns; the ink written is a copy).
Removal cost of the middle = open - idle. Read-only; unset = untouched."""
import sys
P = "scripts/phase1_algebra_head.py"; s = open(P).read()
if "ALG_IDLE_BREATHS" in s:
    print("apply_idle_breaths: already applied — refusing"); sys.exit(2)
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80]); s = s.replace(old, new)
rep('_GTAP = None      # THE GRADIENT TAP (apply_grad_tap.py): read-only probe leaves per breath\n',
    '_GTAP = None      # THE GRADIENT TAP (apply_grad_tap.py): read-only probe leaves per breath\n'
    '_IDLE = frozenset(int(x) for x in os.environ.get("ALG_IDLE_BREATHS", "").split(",") if x)   # read-only\n')
rep('            _w = math.cos(kb * math.pi / (2 * (K_B - 1))) ** 2   # 1 -> 0\n',
    '            _w = math.cos(kb * math.pi / (2 * (K_B - 1))) ** 2   # 1 -> 0\n'
    '            if kb in _IDLE:\n'
    '                _w = 1.0                     # idle: no junction blend\n')
rep('    g = p["breath_gate"][kb].sigmoid()\n',
    '    g = p["breath_gate"][kb].sigmoid()\n'
    '    if kb in _IDLE:\n'
    '        g = g * 0.0                          # idle: no update\n')
open(P, "w").write(s); print("apply_idle_breaths: 3 anchors applied")
