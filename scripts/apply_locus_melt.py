"""THE LOCUS-DRIVEN MELT (2026-09-14, the NL loop's certificate as a mask):
_LOCUS = {"melt": (B, LT) float mask, "kb": breath} set by a read script
(the fingerpost's disagreement locus per row); at the state entering breath
kb the marked slots are melted through the same _melt organ the wheel uses
(ALG_WHEEL_MELT sets the amplitude). None = off (bit-identical). Idempotent."""
import os, sys
p = os.environ.get("HEAD_PATH", "scripts/phase1_algebra_head.py"); s = open(p).read()
if "_LOCUS" in s:
    print("[apply] already applied"); sys.exit(0)
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80]); s = s.replace(old, new)
rep('''_WHEEL_NOGOOD = int(os.environ.get("ALG_WHEEL_NOGOOD", "0"))
''', '''_WHEEL_NOGOOD = int(os.environ.get("ALG_WHEEL_NOGOOD", "0"))
_LOCUS = None      # THE LOCUS-DRIVEN MELT (2026-09-14): {"melt": (B, LT) numpy, "kb": int} from the fingerpost's disagreement locus
''')
rep('''    _wm = state.get("wheel_melt")
    state["melted"] = _wm if (_WHEEL_MELT is not None and _wm is not None) else None   # this breath's released slots (T2)
''', '''    _wm = state.get("wheel_melt")
    if _LOCUS is not None and kb == _LOCUS["kb"] and _WHEEL_MELT is not None:
        from tinygrad import Tensor as _Tl
        _wm = _Tl(np.asarray(_LOCUS["melt"], np.float32))   # the locus: slots whose reading is unstable across views
    state["melted"] = _wm if (_WHEEL_MELT is not None and _wm is not None) else None   # this breath's released slots (T2)
''')
open(p, "w").write(s); print("[apply] the locus melt hook applied to", p)
