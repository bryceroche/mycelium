"""T2 — THE CLAIM MASK (2026-09-13, the order of attack): the missing back-edge
tokens <- slots, entered as STRUCTURE (no gain to vote down): a token that a
slot attended to strongly at breath k (station 3's head-mean attention > tau)
is CLAIMED, and at breath k+1 every OTHER slot's token score on it takes -beta
(both grounding roads: the bank's pbias and station 3). A melted slot's claims
are released (the melt frees its tokens for re-derivation). ALG_T2_CLAIM=
"beta:tau" (e.g. 3:0.5); unset = bit-identical. Idempotent; HEAD_PATH for a copy."""
import os, sys
p = os.environ.get("HEAD_PATH", "scripts/phase1_algebra_head.py"); s = open(p).read()
if "_T2_CLAIM" in s:
    print("[apply] already applied"); sys.exit(0)
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80]); s = s.replace(old, new)
rep('''_WHEEL_NOGOOD = int(os.environ.get("ALG_WHEEL_NOGOOD", "0"))
''', '''_WHEEL_NOGOOD = int(os.environ.get("ALG_WHEEL_NOGOOD", "0"))
# T2 — THE CLAIM MASK (2026-09-13): tokens <- slots as structure. ALG_T2_CLAIM="beta:tau".
_T2 = os.environ.get("ALG_T2_CLAIM", "")
_T2_CLAIM = bool(_T2)
_T2_BETA = float(_T2.split(":")[0]) if _T2 else 0.0
_T2_TAU = float(_T2.split(":")[1]) if _T2 else 0.5


def _claim_bias(prev_at, melted, B, LT):
    """prev_at: (B, LT, T) last breath's head-mean slots<-tokens attention; melted: (B, LT) or None.
    Returns (B, 1, LT, T): -beta where the token is claimed (> tau) by ANOTHER, unmelted slot."""
    C = (prev_at > _T2_TAU).float()
    if melted is not None:
        C = C * (1.0 - melted.reshape(B, LT, 1))
    anyc = C.sum(1, keepdim=True)                     # (B, 1, T)
    other = ((anyc - C) > 0).float()                  # claimed by someone else
    return (other * -_T2_BETA).reshape(B, 1, LT, -1)
''')
rep('''    _wm = state.get("wheel_melt")
    if _WHEEL_MELT is not None and _wm is not None:
        cur = _melt(cur, _wm)            # THE MELT: the core's slots, re-derived this breath
''', '''    _wm = state.get("wheel_melt")
    state["melted"] = _wm if (_WHEEL_MELT is not None and _wm is not None) else None   # this breath's released slots (T2)
    if _WHEEL_MELT is not None and _wm is not None:
        cur = _melt(cur, _wm)            # THE MELT: the core's slots, re-derived this breath
''')
rep('''    _wb = state.get("wheel_bias")
    if _wb is not None:                    # THE STEERING WHEEL's spotlight
        _pb_kb = _wb if _pb_kb is None else _pb_kb + _wb
''', '''    _wb = state.get("wheel_bias")
    if _wb is not None:                    # THE STEERING WHEEL's spotlight
        _pb_kb = _wb if _pb_kb is None else _pb_kb + _wb
    _t2b = None
    if _T2_CLAIM and state.get("prev_a21") is not None:   # T2: last breath's claims, first road
        _t2b = _claim_bias(state["prev_a21"], state.get("melted"), B, L_TOT)
        _pb_kb = _t2b if _pb_kb is None else _pb_kb + _t2b
''')
rep('''        if state.get("wheel_bias") is not None:   # the wheel's spotlight, second road
            _sa21 = _sa21 + state["wheel_bias"]
''', '''        if state.get("wheel_bias") is not None:   # the wheel's spotlight, second road
            _sa21 = _sa21 + state["wheel_bias"]
        if _t2b is not None:                       # T2: the claims, second road
            _sa21 = _sa21 + _t2b
''')
rep('''        _a21 = _sa21.softmax(-1)
''', '''        _a21 = _sa21.softmax(-1)
        if _T2_CLAIM:
            state["prev_a21"] = _a21.mean(1).detach()   # T2: this breath's claims, for the next breath
''')
open(p, "w").write(s); print("[apply] T2 claim mask applied to", p)
