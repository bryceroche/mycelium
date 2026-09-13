"""THE WHIP (2026-09-12, the word; the laundry analogy): at breath K the state
entering the breath is kicked with Gaussian noise on the CONTENT planes only
(the clock keeps its time), scaled per slot to ALG_WHIP amplitude x the slot's
norm. Training: fresh noise every step through a fixed feed buffer (copyin);
read and val: a SEEDED constant (the same kick every time — diffusion samplers
re-noise at inference too). ALG_WHIP="k:amp" (e.g. 3:0.3); unset = bit-identical.
Idempotent; HEAD_PATH env for a copy."""
import os, sys
p = os.environ.get("HEAD_PATH", "scripts/phase1_algebra_head.py"); s = open(p).read()
if "_WHIP_SRC" in s:
    print("[apply] already applied"); sys.exit(0)
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80]); s = s.replace(old, new)
rep('''_GTAP = None      # THE GRADIENT TAP (apply_grad_tap.py): read-only probe leaves per breath
''', '''_GTAP = None      # THE GRADIENT TAP (apply_grad_tap.py): read-only probe leaves per breath
# THE WHIP (2026-09-12, the word): ALG_WHIP="k:amp" kicks the state ENTERING
# breath k with Gaussian noise on the content planes (per slot: amp x the
# slot's norm); the breaths after it have wrinkles to remove. Training feeds
# fresh noise per step (_WHIP_SRC = the trainer's buffer); the read uses a
# seeded constant (the same kick at train-time val and at read).
_WHIP = os.environ.get("ALG_WHIP", "")
_WHIP_K = int(_WHIP.split(":")[0]) if _WHIP else 0
_WHIP_AMP = float(_WHIP.split(":")[1]) if _WHIP else 0.0
_WHIP_SRC = None
_WHIP_CONST = {}


def _whip_noise(B, LT, HW):
    if _WHIP_SRC is not None:
        return _WHIP_SRC
    key = (B, LT, HW)
    if key not in _WHIP_CONST:
        from tinygrad import Tensor as _Tw
        _WHIP_CONST[key] = _Tw(np.random.RandomState(4242).randn(B, LT, HW).astype(np.float32)).contiguous().realize()
    return _WHIP_CONST[key]


def _whip_kick(cur):
    """cur + amp * ||cur_slot|| * unit-norm content-plane noise, per slot."""
    B, LT, HW = (int(x) for x in cur.shape)
    n = _whip_noise(B, LT, HW)
    if ALG_POLAR:
        n = n * _polar_sink()[2]        # content dims only: the clock keeps its time
    nn = (n * n).sum(-1, keepdim=True).sqrt() + 1e-6
    nc = (cur * cur).sum(-1, keepdim=True).sqrt()
    return cur + n / nn * nc * _WHIP_AMP
''')
rep('''    if _GTAP is not None and kb in _GTAP:
        cur = cur + _GTAP[kb]            # the probe leaf: dL/d(state entering breath kb)
''', '''    if _GTAP is not None and kb in _GTAP:
        cur = cur + _GTAP[kb]            # the probe leaf: dL/d(state entering breath kb)
    if _WHIP_K and kb == _WHIP_K:
        cur = _whip_kick(cur)            # THE WHIP: the state entering breath kb, kicked
''')
rep('''    b_drop = fix(np.ones((1,), np.float32), dtypes.float) \\
        if os.environ.get("BREATH_DROPOUT") else None   # door #52 coin buffer
''', '''    b_drop = fix(np.ones((1,), np.float32), dtypes.float) \\
        if os.environ.get("BREATH_DROPOUT") else None   # door #52 coin buffer
    b_whip = fix(np.zeros((batch, L_TOT, H_W), np.float32), dtypes.float) if _WHIP_K else None   # THE WHIP's fresh noise per step
    if b_whip is not None:
        globals()["_WHIP_SRC"] = b_whip
''')
rep('''        if b_drop is not None:
            _fd(b_drop, np.array(
''', '''        if b_whip is not None:
            _fd(b_whip, rng.randn(batch, L_TOT, H_W).astype(np.float32), _rl)   # THE WHIP: fresh noise this step
        if b_drop is not None:
            _fd(b_drop, np.array(
''')
rep('''            fv = _quick_val()
''', '''            globals()["_WHIP_SRC"] = None          # the val reads with the seeded kick, like the read
            fv = _quick_val()
            globals()["_WHIP_SRC"] = b_whip
''')
open(p, "w").write(s); print("[apply] the whip applied to", p)
