"""bridge_identity_check.py — THE BRIDGE's identity gate (2026-09-14). The
reference is the code that runs TODAY, extracted verbatim from the files:
the step trainer's wheel_bias tail (bias/melt from the pool's answers) and
the head's _wheel_turn tail (the same, per row); plus the head's own
_claim_bias / _melt / _nogood_apply and _token_step's sentence mask. Each
is compared with mycelium/loop_bridge on random inputs; any difference is
a FAIL. CPU only (DEV=CPU in the env); imports the head under the family
env, so run it fenced (MemoryMax) beside an arm."""
import os, re, sys, textwrap, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from mycelium.loop_bridge import Bridge, claims_bias, melt, nogood_apply, same_sentence

rng = np.random.default_rng(11)
B, LT, T = 6, 32, 40
fat_np = rng.random((B, LT, T)).astype(np.float32); fat_np /= fat_np.sum(-1, keepdims=True)
se_np = np.sort(rng.integers(0, 5, (B, T)), axis=1).astype(np.int32)
# the pool's answers: rows 0, 2, 3 refused with cores; core k indexes the row's parse (factors carry _slot)
parses = [[{"_slot": int(j)} for j in sorted(rng.choice(LT, 6, replace=False))] for _ in range(B)]
res = [("unsat", [0, 3]), ("sat", []), ("unsat", [1, 2, 5]), ("unsat", [4]), ("sat", []), ("unsat", [])]
fails = 0

def check(name, a, b):
    global fails
    ok = np.array_equal(np.asarray(a), np.asarray(b))
    print(f"[bridge-id] {name}: {'PASS' if ok else 'FAIL'}" + ("" if ok else f" max|d|={np.abs(np.asarray(a)-np.asarray(b)).max()}"))
    fails += (not ok)

# 1. the step trainer's wheel_bias tail, verbatim
src = open("scripts/step_trainer.py").read()
i0 = src.find("    bias = np.zeros((B, 1, LT, T), np.float32); turned = 0")
if i0 < 0:
    print("[bridge-id] trainer wheel_bias tail: APPLIED (it is the bridge call now; verified pre-apply 2026-09-14)")
i1 = src.index("    return bias, turned, melt", max(i0, 0))
tail = textwrap.dedent(src[i0:i1]) if i0 >= 0 else None
for mode in (("union", "own") if tail else ()):
    g = {"np": np, "B": B, "LT": LT, "T": T, "res": res, "parses": parses, "se_np": se_np, "fat_np": fat_np, "beta": 3.0, "mode": mode}
    exec(tail, g)
    br = Bridge(fat_np, se_np)
    check(f"trainer wheel_bias tail ({mode}) bias", g["bias"], br.spotlight(g["melt"], 3.0, mode))
    slots = np.zeros((B, LT), np.float32)
    for b, (st, core) in enumerate(res):
        if st == "unsat" and core:
            for k in core: slots[b, parses[b][k]["_slot"]] = 1.0
    check(f"trainer wheel_bias tail ({mode}) melt", g["melt"], slots)

# 2. the head's _wheel_turn tail, verbatim (nogood off; the memo answers the rows)
hsrc = open("scripts/phase1_algebra_head.py").read()
j0 = hsrc.index("    for b in range(B):\n        row = _rowd[b]; parse = _parses[b]")
j1 = hsrc.index('    _WHEEL.setdefault("turned", [])', j0)
htail = textwrap.dedent(hsrc[j0:j1])
if "loop_bridge" in htail:
    print("[bridge-id] head _wheel_turn tail: APPLIED (it is the bridge call now; verified pre-apply 2026-09-14)")
    htail = None
for mode in (("union", "own") if htail else ()):
    keys = list(range(B)); memo = {k: res[k] for k in keys}
    g = {"_np": np, "B": B, "L_TOT": LT, "T": T, "_rowd": [None] * B, "_parses": parses, "_memo": memo, "_keys": keys,
         "_WHEEL": {}, "_WHEEL_NOGOOD": 0, "fat_np": fat_np, "sent_np": se_np, "mode": mode, "beta": 3.0, "turned": 0,
         "bias": np.zeros((B, 1, LT, T), np.float32), "melt": np.zeros((B, LT), np.float32), "kb": 2}
    exec(htail, g)
    check(f"head _wheel_turn tail ({mode}) bias", g["bias"], Bridge(fat_np, se_np).spotlight(g["melt"], 3.0, mode))
    check(f"head _wheel_turn tail ({mode}) melt", g["melt"], slots)

# 3. the head's own in-graph roads
os.environ.setdefault("ALG_WHEEL_MELT", "0.5"); os.environ.setdefault("ALG_T2_CLAIM", "2:0.5")
import phase1_algebra_head as H
from tinygrad import Tensor
pa = Tensor(rng.random((B, LT, T)).astype(np.float32)); me = Tensor((rng.random((B, LT)) > 0.7).astype(np.float32))
check("_claim_bias (melted)", H._claim_bias(pa, me, B, LT).numpy(), claims_bias(pa, me, B, LT, H._T2_BETA, H._T2_TAU).numpy())
check("_claim_bias (no melt)", H._claim_bias(pa, None, B, LT).numpy(), claims_bias(pa, None, B, LT, H._T2_BETA, H._T2_TAU).numpy())
HW = H.H_W
cur = Tensor(rng.standard_normal((B, LT, HW)).astype(np.float32))
ref1 = H._melt(cur, me).numpy(); ref2 = H._melt(cur, me).numpy()
check("_melt deterministic", ref1, ref2)
gate = H._polar_sink()[2] if H.ALG_POLAR else 1.0
check("_melt", ref1, melt(cur, me, H._WHEEL_MELT, H._whip_noise(B, LT, HW), gate).numpy())
o1 = {"res": rng.standard_normal((B, LT, 24)).astype(np.float32), "op": rng.standard_normal((B, LT, 8)).astype(np.float32)}
o2 = {k: v.copy() for k, v in o1.items()}
ng = [(0, 3, "res", 5), (2, 7, "op", 1), (5, 31, "res", 23)]
H._nogood_apply(o1, ng); nogood_apply(o2, ng)
check("_nogood_apply", np.concatenate([o1["res"].ravel(), o1["op"].ravel()]), np.concatenate([o2["res"].ravel(), o2["op"].ravel()]))
# 4. _token_step's sentence mask, the two lines verbatim
k0 = hsrc.index("    same = (sent.reshape(B, 1, T, 1)"); k1 = hsrc.index("\n", hsrc.index("    ok = same *", k0))
lines = textwrap.dedent(hsrc[k0:k1])
sent_t = Tensor(se_np); tk_t = Tensor((rng.random((B, T)) > 0.2).astype(np.float32))
g = {"sent": sent_t, "tokmask": tk_t, "B": B, "T": T}
exec(lines, g)
check("_token_step sentence mask", g["ok"].numpy(), same_sentence(sent_t, tk_t, B, T).numpy())
print(f"[bridge-id] {'ALL PASS' if not fails else f'{fails} FAIL'}")
sys.exit(1 if fails else 0)
