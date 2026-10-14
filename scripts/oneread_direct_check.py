"""oneread_direct_check.py -- OR_241's gate, THE DIRECT CHECK (ledger 2026-10-12 18:36,
"THE ENGINE IS MUDDY"): under ALG_ONE_READ=1, verify by direct instrumentation (not
inference from a loss number) that (1) the bank's attention is never computed at loop
breaths and (2) h_tok at every breath equals h_tok at breath 0, bitwise, for every slot.

(1) is checked by monkeypatching phase1_algebra_head._make_bank so the bank closure it
returns counts its own calls; under ALG_ONE_READ=1 the only calls left are the THREE
breath-0 grounding reads (vq, fq, qq) -- forward()'s own `bank(p["vq"], ...)`,
`bank(p["fq"], ..., flat=...)` (breath 0), `bank(p["qq"], 1)` -- never the per-loop-breath
fq re-read (phase1_algebra_head.py's breath_step, the `else: h_tok, fat_cur = bank(...)`
branch, now skipped when ALG_ONE_READ or ctx["tok_override"] is set).

(2) is checked via the existing _CENSUS hook (armed here exactly as port_census.py arms
it): breath_step and forward() both tap `("h_tok", kb)` -> h_tok.realize().numpy()
unconditionally when _CENSUS is not None (phase1_algebra_head.py, forward()'s breath-0
tap right after the busreg/IMP globals section, and breath_step's own tap right after
`state["fat_cur"] = fat_cur`). A direct numpy array_equal between kb=0's array and every
kb>=1 array is the proof; under ALG_ONE_READ=0 (role8) the arrays legitimately DIFFER
breath to breath (the live re-read) -- this script checks the ALG_ONE_READ=1 config only.

Fixture: tiny64 (ALG_TRAIN/ALG_TEST env already set by the caller -- the gate's own $COMMON
string), DEV=CPU, a real two-pass forward (fact_buf + slot_mask) so the breath loop
actually executes (the identity instrument's blind spot, 2026-10-11: a forward() call
without slot_mask never enters the loop at all).
"""
import os
import sys

assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r} -- CPU-only script"
assert int(os.environ.get("ALG_ONE_READ", "0")) == 1, "this script's checks are specific to ALG_ONE_READ=1"

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                 alt2_fact_buf, K_VARS)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

# --- instrument (1): count real bank() calls, by wrapping _make_bank's returned closure ---
_bank_call_count = [0]
_orig_make_bank = H._make_bank
def _counting_make_bank(*a, **kw):
    inner = _orig_make_bank(*a, **kw)
    def wrapped(*a2, **kw2):
        _bank_call_count[0] += 1
        return inner(*a2, **kw2)
    return wrapped
H._make_bank = _counting_make_bank

# --- instrument (2): arm the existing _CENSUS hook (same idiom port_census.py uses) ---
H._CENSUS = []

N = int(os.environ.get("ODC_N", "4"))
vs, vst, vtk, vg, vse = load_alg("test")
n = min(N, len(vs))
p = build_params(0)
ckpt = os.environ.get("ODC_CKPT")
if ckpt:
    sd = safe_load(ckpt)
    assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(p))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

sl = np.arange(n)
nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl])
ma = np.array([vs[int(i)].get("m", 0) for i in sl])
se_np = vse[sl].astype(np.int32)
ts = Tensor(np.ascontiguousarray(vst[sl]), dtype=dtypes.half)
tk = Tensor(vtk[sl].astype(np.float32))
se = Tensor(se_np, dtype=dtypes.int)

o0 = forward(p, ts, tk, se)
onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
mk = build_slot_masks(onp0, se_np)
_oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
fb = alt2_fact_buf(_oa, se_np, nv, ma)
fact_t = Tensor(fb, dtype=dtypes.float)

_bank_call_count[0] = 0   # only count the SECOND (real, slot_mask-bearing) pass
H._CENSUS = []
out = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fact_t)
K_B = int(os.environ.get("ALG_BREATH", "1"))

n_bank_calls = _bank_call_count[0]
expected = 3   # vq, fq(breath0), qq -- exactly once each, zero loop-breath fq re-reads
print(f"[direct-check] bank() call count (the real pass, K_B={K_B}): {n_bank_calls} (expected {expected})")
print("[direct-check] BANK-CALL-COUNT " + ("PASS" if n_bank_calls == expected else "FAIL"))

h_tok_by_kb = {}
for kb, name, arr in H._CENSUS:
    if name == "h_tok":
        h_tok_by_kb.setdefault(kb, arr)
kbs = sorted(h_tok_by_kb)
print(f"[direct-check] h_tok tapped at breaths {kbs}")
assert 0 in h_tok_by_kb, "no breath-0 h_tok tap -- the census hook did not fire"
base = h_tok_by_kb[0]
all_equal = True
for kb in kbs:
    if kb == 0:
        continue
    eq = np.array_equal(base, h_tok_by_kb[kb])
    maxdiff = float(np.abs(base - h_tok_by_kb[kb]).max())
    print(f"[direct-check] h_tok[kb={kb}] vs h_tok[kb=0]: array_equal={eq} maxdiff={maxdiff:.3e}")
    all_equal = all_equal and eq
print("[direct-check] H-TOK-CARRY " + ("PASS" if (all_equal and len(kbs) > 1) else "FAIL"))
H._make_bank = _orig_make_bank
