"""eq_check.py — THE EQUIVALENCE GATE (2026-08-31): before dead code
leaves the branch, the cleaned module must reproduce the current module's
forward EXACTLY on fixed inputs for every LIVING config, on banked
weights. Env: EQ_CFG in {A, B, C}; EQ_OUT = npz path.
  A = BINDBUS=3 D=256 (the bus incumbent 7c's language)
  B = BINDBUS=7 D=512 (the family line, 10n)
  C = BINDBUS=7 + BUSGARAGE=2 + SHELF_CIRCLE=1 (living pressure config, 12r)
"""
import os, sys, json
CFG = os.environ["EQ_CFG"]
base = {"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
        "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
        "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
        "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl", "ALG_TEST_NAME": "bigtest"}
if CFG == "A":
    base.update({"ALG_BINDBUS": "3", "ALG_BIND_D": "256",
                 "BIND_CODES": ".cache/bindbus_codes256.npz"})
    CKPT = ".cache/sharp_bind7c.safetensors"
elif CFG == "B":
    base.update({"ALG_BINDBUS": "7", "ALG_BIND_D": "512",
                 "BIND_CODES": ".cache/bindbus_codes512.npz"})
    CKPT = ".cache/sharp_bind10n.safetensors"
else:
    base.update({"ALG_BINDBUS": "7", "ALG_BIND_D": "512",
                 "BIND_CODES": ".cache/bindbus_codes512.npz",
                 "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "1"})
    CKPT = ".cache/sharp_bind12r.safetensors"
os.environ.update(base)
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, T_ALG, TOKENIZER_JSON,
                                 sent_indices, load_alg, build_slot_masks)
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys()), \
    (sorted(set(sd) - set(p)), sorted(set(p) - set(sd)))
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
rows = [json.loads(l) for l in open('.cache/algebra_nl_test.jsonl')][:8]
ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
snt = np.zeros((8, T_ALG), np.int32)
for i, r in enumerate(rows):
    t = r.get('text') or r.get('original'); e = tok.encode(t)
    Ln = min(len(e.ids), T_ALG)
    ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
    snt[i] = sent_indices(t, list(e.offsets), msk[i])
st = Tensor(np.asarray(recompute_states(ids)).astype(np.float32), dtype=dtypes.float)
tk = Tensor(msk, dtype=dtypes.float)
se = Tensor(snt.astype(np.int32), dtype=dtypes.int)
o0 = forward(p, st, tk, se)
onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
mk = build_slot_masks(onp0, snt)
o = forward(p, st, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
out = {k: o[k].realize().numpy() for k in
       ("pres", "ftype", "op", "args", "res", "query", "bind")
       if k in o}
np.savez(os.environ["EQ_OUT"], **out)
print(f"[eq {CFG}] dumped {sorted(out)} -> {os.environ['EQ_OUT']}")
