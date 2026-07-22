"""band_bars.py — the band-dose probe's photometers (2026-07-22).
TARGET: param-path digit accuracy on alg4test (mag-2 baseline 0.903 ->
bar 0.93; mag-3 baseline 0.837 -> bar 0.88). GUARD: bigtest >= 1208."""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import defaultdict
os.environ.setdefault("ALG2", "1"); os.environ["ALG_FTYPES"] = "8"
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import build_params, forward, decode, N_DIG, L_FAC
from tta_alg2_dials import solve2
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

p = build_params(0)
sd = safe_load(".cache/band_patch.safetensors")
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

# TARGET: the digit curve on alg4test (param path)
z = np.load(".cache/phase1_alg_states_alg4test.npz")
st, tk, se = z["states"], z["tokmask"], z["sent"]
g_pres, g_ft, g_dig = z["g_presence"], z["g_ftype"], z["g_digits"]
FT_GIVEN, FT_MOD, FT_PCT, FT_FDIV = 1, 2, 4, 5
slot_acc = defaultdict(lambda: np.zeros(2))
n = len(st)
for s0 in range(0, n, 8):
    sl = np.arange(s0, min(s0+8, n))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    out = forward(p, Tensor(st[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(tk[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(se[sl_p].astype(np.int32), dtype=dtypes.int))
    dig = out["dig"].realize().numpy()
    for bi, i in enumerate(sl):
        i = int(i)
        for j in range(L_FAC):
            if g_pres[i, j] <= 0: continue
            ft = int(g_ft[i, j])
            path = "given" if ft == FT_GIVEN else ("param" if ft in (FT_MOD, FT_FDIV, FT_PCT) else None)
            if path is None: continue
            gold_d = g_dig[i, j]
            val = int(sum(d * 10 ** (N_DIG - 1 - k_) for k_, d in enumerate(gold_d)))
            mag = len(str(max(val, 0))) if val > 0 else 1
            pred = dig[bi, j].argmax(-1)
            slot_acc[(path, mag)] += (int((pred == gold_d).all()), 1)
for path in ("given", "param"):
    for mag in (1, 2, 3):
        t = slot_acc[(path, mag)]
        if t[1]: print(f"[band] {path} mag-{mag}: {t[0]/t[1]:.3f} (n={int(t[1])})")
p2 = slot_acc[("param", 2)]; p3 = slot_acc[("param", 3)]
a2 = p2[0]/max(p2[1],1); a3 = p3[0]/max(p3[1],1)
target = (a2 >= 0.93) or (a3 >= 0.88)
print(f"[band] TARGET (mag2>=0.93 [was .903] or mag3>=0.88 [was .837]): "
      f"{a2:.3f}/{a3:.3f} -> {'HIT' if target else 'MISS'}")

# GUARD: bigtest
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
zb = np.load(".cache/phase1_alg_states_bigtest.npz")
stb, tkb, seb = zb["states"], zb["tokmask"], zb["sent"]
n_ans = 0
for s0 in range(0, 1500, 8):
    sl = np.arange(s0, min(s0+8, 1500))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    out = forward(p, Tensor(stb[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(tkb[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(seb[sl_p].astype(np.int32), dtype=dtypes.int))
    keys = [k for k in ("pres","ftype","op","islit","dig","dig2","args","res","query","sel","dup","y") if k in out]
    o = {k: out[k].realize().numpy() for k in keys}
    for bi, i in enumerate(sl):
        i = int(i)
        facs, q = decode({k: o[k][bi] for k in o})
        a_ = solve2(facs, q, {"n_vars": 24, "m": rows[i].get("m", 60)})
        n_ans += (a_ == rows[i]["solution"][rows[i]["query_var"]])
guard = n_ans >= 1208
print(f"[band] GUARD bigtest: {n_ans} ({'HELD' if guard else 'BROKEN'} vs 1208)")
verdict = ("SPECTRAL REPAIR — the band takes its medicine without waking the neighbors"
           if target and guard else
           "COUPLED — the coupling law banks" if not target and not guard else
           "TARGET HIT, GUARD BROKEN — repair at displacement cost" if target else
           "GUARD HELD, TARGET MISSED — the dose was too weak or the band too stiff")
print(f"[band] VERDICT: {verdict}")
json.dump({"param_mag2": a2, "param_mag3": a3, "bigtest": n_ans,
           "verdict": verdict}, open(".cache/band_bars.json","w"))
print("[band] banked")
