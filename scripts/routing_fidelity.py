"""routing_fidelity.py — GUT #47's probe (2026-07-22): PHOTOGRAPH THE
MESSAGES. The head's per-slot attention (fat) IS the learned message
routing table; the gold fspans ARE the true graph's edges. FIDELITY =
attention mass landing inside the slot's own gold span. PINNED (i):
fidelity higher on answered-correct than refused; if refusals show
INTACT fidelity, the failure is post-routing (the digit banks) and the
width tax splits. The probe photographs routing; it never steers it."""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ["ALG_FTYPES"] = "8"
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import build_params, forward
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

z = np.load(".cache/phase1_alg_states_bigtest.npz")
st, tk, se = z["states"], z["tokmask"], z["sent"]
g_pres, g_fspan = z["g_presence"], z["g_fspan"]
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
votes = json.load(open(".cache/lattice_gen16_V4.json"))["bigtest"]
def maj(v):
    vs=[x for x in v if x is not None]
    return Counter(vs).most_common(1)[0] if vs else (None,0)

p = build_params(0)
sd = safe_load(".cache/crown_reader_v4.safetensors")
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

fid = np.zeros(1500)
for s0 in range(0, 1500, 8):
    sl = np.arange(s0, min(s0+8, 1500))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    out = forward(p, Tensor(st[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(tk[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(se[sl_p].astype(np.int32), dtype=dtypes.int))
    fat = out["fat"].realize().numpy()          # (B, L_FAC, T)
    for bi, i in enumerate(sl):
        i = int(i)
        pres = g_pres[i] > 0
        if not pres.any(): continue
        sp = g_fspan[i]                          # (L_FAC, T)
        m = []
        for j in np.where(pres)[0]:
            s_ = sp[j]
            if s_.sum() > 0:
                m.append(float((fat[bi, j] * s_).sum()))
        fid[i] = np.mean(m) if m else np.nan
ok = np.array([maj(votes[i])[0] == gold[i] for i in range(1500)])
va = ~np.isnan(fid)
fa, fr = fid[va & ok], fid[va & ~ok]
print(f"[routing] fidelity (attention mass in gold span, mean over slots):")
print(f"  answered-correct (n={len(fa)}): {fa.mean():.3f}  median {np.median(fa):.3f}")
print(f"  refused/wrong    (n={len(fr)}): {fr.mean():.3f}  median {np.median(fr):.3f}")
d = fa.mean() - fr.mean()
verdict = ("FIDELITY TRACKS SUCCESS — routing degrades on failures" if d > 0.05 else
           "INTACT ROUTING ON FAILURES — the wall is POST-ROUTING (the digit banks); the width tax splits" if abs(d) <= 0.05 else "inverted(?)")
print(f"  delta {d:+.3f} -> {verdict}")
json.dump({"acc_mean": float(fa.mean()), "ref_mean": float(fr.mean()),
           "delta": float(d), "verdict": verdict,
           "fid": fid.tolist()}, open(".cache/routing_fidelity.json","w"))
print("[routing] banked")
