"""THE LOCUS READ (2026-09-14): the fingerpost's certificate as the NL loop's
mask. Per wild row: parse the original and four sentence-permuted views (the
trunk recomputed per view); a slot of the original is UNSTABLE if its coarse
factor key (ftype, op, value) is absent from the view's decoded set in >= LR_K
of the 4 views; the unstable slots are melted at the state entering breath
LR_KB (the wheel's _melt organ, ALG_WHEEL_MELT sets the amplitude) and the
row is re-read. Reports: open vs locus-melt fac-exact (paired per slot), the
locus's precision (is an unstable slot more often wrong?), and coverage.
Env: family env + ALG_WHEEL_MELT, LR_CKPT, ALG_TEST(_NAME), LR_K (2), LR_KB (4)."""
import os, sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, decode, sent_indices, L_FAC, K_VARS, T_ALG, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
assert H._WHEEL_MELT is not None, "set ALG_WHEEL_MELT (the melt amplitude)"
LR_K = int(os.environ.get("LR_K", "2")); LR_KB = int(os.environ.get("LR_KB", "4"))
vs, vst, vtk, vg, vse = load_alg("test"); N = len(vs)
p = build_params(0); sd = safe_load(os.environ["LR_CKPT"]); assert set(sd) == set(p)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
tok = Tokenizer.from_file(TOKENIZER_JSON)
KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())

def two_pass(ts, tk, se, nv, ma):
    o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, se.numpy())
    _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
    fb = Tensor(alt2_fact_buf(_oa, se.numpy(), nv, ma), dtype=dtypes.float)
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fb)
    return {k: o[k].numpy() for k in KEYS}

def slot_keys(onp, b):
    """coarse factor key per present slot: (ftype-class, op, value) — pointers excluded (numbering varies across views)"""
    row = {k: onp[k][b] for k in onp}; row["query"] = np.zeros(K_VARS, np.float32); keys = {}
    for j in range(L_FAC):
        if row["pres"][j] <= 0: continue
        rj = dict(row); pr = np.full_like(row["pres"], -1.0); pr[j] = row["pres"][j]; rj["pres"] = pr
        try: facs, _ = decode(rj)
        except Exception: facs = []
        keys[j] = tuple(sorted((f["ftype"], f.get("op", ""), f.get("value", "")) for f in facs))
    return keys

def score(onp, b, i):
    out = {}
    for j in range(L_FAC):
        if vg["presence"][i, j] < 0.5: continue
        ok = (onp["pres"][b, j] > 0) and int(onp["ftype"][b, j].argmax()) == vg["ftype"][i, j] and int(onp["res"][b, j].argmax()) == vg["res"][i, j]
        if vg["ftype"][i, j] == 0:
            ok = ok and int(onp["op"][b, j].argmax()) == vg["op"][i, j]
            gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
            ok = ok and ((bool(onp["dup"][b, j] > 0) and int(np.argmax(onp["args"][b, j])) in gset) if (len(gset) == 1 and "dup" in onp) else set(np.argsort(-onp["args"][b, j])[:2].tolist()) == gset)
        else:
            ok = ok and bool((onp["dig"][b, j].argmax(-1) == vg["digits"][i, j]).all())
        out[j] = bool(ok)
    return out

open_ok, melt_ok, unstable, present = [], [], [], []
n_unst = n_pres = 0
for s0 in range(0, N, 8):
    sl = np.arange(s0, min(s0 + 8, N)); pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
    ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    H._LOCUS = None
    o_orig = two_pass(ts, tk, se, nv, ma); keys0 = [slot_keys(o_orig, b) for b in range(8)]
    miss = np.zeros((8, L_FAC))
    for k in range(1, 5):
        texts = [permuted_view(vs[int(i)]["text"], 40000 + 10 * int(i) + k) for i in sl_p]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32); snt = np.zeros((8, T_ALG), np.int32)
        for b, t in enumerate(texts):
            e = tok.encode(t); Ln = min(len(e.ids), T_ALG); ids[b, :Ln] = e.ids[:Ln]; msk[b, :Ln] = 1.0; snt[b] = sent_indices(t, list(e.offsets), msk[b])
        stv = recompute_states(ids)
        o_v = two_pass(Tensor(stv, dtype=dtypes.half), Tensor(msk), Tensor(snt, dtype=dtypes.int), nv, ma)
        for b in range(8):
            kv = list(slot_keys(o_v, b).values())
            for j, kj in keys0[b].items():
                if kj not in kv: miss[b, j] += 1
                else: kv.remove(kj)
    melt = (miss >= LR_K).astype(np.float32)
    H._LOCUS = {"melt": np.concatenate([melt, np.zeros((0, H.L_TOT - L_FAC))], 1) if H.L_TOT == L_FAC else np.concatenate([melt, np.zeros((8, H.L_TOT - L_FAC), np.float32)], 1), "kb": LR_KB}
    o_melt = two_pass(ts, tk, se, nv, ma); H._LOCUS = None
    for b, i in enumerate(sl):
        i = int(i); so = score(o_orig, b, i); sm = score(o_melt, b, i)
        for j in so:
            open_ok.append(so[j]); melt_ok.append(sm[j]); unstable.append(bool(melt[b, j] > 0)); present.append((i, j))
        n_pres += len(so); n_unst += int(sum(melt[b, j] > 0 for j in so))
    if s0 % 80 == 0: print(f"[locus] {s0}/{N} rows", flush=True)
o = np.array(open_ok); m = np.array(melt_ok); u = np.array(unstable); n = len(o)
d = m.mean() - o.mean(); only_o = int((o & ~m).sum()); only_m = int((~o & m).sum()); z = (only_m - only_o) / max(np.sqrt(only_m + only_o), 1)
name = os.path.basename(os.environ["LR_CKPT"]).replace("sharp_", "").replace(".safetensors", "")
print(f"[locus-read] {name} on {os.environ.get('ALG_TEST_NAME','?')}: {n} slots; unstable (>= {LR_K}/4 views) {u.mean():.3f} of slots; melt at breath {LR_KB} amp {H._WHEEL_MELT}")
print(f"[locus-read] open {o.mean():.4f} -> locus-melt {m.mean():.4f} (diff {d:+.4f}; gained {only_m} / lost {only_o}; McNemar z {z:+.2f})")
print(f"[locus-read] the locus's precision: P(wrong | unstable) = {(~o[u]).mean() if u.any() else float('nan'):.3f} vs P(wrong | stable) = {(~o[~u]).mean() if (~u).any() else float('nan'):.3f}; unstable slots that the melt fixed {int((u & ~o & m).sum())}, broke {int((u & o & ~m).sum())}; stable slots changed {int((~u & (o != m)).sum())}")
