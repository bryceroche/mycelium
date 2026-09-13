"""THE HAPPY-FAMILY READ (2026-09-13, the atlas revival rung 2): per row, the
variance-normalized z-radius of its mean-pooled breath states to the NEAREST
atlas cell (cosine-nearest, then ||s - mu|| / sqrt(mean var)); does tight
membership predict a correct parse? Reports the AUROC of "row correct" vs
-z (per breath and pooled), the z-radius quantiles by outcome, and the class
histogram of nearest cells. A read: no training, no supervision.
Env: family env + ALG_MINE_BREATHS=1, HF_CKPT, HF_ATLAS, ALG_TEST(_NAME), HF_N."""
import os, sys, numpy as np
os.environ.setdefault("ALG_MINE_BREATHS", "1")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, L_FAC, K_VARS
from mycelium.step_atlas import load_atlas
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

def auroc(score, label):
    """rank-based AUROC of label==1 having higher score (ties averaged)."""
    order = np.argsort(score); ranks = np.empty(len(score)); s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    pos = label == 1; n1 = pos.sum(); n0 = (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

vs, vst, vtk, vg, vse = load_alg("test")
N = min(int(os.environ.get("HF_N", "512")), len(vs)); rng = np.random.RandomState(0); idx = np.sort(rng.choice(len(vs), N, replace=False))
p = build_params(0); sd = safe_load(os.environ["HF_CKPT"])
assert set(sd) == set(p), (sorted(set(sd) - set(p))[:3], sorted(set(p) - set(sd))[:3])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
atlas = load_atlas(os.environ.get("HF_ATLAS", ".cache/step_atlas_current.npz"), manifest_path=os.environ.get("MH_ATLAS_MANIFEST", ".cache/RESEARCH_MANIFEST.json"))   # the research era anchor (the miner writes it)
M, V, C = atlas["means"], atlas["vars"], atlas["counts"]          # (K, ncls, D)
K = M.shape[0]
zs = np.full((N, K), np.nan); cls = np.full((N, K), -1); okrow = np.zeros(N, bool); okfrac = np.zeros(N)
for s0 in range(0, N, 8):
    sl = idx[s0:s0 + 8]; pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
    _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
    _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
    fb = Tensor(alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma), dtype=dtypes.float)
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fb)
    br = [b.numpy().mean(1) for b in o["breaths_all"]]                # (B, D) mean-pooled over slots, the miner's idiom
    onp = {k: o[k].numpy() for k in ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in o else ())}
    for bi, i in enumerate(sl):
        i = int(i); ok = True; any_slot = False; n_s = 0; n_ok = 0
        for j in range(L_FAC):
            if vg["presence"][i, j] < 0.5: continue
            any_slot = True; n_s += 1
            sok = (onp["pres"][bi, j] > 0) and int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j] and int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
            if vg["ftype"][i, j] == 0:
                sok = sok and int(onp["op"][bi, j].argmax()) == vg["op"][i, j]
                gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                sok = sok and ((bool(onp["dup"][bi, j] > 0) and int(np.argmax(onp["args"][bi, j])) in gset) if (len(gset) == 1 and "dup" in onp) else set(np.argsort(-onp["args"][bi, j])[:2].tolist()) == gset)
            else:
                sok = sok and bool((onp["dig"][bi, j].argmax(-1) == vg["digits"][i, j]).all())
            ok = ok and sok; n_ok += int(sok)
        okrow[s0 + bi] = ok and any_slot; okfrac[s0 + bi] = n_ok / max(n_s, 1)
        for kb in range(min(K, len(br))):
            s = br[kb][bi]; live = C[kb] > 0
            if not live.any(): continue
            Ml = M[kb][live]; sim = (Ml / (np.linalg.norm(Ml, axis=1, keepdims=True) + 1e-9)) @ (s / (np.linalg.norm(s) + 1e-9))
            c = int(np.argmax(sim)); ci = np.where(live)[0][c]
            zs[s0 + bi, kb] = np.linalg.norm(s - M[kb][ci]) / (np.sqrt(V[kb][ci].mean()) + 1e-9); cls[s0 + bi, kb] = ci
name = os.path.basename(os.environ["HF_CKPT"]).replace("sharp_", "").replace(".safetensors", "")
half = okfrac >= 0.5
print(f"[happy-family] {name} on {os.environ.get('ALG_TEST_NAME','?')} N={N}: rows fully correct {okrow.mean():.3f}, rows >= half correct {half.mean():.3f}; atlas cells live per breath {[int((C[k] > 0).sum()) for k in range(K)]}")
for kb in range(K):
    z = zs[:, kb]; m = ~np.isnan(z)
    if m.sum() < 10: continue
    print(f"  breath {kb}: AUROC(>=half | tighter) = {auroc(-z[m], half[m].astype(int)):.3f} | AUROC(full) = {auroc(-z[m], okrow[m].astype(int)):.3f} | z median >=half {np.median(z[m & half]) if (m & half).any() else float('nan'):.2f} vs below {np.median(z[m & ~half]) if (m & ~half).any() else float('nan'):.2f} | nearest-class hist {np.bincount(cls[m, kb], minlength=M.shape[1]).tolist()}")
zp = np.nanmean(zs, axis=1); m = ~np.isnan(zp)
print(f"  pooled over breaths: AUROC(>=half) = {auroc(-zp[m], half[m].astype(int)):.3f}, AUROC(full) = {auroc(-zp[m], okrow[m].astype(int)):.3f}  (bar >= 0.65 on >=half = a certificate; < 0.55 = not a compass on this register)")
