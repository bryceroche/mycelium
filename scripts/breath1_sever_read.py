"""breath1_sever_read.py — THE BREATH-1 SEVER (2026-09-16, registered after
the visibility read: 67% of the intake breath's step is content the readout
cannot see). Is that null-space write working memory or waste? Two passes
per batch: A. the open loop, capturing s0 and s1 (breaths_all) and the
per-row visible basis Q (the readout's row space on the content planes);
B. the same rows with the null-content part of breath 1's update REMOVED
at the entry of breath 2 (the head's _IMP kick: cur += -n, n = (I - QQ^T)
(s1 - s0) on the content planes), then decoded. Paired slot-exact A vs B;
a large drop = the intake's invisible write is load-bearing (memory the
later breaths read); no drop = waste (a road to close). A read. Env: family
env + ALG_MINE_BREATHS=1 (set), PV_CKPT, ALG_TEST(_NAME), PV_N, LV_PER_SLOT-
style output .cache/ps_b1sever_<name>.npz"""
import os, sys


def _main():
    import numpy as np
    os.environ.setdefault("ALG_MINE_BREATHS", "1")
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    import phase1_algebra_head as H
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, L_FAC, K_VARS
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    vs, vst_, vtk, vg, vse = load_alg("test"); N = min(int(os.environ.get("PV_N", "311")), len(vs)); rng = np.random.RandomState(0); idx = np.sort(rng.choice(len(vs), N, replace=False))
    p = build_params(0); sd = safe_load(os.environ["PV_CKPT"]); assert set(sd) == set(p)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    g = H._polar_sink()[2].numpy().reshape(-1).astype(np.float64) if H.ALG_POLAR else np.ones(H.H_W); content = g > 0.5
    Hm = np.concatenate([p[k].numpy().astype(np.float64).reshape(H.H_W, -1) for k in ("h_pres", "h_ftype", "h_op", "h_dig", "h_islit", "h_dup", "h_sgn", "h_sel") if k in p], axis=1)
    ptr = [p[k].numpy().astype(np.float64) for k in ("W_args", "W_res", "W_query") if k in p]
    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())

    def score(onp, bi, i):
        out = {}
        for j in range(L_FAC):
            if vg["presence"][i, j] < 0.5: continue
            rel = vg["ftype"][i, j] == 0
            ok = (onp["pres"][bi, j] > 0) and int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j] and int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
            if rel:
                ok = ok and int(onp["op"][bi, j].argmax()) == vg["op"][i, j]; gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                ok = ok and ((bool(onp["dup"][bi, j] > 0) and int(np.argmax(onp["args"][bi, j])) in gset) if (len(gset) == 1 and "dup" in onp) else set(np.argsort(-onp["args"][bi, j])[:2].tolist()) == gset)
            else:
                ok = ok and bool((onp["dig"][bi, j].argmax(-1) == vg["digits"][i, j]).all())
            out[j] = bool(ok)
        return out

    rows_, slots_, okA, okB = [], [], [], []; null_share = []
    for s0 in range(0, N, 8):
        sl = idx[s0:s0 + 8]; pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst_[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32)); _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fb = Tensor(alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma), dtype=dtypes.float); mkT = Tensor(mk, dtype=dtypes.float)
        H._IMP = None
        oA = forward(p, ts, tk, se, slot_mask=mkT, fact_buf=fb); onpA = {k: oA[k].numpy() for k in KEYS}
        S0 = oA["breaths_all"][0].numpy().astype(np.float64); S1 = oA["breaths_all"][1].numpy().astype(np.float64); V = oA["vst_mine"].numpy().astype(np.float64)
        kick = np.zeros((8, H.L_TOT, H.H_W), np.float32)
        for bi in range(8):
            M = np.concatenate([Hm] + [W @ V[bi, :int(_nv[bi])].T for W in ptr], axis=1) * content[:, None]; Q, R = np.linalg.qr(M)
            dx = (S1[bi] - S0[bi]) * content; vis = (dx @ Q) @ Q.T; nul = dx - vis
            kick[bi, :L_FAC] = -nul.astype(np.float32)
            null_share.append(float((nul ** 2).sum() / ((S1[bi] - S0[bi]) ** 2).sum().clip(1e-12)))
        H._IMP = (2, Tensor(kick))                                  # at the entry of breath 2: breath 1's invisible write removed
        oB = forward(p, ts, tk, se, slot_mask=mkT, fact_buf=fb); onpB = {k: oB[k].numpy() for k in KEYS}; H._IMP = None
        for bi, i in enumerate(sl):
            i = int(i); a = score(onpA, bi, i); b = score(onpB, bi, i)
            for j in a: rows_.append(i); slots_.append(j); okA.append(a[j]); okB.append(b[j])
        if s0 % 160 == 0: print(f"[b1-sever] {s0 + len(sl)}/{N}", flush=True)
    a = np.array(okA); b = np.array(okB); only_a = int((a & ~b).sum()); only_b = int((~a & b).sum()); z = (only_b - only_a) / max(np.sqrt(only_a + only_b), 1)
    name = os.path.basename(os.environ["PV_CKPT"]).replace("sharp_", "").replace(".safetensors", "")
    print(f"[b1-sever] {name} on {os.environ.get('ALG_TEST_NAME', '?')}: {len(a)} slots; breath 1's null-content share of the step (median) {np.median(null_share):.3f}")
    print(f"[b1-sever] open {a.mean():.4f} -> breath-1 invisible write removed {b.mean():.4f} (diff {b.mean() - a.mean():+.4f}; gained {only_b} / lost {only_a}; McNemar z {z:+.2f})")
    np.savez(f".cache/ps_b1sever_{name}_{os.environ.get('ALG_TEST_NAME', 'x')}.npz", rows=np.array(rows_), slots=np.array(slots_), okA=a, okB=b)


if __name__ == "__main__":
    _main()
