"""trajectory_read.py — THE TRAJECTORY READ (2026-09-14, Bryce's gut: "MLIR,
min(max(step_size)); know where in state space the start and finish is;
baby steps toward the target"). Per row, the loop's state path s_0..s_K:
  step_k      ||s_k - s_{k-1}|| / ||s_{k-1}||           (the step size)
  cos_k       cos(s_k - s_{k-1}, s_{k+1} - s_k)          (direction: march or thrash)
  progress_k  1 - ||s_k - s_K|| / ||s_0 - s_K||          (fraction of the way to the finish)
  field_k     per-field decode accuracy at breath k (the dish line: pres/ftype/res/op/args/dig)
split by rows >= half correct vs not. A read: no training. Env: family env +
PV_CKPT, ALG_TEST(_NAME), PV_N."""
import os, sys


def _main():
    import numpy as np
    os.environ.setdefault("ALG_MINE_BREATHS", "1")
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, L_FAC, K_VARS
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    vs, vst, vtk, vg, vse = load_alg("test")
    N = min(int(os.environ.get("PV_N", "512")), len(vs)); rng = np.random.RandomState(0); idx = np.sort(rng.choice(len(vs), N, replace=False))
    p = build_params(0); sd = safe_load(os.environ["PV_CKPT"]); assert set(sd) == set(p)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    K = None; steps = []; coss = []; prog = []; fields = {f: [] for f in ("pres", "ftype", "res", "op", "args", "dig")}; half = []
    for s0 in range(0, N, 8):
        sl = idx[s0:s0 + 8]; pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma), dtype=dtypes.float))
        br = [b.numpy().astype(np.float64) for b in o["breaths_all"]]; hd = [{k: v.numpy() for k, v in h.items() if k in fields} for h in o["heads_all"]]
        if int(os.environ.get("PV_CONTENT", "0")):
            # THE CONTENT-ONLY PATH (2026-09-15, after the visibility read: 85% of the middle breaths'
            # motion is the clock planes' rotation) — the clock planes are zeroed before the geometry
            import phase1_algebra_head as _H
            _g = _H._polar_sink()[2].numpy().reshape(-1).astype(np.float64)
            br = [b * _g for b in br]
        if K is None: K = len(br); print(f"[traj] K={K} N={N} ckpt={os.path.basename(os.environ['PV_CKPT'])}", flush=True)
        b_n = len(sl)
        for bi in range(b_n):
            i = int(sl[bi]); S = [b[bi].reshape(-1) for b in br]          # K states (L*D,)
            d = [S[k] - S[k - 1] for k in range(1, K)]
            steps.append([np.linalg.norm(d[k - 1]) / (np.linalg.norm(S[k - 1]) + 1e-9) for k in range(1, K)])
            coss.append([float(d[k] @ d[k + 1] / (np.linalg.norm(d[k]) * np.linalg.norm(d[k + 1]) + 1e-9)) for k in range(K - 2)])
            tot = np.linalg.norm(S[0] - S[-1]) + 1e-9
            prog.append([1 - np.linalg.norm(S[k] - S[-1]) / tot for k in range(K)])
            pres = vg["presence"][i] >= 0.5; n_p = max(pres.sum(), 1); rel = pres & (vg["ftype"][i] == 0); giv = pres & (vg["ftype"][i] != 0)
            row_ok = []
            for k in range(K):
                h = hd[k]; f = {}
                f["pres"] = ((h["pres"][bi] > 0) == pres)[pres].mean()
                f["ftype"] = (h["ftype"][bi].argmax(-1) == vg["ftype"][i])[pres].mean()
                f["res"] = (h["res"][bi].argmax(-1) == vg["res"][i])[pres].mean()
                f["op"] = (h["op"][bi].argmax(-1) == vg["op"][i])[rel].mean() if rel.any() else np.nan
                aok = np.array([set(np.argsort(-h["args"][bi, j])[:2].tolist()) == set(np.where(vg["args"][i, j] > .5)[0].tolist()) for j in range(L_FAC)])
                f["args"] = aok[rel].mean() if rel.any() else np.nan
                f["dig"] = (h["dig"][bi].argmax(-1) == vg["digits"][i]).all(-1)[giv].mean() if giv.any() else np.nan
                for kk in fields: fields[kk].append((k, f[kk]))
                if k == K - 1:
                    sok = (h["pres"][bi] > 0) & (h["ftype"][bi].argmax(-1) == vg["ftype"][i]) & (h["res"][bi].argmax(-1) == vg["res"][i]) & np.where(rel, (h["op"][bi].argmax(-1) == vg["op"][i]) & aok, (h["dig"][bi].argmax(-1) == vg["digits"][i]).all(-1))
                    half.append((sok & pres).sum() / n_p >= 0.5)
        if s0 % 160 == 0: print(f"[traj] {s0 + b_n}/{N}", flush=True)
    st = np.array(steps); cs = np.array(coss); pr = np.array(prog); hf = np.array(half)
    name = os.path.basename(os.environ["PV_CKPT"]).replace("sharp_", "").replace(".safetensors", "")
    print(f"[traj-read] {name} on {os.environ.get('ALG_TEST_NAME', '?')}: N={len(hf)} rows, >= half correct {hf.mean():.3f}")
    for lab, m in (("all", np.ones_like(hf)), (">= half", hf), ("< half", ~hf)):
        if m.sum() < 5: continue
        print(f"  [{lab:7s} n={int(m.sum()):3d}] step  " + " ".join(f"b{k + 1}:{np.median(st[m][:, k]):.3f}" for k in range(K - 1)))
        print(f"  [{lab:7s}      ] cos   " + " ".join(f"b{k + 1}>{k + 2}:{np.median(cs[m][:, k]):+.3f}" for k in range(K - 2)) + "   (+ = march, - = thrash)")
        print(f"  [{lab:7s}      ] prog  " + " ".join(f"b{k}:{np.median(pr[m][:, k]):.3f}" for k in range(K)))
    print("  the dish line (median per-field accuracy per breath):")
    for f, vals in fields.items():
        A = np.full((len(hf), K), np.nan)
        for n_, (k, v) in enumerate(vals): A[n_ // K, k] = v
        print(f"    {f:5s} " + " ".join(f"b{k}:{np.nanmean(A[:, k]):.3f}" for k in range(K)))
    np.savez(f".cache/traj_{name}_{os.environ.get('ALG_TEST_NAME', 'x')}.npz", step=st, cos=cs, prog=pr, half=hf)


if __name__ == "__main__":
    _main()
