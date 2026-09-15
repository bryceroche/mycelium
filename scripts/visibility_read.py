"""visibility_read.py — THE VISIBILITY READ (2026-09-15, word given; the
projection theorem as a meter). Per row, per breath k >= 1, per factor slot,
the state velocity dx = s_k - s_{k-1} (512-d) is decomposed into THREE
orthogonal parts:
  CLOCK    dx on the polar sink's clock planes (rotated by design; invisible
           to the content readout on purpose)
  VISIBLE  the content part projected onto the readout's row space — the
           linear heads' columns (pres/ftype/op/dig/islit/dup/sgn/sel) and,
           per row, the pointer directions W_args v_i / W_res v_i / W_query
           v_i for the row's variable states v_i — what the decode can see
  NULL     the rest of the content part — motion no readout sees
Reports the median share of ||dx||^2 in each part per breath, split by
register and by row correctness (>= half), plus the visible-only progress
toward the final state. THE PREDICTION (pinned): the middle breaths are
mostly clock + null; the visible share is small (< 20%) but not zero.
A read. Env: family env + ALG_MINE_BREATHS=1 (set), PV_CKPT, ALG_TEST(_NAME), PV_N."""
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
    g = H._polar_sink()[2].numpy().reshape(-1).astype(np.float64) if H.ALG_POLAR else np.ones(H.H_W)
    content = g > 0.5; clock = ~content
    heads = [p[k].numpy().astype(np.float64).reshape(H.H_W, -1) for k in ("h_pres", "h_ftype", "h_op", "h_dig", "h_islit", "h_dup", "h_sgn", "h_sel") if k in p]
    Hm = np.concatenate(heads, axis=1)                      # (512, ~95) the linear heads' columns
    ptr = [p[k].numpy().astype(np.float64) for k in ("W_args", "W_res", "W_query") if k in p]
    print(f"[visibility] content planes {int(content.sum())}, clock planes {int(clock.sum())}; linear head columns {Hm.shape[1]}; pointer matrices {len(ptr)}", flush=True)
    K = None; shares = []; prog_vis = []; half = []
    for s0 in range(0, N, 8):
        sl = idx[s0:s0 + 8]; pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst_[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32)); _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma), dtype=dtypes.float))
        br = [b.numpy().astype(np.float64) for b in o["breaths_all"]]; V = o["vst_mine"].numpy().astype(np.float64)   # (8, K_VARS, 512)
        hd = {k: v.numpy() for k, v in o["heads_all"][-1].items() if k in ("pres", "ftype", "op", "dig", "args", "res", "dup")}
        if K is None: K = len(br)
        for bi, i in enumerate(sl):
            i = int(i); nv = int(_nv[bi])
            cols = [Hm] + [W @ V[bi, :nv].T for W in ptr]         # (512, m) each
            M = np.concatenate(cols, axis=1) * content[:, None]     # the readout restricted to the content planes
            Q, _ = np.linalg.qr(M); Q = Q[:, np.abs(np.diag(_)) > 1e-8] if _.shape[0] == _.shape[1] else Q
            pres = vg["presence"][i] >= 0.5; js = np.where(pres)[0]
            if len(js) == 0: continue
            row_sh = []
            for k in range(1, K):
                dx = br[k][bi][js] - br[k - 1][bi][js]              # (n_slots, 512)
                dxc = dx * content; dxk = dx * clock
                vis = (dxc @ Q) @ Q.T; nul = dxc - vis
                tot = (dx ** 2).sum(1) + 1e-12
                row_sh.append([np.median((dxk ** 2).sum(1) / tot), np.median((vis ** 2).sum(1) / tot), np.median((nul ** 2).sum(1) / tot)])
            shares.append(row_sh)
            # visible-only progress toward the final state
            fin = br[-1][bi][js] * content; pv = []
            for k in range(K):
                cur = br[k][bi][js] * content; d = ((cur - fin) @ Q); d0 = ((br[0][bi][js] * content - fin) @ Q)
                pv.append(1 - np.linalg.norm(d) / (np.linalg.norm(d0) + 1e-9))
            prog_vis.append(pv)
            ok_n = 0
            for j in js:
                rel = vg["ftype"][i, j] == 0
                ok = (hd["pres"][bi, j] > 0) and int(hd["ftype"][bi, j].argmax()) == vg["ftype"][i, j] and int(hd["res"][bi, j].argmax()) == vg["res"][i, j]
                if rel:
                    ok = ok and int(hd["op"][bi, j].argmax()) == vg["op"][i, j]; gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                    ok = ok and ((bool(hd["dup"][bi, j] > 0) and int(np.argmax(hd["args"][bi, j])) in gset) if (len(gset) == 1 and "dup" in hd) else set(np.argsort(-hd["args"][bi, j])[:2].tolist()) == gset)
                else:
                    ok = ok and bool((hd["dig"][bi, j].argmax(-1) == vg["digits"][i, j]).all())
                ok_n += int(ok)
            half.append(ok_n / len(js) >= 0.5)
        if s0 % 160 == 0: print(f"[visibility] {s0 + len(sl)}/{N}", flush=True)
    S = np.array(shares); P = np.array(prog_vis); hf = np.array(half)
    name = os.path.basename(os.environ["PV_CKPT"]).replace("sharp_", "").replace(".safetensors", "")
    print(f"[visibility] {name} on {os.environ.get('ALG_TEST_NAME', '?')}: N={len(hf)} rows, >= half {hf.mean():.3f}; share of ||dx||^2 per breath (median over rows):")
    for lab, m in (("all", np.ones_like(hf)), (">= half", hf), ("< half", ~hf)):
        if m.sum() < 5: continue
        for ci, cname in enumerate(("clock", "visible", "null")):
            print(f"  [{lab:7s} n={int(m.sum()):3d}] {cname:8s} " + " ".join(f"b{k + 1}:{np.median(S[m][:, k, ci]):.3f}" for k in range(K - 1)))
        print(f"  [{lab:7s}      ] visible-only progress " + " ".join(f"b{k}:{np.median(P[m][:, k]):.3f}" for k in range(K)))
    np.savez(f".cache/visibility_{name}_{os.environ.get('ALG_TEST_NAME', 'x')}.npz", shares=S, prog_vis=P, half=hf)


if __name__ == "__main__":
    _main()
