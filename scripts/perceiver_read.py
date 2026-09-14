"""perceiver_read.py — THE PERCEIVER's first read (2026-09-14, word given,
step 1 of the order): the LoopHealth record on a checkpoint's rows and the
calibration table — which meters are a compass for correctness (AUROC per
breath, bar 0.65) — with THE CHART GAP as the registered hypothesis. A
read: no training, no supervision; gold is used to SCORE, never to steer.
Env: family env + ALG_MINE_BREATHS=1 (set here), PV_CKPT, PV_ATLAS
(.cache/step_atlas_ds242.npz), MH_ATLAS_MANIFEST (.cache/RESEARCH_MANIFEST.json),
ALG_TEST(_NAME), PV_N (512), PV_SOLVE=1 (the solver's verdict on the final
parse, the pool), PV_OUT (npz)."""
import os, sys


def _main():
    """The read. Under the main guard: alternator_bridge.core_rows spawns a pool whose
    workers re-import this module — nothing may run at import (the spawn-storm rule)."""
    import time, numpy as np
    os.environ.setdefault("ALG_MINE_BREATHS", "1")
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, _slot_margins, _decode_slots, L_FAC, K_VARS
    import phase1_algebra_head as H
    from mycelium.step_atlas import load_atlas
    from mycelium.perceiver import LoopHealth, METERS
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    vs, vst, vtk, vg, vse = load_alg("test")
    N = min(int(os.environ.get("PV_N", "512")), len(vs)); rng = np.random.RandomState(0); idx = np.sort(rng.choice(len(vs), N, replace=False))
    p = build_params(0); sd = safe_load(os.environ["PV_CKPT"])
    assert set(sd) == set(p), (sorted(set(sd) - set(p))[:3], sorted(set(p) - set(sd))[:3])
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    atlas = load_atlas(os.environ.get("PV_ATLAS", ".cache/step_atlas_ds242.npz"), manifest_path=os.environ.get("MH_ATLAS_MANIFEST", ".cache/RESEARCH_MANIFEST.json"))
    SOLVE = int(os.environ.get("PV_SOLVE", "1")); TAU = 0.5
    if SOLVE:
        from alternator_bridge import core_rows
    K = None; hl = None; t0 = time.time()
    CENSUS_ORGANS = ("notebook", "garage", "mixer", "maskhead", "alt21_s3", "tokloop")

    def score_rows(onp, sl):
        """happy_family_read's slot scoring: slot_ok (b, L), present (b, L), row_ok (b,)."""
        b_n = len(sl); sok = np.zeros((b_n, L_FAC), bool); pres = np.zeros((b_n, L_FAC), bool); rok = np.zeros(b_n, bool)
        for bi, i in enumerate(sl):
            i = int(i); ok = True; any_slot = False
            for j in range(L_FAC):
                if vg["presence"][i, j] < 0.5: continue
                any_slot = True; pres[bi, j] = True
                s = (onp["pres"][bi, j] > 0) and int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j] and int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
                if vg["ftype"][i, j] == 0:
                    s = s and int(onp["op"][bi, j].argmax()) == vg["op"][i, j]
                    gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                    s = s and ((bool(onp["dup"][bi, j] > 0) and int(np.argmax(onp["args"][bi, j])) in gset) if (len(gset) == 1 and "dup" in onp) else set(np.argsort(-onp["args"][bi, j])[:2].tolist()) == gset)
                else:
                    s = s and bool((onp["dig"][bi, j].argmax(-1) == vg["digits"][i, j]).all())
                sok[bi, j] = s; ok = ok and s
            rok[bi] = ok and any_slot
        return sok, pres, rok

    for s0 in range(0, N, 8):
        sl = idx[s0:s0 + 8]; pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fb = Tensor(alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma), dtype=dtypes.float)
        H._CENSUS = []
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fb)
        br = [b.numpy() for b in o["breaths_all"]]                        # K x (8, L, D)
        nl = [t.numpy() for t in o["nl_all"]]                             # K x (8, D)
        fa = [f.numpy() for f in o["fat_all"]]                            # K x (8, L, T)
        hd = [{k: v.numpy() for k, v in h.items() if k in ("pres", "ftype", "op", "args", "res", "dig", "dup")} for h in o["heads_all"]]
        if K is None:
            K = len(br); hl = LoopHealth(K); print(f"[perceiver-read] K={K} breaths, N={N} rows, ckpt={os.path.basename(os.environ['PV_CKPT'])}", flush=True)
        b_n = len(sl); tkm = vtk[sl_p].astype(bool)
        met = {k: np.full((K, b_n), np.nan) for k in METERS}; cs = np.full((K, b_n), -1); ct = np.full((K, b_n), -1)
        for k in range(K):
            S = br[k][:b_n].mean(1)
            z, c = atlas.radius("slot", k, S); met["z_slot"][k] = z; cs[k] = c
            zt, ctk = atlas.radius("token", k, nl[k][:b_n]); met["z_tok"][k] = zt; ct[k] = ctk
            met["gap"][k] = zt - z
            if k >= 1:
                d = np.linalg.norm(br[k][:b_n] - br[k - 1][:b_n], axis=-1); n0 = np.linalg.norm(br[k - 1][:b_n], axis=-1) + 1e-6
                met["settle"][k] = (d / n0).mean(1)
            for bi in range(b_n):
                row = {kk: hd[k][kk][bi] for kk in hd[k]}
                pres_j = [j for j in range(L_FAC) if row["pres"][j] > 0]
                mm = {"res": [], "op": [], "args": []}
                for j in pres_j:
                    for f, m, _ in _slot_margins(row, j): mm[f].append(m)
                for f in mm: met[f"margin_{f}"][k, bi] = min(mm[f]) if mm[f] else np.nan
                if pres_j:
                    C = (fa[k][bi][pres_j] > TAU)                          # (n_pres, T)
                    real = tkm[bi]
                    met["claim_conflict"][k, bi] = float(((C.sum(0) > 1) & real).sum() / max(real.sum(), 1))
        # the census: per-row injection RMS over the "state" band's RMS, per breath
        cen = {}
        state_rms = {}
        for (kb, organ, arr) in H._CENSUS:
            if organ == "state": state_rms[kb] = np.sqrt((arr.reshape(arr.shape[0], -1) ** 2).mean(1))[:b_n]
        for (kb, organ, arr) in H._CENSUS:
            if organ in CENSUS_ORGANS and kb in state_rms and arr.ndim >= 2:
                r = np.sqrt((arr.reshape(arr.shape[0], -1) ** 2).mean(1))[:b_n] / (state_rms[kb] + 1e-9)
                cen.setdefault(organ, np.full((K, b_n), np.nan))[kb] = r
        H._CENSUS = None
        onp = hd[-1]
        sok, pres, rok = score_rows({k: v[:b_n] for k, v in onp.items()}, sl)
        wheel = None
        if SOLVE:
            rows = [(int(_nv[bi]), _decode_slots({kk: onp[kk][bi] for kk in onp}), int(_ma[bi])) for bi in range(b_n)]
            wheel = core_rows(rows, None)
        hl.add(sl, met, census=cen, cls_slot=cs, cls_tok=ct, wheel=wheel, row_ok=rok, slot_ok=sok, present=pres)
        if (s0 // 8) % 16 == 0: print(f"[perceiver-read] {s0 + b_n}/{N} ({time.time() - t0:.0f}s)", flush=True)

    name = os.path.basename(os.environ["PV_CKPT"]).replace("sharp_", "").replace(".safetensors", "")
    print(f"[perceiver-read] {name} on {os.environ.get('ALG_TEST_NAME', '?')}:")
    tab = hl.report()
    gk = [tab[("gap", k)] for k in range(K)]; zs = [tab[("z_slot", k)] for k in range(K)]; zt = [tab[("z_tok", k)] for k in range(K)]
    print(f"[perceiver-read] THE CHART GAP: best AUROC gap {np.nanmax(gk):.3f} vs z_slot {np.nanmax(zs):.3f} vs z_tok {np.nanmax(zt):.3f}")
    out = os.environ.get("PV_OUT", f".cache/perceiver_{name}_{os.environ.get('ALG_TEST_NAME', 'x')}.npz"); hl.save(out); print(f"[perceiver-read] saved {out}")


if __name__ == "__main__":
    _main()
