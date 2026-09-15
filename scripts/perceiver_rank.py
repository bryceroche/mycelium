"""perceiver_rank.py — THE ANNOTATION QUEUE, second pass (2026-09-15): rank
rows WITHOUT gold by the perceiver's inference-time meters — the least
confident relation slot's args margin (the compass, 0.76 at slot level),
settle at breath 3 (0.73), and a free SYNTACTIC meter: DANGLING VARIABLES,
the count of variables the decoded relations reference that no decoded
given defines (4-9 on the holdout's multiplier rows). With gold present
(the holdout) the meters are CALIBRATED (AUROC vs >= half correct) before
they rank the unannotated pool. Output: JSONL, ranked. A read.
Env: family env + ALG_MINE_BREATHS=1 (set), PR_CKPT, ALG_TEST(_NAME), PR_N, PR_OUT."""
import os, sys


def _main():
    import json, numpy as np
    os.environ.setdefault("ALG_MINE_BREATHS", "1")
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, _slot_margins, _decode_slots, L_FAC, K_VARS
    from mycelium.step_atlas import load_atlas
    from mycelium.perceiver import auroc
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    vs, vst, vtk, vg, vse = load_alg("test"); N = min(int(os.environ.get("PR_N", "100000")), len(vs)); idx = np.arange(N)
    has_gold = bool(vg["presence"][:N].sum() > 0)
    p = build_params(0); sd = safe_load(os.environ["PR_CKPT"]); assert set(sd) == set(p)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    atlas = load_atlas(os.environ.get("PV_ATLAS", ".cache/step_atlas_ds242.npz"), manifest_path=os.environ.get("MH_ATLAS_MANIFEST", ".cache/RESEARCH_MANIFEST.json")); assert atlas.stamp == os.path.basename(os.environ["PR_CKPT"])
    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())
    out = []
    for s0 in range(0, N, 8):
        sl = idx[s0:s0 + 8]; pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32)); _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) or K_VARS for i in sl_p]); _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma), dtype=dtypes.float))
        hd = {k: v.numpy() for k, v in o["heads_all"][-1].items() if k in KEYS}; br = [b.numpy() for b in o["breaths_all"]]; NL = o["nl_all"][-1].numpy()
        S6 = br[-1].mean(1); _, cs = atlas.radius("slot", 6, S6); _, ct = atlas.radius("token", 6, NL)
        for bi, i in enumerate(sl):
            i = int(i); row = {k: hd[k][bi] for k in KEYS}; parse = _decode_slots(row)
            giv = {f["var"] for f in parse if f["ftype"] == "given"}; rels = [f for f in parse if f["ftype"] == "rel"]
            refd = {a for f in rels for a in list(f.get("args", [])) + [f.get("result")] if a is not None}
            dangling = len(refd - giv - {f.get("result") for f in rels})      # referenced, defined by no given and produced by no relation
            margs = [m for j in range(L_FAC) if row["pres"][j] > 0 for f, m, _ in _slot_margins(row, j) if f == "args"]
            settle3 = float((np.linalg.norm(br[3][bi] - br[2][bi], axis=-1) / (np.linalg.norm(br[2][bi], axis=-1) + 1e-6)).mean())
            rec = {"row": i, "text": vs[i]["text"], "n_given": len(giv), "n_rel": len(rels), "dangling": dangling, "min_args_margin": (float(min(margs)) if margs else None),
                   "settle_b3": settle3, "kind_slot": atlas.classes[int(cs[bi])], "kind_token": atlas.classes[int(ct[bi])]}
            if has_gold:
                pres = vg["presence"][i] >= 0.5; ok_n = 0
                for j in np.where(pres)[0]:
                    rel = vg["ftype"][i, j] == 0
                    ok = (row["pres"][j] > 0) and int(row["ftype"][j].argmax()) == vg["ftype"][i, j] and int(row["res"][j].argmax()) == vg["res"][i, j]
                    if rel:
                        ok = ok and int(row["op"][j].argmax()) == vg["op"][i, j]; gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                        ok = ok and ((bool(row["dup"][j] > 0) and int(np.argmax(row["args"][j])) in gset) if (len(gset) == 1 and "dup" in row) else set(np.argsort(-row["args"][j])[:2].tolist()) == gset)
                    else:
                        ok = ok and bool((row["dig"][j].argmax(-1) == vg["digits"][i, j]).all())
                    ok_n += int(ok)
                rec["half_ok"] = bool(ok_n / max(pres.sum(), 1) >= 0.5)
            out.append(rec)
        if s0 % 800 == 0: print(f"[rank] {s0 + len(sl)}/{N}", flush=True)
    name = os.environ.get("ALG_TEST_NAME", "x")
    if has_gold:
        y = np.array([r["half_ok"] for r in out]); d = np.array([r["dangling"] for r in out], float); m = np.array([r["min_args_margin"] if r["min_args_margin"] is not None else np.nan for r in out]); s3 = np.array([r["settle_b3"] for r in out])
        mm = ~np.isnan(m)
        print(f"[rank] CALIBRATION on {name} (n={len(y)}, >= half {y.mean():.3f}): AUROC dangling {auroc(-d, y):.3f} | args margin {auroc(m[mm], y[mm]):.3f} | settle b3 {auroc(-s3, y):.3f}; dangling median ok {np.median(d[y]):.1f} vs wrong {np.median(d[~y]):.1f}")
    out.sort(key=lambda r: (-r["dangling"], r["min_args_margin"] if r["min_args_margin"] is not None else 1e9))
    path = os.environ.get("PR_OUT", f".cache/annotation_rank_{name}.jsonl")
    with open(path, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    import collections
    print(f"[rank] wrote {path}: {len(out)} rows; dangling median {np.median([r['dangling'] for r in out]):.1f}; kinds {collections.Counter(r['kind_slot'] for r in out).most_common(4)}")


if __name__ == "__main__":
    _main()
